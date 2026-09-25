import { W, H, WATER_LEVEL, type Grid } from '../types.ts';
import { readBufferAsync, readTextureAsync } from '../utils/gpu-debug.ts';
import { CAUSTIC_RECEIVERS, type CausticFace, type CausticTextures } from './receivers.ts';
import {
  makeFootprintShader, makeResolveShader, RECEIVER_REGIONS, RECEIVER_PIXEL_COUNT,
  SEAM_DISPATCH_SHADER,
} from './caustic-footprints.ts';

// Gradient -> projected source triangles -> receiver-map resolve.
// Triangle/texel intersections integrate irradiance instead of depositing a
// unit-pixel kernel at each ray center. Source-space subdivision splits seams;
// local source-area / projected-area ratios conserve power without frame scaling.

// Light direction (toward the light, in air). Matches the water/tub fragment
// shaders' hardcoded lightDir = normalize(vec3(0.2, 0.8, 0.3)).
const LIGHT_DIR = (() => {
  const l = [0.2, 0.8, 0.3];
  const m = Math.hypot(...l);
  return l.map((v) => v / m);
})();
const LX = LIGHT_DIR[0], LY = LIGHT_DIR[1], LZ = LIGHT_DIR[2];
// n_air / n_water (eta in the refract formula).
const ETA = 1 / 1.333;

// Central differences inside the grid; one-sided differences at its boundary.
// Reads the height from the state buffer's .y component.
function makeGradientShader(nx: number, nz: number, dx: number, dz: number) {
  return `
@group(0) @binding(0) var<storage, read> state: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> grad: array<vec2f>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let ix = i32(id.x);
  let iz = i32(id.y);
  let nx = ${nx};
  let nz = ${nz};
  if (ix >= nx || iz >= nz) { return; }

  let dx = ${dx.toFixed(6)};
  let dz = ${dz.toFixed(6)};

  // Boundary derivatives use the actual distance between available neighbors.
  let iL = iz * nx + max(0, ix - 1);
  let iR = iz * nx + min(nx - 1, ix + 1);
  let iD = max(0, iz - 1) * nx + ix;
  let iU = min(nz - 1, iz + 1) * nx + ix;

  grad[iz * nx + ix] = vec2f(
    (state[iR].y - state[iL].y) / (f32(min(nx - 1, ix + 1) - max(0, ix - 1)) * dx),
    (state[iU].y - state[iD].y) / (f32(min(nz - 1, iz + 1) - max(0, iz - 1)) * dz),
  );
}
`;
}

// Bilinear sampling in world space and refraction constants for the footprint pass.
function makeCommonWgsl(nx: number, nz: number, dx: number, dz: number) {
  return `
const NX: i32 = ${nx};
const NZ: i32 = ${nz};
const DX: f32 = ${dx.toFixed(6)};
const DZ: f32 = ${dz.toFixed(6)};
const W_HALF: f32 = ${W / 2};
const H_HALF: f32 = ${H / 2};
const WATER_LEVEL: f32 = ${WATER_LEVEL};
const ETA: f32 = ${ETA.toFixed(6)}; // n_air / n_water
const L: vec3f = normalize(vec3f(${LX.toFixed(6)}, ${LY.toFixed(6)}, ${LZ.toFixed(6)})); // toward the light

// Bilinear sample of the height field at world (wx, wz).
fn sampleHeight(wx: f32, wz: f32) -> f32 {
  let fx = (wx + W_HALF) / DX;
  let fz = (wz + H_HALF) / DZ;
  let ix0 = clamp(i32(floor(fx)), 0, NX - 2);
  let iz0 = clamp(i32(floor(fz)), 0, NZ - 2);
  let tx = clamp(fx - f32(ix0), 0.0, 1.0);
  let tz = clamp(fz - f32(iz0), 0.0, 1.0);
  let a = state[ix0 + iz0 * NX].y;
  let b = state[ix0 + 1 + iz0 * NX].y;
  let c = state[ix0 + (iz0 + 1) * NX].y;
  let d = state[ix0 + 1 + (iz0 + 1) * NX].y;
  return mix(mix(a, b, tx), mix(c, d, tx), tz);
}

// Bilinear sample of the gradient field at world (wx, wz).
fn sampleGrad(wx: f32, wz: f32) -> vec2f {
  let fx = (wx + W_HALF) / DX;
  let fz = (wz + H_HALF) / DZ;
  let ix0 = clamp(i32(floor(fx)), 0, NX - 2);
  let iz0 = clamp(i32(floor(fz)), 0, NZ - 2);
  let tx = clamp(fx - f32(ix0), 0.0, 1.0);
  let tz = clamp(fz - f32(iz0), 0.0, 1.0);
  let a = grad[ix0 + iz0 * NX];
  let b = grad[ix0 + 1 + iz0 * NX];
  let c = grad[ix0 + (iz0 + 1) * NX];
  let d = grad[ix0 + 1 + (iz0 + 1) * NX];
  return mix(mix(a, b, tx), mix(c, d, tx), tz);
}

`;
}


export class Caustics {
  private _device: GPUDevice;
  private _grid: Grid;
  private _gradientPipeline: GPUComputePipeline;
  private _gradientBuffer: GPUBuffer;
  private _irradianceBuffer: GPUBuffer;
  private _seamQueue: GPUBuffer;
  private _seamDispatch: GPUBuffer;
  private _causticTextures: CausticTextures;
  private _footprintPipeline: GPUComputePipeline;
  private _seamPipeline: GPUComputePipeline;
  private _dispatchPipeline: GPUComputePipeline;
  private _dispatchBindGroup: GPUBindGroup;
  private _resolvePipelines: GPUComputePipeline[];
  private _resolveBindGroups: GPUBindGroup[];
  private _stateBindings = new Map<GPUBuffer, { gradient: GPUBindGroup, footprint: GPUBindGroup }>();

  constructor(device: GPUDevice, grid: Grid) {
    this._device = device;
    this._grid = grid;
    this._gradientBuffer = device.createBuffer({
      size: grid.NUM_VERTS * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this._irradianceBuffer = device.createBuffer({
      size: RECEIVER_PIXEL_COUNT * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this._seamQueue = device.createBuffer({
      size: (W * H * 2 + 1) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this._seamDispatch = device.createBuffer({
      size: 12,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
    });
    const createMap = (face: CausticFace) => device.createTexture({
      size: [CAUSTIC_RECEIVERS[face].width, CAUSTIC_RECEIVERS[face].height],
      format: 'r32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    this._causticTextures = [createMap(0), createMap(1), createMap(2), createMap(3), createMap(4)];
    this._gradientPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({
          code: makeGradientShader(grid.NX, grid.NZ, grid.DELTA_X, grid.DELTA_Z),
        }),
        entryPoint: 'main',
      },
    });
    const footprintModule = device.createShaderModule({
      code: makeFootprintShader(makeCommonWgsl(grid.NX, grid.NZ, grid.DELTA_X, grid.DELTA_Z)),
    });
    const footprintLayout = device.createPipelineLayout({
      bindGroupLayouts: [device.createBindGroupLayout({
        entries: [0, 1, 2, 3].map(binding => ({
          binding, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: binding < 2 ? 'read-only-storage' as const : 'storage' as const },
        })),
      })],
    });
    this._footprintPipeline = device.createComputePipeline({
      layout: footprintLayout, compute: { module: footprintModule, entryPoint: 'main' },
    });
    this._seamPipeline = device.createComputePipeline({
      layout: footprintLayout, compute: { module: footprintModule, entryPoint: 'seams' },
    });
    this._dispatchPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: SEAM_DISPATCH_SHADER }), entryPoint: 'main' },
    });
    this._dispatchBindGroup = device.createBindGroup({
      layout: this._dispatchPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._seamQueue } },
        { binding: 1, resource: { buffer: this._seamDispatch } },
      ],
    });
    this._resolvePipelines = RECEIVER_REGIONS.map(region => device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: makeResolveShader(region) }), entryPoint: 'main' },
    }));
    this._resolveBindGroups = this._resolvePipelines.map((pipeline, face) => device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._irradianceBuffer } },
        { binding: 1, resource: this._causticTextures[face].createView() },
      ],
    }));
  }

  get grid(): Grid {
    return this._grid;
  }

  get gradientBuffer(): GPUBuffer {
    return this._gradientBuffer;
  }

  get causticTextures(): CausticTextures {
    return this._causticTextures;
  }

  update(encoder: GPUCommandEncoder, stateBuffer: GPUBuffer) {
    let bindings = this._stateBindings.get(stateBuffer);
    if (!bindings) {
      bindings = {
        gradient: this._device.createBindGroup({
          layout: this._gradientPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: stateBuffer } },
            { binding: 1, resource: { buffer: this._gradientBuffer } },
          ],
        }),
        footprint: this._device.createBindGroup({
          layout: this._footprintPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: stateBuffer } },
            { binding: 1, resource: { buffer: this._gradientBuffer } },
            { binding: 2, resource: { buffer: this._irradianceBuffer } },
            { binding: 3, resource: { buffer: this._seamQueue } },
          ],
        }),
      };
      this._stateBindings.set(stateBuffer, bindings);
    }
    encoder.clearBuffer(this._irradianceBuffer);
    encoder.clearBuffer(this._seamQueue, 0, 4);
    let pass = encoder.beginComputePass();
    pass.setPipeline(this._gradientPipeline);
    pass.setBindGroup(0, bindings.gradient);
    pass.dispatchWorkgroups(Math.ceil(this._grid.NX / 8), Math.ceil(this._grid.NZ / 8));
    pass.end();
    pass = encoder.beginComputePass();
    pass.setPipeline(this._footprintPipeline);
    pass.setBindGroup(0, bindings.footprint);
    pass.dispatchWorkgroups(Math.ceil(W * H * 2 / 64));
    pass.end();
    pass = encoder.beginComputePass();
    pass.setPipeline(this._dispatchPipeline);
    pass.setBindGroup(0, this._dispatchBindGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
    pass = encoder.beginComputePass();
    pass.setPipeline(this._seamPipeline);
    pass.setBindGroup(0, bindings.footprint);
    pass.dispatchWorkgroupsIndirect(this._seamDispatch, 0);
    pass.end();
    pass = encoder.beginComputePass();
    for (let face = 0; face < RECEIVER_REGIONS.length; face++) {
      const receiver = RECEIVER_REGIONS[face];
      pass.setPipeline(this._resolvePipelines[face]);
      pass.setBindGroup(0, this._resolveBindGroups[face]);
      pass.dispatchWorkgroups(Math.ceil(receiver.width / 8), Math.ceil(receiver.height / 8));
    }
    pass.end();
  }

  async readGradientAsync(): Promise<Float32Array> {
    return readBufferAsync(this._device, this._gradientBuffer);
  }

  async readCausticAsync(face: CausticFace): Promise<Float32Array> {
    const { width, height } = CAUSTIC_RECEIVERS[face];
    return readTextureAsync(this._device, this._causticTextures[face], width, height);
  }

  dispose() {
    this._stateBindings.clear();
    this._resolveBindGroups.length = 0;
    this._gradientBuffer.destroy();
    this._irradianceBuffer.destroy();
    this._seamQueue.destroy();
    this._seamDispatch.destroy();
    for (const texture of this._causticTextures) texture.destroy();
  }
}

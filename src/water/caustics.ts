import { W, H, WATER_LEVEL, type Grid } from '../types.ts';
import { readBufferAsync } from '../utils/gpu-debug.ts';

// Height-field water caustics (Yuksel & Keyser, "Fast Real-Time Caustics from
// Height Fields", CGI 2009). Backward method: for each caustic-receiving ground
// pixel, sum refracted radiance from a small rectangle on the water surface.
//
// Pipeline (each step verified against a CPU reference):
//   1. gradient pass - central-difference gradient of the height field
//   2. pass 1        - per ground pixel, 7 x-samples -> 7 y-neighbor intensities
//   3. pass 2        - sum the 7 y-neighbor taps -> final caustic map
//
// Storage convention (clean, verified on CPU to match a brute-force 49-sample
// sum to machine precision): pass 1 writes channel `j` (j in -3..3) to the
// neighbor at row +j. Pass 2, for pixel P, sums channel (-d)+3 from the
// neighbor at row +d (d in -3..3). This sidesteps the paper's confusing
// channel-permutation pseudocode.
//
// The 7 channels are packed into two vec4f per caustic pixel:
//   channels0 = (j=-3, j=-2, j=-1, j=0)
//   channels1 = (j=1, j=2, j=3, 0)

// Caustic map: one world unit per pixel, covering the tub bottom (W x H) plus
// a margin so the map's edges (sampled by the tub's vertical sides) fall
// outside the water and read ~0. Centered on the origin (the tub's center).
export const CAUSTIC_MARGIN = 10;
export const CAUSTIC_NX = W + 2 * CAUSTIC_MARGIN;
export const CAUSTIC_NZ = H + 2 * CAUSTIC_MARGIN;

// Light direction (toward the light, in air). Matches the water/floor fragment
// shaders' hardcoded lightDir = normalize(vec3(0.2, 0.8, 0.3)).
const LIGHT_DIR = (() => {
  const l = [0.2, 0.8, 0.3];
  const m = Math.hypot(...l);
  return l.map((v) => v / m);
})();
const LX = LIGHT_DIR[0], LY = LIGHT_DIR[1], LZ = LIGHT_DIR[2];
// n_air / n_water (eta in the refract formula).
const ETA = 1 / 1.333;
// Refracted direction for flat water (normal +y), and its slope (xz/y).
// Precomputed in JS because WGSL module-scope const initializers cannot call
// functions. Matches the CPU reference's refract().
const DFLAT = (() => {
  const I = [-LX, -LY, -LZ];
  const N = [0, 1, 0];
  const cosI = -(I[0] * N[0] + I[1] * N[1] + I[2] * N[2]);
  const k = 1 - ETA * ETA * (1 - cosI * cosI);
  const c = Math.sqrt(Math.max(0, k));
  return [ETA * I[0] + (ETA * cosI - c) * N[0], ETA * I[1] + (ETA * cosI - c) * N[1], ETA * I[2] + (ETA * cosI - c) * N[2]];
})();
const SLOPE_X = DFLAT[0] / DFLAT[1];
const SLOPE_Z = DFLAT[2] / DFLAT[1];

// Central-difference gradient of the height field, with clamped (edge) indices
// so boundary cells get a one-sided gradient instead of reading out of bounds.
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

  // Clamped neighbor indices (edge cells reuse themselves -> one-sided gradient).
  let iL = iz * nx + max(0, ix - 1);
  let iR = iz * nx + min(nx - 1, ix + 1);
  let iD = max(0, iz - 1) * nx + ix;
  let iU = min(nz - 1, iz + 1) * nx + ix;

  grad[iz * nx + ix] = vec2f(
    (state[iR].y - state[iL].y) / (2.0 * dx),
    (state[iU].y - state[iD].y) / (2.0 * dz),
  );
}
`;
}

// Shared WGSL helpers: bilinear sampling of the height/gradient fields (in
// world space) and the light-refraction setup. Inlined into both passes.
function makeCommonWgsl(nx: number, nz: number, dx: number, dz: number) {
  return `
const NX: i32 = ${nx};
const NZ: i32 = ${nz};
const DX: f32 = ${dx.toFixed(6)};
const DZ: f32 = ${dz.toFixed(6)};
const W_HALF: f32 = ${W / 2};
const H_HALF: f32 = ${H / 2};
const WATER_LEVEL: f32 = ${WATER_LEVEL};
const S: f32 = 1.0; // caustic pixel size (world units)
const H0: f32 = WATER_LEVEL; // rest depth above the ground plane (y=0)
const ETA: f32 = ${ETA.toFixed(6)}; // n_air / n_water
const L: vec3f = normalize(vec3f(${LX.toFixed(6)}, ${LY.toFixed(6)}, ${LZ.toFixed(6)})); // toward the light
const DFLAT: vec3f = vec3f(${DFLAT[0].toFixed(6)}, ${DFLAT[1].toFixed(6)}, ${DFLAT[2].toFixed(6)}); // refracted direction for flat water (precomputed)
const SLOPE: vec2f = vec2f(${SLOPE_X.toFixed(6)}, ${SLOPE_Z.toFixed(6)}); // dflat.xz / dflat.y (precomputed)

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

// Pass 1: for each caustic pixel, 7 x-samples -> 7 y-neighbor intensities.
// Writes two vec4f (channels j=-3..0 and j=1..3) per caustic pixel.
function makePass1Shader(nx: number, nz: number, dx: number, dz: number) {
  return `
@group(0) @binding(0) var<storage, read> state: array<vec2f>;
@group(0) @binding(1) var<storage, read> grad: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> channels0: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> channels1: array<vec4f>;
${makeCommonWgsl(nx, nz, dx, dz)}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = i32(id.x);
  let j = i32(id.y);
  if (i >= ${CAUSTIC_NX} || j >= ${CAUSTIC_NZ}) { return; }

  let X = f32(i) - f32(${CAUSTIC_NX / 2});
  let Z = f32(j) - f32(${CAUSTIC_NZ / 2});

  // Illumination center: where a flat surface would focus the light.
  let PC = vec2f(X, Z) + H0 * SLOPE;

  var ch = array<f32, 8>();
  for (var k = -3; k <= 3; k++) {
    let sx = PC.x + f32(k) * S;
    let sz = PC.y;
    let h = sampleHeight(sx, sz);
    let g = sampleGrad(sx, sz);
    let N = normalize(vec3f(-g.x, 1.0, -g.y));
    let d = refract(-L, N, ETA);
    if (d.y >= 0.0) { continue; } // no downward component (total internal reflection)
    let t = (WATER_LEVEL + h) / (-d.y);
    let ix = sx + d.x * t;
    let iz = sz + d.z * t;
    let ax = max(0.0, 1.0 - abs(X - ix) / S);
    for (var jj = -3; jj <= 3; jj++) {
      let rowZ = Z + f32(jj) * S;
      let ay = max(0.0, 1.0 - abs(rowZ - iz) / S);
      ch[jj + 3] += ax * ay;
    }
  }
  channels0[i + j * ${CAUSTIC_NX}] = vec4f(ch[0], ch[1], ch[2], ch[3]);
  channels1[i + j * ${CAUSTIC_NX}] = vec4f(ch[4], ch[5], ch[6], 0.0);
}
`;
}

// Pass 2: for each caustic pixel, sum the 7 y-neighbor taps -> final caustic map.
function makePass2Shader() {
  return `
@group(0) @binding(0) var<storage, read> channels0: array<vec4f>;
@group(0) @binding(1) var<storage, read> channels1: array<vec4f>;
@group(0) @binding(2) var caustic: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = i32(id.x);
  let j = i32(id.y);
  if (i >= ${CAUSTIC_NX} || j >= ${CAUSTIC_NZ}) { return; }

  var val = 0.0;
  for (var d = -3; d <= 3; d++) {
    let row = clamp(j + d, 0, ${CAUSTIC_NZ} - 1);
    let c0 = channels0[i + row * ${CAUSTIC_NX}];
    let c1 = channels1[i + row * ${CAUSTIC_NX}];
    // channel for jj = -d: index (-d)+3 (0..6)
    let idx = (-d) + 3;
    var w = 0.0;
    switch idx {
      case 0: { w = c0.x; }
      case 1: { w = c0.y; }
      case 2: { w = c0.z; }
      case 3: { w = c0.w; }
      case 4: { w = c1.x; }
      case 5: { w = c1.y; }
      case 6: { w = c1.z; }
      default: {}
    }
    val += w;
  }
  textureStore(caustic, vec2i(i, j), vec4f(val));
}
`;
}

export class Caustics {
  private _device: GPUDevice;
  private _grid: Grid;
  private _workgroupX: number;
  private _workgroupZ: number;
  private _gradientPipeline: GPUComputePipeline;
  private _gradientBuffer: GPUBuffer;
  private _gradientBindGroup: GPUBindGroup | null = null;
  private _stateBuffer: GPUBuffer | null = null;
  private _channel0Buffer: GPUBuffer;
  private _channel1Buffer: GPUBuffer;
  private _causticTexture: GPUTexture;
  private _pass1Pipeline: GPUComputePipeline;
  private _pass2Pipeline: GPUComputePipeline;
  private _pass1BindGroup: GPUBindGroup | null = null;
  private _pass2BindGroup: GPUBindGroup;

  constructor(device: GPUDevice, grid: Grid) {
    this._device = device;
    this._grid = grid;
    this._workgroupX = Math.ceil(grid.NX / 8);
    this._workgroupZ = Math.ceil(grid.NZ / 8);

    this._gradientBuffer = device.createBuffer({
      size: grid.NUM_VERTS * 8, // vec2f (dh/dx, dh/dz) per sim cell
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this._gradientPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({
          code: makeGradientShader(grid.NX, grid.NZ, grid.DELTA_X, grid.DELTA_Z),
        }),
        entryPoint: 'main',
      },
    });

    const cn = CAUSTIC_NX * CAUSTIC_NZ;
    this._channel0Buffer = device.createBuffer({
      size: cn * 16, // vec4f per caustic pixel (channels j=-3..0)
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this._channel1Buffer = device.createBuffer({
      size: cn * 16, // vec4f per caustic pixel (channels j=1..3)
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this._causticTexture = device.createTexture({
      size: { width: CAUSTIC_NX, height: CAUSTIC_NZ },
      format: 'r32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    this._pass1Pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({
          code: makePass1Shader(grid.NX, grid.NZ, grid.DELTA_X, grid.DELTA_Z),
        }),
        entryPoint: 'main',
      },
    });
    this._pass2Pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({
          code: makePass2Shader(),
        }),
        entryPoint: 'main',
      },
    });
    this._pass2BindGroup = this._device.createBindGroup({
      layout: this._pass2Pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._channel0Buffer } },
        { binding: 1, resource: { buffer: this._channel1Buffer } },
        { binding: 2, resource: this._causticTexture.createView() },
      ],
    });
  }

  get grid(): Grid {
    return this._grid;
  }

  get gradientBuffer(): GPUBuffer {
    return this._gradientBuffer;
  }

  get causticTexture(): GPUTexture {
    return this._causticTexture;
  }

  // Run the full pipeline: gradient -> pass 1 -> pass 2. The bind groups are
  // rebuilt only when the state buffer changes (it ping-pongs).
  update(encoder: GPUCommandEncoder, stateBuffer: GPUBuffer) {
    if (stateBuffer !== this._stateBuffer) {
      this._stateBuffer = stateBuffer;
      this._gradientBindGroup = this._device.createBindGroup({
        layout: this._gradientPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: stateBuffer } },
          { binding: 1, resource: { buffer: this._gradientBuffer } },
        ],
      });
      this._pass1BindGroup = this._device.createBindGroup({
        layout: this._pass1Pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: stateBuffer } },
          { binding: 1, resource: { buffer: this._gradientBuffer } },
          { binding: 2, resource: { buffer: this._channel0Buffer } },
          { binding: 3, resource: { buffer: this._channel1Buffer } },
        ],
      });
    }
    // Pass 1: gradient (central differences)
    let pass = encoder.beginComputePass();
    pass.setPipeline(this._gradientPipeline);
    pass.setBindGroup(0, this._gradientBindGroup!);
    pass.dispatchWorkgroups(this._workgroupX, this._workgroupZ);
    pass.end();
    // Pass 2: 7 x-samples -> 7 y-neighbor channels
    pass = encoder.beginComputePass();
    pass.setPipeline(this._pass1Pipeline);
    pass.setBindGroup(0, this._pass1BindGroup!);
    pass.dispatchWorkgroups(Math.ceil(CAUSTIC_NX / 8), Math.ceil(CAUSTIC_NZ / 8));
    pass.end();
    // Pass 3: sum 7 y-neighbor taps -> caustic map
    pass = encoder.beginComputePass();
    pass.setPipeline(this._pass2Pipeline);
    pass.setBindGroup(0, this._pass2BindGroup);
    pass.dispatchWorkgroups(Math.ceil(CAUSTIC_NX / 8), Math.ceil(CAUSTIC_NZ / 8));
    pass.end();
  }

  // Read the gradient buffer back to the CPU (for verification).
  async readGradientAsync(): Promise<Float32Array> {
    return readBufferAsync(this._device, this._gradientBuffer);
  }

  // Read the caustic map back to the CPU (for verification).
  async readCausticAsync(): Promise<Float32Array> {
    const dev = this._device;
    const buf = dev.createBuffer({
      size: CAUSTIC_NX * CAUSTIC_NZ * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = dev.createCommandEncoder();
    enc.copyTextureToBuffer(
      { texture: this._causticTexture },
      { buffer: buf },
      { width: CAUSTIC_NX, height: CAUSTIC_NZ },
    );
    dev.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(buf.getMappedRange().slice(0));
    buf.unmap();
    buf.destroy();
    return out;
  }

  dispose() {
    this._gradientBuffer.destroy();
    this._channel0Buffer.destroy();
    this._channel1Buffer.destroy();
    this._causticTexture.destroy();
  }
}

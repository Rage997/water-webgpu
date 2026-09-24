import { W, H, WATER_LEVEL, type Grid } from '../types.ts';
import { readBufferAsync, readTextureAsync } from '../utils/gpu-debug.ts';

// Yuksel & Keyser's unit-pixel footprint overlap model, with tiled receiver
// gathers instead of a fixed 7x7 search window. Strong waves can move light
// well outside that window. Bin actual ray landings, then sum all footprints
// touching each receiver; no radius cutoff or per-frame energy normalization.
//
// Gradient -> refract/bin -> receiver gather. Each footprint touches at most
// four tiles, so four linked-list nodes per source sample suffice without
// overflow, a global allocator, or floating-point atomics.

// One world unit per caustic texel, centered on the tub's floor (Y=0).
// The margin is receiver coverage, not a substitute for wall caustics.
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

const TILE_SIZE = 8;
const TILES_X = Math.ceil(CAUSTIC_NX / TILE_SIZE);
const TILES_Z = Math.ceil(CAUSTIC_NZ / TILE_SIZE);
// Preserve the paper's illumination-center lattice, but cover the entire
// finite water surface rather than the receiver-centered search rectangle.
const OFFSET_X = WATER_LEVEL * Number(SLOPE_X.toFixed(6));
const OFFSET_Z = WATER_LEVEL * Number(SLOPE_Z.toFixed(6));
const SOURCE_X = Math.floor(-W / 2 - 0.5 - OFFSET_X) + 1;
const SOURCE_Z = Math.floor(-H / 2 - 0.5 - OFFSET_Z) + 1;
const SOURCE_NX = Math.ceil(W / 2 + 0.5 - OFFSET_X) - SOURCE_X;
const SOURCE_NZ = Math.ceil(H / 2 + 0.5 - OFFSET_Z) - SOURCE_Z;

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

// Bilinear sampling in world space and refraction constants for the bin pass.
function makeCommonWgsl(nx: number, nz: number, dx: number, dz: number) {
  return `
const NX: i32 = ${nx};
const NZ: i32 = ${nz};
const DX: f32 = ${dx.toFixed(6)};
const DZ: f32 = ${dz.toFixed(6)};
const W_HALF: f32 = ${W / 2};
const H_HALF: f32 = ${H / 2};
const WATER_LEVEL: f32 = ${WATER_LEVEL};
const H0: f32 = WATER_LEVEL; // rest depth above the ground plane (y=0)
const ETA: f32 = ${ETA.toFixed(6)}; // n_air / n_water
const L: vec3f = normalize(vec3f(${LX.toFixed(6)}, ${LY.toFixed(6)}, ${LZ.toFixed(6)})); // toward the light
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

function makeBinShader(nx: number, nz: number, dx: number, dz: number) {
  return `
struct Link { sample: u32, next: u32 };
@group(0) @binding(0) var<storage, read> state: array<vec2f>;
@group(0) @binding(1) var<storage, read> grad: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> landings: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> heads: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> links: array<Link>;
${makeCommonWgsl(nx, nz, dx, dz)}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= ${SOURCE_NX}u || id.y >= ${SOURCE_NZ}u) { return; }
  let sampleIndex = id.x + id.y * ${SOURCE_NX}u;
  let source = vec2f(f32(id.x) + ${SOURCE_X}.0, f32(id.y) + ${SOURCE_Z}.0) + H0 * SLOPE;
  // Boundary patches carry only the portion of their area inside the water.
  let extent = max(vec2f(0.0),
    min(source + vec2f(0.5), vec2f(W_HALF, H_HALF)) -
    max(source - vec2f(0.5), -vec2f(W_HALF, H_HALF)));
  let weight = extent.x * extent.y;
  if (weight <= 0.0) { return; }
  let h = sampleHeight(source.x, source.y);
  let g = sampleGrad(source.x, source.y);
  let normal = normalize(vec3f(-g.x, 1.0, -g.y));
  let direction = refract(-L, normal, ETA);
  if (direction.y >= 0.0 || WATER_LEVEL + h <= 0.0) { return; }
  let distance = (WATER_LEVEL + h) / -direction.y;
  let q = source + direction.xz * distance + vec2f(${CAUSTIC_NX / 2}.0, ${CAUSTIC_NZ / 2}.0);
  // A unit footprint contributes only to floor(q) and floor(q)+1 per axis.
  // Reject off-map rays before converting potentially distant coordinates.
  if (q.x <= -1.0 || q.y <= -1.0 || q.x >= ${CAUSTIC_NX}.0 || q.y >= ${CAUSTIC_NZ}.0) { return; }
  landings[sampleIndex] = vec4f(q, weight, 0.0);
  let pixel = vec2i(floor(q));
  let firstTile = max(pixel, vec2i(0)) / ${TILE_SIZE};
  let lastTile = min(pixel + vec2i(1), vec2i(${CAUSTIC_NX - 1}, ${CAUSTIC_NZ - 1})) / ${TILE_SIZE};
  var slot = 0u;
  for (var ty = firstTile.y; ty <= lastTile.y; ty++) {
    for (var tx = firstTile.x; tx <= lastTile.x; tx++) {
      let node = sampleIndex * 4u + slot;
      let previous = atomicExchange(&heads[u32(tx + ty * ${TILES_X})], node + 1u);
      links[node] = Link(sampleIndex, previous);
      slot++;
    }
  }
}
`;
}

function makeGatherShader() {
  return `
struct Link { sample: u32, next: u32 };
@group(0) @binding(0) var<storage, read> landings: array<vec4f>;
@group(0) @binding(1) var<storage, read> heads: array<u32>;
@group(0) @binding(2) var<storage, read> links: array<Link>;
@group(0) @binding(3) var caustic: texture_storage_2d<r32float, write>;

@compute @workgroup_size(${TILE_SIZE}, ${TILE_SIZE})
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= ${CAUSTIC_NX}u || id.y >= ${CAUSTIC_NZ}u) { return; }
  let tile = id.xy / ${TILE_SIZE}u;
  var entry = heads[tile.x + tile.y * ${TILES_X}u];
  var intensity = 0.0;
  // Binning and gathering are separate passes: all links/landings are visible.
  // Each sample appears at most once in a tile's list.
  while (entry != 0u) {
    let link = links[entry - 1u];
    let landing = landings[link.sample];
    let overlap = max(vec2f(0.0), vec2f(1.0) - abs(vec2f(id.xy) - landing.xy));
    intensity += overlap.x * overlap.y * landing.z;
    entry = link.next;
  }
  textureStore(caustic, vec2i(id.xy), vec4f(intensity));
}
`;
}

export class Caustics {
  private _device: GPUDevice;
  private _grid: Grid;
  private _gradientPipeline: GPUComputePipeline;
  private _gradientBuffer: GPUBuffer;
  private _landingBuffer: GPUBuffer;
  private _headBuffer: GPUBuffer;
  private _linkBuffer: GPUBuffer;
  private _causticTexture: GPUTexture;
  private _binPipeline: GPUComputePipeline;
  private _gatherPipeline: GPUComputePipeline;
  private _gatherBindGroup: GPUBindGroup;
  private _stateBindings = new Map<GPUBuffer, { gradient: GPUBindGroup; bin: GPUBindGroup }>();

  constructor(device: GPUDevice, grid: Grid) {
    this._device = device;
    this._grid = grid;
    this._gradientBuffer = device.createBuffer({
      size: grid.NUM_VERTS * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this._landingBuffer = device.createBuffer({
      size: SOURCE_NX * SOURCE_NZ * 16,
      usage: GPUBufferUsage.STORAGE,
    });
    this._headBuffer = device.createBuffer({
      size: TILES_X * TILES_Z * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this._linkBuffer = device.createBuffer({
      size: SOURCE_NX * SOURCE_NZ * 4 * 8,
      usage: GPUBufferUsage.STORAGE,
    });
    this._causticTexture = device.createTexture({
      size: { width: CAUSTIC_NX, height: CAUSTIC_NZ },
      format: 'r32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
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
    this._binPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({
          code: makeBinShader(grid.NX, grid.NZ, grid.DELTA_X, grid.DELTA_Z),
        }),
        entryPoint: 'main',
      },
    });
    this._gatherPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: makeGatherShader() }),
        entryPoint: 'main',
      },
    });
    this._gatherBindGroup = device.createBindGroup({
      layout: this._gatherPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._landingBuffer } },
        { binding: 1, resource: { buffer: this._headBuffer } },
        { binding: 2, resource: { buffer: this._linkBuffer } },
        { binding: 3, resource: this._causticTexture.createView() },
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
        bin: this._device.createBindGroup({
          layout: this._binPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: stateBuffer } },
            { binding: 1, resource: { buffer: this._gradientBuffer } },
            { binding: 2, resource: { buffer: this._landingBuffer } },
            { binding: 3, resource: { buffer: this._headBuffer } },
            { binding: 4, resource: { buffer: this._linkBuffer } },
          ],
        }),
      };
      this._stateBindings.set(stateBuffer, bindings);
    }
    // Zero is the empty-list sentinel; stale nodes become unreachable.
    encoder.clearBuffer(this._headBuffer);
    let pass = encoder.beginComputePass();
    pass.setPipeline(this._gradientPipeline);
    pass.setBindGroup(0, bindings.gradient);
    pass.dispatchWorkgroups(Math.ceil(this._grid.NX / 8), Math.ceil(this._grid.NZ / 8));
    pass.end();
    pass = encoder.beginComputePass();
    pass.setPipeline(this._binPipeline);
    pass.setBindGroup(0, bindings.bin);
    pass.dispatchWorkgroups(Math.ceil(SOURCE_NX / 8), Math.ceil(SOURCE_NZ / 8));
    pass.end();
    pass = encoder.beginComputePass();
    pass.setPipeline(this._gatherPipeline);
    pass.setBindGroup(0, this._gatherBindGroup);
    pass.dispatchWorkgroups(TILES_X, TILES_Z);
    pass.end();
  }

  async readGradientAsync(): Promise<Float32Array> {
    return readBufferAsync(this._device, this._gradientBuffer);
  }

  async readCausticAsync(): Promise<Float32Array> {
    return readTextureAsync(this._device, this._causticTexture, CAUSTIC_NX, CAUSTIC_NZ);
  }

  dispose() {
    this._stateBindings.clear();
    this._gradientBuffer.destroy();
    this._landingBuffer.destroy();
    this._headBuffer.destroy();
    this._linkBuffer.destroy();
    this._causticTexture.destroy();
  }
}

import { C2, DAMPING, SIM_SPEED, MAX_ITERATED_DT, MAX_Y, SIGMA, W, H, type Grid } from '../types.ts';
import type { ClickData } from '../types.ts';

// Max drag rate the damping slider (0..1) maps to; keeps waves from over-damping.
const MAX_GAMMA = 0.05;
// CFL safety factor (<1). Caps dt so the explicit wave term stays stable.
const CFL_SAFETY = 0.5;

const COMPUTE_SHADER = `
@group(0) @binding(0) var<storage, read> stateA: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> stateB: array<vec2f>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
  dt: f32,
  c2: f32,
  damping: f32,
  nx: f32,
  nz: f32,
  delta_x2: f32,
  delta_z2: f32,
  _pad: f32,
};

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let ix = i32(id.x);
  let iz = i32(id.y);
  let nx = i32(params.nx);
  let nz = i32(params.nz);

  if (ix >= nx || iz >= nz) { return; }

  let i = iz * nx + ix;

  if (ix == 0 || ix == nx - 1 || iz == 0 || iz == nz - 1) {
    stateB[i] = vec2f(stateA[i].y, 0.0);
    return;
  }

  let prev = stateA[i].x;
  let cur = stateA[i].y;

  let left = stateA[i - 1].y;
  let right = stateA[i + 1].y;
  let down = stateA[i - nx].y;
  let up = stateA[i + nx].y;

  let d2x = (right - 2.0 * cur + left) / params.delta_x2;
  let d2z = (up - 2.0 * cur + down) / params.delta_z2;

  let vel = (cur - prev) / params.dt;
  let accel = params.c2 * (d2x + d2z);
  // Semi-implicit (backward-Euler) drag: velocity retention = 1/(1 + damping*dt),
  // always in (0,1] for any damping >= 0 -> unconditionally stable. The old explicit
  // form (1 - damping*dt) went negative and exploded once damping*dt > 2.
  let uy = (vel + accel * params.dt) / (1.0 + params.damping * params.dt);
  let nextY = cur + uy * params.dt;

  stateB[i] = vec2f(cur, nextY);
}
`;

// Ambient wave shader: continuously injects small random ripples
function makeAmbientWaveShader(nx: number, nz: number, w: number, h: number) {
  const dx = w / (nx - 1);
  const dz = h / (nz - 1);
  return `
@group(0) @binding(0) var<storage, read_write> state: array<vec2f>;
@group(0) @binding(1) var<uniform> ambient: AmbientParams;

struct AmbientParams {
  time: f32,          // Current time in seconds for randomness
  frequency: f32,     // How often to spawn ripples (0-1, where 1 = very frequent)
  strength: f32,      // Amplitude of ambient ripples (typically 1-5)
  sigma: f32,         // Gaussian spread (reuse SIGMA constant)
};

// Hash-based pseudo-random function (returns 0-1)
fn random(seed: vec2f) -> f32 {
  return fract(sin(dot(seed, vec2f(12.9898, 78.233))) * 43758.5453);
}

// Gaussian disturbance function
fn gaussian2D(dx: f32, dz: f32, sigma: f32) -> f32 {
  let r2 = dx * dx + dz * dz;
  return exp(-r2 / (2.0 * sigma * sigma));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let ix = i32(id.x);
  let iz = i32(id.y);
  let nx = ${nx};
  let nz = ${nz};

  if (ix >= nx || iz >= nz) { return; }

  let i = iz * nx + ix;
  
  // Skip if no ambient waves enabled
  if (ambient.strength <= 0.0 || ambient.frequency <= 0.0) {
    return;
  }
  
  // Generate multiple potential ripple centers based on time
  // Use frequency to determine how many ripples to check
  let numRipples = i32(ambient.frequency * 5.0) + 1;
  
  var totalDisplacement = 0.0;
  
  for (var r = 0; r < numRipples; r++) {
    // Generate pseudo-random ripple position using time and ripple index
    let seed = vec2f(ambient.time * 0.7 + f32(r) * 123.456, f32(r) * 78.9);
    let randX = random(seed);
    let randZ = random(seed + vec2f(1.0, 0.0));
    
    // Map random position to grid coordinates
    let rippleX = randX * ${w.toFixed(1)};
    let rippleZ = randZ * ${h.toFixed(1)};
    
    // Current vertex world position
    let dx = ${dx.toFixed(6)};
    let dz = ${dz.toFixed(6)};
    let vertX = f32(ix) * dx;
    let vertZ = f32(iz) * dz;
    
    // Distance from this vertex to the ripple center
    let distX = vertX - rippleX;
    let distZ = vertZ - rippleZ;
    
    // Add Gaussian displacement (much smaller than click ripples)
    let displacement = gaussian2D(distX, distZ, ambient.sigma) * ambient.strength;
    
    // Fade in/out over time for each ripple (so they don't last forever)
    let rippleAge = fract(ambient.time * 0.3 + f32(r) * 0.17);
    let fade = sin(rippleAge * 3.14159) * 0.5; // Pulse each ripple
    
    totalDisplacement += displacement * fade;
  }
  
  // Apply the ambient displacement to the current height
  let current = state[i];
  state[i] = vec2f(current.x, current.y + totalDisplacement);
}
`;
}

function makeClickShader(nx: number, nz: number, dx: number, dz: number) {
  // The grid is centered at world origin, spanning [-W/2, W/2] in X and [-H/2, H/2] in Z.
  // Grid vertex (ix, iz) maps to world space as:
  //   wx = -W/2 + ix * dx
  //   wz = -H/2 + iz * dz
  return `
@group(0) @binding(0) var<storage, read_write> state: array<vec2f>;
@group(0) @binding(1) var<uniform> click: ClickParams;

struct ClickParams {
  active_: f32,
  x: f32,
  z: f32,
  max_y: f32,
  sigma: f32,
};

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (click.active_ == 0.0) { return; }

  let ix = i32(id.x);
  let iz = i32(id.y);
  let nx = ${nx}u;
  let nz = ${nz}u;

  if (ix >= i32(nx) || iz >= i32(nz)) { return; }

  let i = iz * i32(nx) + ix;

  // Map grid coordinates to world space (grid is centered at origin)
  let wx = ${(-dx * (nx - 1) / 2).toFixed(6)} + f32(ix) * ${dx.toFixed(6)};
  let wz = ${(-dz * (nz - 1) / 2).toFixed(6)} + f32(iz) * ${dz.toFixed(6)};

  let dx = wx - click.x;
  let dz = wz - click.z;
  let bump = click.max_y * exp(-click.sigma * (dx * dx + dz * dz));

  state[i].x += bump;
  state[i].y += bump;
}
`;
}

export class WaterCompute {
  pipeline!: GPUComputePipeline;
  clickPipeline!: GPUComputePipeline;
  ambientPipeline!: GPUComputePipeline;
  stateA!: GPUBuffer;
  stateB!: GPUBuffer;
  paramsBuffer!: GPUBuffer;
  clickBuffer!: GPUBuffer;
  ambientBuffer!: GPUBuffer;
  bindGroup!: GPUBindGroup;
  clickBindGroup!: GPUBindGroup;
  ambientBindGroup!: GPUBindGroup;
  paramsArray: Float32Array;
  clickArray: Float32Array;
  ambientArray: Float32Array;
  workgroupX: number;
  workgroupZ: number;

  // Public parameters for UI control (user-facing; mapped to physics in simulate()).
  waveSpeed = 1.0;          // propagation speed multiplier (c2 = baseC2 * waveSpeed^2)
  damping = 0.2;            // 0..1 wave decay (mapped to a stable drag rate)
  rippleAmplitude = MAX_Y;  // click splash height (world units)
  rippleSize = 1.0;         // click splash width multiplier
  ambientFrequency = 0.3;   // 0-1, how often ambient ripples spawn
  ambientStrength = 2.0;    // Amplitude of ambient ripples

  private _device: GPUDevice;
  private _useA = true;
  // Base c2 value from constants
  private _baseC2 = C2;
  // 1/dx^2 + 1/dz^2, precomputed for the CFL dt clamp.
  private _invSpaceSum = 0;

  constructor(device: GPUDevice, grid: Grid) {
    this._device = device;

    this.workgroupX = Math.ceil(grid.NX / 8);
    this.workgroupZ = Math.ceil(grid.NZ / 8);
    this._invSpaceSum = 1 / grid.DELTA_X2 + 1 / grid.DELTA_Z2;

    this.paramsArray = new Float32Array([0, C2, DAMPING, grid.NX, grid.NZ, grid.DELTA_X2, grid.DELTA_Z2, 0]);
    this.clickArray = new Float32Array([0, 0, 0, MAX_Y, SIGMA]);
    this.ambientArray = new Float32Array([0, 0.3, 2.0, SIGMA * 400]); // time, frequency, strength, sigma

    this.stateA = device.createBuffer({
      size: grid.NUM_VERTS * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this.stateA.getMappedRange()).fill(0);
    this.stateA.unmap();

    this.stateB = device.createBuffer({
      size: grid.NUM_VERTS * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this.stateB.getMappedRange()).fill(0);
    this.stateB.unmap();

    this.paramsBuffer = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.clickBuffer = device.createBuffer({
      size: 24,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.ambientBuffer = device.createBuffer({
      size: 16, // 4 floats: time, frequency, strength, sigma
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: COMPUTE_SHADER }),
        entryPoint: 'main',
      },
    });

    this.clickPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: makeClickShader(grid.NX, grid.NZ, grid.DELTA_X, grid.DELTA_Z) }),
        entryPoint: 'main',
      },
    });

    this.ambientPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: makeAmbientWaveShader(grid.NX, grid.NZ, W, H) }),
        entryPoint: 'main',
      },
    });

    this._rebind();
  }

  get currentStateBuffer(): GPUBuffer {
    return this._useA ? this.stateA : this.stateB;
  }

  dispose() {
    this.stateA.destroy();
    this.stateB.destroy();
    this.paramsBuffer.destroy();
    this.clickBuffer.destroy();
    this.ambientBuffer.destroy();
  }

  simulate(commandEncoder: GPUCommandEncoder, rawDtMs: number, click: ClickData | null, elapsedTime: number) {
    // Map user-facing controls to physical params.
    const c2 = this._baseC2 * this.waveSpeed * this.waveSpeed;
    const gamma = MAX_GAMMA * this.damping * this.damping; // quadratic: fine low-end control
    const sigma = SIGMA / this.rippleSize;

    // Clamp dt to the CFL-stable limit for the current wave speed (plus an absolute
    // ceiling) so long frames / high wave speeds slow down instead of exploding.
    const dtCflMax = Math.sqrt(CFL_SAFETY / (c2 * this._invSpaceSum));
    let dt = rawDtMs * SIM_SPEED;
    if (dt > dtCflMax) dt = dtCflMax;
    if (dt > MAX_ITERATED_DT) dt = MAX_ITERATED_DT;

    if (click?.active) {
      this.clickArray[0] = 1;
      this.clickArray[1] = click.x;
      this.clickArray[2] = click.z;
      this.clickArray[3] = this.rippleAmplitude;
      this.clickArray[4] = sigma;
      this._device.queue.writeBuffer(this.clickBuffer, 0, this.clickArray);

      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.clickPipeline);
      pass.setBindGroup(0, this.clickBindGroup);
      pass.dispatchWorkgroups(this.workgroupX, this.workgroupZ);
      pass.end();

      this.clickArray[0] = 0;
    }

    // Apply ambient waves before the main simulation
    this.ambientArray[0] = elapsedTime;
    this.ambientArray[1] = this.ambientFrequency;
    this.ambientArray[2] = this.ambientStrength;
    this._device.queue.writeBuffer(this.ambientBuffer, 0, this.ambientArray);

    if (this.ambientStrength > 0 && this.ambientFrequency > 0) {
      const ambientPass = commandEncoder.beginComputePass();
      ambientPass.setPipeline(this.ambientPipeline);
      ambientPass.setBindGroup(0, this.ambientBindGroup);
      ambientPass.dispatchWorkgroups(this.workgroupX, this.workgroupZ);
      ambientPass.end();
    }

    // Upload physical params for this frame.
    this.paramsArray[0] = dt;
    this.paramsArray[1] = c2;
    this.paramsArray[2] = gamma;
    this._device.queue.writeBuffer(this.paramsBuffer, 0, this.paramsArray);

    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(this.workgroupX, this.workgroupZ);
    pass.end();

    this._swap();
  }

  private _swap() {
    this._useA = !this._useA;
    this._rebind();
  }

  private _rebind() {
    this.bindGroup = this._device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._useA ? this.stateA : this.stateB } },
        { binding: 1, resource: { buffer: this._useA ? this.stateB : this.stateA } },
        { binding: 2, resource: { buffer: this.paramsBuffer } },
      ],
    });

    this.clickBindGroup = this._device.createBindGroup({
      layout: this.clickPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._useA ? this.stateA : this.stateB } },
        { binding: 1, resource: { buffer: this.clickBuffer } },
      ],
    });

    this.ambientBindGroup = this._device.createBindGroup({
      layout: this.ambientPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._useA ? this.stateA : this.stateB } },
        { binding: 1, resource: { buffer: this.ambientBuffer } },
      ],
    });
  }
}

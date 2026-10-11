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

  if (ix <= 0 || iz <= 0 || ix >= i32(nx) - 1 || iz >= i32(nz) - 1) { return; }

  let i = iz * i32(nx) + ix;

  // Map grid coordinates to world space (grid is centered at origin)
  let wx = ${(-dx * (nx - 1) / 2).toFixed(6)} + f32(ix) * ${dx.toFixed(6)};
  let wz = ${(-dz * (nz - 1) / 2).toFixed(6)} + f32(iz) * ${dz.toFixed(6)};

  let dx = wx - click.x;
  let dz = wz - click.z;
  let radius2 = dx * dx + dz * dz;
  if (click.sigma * radius2 > 9.0) { return; }
  // A shallow pressure brush, not an additive splash on every pointer event.
  // Repeated samples at the same location cannot stack height indefinitely.
  let targetHeight = -0.2 * click.max_y * exp(-click.sigma * radius2);
  let displacement = min(0.0, targetHeight - state[i].y);
  // Move both time levels equally to preserve the existing wave velocity.
  state[i].x += displacement;
  state[i].y += displacement;
}
`;
}

export class WaterCompute {
  pipeline!: GPUComputePipeline;
  clickPipeline!: GPUComputePipeline;
  stateA!: GPUBuffer;
  stateB!: GPUBuffer;
  paramsBuffer!: GPUBuffer;
  clickBuffer!: GPUBuffer;
  bindGroup!: GPUBindGroup;
  clickBindGroup!: GPUBindGroup;
  paramsArray: Float32Array;
  clickArray: Float32Array;
  workgroupX: number;
  workgroupZ: number;

  // Public parameters for UI control (user-facing; mapped to physics in simulate()).
  waveSpeed = 1.0;          // propagation speed multiplier (c2 = baseC2 * waveSpeed^2)
  damping = 0.2;            // 0..1 wave decay (mapped to a stable drag rate)
  rippleAmplitude = MAX_Y;  // pointer brush strength; depth = 0.2 * amplitude
  rippleSize = 2.0;         // pointer brush width multiplier

  private _device: GPUDevice;
  private _grid: Grid;
  private _useA = true;
  paused = false;             // when true, simulate() is skipped (debug/verification)
  // Base c2 value from constants
  private _baseC2 = C2;
  // 1/dx^2 + 1/dz^2, precomputed for the CFL dt clamp.
  private _invSpaceSum = 0;

  constructor(device: GPUDevice, grid: Grid) {
    this._device = device;
    this._grid = grid;

    this.workgroupX = Math.ceil(grid.NX / 8);
    this.workgroupZ = Math.ceil(grid.NZ / 8);
    this._invSpaceSum = 1 / grid.DELTA_X2 + 1 / grid.DELTA_Z2;

    this.paramsArray = new Float32Array([0, C2, DAMPING, grid.NX, grid.NZ, grid.DELTA_X2, grid.DELTA_Z2, 0]);
    this.clickArray = new Float32Array([0, 0, 0, MAX_Y, SIGMA]);

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

    this._rebind();
  }

  get currentStateBuffer(): GPUBuffer {
    return this._useA ? this.stateA : this.stateB;
  }

  get grid(): Grid {
    return this._grid;
  }

  // Zero both ping-pong state buffers (flat water). For verification.
  zeroState() {
    const zeros = new Float32Array(this._grid.NUM_VERTS * 2);
    this._device.queue.writeBuffer(this.stateA, 0, zeros);
    this._device.queue.writeBuffer(this.stateB, 0, zeros);
  }

  // Inject an analytic Gaussian height field into the current state buffer,
  // using h = amp * exp(-sigma * r^2).
  // Both prev and cur are set so the field is static (no velocity).
  // For verification of the gradient pass.
  injectGaussian(x: number, z: number, amp: number, sigma: number) {
    const g = this._grid;
    const data = new Float32Array(g.NUM_VERTS * 2);
    for (let iz = 0; iz < g.NZ; iz++) {
      for (let ix = 0; ix < g.NX; ix++) {
        const wx = -W / 2 + ix * g.DELTA_X;
        const wz = -H / 2 + iz * g.DELTA_Z;
        const dx = wx - x;
        const dz = wz - z;
        const h = amp * Math.exp(-sigma * (dx * dx + dz * dz));
        const i = (iz * g.NX + ix) * 2;
        data[i] = h;     // prev
        data[i + 1] = h; // cur
      }
    }
    this._device.queue.writeBuffer(this.currentStateBuffer, 0, data);
  }

  dispose() {
    this.stateA.destroy();
    this.stateB.destroy();
    this.paramsBuffer.destroy();
    this.clickBuffer.destroy();
  }

  simulate(commandEncoder: GPUCommandEncoder, rawDtMs: number, click: ClickData | null) {
    if (this.paused) return;
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
  }
}

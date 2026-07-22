import { NX, NZ, NUM_VERTS, C2, DAMPING, DELTA_X2, DELTA_Z2, SIM_SPEED, MAX_ITERATED_DT, MAX_Y, SIGMA, DELTA_X, DELTA_Z } from '../types.ts';
import type { ClickData } from '../types.ts';

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

  let ay = params.c2 * (d2x + d2z) - params.damping * (cur - prev) / params.dt;
  let uy = (cur - prev) / params.dt + ay * params.dt;
  let nextY = cur + uy * params.dt;

  stateB[i] = vec2f(cur, nextY);
}
`;

function makeClickShader(nx: number, nz: number) {
  const halfX = (nx - 1) / 2;
  const halfZ = (nz - 1) / 2;
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

  let wx = (f32(ix) - ${halfX}) * ${DELTA_X};
  let wz = (f32(iz) - ${halfZ}) * ${DELTA_Z};

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

  private _device: GPUDevice;
  private _useA = true;

  constructor(device: GPUDevice) {
    this._device = device;

    this.workgroupX = Math.ceil(NX / 8);
    this.workgroupZ = Math.ceil(NZ / 8);

    this.paramsArray = new Float32Array([0, C2, DAMPING, NX, NZ, DELTA_X2, DELTA_Z2, 0]);
    this.clickArray = new Float32Array([0, 0, 0, MAX_Y, SIGMA]);

    this.stateA = device.createBuffer({
      size: NUM_VERTS * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this.stateA.getMappedRange()).fill(0);
    this.stateA.unmap();

    this.stateB = device.createBuffer({
      size: NUM_VERTS * 8,
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
        module: device.createShaderModule({ code: makeClickShader(NX, NZ) }),
        entryPoint: 'main',
      },
    });

    this._rebind();
  }

  get currentStateBuffer(): GPUBuffer {
    return this._useA ? this.stateA : this.stateB;
  }

  simulate(commandEncoder: GPUCommandEncoder, rawDtMs: number, click: ClickData | null) {
    let dt = rawDtMs * SIM_SPEED;
    if (dt > MAX_ITERATED_DT) dt = MAX_ITERATED_DT;

    if (click?.active) {
      this.clickArray[0] = 1;
      this.clickArray[1] = click.x;
      this.clickArray[2] = click.z;
      this._device.queue.writeBuffer(this.clickBuffer, 0, this.clickArray);

      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.clickPipeline);
      pass.setBindGroup(0, this.clickBindGroup);
      pass.dispatchWorkgroups(this.workgroupX, this.workgroupZ);
      pass.end();

      this.clickArray[0] = 0;
    }

    this.paramsArray[0] = dt;
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

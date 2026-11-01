import { W, H, type Grid } from '../types.ts';

// Stam, Real-Time Fluid Dynamics for Games, §§ Moving Densities / Evolving Velocities.
// Cell-centred X–Z velocity (world units/second). Ghost cells are sampled implicitly:
// odd normal velocity, even tangential velocity and pressure at solid, free-slip walls.
const COMMON = `
struct Params { dt: f32, viscosity: f32, nx: u32, nz: u32, dx: f32, dz: f32, pad: vec2f }
@group(0) @binding(0) var<storage, read> src: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec2f>;
@group(0) @binding(2) var<storage, read> rhs: array<vec2f>;
@group(0) @binding(3) var<uniform> p: Params;
fn index(q: vec2i) -> u32 {
  let c = clamp(q, vec2i(0), vec2i(i32(p.nx)-1, i32(p.nz)-1));
  return u32(c.y) * p.nx + u32(c.x);
}
fn velocity(q: vec2i) -> vec2f {
  var v = src[index(q)];
  if (q.x < 0 || q.x >= i32(p.nx)) { v.x = -v.x; }
  if (q.y < 0 || q.y >= i32(p.nz)) { v.y = -v.y; }
  return v;
}
fn pressure(q: vec2i) -> f32 { return src[index(q)].x; }
fn sampleVelocity(q: vec2f) -> vec2f {
  let c = clamp(q, vec2f(-0.5), vec2f(f32(p.nx), f32(p.nz))-0.5);
  let lo = vec2i(floor(c)); let t = fract(c);
  return mix(mix(velocity(lo), velocity(lo+vec2i(1,0)), t.x),
             mix(velocity(lo+vec2i(0,1)), velocity(lo+vec2i(1,1)), t.x), t.y);
}
`;
const KERNELS = {
  diffuse: `
  let a = p.dt * p.viscosity / vec2f(p.dx*p.dx, p.dz*p.dz);
  dst[i] = (rhs[i] + a.x*(velocity(q+ex)+velocity(q-ex))
                   + a.y*(velocity(q+ez)+velocity(q-ez))) / (1.0+2.0*(a.x+a.y));`,
  divergence: `
  let div = (velocity(q+ex).x-velocity(q-ex).x)/(2.0*p.dx)
          + (velocity(q+ez).y-velocity(q-ez).y)/(2.0*p.dz);
  dst[i] = vec2f(0.0, div);`,
  pressure: `
  let a = 1.0 / vec2f(p.dx*p.dx, p.dz*p.dz);
  dst[i] = vec2f((a.x*(pressure(q+ex)+pressure(q-ex))
                +a.y*(pressure(q+ez)+pressure(q-ez))-src[i].y)/(2.0*(a.x+a.y)), src[i].y);`,
  project: `
  let gradient = vec2f(rhs[index(q+ex)].x-rhs[index(q-ex)].x,
                       rhs[index(q+ez)].x-rhs[index(q-ez)].x) / (2.0*vec2f(p.dx,p.dz));
  dst[i] = src[i] - gradient;`,
  advect: `
  dst[i] = sampleVelocity(vec2f(q)-p.dt*src[i]/vec2f(p.dx,p.dz));`,
};
type Kernel = keyof typeof KERNELS;
type Dispatch = { pipeline: GPUComputePipeline; group: GPUBindGroup };

export class StableFlow {
  readonly nx: number;
  readonly nz: number;
  readonly velocityBuffer: GPUBuffer;
  viscosity = 8; // Kinematic viscosity in world units²/second.
  private readonly buffers: GPUBuffer[];
  private readonly params: GPUBuffer;
  private readonly paramsData = new ArrayBuffer(32);
  private readonly floats = new Float32Array(this.paramsData);
  private readonly uints = new Uint32Array(this.paramsData);
  private readonly forceParams: GPUBuffer;
  private readonly forceData = new Float32Array(8);
  private readonly forceDispatch: Dispatch;
  private readonly diffusion: Dispatch[];
  private readonly firstProjection: Dispatch[];
  private readonly advection: Dispatch;
  private readonly lastProjection: Dispatch[];
  private readonly surfacePipeline: GPUComputePipeline;
  private readonly surfaceGroups: GPUBindGroup[];
  private readonly device: GPUDevice;

  constructor(device: GPUDevice, grid: Grid, states: GPUBuffer[]) {
    this.device = device;
    // Pressure cost is independent of the much denser surface/caustic mesh.
    this.nz = Math.min(100, grid.NH);
    this.nx = Math.ceil(this.nz * W / H);
    const buffer = (label: string, size = this.nx*this.nz*8) => device.createBuffer({
      label, size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.velocityBuffer = buffer('Flow velocity');
    const a = buffer('Flow scratch A'), b = buffer('Flow scratch B');
    const pressureA = buffer('Flow pressure A'), pressureB = buffer('Flow pressure B');
    this.buffers = [this.velocityBuffer, a, b, pressureA, pressureB];
    this.params = device.createBuffer({size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
    this.forceParams = device.createBuffer({size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
    this.uints[2] = this.nx; this.uints[3] = this.nz;
    this.floats[4] = W / this.nx; this.floats[5] = H / this.nz;
    const layout = device.createBindGroupLayout({entries: [
      {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: 'read-only-storage'}},
      {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: {type: 'storage'}},
      {binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: {type: 'read-only-storage'}},
      {binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: {type: 'uniform'}},
    ]});
    const pipelines = {} as Record<Kernel, GPUComputePipeline>;
    for (const name of Object.keys(KERNELS) as Kernel[]) {
      const code = COMMON + `
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= p.nx || id.y >= p.nz) { return; }
  let q = vec2i(id.xy); let i = id.y*p.nx+id.x;
  let ex = vec2i(1,0); let ez = vec2i(0,1);
  ${KERNELS[name]}
}`;
      pipelines[name] = device.createComputePipeline({label: `Flow ${name}`,
        layout: device.createPipelineLayout({bindGroupLayouts: [layout]}),
        compute: {module: device.createShaderModule({code}), entryPoint: 'main'}});
    }
    const dispatch = (name: Kernel, src: GPUBuffer, dst: GPUBuffer, rhs = src): Dispatch => ({
      pipeline: pipelines[name], group: device.createBindGroup({layout, entries:
        [src, dst, rhs, this.params].map((buffer, binding) => ({binding, resource: {buffer}}))}),
    });
    // Jacobi rather than in-place Gauss–Seidel: no cross-workgroup data races.
    // Keep the original velocity as the fixed diffusion RHS throughout all iterations.
    this.diffusion = [dispatch('diffuse', this.velocityBuffer, a),
      dispatch('diffuse', a, b, this.velocityBuffer), dispatch('diffuse', b, a, this.velocityBuffer)];
    const projection = (src: GPUBuffer, dst: GPUBuffer) => [
      dispatch('divergence', src, pressureA),
      dispatch('pressure', pressureA, pressureB), dispatch('pressure', pressureB, pressureA),
      dispatch('project', src, dst, pressureA),
    ];
    this.firstProjection = projection(b, a);
    this.advection = dispatch('advect', a, b);
    this.lastProjection = projection(b, this.velocityBuffer);

    const forceCode = `
@group(0) @binding(0) var<storage, read_write> v: array<vec2f>;
@group(0) @binding(1) var<uniform> brush: array<vec4f,2>;
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= ${this.nx}u || id.y >= ${this.nz}u) { return; }
  let world = (vec2f(id.xy)+0.5)*vec2f(${W/this.nx},${H/this.nz})-vec2f(${W/2},${H/2});
  let delta = world-brush[0].xy;
  v[id.y*${this.nx}u+id.x] += brush[0].zw * exp(-dot(delta,delta)*brush[1].x);
}`;
    const forcePipeline = device.createComputePipeline({layout: 'auto',
      compute: {module: device.createShaderModule({code: forceCode}), entryPoint: 'main'}});
    this.forceDispatch = {pipeline: forcePipeline, group: device.createBindGroup({
      layout: forcePipeline.getBindGroupLayout(0), entries: [
        {binding: 0, resource: {buffer: this.velocityBuffer}},
        {binding: 1, resource: {buffer: this.forceParams}},
      ]})};

    // Transport both Verlet time levels together, so currents carry wave momentum,
    // rather than spuriously turning height advection into a vertical velocity impulse.
    const surfaceCode = `
@group(0) @binding(0) var<storage, read> height: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> result: array<vec2f>;
@group(0) @binding(2) var<storage, read> v: array<vec2f>;
@group(0) @binding(3) var<uniform> p: vec4f;
fn velocity(q: vec2i) -> vec2f {
  let c = clamp(q, vec2i(0), vec2i(${this.nx-1},${this.nz-1}));
  var value = v[c.y*${this.nx}+c.x];
  if (q.x<0 || q.x>=${this.nx}) { value.x = -value.x; }
  if (q.y<0 || q.y>=${this.nz}) { value.y = -value.y; }
  return value;
}
fn h(q: vec2i) -> vec2f { return height[q.y*${grid.NX}+q.x]; }
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x>=${grid.NX}u || id.y>=${grid.NZ}u) { return; }
  let uv = vec2f(id.xy)/vec2f(${grid.NX-1},${grid.NZ-1});
  let c = uv*vec2f(${this.nx},${this.nz})-0.5;
  let lo = vec2i(floor(c)); let t = fract(c);
  let vel = mix(mix(velocity(lo),velocity(lo+vec2i(1,0)),t.x),
                mix(velocity(lo+vec2i(0,1)),velocity(lo+vec2i(1,1)),t.x),t.y);
  let q = clamp(vec2f(id.xy)-p.x*vel/vec2f(${grid.DELTA_X},${grid.DELTA_Z}),
                vec2f(0),vec2f(${grid.NX-1},${grid.NZ-1}));
  let a = vec2i(floor(q)); let b = min(a+1,vec2i(${grid.NX-1},${grid.NZ-1})); let f = fract(q);
  result[id.y*${grid.NX}u+id.x] = mix(mix(h(a),h(vec2i(b.x,a.y)),f.x),
                                     mix(h(vec2i(a.x,b.y)),h(b),f.x),f.y);
}`;
    this.surfacePipeline = device.createComputePipeline({layout: 'auto',
      compute: {module: device.createShaderModule({code: surfaceCode}), entryPoint: 'main'}});
    this.surfaceGroups = states.map((state, i) => device.createBindGroup({
      layout: this.surfacePipeline.getBindGroupLayout(0),
      entries: [state, states[1-i], this.velocityBuffer, this.params]
        .map((buffer, binding) => ({binding, resource: {buffer}})),
    }));
  }

  prepare(dtSeconds: number) {
    this.floats[0] = dtSeconds;
    this.floats[1] = Math.max(0, this.viscosity);
    this.device.queue.writeBuffer(this.params, 0, this.paramsData);
  }

  // Integrated impulse (not a per-frame force); retained even on a zero-step frame.
  addImpulse(encoder: GPUCommandEncoder, x: number, z: number, vx: number, vz: number, sigma: number) {
    this.forceData.set([x, z, vx, vz, sigma]);
    this.device.queue.writeBuffer(this.forceParams, 0, this.forceData);
    const pass = encoder.beginComputePass();
    this.dispatch(pass, this.forceDispatch);
    pass.end();
  }

  step(pass: GPUComputePassEncoder) {
    this.dispatch(pass, this.diffusion[0]);
    for (let i = 1; i < 20; i++) this.dispatch(pass, this.diffusion[i % 2 === 1 ? 1 : 2]);
    this.project(pass, this.firstProjection);
    this.dispatch(pass, this.advection);
    this.project(pass, this.lastProjection);
  }

  advectSurface(pass: GPUComputePassEncoder, stateIndex: number, x: number, z: number) {
    pass.setPipeline(this.surfacePipeline);
    pass.setBindGroup(0, this.surfaceGroups[stateIndex]);
    pass.dispatchWorkgroups(x, z);
  }

  clear() {
    const encoder = this.device.createCommandEncoder();
    for (const buffer of this.buffers) encoder.clearBuffer(buffer);
    this.device.queue.submit([encoder.finish()]);
  }

  dispose() {
    for (const buffer of this.buffers) buffer.destroy();
    this.params.destroy(); this.forceParams.destroy();
  }

  private project(pass: GPUComputePassEncoder, stages: Dispatch[]) {
    this.dispatch(pass, stages[0]);
    for (let i = 0; i < 60; i++) this.dispatch(pass, stages[1 + i % 2]);
    this.dispatch(pass, stages[3]);
  }

  private dispatch(pass: GPUComputePassEncoder, stage: Dispatch) {
    pass.setPipeline(stage.pipeline); pass.setBindGroup(0, stage.group);
    pass.dispatchWorkgroups(Math.ceil(this.nx/8), Math.ceil(this.nz/8));
  }
}

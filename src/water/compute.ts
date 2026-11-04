import { C2, SIM_SPEED, MAX_Y, SIGMA, W, H, WATER_LEVEL, type Grid } from '../types.ts';
import type { ClickData, SphereState } from '../types.ts';
import { StableFlow } from './flow.ts';
import { CONSERVATIVE_SPLASH_WGSL } from './splash.ts';

// Max drag rate the damping slider (0..1) maps to; keeps waves from over-damping.
const MAX_GAMMA = 0.05;
// omega_max² = c² lambda_max (1 + length² lambda_max), lambda_max = 4*(1/dx²+1/dz²).
const CFL_SAFETY = 0.5;
const DISPERSION_LENGTH = 3; // World units; shallow-water capillary-style dispersion.
const MAX_WAVE_SPEED = 2; // Supported UI range; fixes the timestep across speed changes.
const MAX_FRAME_MS = 100; // Discard suspended-tab time, not ordinary frame remainder.
// Wake: fraction of the ball's horizontal displacement transferred to the
// current, scaled by the fraction of the ball that is submerged.
const WAKE_COUPLING = 3;
const WAKE_INFLUENCE = 1.5;
// Keep the sphere-sized disturbance, but only convert part of the displaced
// volume into local waves. The rest contributes to the pool's mean level.
const SPHERE_WAVE_COUPLING = 0.15;

// Fraction of the sphere's volume below the rest waterline (spherical cap over
// the full volume). 0 when the sphere is absent or entirely above the waterline.
function submergedFraction(centerY: number, radius: number) {
  if (radius <= 0) return 0;
  const h = Math.min(2 * radius, Math.max(0, WATER_LEVEL - (centerY - radius)));
  return (h * h * (3 * radius - h)) / (4 * radius * radius * radius);
}

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
  dispersion2: f32,
};

// Even extension at rigid walls: dh/dn = 0, including the biharmonic stencil.
fn height(x: i32, z: i32) -> f32 {
  let nx = i32(params.nx);
  let nz = i32(params.nz);
  let ix = min(abs(x), 2 * (nx - 1) - x);
  let iz = min(abs(z), 2 * (nz - 1) - z);
  return stateA[iz * nx + ix].y;
}

fn laplacian(x: i32, z: i32) -> f32 {
  let h = height(x, z);
  return (height(x-1, z) - 2.0*h + height(x+1, z)) / params.delta_x2
       + (height(x, z-1) - 2.0*h + height(x, z+1)) / params.delta_z2;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let ix = i32(id.x);
  let iz = i32(id.y);
  let nx = i32(params.nx);
  let nz = i32(params.nz);

  if (ix >= nx || iz >= nz) { return; }

  let i = iz * nx + ix;


  let prev = stateA[i].x;
  let cur = stateA[i].y;

  let lap = laplacian(ix, iz);
  let biharmonic =
      (laplacian(ix-1, iz) - 2.0*lap + laplacian(ix+1, iz)) / params.delta_x2
    + (laplacian(ix, iz-1) - 2.0*lap + laplacian(ix, iz+1)) / params.delta_z2;
  // h_tt + gamma*h_t = c² (L h - length² L² h).
  // Centered damped Verlet: second order for the fixed simulation timestep.
  let accel = params.c2 * (lap - params.dispersion2 * biharmonic);
  let drag = 0.5 * params.damping * params.dt;
  // Increment form avoids cancellation between large absolute height terms.
  let nextY = cur + ((1.0-drag)*(cur-prev) + accel*params.dt*params.dt) / (1.0+drag);

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
@group(0) @binding(2) var<storage, read> positions: array<vec4f>;

struct ClickParams {
  count: f32,
  max_y: f32,
  sigma: f32,
};

${CONSERVATIVE_SPLASH_WGSL}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let ix = i32(id.x);
  let iz = i32(id.y);
  let nx = ${nx}u;
  let nz = ${nz}u;

  if (ix >= i32(nx) || iz >= i32(nz)) { return; }

  let i = iz * i32(nx) + ix;

  let q = vec2i(ix, iz);
  let last = vec2i(i32(nx)-1, i32(nz)-1);
  let spacing = vec2f(${dx}, ${dz});
  let inverseSpacing2 = 1.0 / (spacing*spacing);
  let width = 1.0 / click.sigma;

  // Opposite sign to rain: a raised center borrows water from a surrounding trough.
  // The mirrored discrete Laplacian conserves volume even at walls and corners.
  var bump = 0.0;
  for (var s = 0u; s < u32(click.count); s++) {
    let sample = positions[s];
    let laplacian = splashLaplacian(q, last, vec2f(${W}.0, ${H}.0),
      sample.xy+vec2f(${W/2}, ${H/2}), width, inverseSpacing2);
    bump -= click.max_y * sample.z * (0.25*width) * laplacian;
  }
  // Preserve the existing vertical velocity while adding the displacement.
  state[i].x += bump;
  state[i].y += bump;
}
`;
}

// Retain the sphere's local footprint, separating its wave-producing part
// from the equilibrium water-level change. Never relax or clamp existing waves.
function makeSphereShader(nx: number, nz: number, dx: number, dz: number) {
  return `
@group(0) @binding(0) var<storage, read_write> state: array<vec2f>;
@group(0) @binding(1) var<uniform> sp: SphereParams;

struct SphereParams {
  current: vec4f,  // (x, y, z, radius); radius 0 = absent
  previous: vec4f,
  levelChange: f32,
};

fn thickness(center: vec4f, point: vec2f) -> f32 {
  let radius = center.w;
  if (radius <= 0.0) { return 0.0; }
  // Four samples average the column over a surface cell, avoiding a jagged
  // rim as the sphere crosses grid vertices. The footprint stays local.
  let dx = point.x - center.x + vec4f(-${dx / 4}, ${dx / 4}, -${dx / 4}, ${dx / 4});
  let dz = point.y - center.z + vec4f(-${dz / 4}, -${dz / 4}, ${dz / 4}, ${dz / 4});
  let halfChord = sqrt(max(vec4f(radius * radius) - dx * dx - dz * dz, vec4f(0.0)));
  let columns = clamp(vec4f(${WATER_LEVEL}.0 - center.y) + halfChord, vec4f(0.0), 2.0 * halfChord);
  return dot(columns, vec4f(0.25));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let ix = i32(id.x);
  let iz = i32(id.y);
  let nx = ${nx}u;
  let nz = ${nz}u;

  if (ix >= i32(nx) || iz >= i32(nz)) { return; }

  let i = iz * i32(nx) + ix;
  let point = vec2f(-${W/2} + f32(ix)*${dx}, -${H/2} + f32(iz)*${dz});
  let localChange = thickness(sp.current, point) - thickness(sp.previous, point);
  let d = ${SPHERE_WAVE_COUPLING} * localChange + sp.levelChange;
  if (d == 0.0) { return; }
  // Add the delta to both Verlet time levels: displace without injecting
  // vertical velocity.
  state[i].x += d;
  state[i].y += d;
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
  spherePipeline!: GPUComputePipeline;
  sphereBuffer!: GPUBuffer;
  sphereBindGroup!: GPUBindGroup;
  paramsArray: Float32Array;
  clickArray: Float32Array;
  sphereArray: Float32Array;
  workgroupX: number;
  workgroupZ: number;

  // Public parameters for UI control (user-facing; mapped to physics in simulate()).
  simulationSpeed = 1.0;    // clock multiplier; zero freezes waves, currents, and pointer input
  waveSpeed = 1.0;          // propagation speed multiplier (c2 = baseC2 * waveSpeed^2)
  damping = 0.2;            // 0..1 wave decay (mapped to a stable drag rate)
  rippleAmplitude = MAX_Y;  // volume-neutral splash height scale (world units)
  rippleSize = 1.0;         // click splash width multiplier
  readonly flow: StableFlow;

  private _device: GPUDevice;
  private _grid: Grid;
  private _useA = true;
  paused = false;             // when true, simulate() is skipped (debug/verification)
  // Base c2 value from constants
  private _baseC2 = C2;
  private _accumulator = 0;
  private _stepDt: number;
  private _waveBindings!: [GPUBindGroup, GPUBindGroup];
  private _clickBindings!: [GPUBindGroup, GPUBindGroup];
  private _sphereBindings!: [GPUBindGroup, GPUBindGroup];
  private _clickPositions = new Float32Array(256);
  private _clickPositionsBuffer: GPUBuffer;
  // Last sphere pose actually applied to the surface (x, y, z, radius); radius
  // 0 = absent. A paused frame leaves it untouched, so resuming applies the net
  // movement since the last accepted pose.
  private _lastSphere = new Float32Array(4);

  constructor(device: GPUDevice, grid: Grid) {
    this._device = device;
    this._grid = grid;

    this.workgroupX = Math.ceil(grid.NX / 8);
    this.workgroupZ = Math.ceil(grid.NZ / 8);
    const lambdaMax = 4 * (1 / grid.DELTA_X2 + 1 / grid.DELTA_Z2);
    const omegaMax2 = C2 * MAX_WAVE_SPEED ** 2 * lambdaMax
      * (1 + DISPERSION_LENGTH ** 2 * lambdaMax);
    this._stepDt = Math.min(1000 / 120 * SIM_SPEED, Math.sqrt(4 * CFL_SAFETY / omegaMax2));

    this.paramsArray = new Float32Array([this._stepDt, C2, 0, grid.NX, grid.NZ,
      grid.DELTA_X2, grid.DELTA_Z2, DISPERSION_LENGTH ** 2]);
    this.clickArray = new Float32Array([0, MAX_Y, SIGMA, 0]);
    this.sphereArray = new Float32Array(12);

    this.stateA = device.createBuffer({
      size: grid.NUM_VERTS * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
    });
    new Float32Array(this.stateA.getMappedRange()).fill(0);
    this.stateA.unmap();

    this.stateB = device.createBuffer({
      size: grid.NUM_VERTS * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
    });
    new Float32Array(this.stateB.getMappedRange()).fill(0);
    this.stateB.unmap();
    this.flow = new StableFlow(device, grid, [this.stateA, this.stateB]);

    this.paramsBuffer = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.clickBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.sphereBuffer = device.createBuffer({
      size: this.sphereArray.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._clickPositionsBuffer = device.createBuffer({
      size: this._clickPositions.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
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

    this.spherePipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({ code: makeSphereShader(grid.NX, grid.NZ, grid.DELTA_X, grid.DELTA_Z) }),
        entryPoint: 'main',
      },
    });

    this._rebind();
    this._waveBindings = [this.bindGroup, this._makeWaveBinding(this.stateB, this.stateA)];
    this._clickBindings = [this.clickBindGroup, this._makeClickBinding(this.stateB)];
    this._sphereBindings = [this.sphereBindGroup, this._makeSphereBinding(this.stateB)];
  }

  get currentStateBuffer(): GPUBuffer {
    return this._useA ? this.stateA : this.stateB;
  }

  get grid(): Grid {
    return this._grid;
  }

  // Zero both ping-pong state buffers (flat water). For verification.
  zeroState() {
    this._accumulator = 0;
    this._lastSphere.fill(0);
    this.flow.clear();
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
    this.flow.dispose();
    this.stateA.destroy();
    this.stateB.destroy();
    this.paramsBuffer.destroy();
    this.clickBuffer.destroy();
    this.sphereBuffer.destroy();
    this._clickPositionsBuffer.destroy();
  }

  simulate(commandEncoder: GPUCommandEncoder, rawDtMs: number, clicks: readonly ClickData[], sphere?: SphereState) {
    if (this.paused || this.simulationSpeed === 0) return;
    // Map user-facing controls to physical params.
    const c2 = this._baseC2 * Math.min(MAX_WAVE_SPEED, Math.max(0, this.waveSpeed)) ** 2;
    const gamma = MAX_GAMMA * this.damping * this.damping; // quadratic: fine low-end control
    const sigma = SIGMA / this.rippleSize;

    // Cap wall time before scaling; speed changes substep count, not the stable timestep.
    this._accumulator += Math.min(MAX_FRAME_MS, Math.max(0, rawDtMs)) * SIM_SPEED * this.simulationSpeed;
    const steps = Math.floor((this._accumulator + 1e-9) / this._stepDt);
    this._accumulator = Math.max(0, this._accumulator - steps * this._stepDt);

    if (clicks.length > 0) {
      if (clicks.length * 4 > this._clickPositions.length) {
        this._clickPositions = new Float32Array(Math.max(clicks.length * 4, this._clickPositions.length * 2));
        this._clickPositionsBuffer.destroy();
        this._clickPositionsBuffer = this._device.createBuffer({
          size: this._clickPositions.byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this._clickBindings = [this._makeClickBinding(this.stateA), this._makeClickBinding(this.stateB)];
        this.clickBindGroup = this._clickBindings[this._useA ? 0 : 1];
      }
      for (let i = 0; i < clicks.length; i++) {
        this._clickPositions[i * 4] = clicks[i].x;
        this._clickPositions[i * 4 + 1] = clicks[i].z;
        this._clickPositions[i * 4 + 2] = clicks[i].strength;
      }
      this._device.queue.writeBuffer(this._clickPositionsBuffer, 0, this._clickPositions, 0, clicks.length * 4);
      this.clickArray[0] = clicks.length;
      this.clickArray[1] = this.rippleAmplitude;
      this.clickArray[2] = sigma;
      this._device.queue.writeBuffer(this.clickBuffer, 0, this.clickArray);

      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.clickPipeline);
      pass.setBindGroup(0, this.clickBindGroup);
      pass.dispatchWorkgroups(this.workgroupX, this.workgroupZ);
      pass.end();
    }

    // Displacement + wake from the ball, applied even on zero-step frames.
    // The accepted pose is the last one actually applied; a paused frame leaves
    // it untouched, so resuming applies the net movement since then.
    const present = !!sphere && sphere.active && sphere.radius > 0;
    const fx = Math.fround(present ? sphere.x : 0);
    const fy = Math.fround(present ? sphere.y : 0);
    const fz = Math.fround(present ? sphere.z : 0);
    const fr = Math.fround(present ? sphere.radius : 0);
    const last = this._lastSphere;
    if (fx !== last[0] || fy !== last[1] || fz !== last[2] || fr !== last[3]) {
      const wetFraction = submergedFraction(fy, fr);
      // Wake: transfer horizontal displacement to the current, scaled by the
      // submerged volume fraction. Only when both the old and new poses are
      // present and wet; airborne or disabled movement injects no current.
      if (last[3] > 0 && fr > 0 && last[1] - last[3] < WATER_LEVEL && fy - fr < WATER_LEVEL) {
        const ddx = fx - last[0], ddz = fz - last[2];
        if (ddx !== 0 || ddz !== 0) {
          const k = WAKE_COUPLING * wetFraction;
          this.flow.addImpulse(commandEncoder, fx, fz, ddx * k, ddz * k,
            1 / (WAKE_INFLUENCE * fr) ** 2);
        }
      }
      this.sphereArray[0] = fx; this.sphereArray[1] = fy; this.sphereArray[2] = fz; this.sphereArray[3] = fr;
      this.sphereArray[4] = last[0]; this.sphereArray[5] = last[1]; this.sphereArray[6] = last[2]; this.sphereArray[7] = last[3];
      const volumeChange = (4 * Math.PI / 3) * (fr ** 3 * wetFraction
        - last[3] ** 3 * submergedFraction(last[1], last[3]));
      this.sphereArray[8] = (1 - SPHERE_WAVE_COUPLING) * volumeChange / (W * H);
      this._device.queue.writeBuffer(this.sphereBuffer, 0, this.sphereArray);
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.spherePipeline);
      pass.setBindGroup(0, this.sphereBindGroup);
      pass.dispatchWorkgroups(this.workgroupX, this.workgroupZ);
      pass.end();
      last[0] = fx; last[1] = fy; last[2] = fz; last[3] = fr;
    }

    // Upload physical params for this frame.
    this.paramsArray[0] = this._stepDt;
    this.paramsArray[1] = c2;
    this.paramsArray[2] = gamma;
    this._device.queue.writeBuffer(this.paramsBuffer, 0, this.paramsArray);

    this.flow.prepare(this._stepDt / 1000);
    if (steps === 0) return;
    const pass = commandEncoder.beginComputePass();
    // Stam's flow step precedes passive transport of the wave surface.
    for (let step = 0; step < steps; step++) {
      this.flow.step(pass);
      this.flow.advectSurface(pass, this._useA ? 0 : 1, this.workgroupX, this.workgroupZ);
      this._swap();
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, this.bindGroup);
      pass.dispatchWorkgroups(this.workgroupX, this.workgroupZ);
      this._swap();
    }
    pass.end();
  }

  private _swap() {
    this._useA = !this._useA;
    this.bindGroup = this._waveBindings[this._useA ? 0 : 1];
    this.clickBindGroup = this._clickBindings[this._useA ? 0 : 1];
    this.sphereBindGroup = this._sphereBindings[this._useA ? 0 : 1];
  }

  private _rebind() {
    this.bindGroup = this._makeWaveBinding(this.stateA, this.stateB);
    this.clickBindGroup = this._makeClickBinding(this.stateA);
    this.sphereBindGroup = this._makeSphereBinding(this.stateA);
  }

  private _makeWaveBinding(source: GPUBuffer, destination: GPUBuffer) {
    return this._device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: source } },
        { binding: 1, resource: { buffer: destination } },
        { binding: 2, resource: { buffer: this.paramsBuffer } },
      ],
    });
  }

  private _makeClickBinding(state: GPUBuffer) {
    return this._device.createBindGroup({
      layout: this.clickPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: state } },
        { binding: 1, resource: { buffer: this.clickBuffer } },
        { binding: 2, resource: { buffer: this._clickPositionsBuffer } },
      ],
    });
  }

  private _makeSphereBinding(state: GPUBuffer) {
    return this._device.createBindGroup({
      layout: this.spherePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: state } },
        { binding: 1, resource: { buffer: this.sphereBuffer } },
      ],
    });
  }
}

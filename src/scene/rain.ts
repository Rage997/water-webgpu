import { W, H, WATER_LEVEL, type Grid } from '../types.ts';

const DROP_COUNT = 96;
const FALL_HEIGHT = 300;
const FALL_SPEED = 420;

const RENDER_SHADER = `
struct Camera { vp: mat4x4f };
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> drops: array<vec4f>;
struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) alpha: f32,
};
@vertex fn vs(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> VertexOutput {
  let drop = drops[instance];
  let tail = f32(vertex);
  var out: VertexOutput;
  out.position = camera.vp * vec4f(drop.xyz + vec3f(0.0, tail * 14.0 * drop.w, 0.0), 1.0);
  out.alpha = mix(min(0.95, 0.8 * drop.w), 0.05, tail);
  return out;
}
@fragment fn fs(in: VertexOutput) -> @location(0) vec4f {
  return vec4f(0.7, 0.85, 1.0, in.alpha);
}
`;

const IMPACT_SHADER = `
struct Params { nx: u32, nz: u32, count: u32, padding: u32 };
@group(0) @binding(0) var<storage, read_write> state: array<vec2f>;
@group(0) @binding(1) var<storage, read> impacts: array<vec4f>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  // Keep the fixed pool boundary undisturbed.
  if (id.x == 0u || id.y == 0u || id.x >= params.nx - 1u || id.y >= params.nz - 1u) { return; }
  let p = vec2f(f32(id.x) / f32(params.nx - 1u) * ${W}.0 - ${W / 2},
                f32(id.y) / f32(params.nz - 1u) * ${H}.0 - ${H / 2});
  var displacement = 0.0;
  for (var i = 0u; i < params.count; i++) {
    let delta = p - impacts[i].xy;
    let r2 = dot(delta, delta);
    // impacts.z is depth; impacts.w is 2*sigma^2. Truncate at five sigma.
    if (r2 < 12.5 * impacts[i].w) {
      displacement -= impacts[i].z * exp(-r2 / impacts[i].w);
    }
  }
  let index = id.y * params.nx + id.x;
  // Displace both time levels, like click splashes, without a dt-dependent kick.
  state[index] += vec2f(displacement);
}
`;

/** Falling streaks and matching, batched disturbances of the simulated surface. */
export class Rain {
  enabled = true;
  strength = 14; // Base impact depth in world units.
  size = 8;      // Base Gaussian standard deviation in world units.
  variety = 0.5; // 0 = uniform, 1 = full per-drop variation.
  private readonly drops = new Float32Array(DROP_COUNT * 4);
  private readonly sizeSamples = new Float32Array(DROP_COUNT);
  private readonly speedSamples = new Float32Array(DROP_COUNT);
  private readonly impacts = new Float32Array(DROP_COUNT * 4);
  private readonly params = new Uint32Array(4);
  private readonly dropBuffer: GPUBuffer;
  private readonly impactBuffer: GPUBuffer;
  private readonly paramsBuffer: GPUBuffer;
  private readonly cameraBuffer: GPUBuffer;
  private readonly pipeline: GPURenderPipeline;
  private readonly impactPipeline: GPUComputePipeline;
  private readonly renderGroup: GPUBindGroup;
  private readonly impactGroups = new WeakMap<GPUBuffer, GPUBindGroup>();
  private readonly device: GPUDevice;

  constructor(device: GPUDevice, format: GPUTextureFormat, depthFormat: GPUTextureFormat) {
    this.device = device;
    this.dropBuffer = device.createBuffer({ size: this.drops.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.impactBuffer = device.createBuffer({ size: this.impacts.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.paramsBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.cameraBuffer = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const module = device.createShaderModule({ code: RENDER_SHADER });
    this.pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: {
        module, entryPoint: 'fs',
        targets: [{ format, blend: {
          color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        } }],
      },
      primitive: { topology: 'line-list' },
      depthStencil: { format: depthFormat, depthWriteEnabled: false, depthCompare: 'less-equal' },
    });
    this.impactPipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code: IMPACT_SHADER }), entryPoint: 'main' },
    });
    this.renderGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.cameraBuffer } },
        { binding: 1, resource: { buffer: this.dropBuffer } },
      ],
    });
    for (let i = 0; i < DROP_COUNT; i++) {
      this.spawn(i * 4);
      this.drops[i * 4 + 1] = WATER_LEVEL + Math.random() * FALL_HEIGHT;
    }
  }

  private spawn(offset: number) {
    // Keep impacts inside the pool, away from its pinned edge vertices.
    this.drops[offset] = (Math.random() - 0.5) * (W - 24);
    this.drops[offset + 1] = WATER_LEVEL + FALL_HEIGHT;
    this.drops[offset + 2] = (Math.random() - 0.5) * (H - 24);
    const index = offset / 4;
    this.sizeSamples[index] = Math.random() * 2 - 1;
    this.speedSamples[index] = Math.random() * 2 - 1;
  }

  update(encoder: GPUCommandEncoder, rawDtMs: number, state: GPUBuffer, grid: Grid, vp: Float32Array, paused: boolean) {
    if (!this.enabled) return;
    // No catch-up storm on returning from a background tab.
    const dt = paused ? 0 : Math.min(Math.max(rawDtMs, 0), 50) * 0.001;
    let count = 0;
    for (let i = 0; i < DROP_COUNT; i++) {
      const offset = i * 4;
      // Samples stay fixed during a fall; controls still take effect immediately.
      const scale = 1 + this.variety * this.sizeSamples[i] * 0.6;
      const speed = FALL_SPEED * (1 + this.variety * this.speedSamples[i] * 0.35);
      this.drops[offset + 3] = scale;
      this.drops[offset + 1] -= speed * dt;
      if (this.drops[offset + 1] <= WATER_LEVEL) {
        this.impacts[count * 4] = this.drops[offset];
        this.impacts[count * 4 + 1] = this.drops[offset + 2];
        this.impacts[count * 4 + 2] = this.strength * scale * scale;
        const sigma = this.size * scale;
        this.impacts[count * 4 + 3] = 2 * sigma * sigma;
        count++;
        const overshoot = WATER_LEVEL - this.drops[offset + 1];
        this.spawn(offset);
        const nextSpeed = FALL_SPEED * (1 + this.variety * this.speedSamples[i] * 0.35);
        this.drops[offset + 1] -= overshoot / speed * nextSpeed;
        this.drops[offset + 3] = 1 + this.variety * this.sizeSamples[i] * 0.6;
      }
    }
    this.device.queue.writeBuffer(this.dropBuffer, 0, this.drops);
    this.device.queue.writeBuffer(this.cameraBuffer, 0, vp as Float32Array<ArrayBuffer>);
    if (count === 0) return;
    this.params[0] = grid.NX;
    this.params[1] = grid.NZ;
    this.params[2] = count;
    this.device.queue.writeBuffer(this.paramsBuffer, 0, this.params);
    this.device.queue.writeBuffer(this.impactBuffer, 0, this.impacts, 0, count * 4);
    let group = this.impactGroups.get(state);
    if (!group) {
      group = this.device.createBindGroup({
        layout: this.impactPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: state } },
          { binding: 1, resource: { buffer: this.impactBuffer } },
          { binding: 2, resource: { buffer: this.paramsBuffer } },
        ],
      });
      this.impactGroups.set(state, group);
    }
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.impactPipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(grid.NX / 8), Math.ceil(grid.NZ / 8));
    pass.end();
  }

  render(pass: GPURenderPassEncoder) {
    if (!this.enabled) return;
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.renderGroup);
    pass.draw(2, DROP_COUNT);
  }
}

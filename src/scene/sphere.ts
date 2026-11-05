import { W, H, WATER_LEVEL } from '../types.ts';

// A draggable ball that can be lifted out of the pool or pushed underwater.
// Its diameter fits below the rest waterline without intersecting the floor.
const RADIUS = 32;
const FREEBOARD = 0.35; // Initial center height above the waterline, in radii.
const FOLLOW_RATE = 20; // 1/s; how quickly the ball chases the pointer target.
const MAX_STEP_MS = 100; // Discard suspended-tab time, like the sim does.

const SPHERE_VERT = `
@group(0) @binding(0) var<uniform> cam: Camera;

struct Camera {
  viewProj: mat4x4f,
  center: vec3f,
  radius: f32,
  eye: vec3f,
};

struct VertexInput {
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
};

struct VertexOutput {
  @builtin(position) clipPos: vec4f,
  @location(0) worldPos: vec3f,
  @location(1) normal: vec3f,
};

@vertex
fn main(input: VertexInput) -> VertexOutput {
  var out: VertexOutput;
  let world = cam.center + input.position * cam.radius;
  out.clipPos = cam.viewProj * vec4f(world, 1.0);
  out.worldPos = world;
  out.normal = input.normal;
  return out;
}
`;

// Single source of truth for the ball's look: the rasterized sphere and the
// water's refracted ray hits both call shadeSphere with the same coral color
// and lighting.
export const SPHERE_SHADING = `
const SPHERE_COLOR: vec3f = vec3f(0.93, 0.45, 0.30);

fn shadeSphere(normal: vec3f, viewDir: vec3f) -> vec3f {
  let n = normalize(normal);
  let v = normalize(viewDir);
  let lightDir = normalize(vec3f(0.2, 0.8, 0.3));
  let ndl = max(dot(n, lightDir), 0.0);
  let h = normalize(lightDir + v);
  let spec = pow(max(dot(n, h), 0.0), 64.0) * 0.6;
  let rim = pow(1.0 - max(dot(n, v), 0.0), 3.0);
  return SPHERE_COLOR * (0.25 + 0.75 * ndl) + vec3f(spec) + SPHERE_COLOR * rim * 0.2;
}
`;

const SPHERE_FRAG = `
@group(0) @binding(0) var<uniform> cam: Camera;

struct Camera {
  viewProj: mat4x4f,
  center: vec3f,
  radius: f32,
  eye: vec3f,
};

${SPHERE_SHADING}

struct VertexOutput {
  @builtin(position) clipPos: vec4f,
  @location(0) worldPos: vec3f,
  @location(1) normal: vec3f,
};

@fragment
fn main(in: VertexOutput) -> @location(0) vec4f {
  return vec4f(shadeSphere(in.normal, cam.eye - in.worldPos), 1.0);
}
`;

// Unit UV sphere. Index winding matches the water grid convention
// (a, b, c, b, d, c with b = +first axis, c = +second axis), which renders
// outward-facing in this pipeline.
function makeSphereMesh(lat: number, lon: number) {
  const positions: number[] = [];
  for (let i = 0; i <= lat; i++) {
    const theta = (i / lat) * Math.PI;
    const st = Math.sin(theta), ct = Math.cos(theta);
    for (let j = 0; j <= lon; j++) {
      const phi = (j / lon) * 2 * Math.PI;
      positions.push(st * Math.cos(phi), ct, st * Math.sin(phi));
    }
  }
  const indices: number[] = [];
  for (let i = 0; i < lat; i++) {
    for (let j = 0; j < lon; j++) {
      const a = i * (lon + 1) + j;
      const b = a + 1;
      const c = a + lon + 1;
      const d = c + 1;
      indices.push(a, b, c, b, d, c);
    }
  }
  return { positions: new Float32Array(positions), indices: new Uint16Array(indices) };
}

export class DragSphere {
  readonly radius = RADIUS;
  enabled = true;
  position: [number, number, number] = [0, WATER_LEVEL + RADIUS * FREEBOARD, 0];

  private _target: [number, number, number] = [...this.position];
  private _device: GPUDevice;
  private _pipeline: GPURenderPipeline;
  private _bindGroup: GPUBindGroup;
  private _cameraBuffer: GPUBuffer;
  private _vertexBuffer: GPUBuffer;
  private _indexBuffer: GPUBuffer;
  private _indexCount: number;
  private _cameraData = new Float32Array(24); // 96 bytes: WGSL pads the struct to 16-byte alignment

  constructor(device: GPUDevice, format: GPUTextureFormat, depthFormat: GPUTextureFormat) {
    this._device = device;
    const mesh = makeSphereMesh(32, 48);
    this._indexCount = mesh.indices.length;

    // Unit-sphere positions are also normals; both attributes share the data.
    this._vertexBuffer = device.createBuffer({
      size: mesh.positions.byteLength,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });
    new Float32Array(this._vertexBuffer.getMappedRange()).set(mesh.positions);
    this._vertexBuffer.unmap();

    this._indexBuffer = device.createBuffer({
      size: mesh.indices.byteLength,
      usage: GPUBufferUsage.INDEX,
      mappedAtCreation: true,
    });
    new Uint16Array(this._indexBuffer.getMappedRange()).set(mesh.indices);
    this._indexBuffer.unmap();

    this._cameraBuffer = device.createBuffer({
      size: this._cameraData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const vertexModule = device.createShaderModule({ code: SPHERE_VERT });
    const fragmentModule = device.createShaderModule({ code: SPHERE_FRAG });
    this._pipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: vertexModule,
        entryPoint: 'main',
        buffers: [{
          arrayStride: 12,
          attributes: [
            { format: 'float32x3', offset: 0, shaderLocation: 0 },
            { format: 'float32x3', offset: 0, shaderLocation: 1 },
          ],
        }],
      },
      fragment: {
        module: fragmentModule,
        entryPoint: 'main',
        targets: [{ format }],
      },
      primitive: { topology: 'triangle-list' },
      depthStencil: {
        format: depthFormat,
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
    });

    this._bindGroup = device.createBindGroup({
      layout: this._pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this._cameraBuffer } }],
    });
  }

  // Follow the pointer in 3D; the sim uses the change in submerged volume.
  update(rawDtMs: number) {
    const dt = Math.min(MAX_STEP_MS, Math.max(0, rawDtMs)) / 1000;
    if (dt <= 0) return;
    const k = 1 - Math.exp(-FOLLOW_RATE * dt);
    for (let axis = 0; axis < 3; axis++) {
      this.position[axis] += (this._target[axis] - this.position[axis]) * k;
    }
  }

  setTarget(x: number, y: number, z: number) {
    // Stay inside the tub walls and above its floor, but allow lifting out.
    this._target[0] = Math.max(-W / 2 + this.radius, Math.min(W / 2 - this.radius, x));
    this._target[1] = Math.max(this.radius, y);
    this._target[2] = Math.max(-H / 2 + this.radius, Math.min(H / 2 - this.radius, z));
  }

  // Analytic ray/sphere intersection; nearest positive t, or null.
  raycast(ox: number, oy: number, oz: number, dx: number, dy: number, dz: number): number | null {
    const [cx, cy, cz] = this.position;
    const fx = cx - ox, fy = cy - oy, fz = cz - oz;
    const b = fx * dx + fy * dy + fz * dz;
    const c = fx * fx + fy * fy + fz * fz - this.radius * this.radius;
    const disc = b * b - c;
    if (disc < 0) return null;
    const s = Math.sqrt(disc);
    const t0 = b - s;
    const t1 = b + s;
    if (t0 > 1e-4) return t0;
    if (t1 > 1e-4) return t1;
    return null;
  }

  updateCamera(viewProj: Float32Array, eyeX: number, eyeY: number, eyeZ: number) {
    const d = this._cameraData;
    d.set(viewProj, 0);
    d[16] = this.position[0]; d[17] = this.position[1]; d[18] = this.position[2];
    d[19] = this.radius;
    d[20] = eyeX; d[21] = eyeY; d[22] = eyeZ;
    this._device.queue.writeBuffer(this._cameraBuffer, 0, d);
  }

  render(pass: GPURenderPassEncoder) {
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, this._bindGroup);
    pass.setVertexBuffer(0, this._vertexBuffer);
    pass.setIndexBuffer(this._indexBuffer, 'uint16');
    pass.drawIndexed(this._indexCount);
  }

  dispose() {
    this._vertexBuffer.destroy();
    this._indexBuffer.destroy();
    this._cameraBuffer.destroy();
  }
}

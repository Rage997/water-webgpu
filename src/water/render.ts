import { NX, NZ, NUM_VERTS, DELTA_X, DELTA_Z, NW, NH, W, H } from '../types.ts';

function makeWaterVertexShader(nx: number, nz: number, dx: number, dz: number) {
  return `
@group(0) @binding(0) var<storage, read> heightState: array<vec2f>;
@group(0) @binding(1) var<uniform> camera: Camera;

struct Camera {
  viewProj: mat4x4f,
  eye: vec3f,
};

struct VertexInput {
  @location(0) position: vec2f,
};

struct VertexOutput {
  @builtin(position) clipPos: vec4f,
  @location(0) worldPos: vec3f,
  @location(1) normal: vec3f,
  @location(2) viewDir: vec3f,
};

@vertex
fn main(@builtin(vertex_index) idx: u32, input: VertexInput) -> VertexOutput {
  let h = heightState[idx].y;
  let worldPos = vec3f(input.position.x, h, input.position.y);

  let nx = i32(${nx}u);
  let nz = i32(${nz}u);
  let i = i32(idx);

  let ix = i % nx;
  let iz = i / nx;

  let hL = select(heightState[i - 1].y, h, ix == 0);
  let hR = select(heightState[i + 1].y, h, ix == nx - 1);
  let hD = select(heightState[i - nx].y, h, iz == 0);
  let hU = select(heightState[i + nx].y, h, iz == nz - 1);

  let ddx = (hR - hL) / (2.0 * ${dx});
  let ddz = (hU - hD) / (2.0 * ${dz});
  let normal = normalize(vec3f(-ddx, 1.0, -ddz));

  let viewDir = normalize(camera.eye - worldPos);

  var out: VertexOutput;
  out.clipPos = camera.viewProj * vec4f(worldPos, 1.0);
  out.worldPos = worldPos;
  out.normal = normal;
  out.viewDir = viewDir;
  return out;
}
`;
}

const WATER_FRAG = `
@fragment
fn main(in: VertexOutput) -> @location(0) vec4f {
  let waterColor = vec3f(0.0, 0.376, 0.502);
  let specColor = vec3f(0.063, 0.063, 0.063);
  let ambientColor = vec3f(0.565, 0.565, 0.565);
  let lightDir = normalize(vec3f(0.2, 0.8, 0.3));

  let n = normalize(in.normal);
  let v = normalize(in.viewDir);

  let ambient = ambientColor * waterColor;

  let nDotL = max(dot(n, lightDir), 0.0);
  let diffuse = nDotL * waterColor;

  let h = normalize(lightDir + v);
  let nDotH = max(dot(n, h), 0.0);
  let specular = pow(nDotH, 32.0) * specColor;

  let finalRgb = ambient + diffuse + specular;
  return vec4f(finalRgb, 0.7);
}

struct VertexOutput {
  @builtin(position) clipPos: vec4f,
  @location(0) worldPos: vec3f,
  @location(1) normal: vec3f,
  @location(2) viewDir: vec3f,
};
`;

export class WaterRender {
  pipeline!: GPURenderPipeline;
  vertexBuffer!: GPUBuffer;
  indexBuffer!: GPUBuffer;
  indexCount!: number;
  bindGroupLayout!: GPUBindGroupLayout;
  bindGroup!: GPUBindGroup;
  cameraBuffer!: GPUBuffer;

  private _device: GPUDevice;

  constructor(device: GPUDevice, format: GPUTextureFormat, depthFormat: GPUTextureFormat) {
    this._device = device;

    const positions = new Float32Array(NUM_VERTS * 2);
    const indices: number[] = [];

    for (let iz = 0; iz < NZ; iz++) {
      for (let ix = 0; ix < NX; ix++) {
        const i = iz * NX + ix;
        positions[i * 2] = -W / 2 + ix * DELTA_X;
        positions[i * 2 + 1] = -H / 2 + iz * DELTA_Z;
      }
    }

    for (let iz = 0; iz < NH; iz++) {
      for (let ix = 0; ix < NW; ix++) {
        const a = iz * NX + ix;
        const b = a + 1;
        const c = a + NX;
        const d = c + 1;
        indices.push(a, b, c, b, d, c);
      }
    }

    this.indexCount = indices.length;

    this.vertexBuffer = device.createBuffer({
      size: positions.byteLength,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });
    new Float32Array(this.vertexBuffer.getMappedRange()).set(positions);
    this.vertexBuffer.unmap();

    this.indexBuffer = device.createBuffer({
      size: indices.length * 2,
      usage: GPUBufferUsage.INDEX,
      mappedAtCreation: true,
    });
    new Uint16Array(this.indexBuffer.getMappedRange()).set(indices);
    this.indexBuffer.unmap();

    this.cameraBuffer = device.createBuffer({
      size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const vertSrc = makeWaterVertexShader(NX, NZ, DELTA_X, DELTA_Z);
    const vertexModule = device.createShaderModule({ code: vertSrc });
    const fragmentModule = device.createShaderModule({ code: WATER_FRAG });

    this.bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      ],
    });

    this.pipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
      vertex: {
        module: vertexModule,
        entryPoint: 'main',
        buffers: [{
          arrayStride: 8,
          attributes: [{ format: 'float32x2', offset: 0, shaderLocation: 0 }],
        }],
      },
      fragment: {
        module: fragmentModule,
        entryPoint: 'main',
        targets: [{
          format,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list' },
      depthStencil: {
        format: depthFormat,
        depthWriteEnabled: false,
        depthCompare: 'less',
      },
    });
  }

  updateCamera(viewProj: Float32Array, eyeX: number, eyeY: number, eyeZ: number) {
    const data = new Float32Array(20);
    data.set(viewProj, 0);
    data[16] = eyeX;
    data[17] = eyeY;
    data[18] = eyeZ;
    this._device.queue.writeBuffer(this.cameraBuffer, 0, data);
  }

  setStateBuffer(buffer: GPUBuffer) {
    this.bindGroup = this._device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer } },
        { binding: 1, resource: { buffer: this.cameraBuffer } },
      ],
    });
  }

  render(pass: GPURenderPassEncoder) {
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setVertexBuffer(0, this.vertexBuffer);
    pass.setIndexBuffer(this.indexBuffer, 'uint16');
    pass.drawIndexed(this.indexCount);
  }
}

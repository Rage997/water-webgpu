const SKYBOX_VERT = `
@group(0) @binding(0) var<uniform> camera: Camera;

struct Camera {
  viewProj: mat4x4f,
  eye: vec3f,
};

struct VertexInput {
  @location(0) position: vec3f,
};

struct VertexOutput {
  @builtin(position) clipPos: vec4f,
  @location(0) uv: vec3f,
};

@vertex
fn main(input: VertexInput) -> VertexOutput {
  var vp = camera.viewProj;
  vp[3] = vec4f(0.0, 0.0, 0.0, 1.0);
  let clipPos = vp * vec4f(input.position, 1.0);
  var out: VertexOutput;
  out.clipPos = clipPos.xyww;
  out.uv = input.position;
  return out;
}
`;

const SKYBOX_FRAG = `
@fragment
fn main(in: VertexOutput) -> @location(0) vec4f {
  let skyTop = vec3f(0.4, 0.6, 0.9);
  let skyBot = vec3f(0.7, 0.8, 1.0);
  let t = max(0.0, min(1.0, in.uv.y * 0.5 + 0.5));
  return vec4f(mix(skyBot, skyTop, t), 1.0);
}

struct VertexOutput {
  @builtin(position) clipPos: vec4f,
  @location(0) uv: vec3f,
};
`;

function makeCube(): { positions: number[], indices: number[] } {
  const positions: number[] = [];
  const indices: number[] = [];
  const s = 5000;

  const faces = [
    [ s,-s,-s, s,-s, s, s, s, s, s, s,-s],
    [-s,-s, s,-s,-s,-s,-s, s,-s,-s, s, s],
    [-s, s,-s, s, s,-s, s, s, s,-s, s, s],
    [-s,-s, s, s,-s, s, s,-s,-s,-s,-s,-s],
    [-s,-s, s, s,-s, s, s, s, s,-s, s, s],
    [ s,-s,-s,-s,-s,-s,-s, s,-s, s, s,-s],
  ];

  let idx = 0;
  for (const f of faces) {
    positions.push(...f);
    indices.push(idx, idx+1, idx+2, idx+2, idx+1, idx+3);
    idx += 4;
  }

  return { positions, indices };
}

export class Skybox {
  pipeline!: GPURenderPipeline;
  vertexBuffer!: GPUBuffer;
  indexBuffer!: GPUBuffer;
  indexCount!: number;
  bindGroup!: GPUBindGroup;
  cameraBuffer!: GPUBuffer;

  private _device: GPUDevice;

  constructor(device: GPUDevice, format: GPUTextureFormat, depthFormat: GPUTextureFormat) {
    this._device = device;

    const cube = makeCube();
    this.indexCount = cube.indices.length;

    this.vertexBuffer = device.createBuffer({
      size: cube.positions.length * 4,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });
    new Float32Array(this.vertexBuffer.getMappedRange()).set(cube.positions);
    this.vertexBuffer.unmap();

    this.indexBuffer = device.createBuffer({
      size: cube.indices.length * 2,
      usage: GPUBufferUsage.INDEX,
      mappedAtCreation: true,
    });
    new Uint16Array(this.indexBuffer.getMappedRange()).set(cube.indices);
    this.indexBuffer.unmap();

    this.cameraBuffer = device.createBuffer({
      size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const vertModule = device.createShaderModule({ code: SKYBOX_VERT });
    const fragModule = device.createShaderModule({ code: SKYBOX_FRAG });

    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });

    this.pipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      vertex: {
        module: vertModule,
        entryPoint: 'main',
        buffers: [{
          arrayStride: 12,
          attributes: [{ format: 'float32x3', offset: 0, shaderLocation: 0 }],
        }],
      },
      fragment: {
        module: fragModule,
        entryPoint: 'main',
        targets: [{ format }],
      },
      primitive: { topology: 'triangle-list' },
      depthStencil: {
        format: depthFormat,
        depthWriteEnabled: false,
        depthCompare: 'less-equal',
      },
    });

    this.bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.cameraBuffer } },
      ],
    });
  }

  updateCamera(viewProj: Float32Array) {
    const data = new Float32Array(20);
    data.set(viewProj, 0);
    this._device.queue.writeBuffer(this.cameraBuffer, 0, data);
  }

  render(pass: GPURenderPassEncoder) {
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setVertexBuffer(0, this.vertexBuffer);
    pass.setIndexBuffer(this.indexBuffer, 'uint16');
    pass.drawIndexed(this.indexCount);
  }
}

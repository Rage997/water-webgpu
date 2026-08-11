const FLOOR_VERT = `
@group(0) @binding(0) var<uniform> camera: Camera;

struct Camera {
  viewProj: mat4x4f,
  eye: vec3f,
};

struct VertexInput {
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) texcoord: vec2f,
};

struct VertexOutput {
  @builtin(position) clipPos: vec4f,
  @location(0) worldPos: vec3f,
  @location(1) normal: vec3f,
  @location(2) texcoord: vec2f,
};

@vertex
fn main(input: VertexInput) -> VertexOutput {
  var out: VertexOutput;
  out.clipPos = camera.viewProj * vec4f(input.position, 1.0);
  out.worldPos = input.position;
  out.normal = input.normal;
  out.texcoord = input.texcoord;
  return out;
}
`;

const FLOOR_FRAG = `
@group(0) @binding(1) var ourTexture: texture_2d<f32>;
@group(0) @binding(2) var ourSampler: sampler;

@fragment
fn main(in: VertexOutput) -> @location(0) vec4f {
  let texColor = textureSample(ourTexture, ourSampler, in.texcoord).rgb;
  let lightDir = normalize(vec3f(0.2, 0.8, 0.3));
  let n = normalize(in.normal);
  let nDotL = max(dot(n, lightDir), 0.0);
  let ambient = 0.3;
  let lit = texColor * (ambient + nDotL * 0.7);
  return vec4f(lit, 1.0);
}

struct VertexOutput {
  @builtin(position) clipPos: vec4f,
  @location(0) worldPos: vec3f,
  @location(1) normal: vec3f,
  @location(2) texcoord: vec2f,
};
`;

export class Floor {
  pipeline!: GPURenderPipeline;
  vertexBuffer!: GPUBuffer;
  indexBuffer!: GPUBuffer;
  indexCount!: number;
  bindGroup!: GPUBindGroup;
  texture!: GPUTexture;
  sampler!: GPUSampler;
  cameraBuffer!: GPUBuffer;

  private _device: GPUDevice;

  constructor(device: GPUDevice, format: GPUTextureFormat, depthFormat: GPUTextureFormat) {
    this._device = device;

    // Floor: large plane at y = -1, extends beyond the tub
    const size = 1500;
    const positions = new Float32Array([
      -size, -1, -size,
       size, -1, -size,
      -size, -1,  size,
       size, -1,  size,
    ]);
    const normals = new Float32Array([
      0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0,
    ]);
    const texcoords = new Float32Array([
      0, 0, 20, 0, 0, 20, 20, 20,
    ]);
    const indices = new Uint16Array([0, 1, 2, 1, 3, 2]);

    this.indexCount = 6;

    // Interleave: [pos3, normal3, tex2] per vertex = 8 floats per vertex
    const vertexData = new Float32Array(4 * 8);
    for (let i = 0; i < 4; i++) {
      const i8 = i * 8;
      const i3 = i * 3;
      const i2 = i * 2;
      vertexData[i8]   = positions[i3];
      vertexData[i8+1] = positions[i3+1];
      vertexData[i8+2] = positions[i3+2];
      vertexData[i8+3] = normals[i3];
      vertexData[i8+4] = normals[i3+1];
      vertexData[i8+5] = normals[i3+2];
      vertexData[i8+6] = texcoords[i2];
      vertexData[i8+7] = texcoords[i2+1];
    }

    this.vertexBuffer = device.createBuffer({
      size: vertexData.byteLength,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });
    new Float32Array(this.vertexBuffer.getMappedRange()).set(vertexData);
    this.vertexBuffer.unmap();

    this.indexBuffer = device.createBuffer({
      size: indices.byteLength,
      usage: GPUBufferUsage.INDEX,
      mappedAtCreation: true,
    });
    new Uint16Array(this.indexBuffer.getMappedRange()).set(indices);
    this.indexBuffer.unmap();

    this.cameraBuffer = device.createBuffer({
      size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.sampler = device.createSampler({
      addressModeU: 'repeat',
      addressModeV: 'repeat',
      magFilter: 'linear',
      minFilter: 'linear',
    });

    // Create fallback texture
    this.texture = device.createTexture({
      size: { width: 1, height: 1 },
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    const white = new Uint8Array([200, 200, 200, 255]);
    device.queue.writeTexture(
      { texture: this.texture },
      white,
      { bytesPerRow: 4 },
      { width: 1, height: 1 }
    );

    this._loadTexture(`${import.meta.env.BASE_URL}textures/floortiles.jpg`);

    const vertModule = device.createShaderModule({ code: FLOOR_VERT });
    const fragModule = device.createShaderModule({ code: FLOOR_FRAG });

    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });

    this.pipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      vertex: {
        module: vertModule,
        entryPoint: 'main',
        buffers: [{
          arrayStride: 32,
          attributes: [
            { format: 'float32x3', offset: 0, shaderLocation: 0 },
            { format: 'float32x3', offset: 12, shaderLocation: 1 },
            { format: 'float32x2', offset: 24, shaderLocation: 2 },
          ],
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
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
    });

    this.bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.cameraBuffer } },
        { binding: 1, resource: this.texture.createView() },
        { binding: 2, resource: this.sampler },
      ],
    });
  }

  private async _loadTexture(url: string) {
    try {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.src = url;
      await img.decode();
      const bitmap = await createImageBitmap(img);
      const device = this._device;

      const newTexture = device.createTexture({
        size: { width: bitmap.width, height: bitmap.height },
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });

      device.queue.copyExternalImageToTexture(
        { source: bitmap },
        { texture: newTexture },
        { width: bitmap.width, height: bitmap.height }
      );

      this.texture.destroy();
      this.texture = newTexture;

      this.bindGroup = this._device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.cameraBuffer } },
          { binding: 1, resource: this.texture.createView() },
          { binding: 2, resource: this.sampler },
        ],
      });
    } catch {
      // Fallback texture already set
    }
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

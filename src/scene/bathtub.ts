import { W, H } from '../types.ts';

const TUB_VERT = `
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

const TUB_FRAG = `
@group(0) @binding(1) var ourTexture: texture_2d<f32>;
@group(0) @binding(2) var ourSampler: sampler;

@fragment
fn main(in: VertexOutput) -> @location(0) vec4f {
  let texColor = textureSample(ourTexture, ourSampler, in.texcoord).rgb;
  let lightDir = normalize(vec3f(0.2, 0.8, 0.3));
  let n = normalize(in.normal);
  let nDotL = max(dot(n, lightDir), 0.0);
  let ambient = 0.4;
  let lit = texColor * (ambient + nDotL * 0.6);
  return vec4f(lit, 1.0);
}

struct VertexOutput {
  @builtin(position) clipPos: vec4f,
  @location(0) worldPos: vec3f,
  @location(1) normal: vec3f,
  @location(2) texcoord: vec2f,
};
`;

interface BoxSide {
  positions: number[];
  normals: number[];
  texcoords: number[];
  indices: number[];
}

function makeBoxSide(px: number, py: number, pz: number, nx: number, ny: number, nz: number, ux: number, uy: number, uz: number, vx: number, vy: number, vz: number, w: number, h: number, repU: number, repV: number): BoxSide {
  const halfW = w / 2;
  const halfH = h / 2;

  const c = [px, py, pz]; // center

  const u = [ux, uy, uz];
  const v = [vx, vy, vz];
  const n = [nx, ny, nz];

  function pt(du: number, dv: number): number[] {
    return [
      c[0] + u[0]*du + v[0]*dv,
      c[1] + u[1]*du + v[1]*dv,
      c[2] + u[2]*du + v[2]*dv,
    ];
  }

  const p00 = pt(-halfW, -halfH);
  const p10 = pt(halfW, -halfH);
  const p01 = pt(-halfW, halfH);
  const p11 = pt(halfW, halfH);

  const positions = [...p00, ...p10, ...p01, ...p11];
  const normals = [...n, ...n, ...n, ...n];
  const texcoords = [0, 0, repU, 0, 0, repV, repU, repV];
  const indices = [0, 1, 2, 1, 3, 2];

  return { positions, normals, texcoords, indices };
}

export class Bathtub {
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

    const tubW = W + 4;
    const tubD = H + 4;
    const wallH = 100;

    // 5 sides: bottom, 2 short sides (front/back), 2 long sides (left/right)
    const sides: BoxSide[] = [
      // Bottom: Y=0, normal (0,1,0), tangent along X and Z
      makeBoxSide(0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, -1, tubW, tubD, 3, 6),
      // Short side front: Z=-H/2, normal (0,0,1), facing forward
      makeBoxSide(0, wallH/2, -tubD/2, 0, 0, 1, 1, 0, 0, 0, 1, 0, tubW, wallH, 3, 1),
      // Short side back: Z=H/2, normal (0,0,-1)
      makeBoxSide(0, wallH/2, tubD/2, 0, 0, -1, -1, 0, 0, 0, 1, 0, tubW, wallH, 3, 1),
      // Long side left: X=-W/2, normal (1,0,0)
      makeBoxSide(-tubW/2, wallH/2, 0, 1, 0, 0, 0, 0, -1, 0, 1, 0, tubD, wallH, 6, 1),
      // Long side right: X=W/2, normal (-1,0,0)
      makeBoxSide(tubW/2, wallH/2, 0, -1, 0, 0, 0, 0, 1, 0, 1, 0, tubD, wallH, 6, 1),
    ];

    const vertexComponents: number[] = [];
    const allIndices: number[] = [];
    let baseIdx = 0;

    for (const s of sides) {
      const n = s.positions.length / 3;
      for (let i = 0; i < n; i++) {
        const i3 = i * 3;
        const i2 = i * 2;
        vertexComponents.push(
          s.positions[i3], s.positions[i3+1], s.positions[i3+2],
          s.normals[i3], s.normals[i3+1], s.normals[i3+2],
          s.texcoords[i2], s.texcoords[i2+1],
        );
      }
      for (const idx of s.indices) allIndices.push(idx + baseIdx);
      baseIdx += n;
    }

    this.indexCount = allIndices.length;

    this.vertexBuffer = device.createBuffer({
      size: vertexComponents.length * 4,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });
    new Float32Array(this.vertexBuffer.getMappedRange()).set(vertexComponents);
    this.vertexBuffer.unmap();

    this.indexBuffer = device.createBuffer({
      size: allIndices.length * 2,
      usage: GPUBufferUsage.INDEX,
      mappedAtCreation: true,
    });
    new Uint16Array(this.indexBuffer.getMappedRange()).set(allIndices);
    this.indexBuffer.unmap();

    this.cameraBuffer = device.createBuffer({
      size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Load texture
    this.texture = device.createTexture({
      size: { width: 1, height: 1 },
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    this.sampler = device.createSampler({
      addressModeU: 'repeat',
      addressModeV: 'repeat',
      magFilter: 'linear',
      minFilter: 'linear',
    });

    // Create a fallback 1x1 white texture
    const whitePixel = new Uint8Array([255, 255, 255, 255]);
    device.queue.writeTexture(
      { texture: this.texture },
      whitePixel,
      { bytesPerRow: 4 },
      { width: 1, height: 1 }
    );

    // Load actual texture
    this._loadTexture(`${import.meta.env.BASE_URL}tiles.jpg`);

    const vertModule = device.createShaderModule({ code: TUB_VERT });
    const fragModule = device.createShaderModule({ code: TUB_FRAG });

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

    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.cameraBuffer } },
        { binding: 1, resource: this.texture.createView() },
        { binding: 2, resource: this.sampler },
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

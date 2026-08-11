import { W, H, type Grid } from '../types.ts';

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
  let baseWaterLevel = 80.0;
  let worldPos = vec3f(input.position.x, baseWaterLevel + h, input.position.y);

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
@group(0) @binding(2) var skyboxTexture: texture_2d<f32>;
@group(0) @binding(3) var skyboxSampler: sampler;
@group(0) @binding(4) var<uniform> timeData: vec4f;

const PI = 3.14159265359;

// Map a unit direction to equirectangular UV (latitude-longitude).
fn dirToEquirectUV(dir: vec3f) -> vec2f {
  let u = atan2(dir.z, dir.x) / (2.0 * PI) + 0.5;
  let v = acos(clamp(dir.y, -1.0, 1.0)) / PI;
  return vec2f(u, v);
}

fn fresnel_schlick(cosTheta: f32, F0: f32) -> f32 {
  return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

fn perturb_normal(worldPos: vec3f, baseNormal: vec3f, time: f32) -> vec3f {
  // Multiple octaves of sine waves for fine ripple detail
  let freq1 = 0.3;
  let freq2 = 0.7;
  let speed = 0.1;
  
  let p = worldPos.xz * 0.05 + time * speed;
  
  let n1 = vec2f(
    sin(p.x * freq1 + p.y * 0.5),
    sin(p.y * freq1 - p.x * 0.3)
  ) * 0.08;
  
  let n2 = vec2f(
    sin(p.x * freq2 - p.y * 0.7),
    sin(p.y * freq2 + p.x * 0.4)
  ) * 0.04;
  
  let perturbation = n1 + n2;
  return normalize(vec3f(
    baseNormal.x + perturbation.x,
    baseNormal.y,
    baseNormal.z + perturbation.y
  ));
}

@fragment
fn main(in: VertexOutput) -> @location(0) vec4f {
  let time = timeData.x;
  
  // Perturb normal for fine detail
  let n = perturb_normal(in.worldPos, normalize(in.normal), time);
  let v = normalize(in.viewDir);
  let lightDir = normalize(vec3f(0.2, 0.8, 0.3));
  
  // Fresnel effect
  let F0 = 0.02; // Water base reflectivity cosntant
  let fresnel = fresnel_schlick(max(dot(n, v), 0.0), F0);
  
  // Sample skybox reflection via equirect UV (scaled by sky intensity in timeData.y)
  let reflectDir = reflect(-v, n);
  let skyColor = textureSample(skyboxTexture, skyboxSampler, dirToEquirectUV(reflectDir)).rgb * timeData.y;
  
  // Water colors - darker for depth
  let waterColor = vec3f(0.0, 0.4, 0.55);
  let deepColor = vec3f(0.0, 0.15, 0.3);
  let depth = clamp(-in.worldPos.y * 0.05, 0.0, 1.0);
  let baseColor = mix(waterColor, deepColor, depth);
  
  // Ambient
  let ambient = vec3f(0.3) * baseColor;
  
  // Diffuse
  let nDotL = max(dot(n, lightDir), 0.0);
  let diffuse = nDotL * baseColor * (1.0 - fresnel * 0.8);
  
  // Specular - sharp highlight (128 exponent for crisp ripple sparkles)
  let h = normalize(lightDir + v);
  let spec = pow(max(dot(n, h), 0.0), 128.0) * fresnel * 1.5;
  
  // Combine: blend water color with sky reflection based on Fresnel
  let refracted = ambient + diffuse;
  let reflected = skyColor * fresnel;
  let finalColor = refracted + reflected + vec3f(spec);
  
  return vec4f(finalColor, 0.85);
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
  indexFormat!: GPUIndexFormat;
  bindGroupLayout!: GPUBindGroupLayout;
  bindGroup!: GPUBindGroup;
  cameraBuffer!: GPUBuffer;
  timeBuffer!: GPUBuffer;
  skyboxTexture!: GPUTexture;
  skyboxSampler!: GPUSampler;

  private _device: GPUDevice;

  constructor(device: GPUDevice, format: GPUTextureFormat, depthFormat: GPUTextureFormat, grid: Grid) {
    this._device = device;

    const positions = new Float32Array(grid.NUM_VERTS * 2);
    const indices: number[] = [];

    for (let iz = 0; iz < grid.NZ; iz++) {
      for (let ix = 0; ix < grid.NX; ix++) {
        const i = iz * grid.NX + ix;
        positions[i * 2] = -W / 2 + ix * grid.DELTA_X;
        positions[i * 2 + 1] = -H / 2 + iz * grid.DELTA_Z;
      }
    }

    for (let iz = 0; iz < grid.NH; iz++) {
      for (let ix = 0; ix < grid.NW; ix++) {
        const a = iz * grid.NX + ix;
        const b = a + 1;
        const c = a + grid.NX;
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

    // Use Uint32 for index buffer if vertex count exceeds Uint16 max (65535)
    const indexFormat: GPUIndexFormat = grid.NUM_VERTS > 65535 ? 'uint32' : 'uint16';
    this.indexFormat = indexFormat;
    const indexBufferSize = indices.length * (grid.NUM_VERTS > 65535 ? 4 : 2);
    const IndexArray = grid.NUM_VERTS > 65535 ? Uint32Array : Uint16Array;
    this.indexBuffer = device.createBuffer({
      size: indexBufferSize,
      usage: GPUBufferUsage.INDEX,
      mappedAtCreation: true,
    });
    new IndexArray(this.indexBuffer.getMappedRange()).set(indices);
    this.indexBuffer.unmap();

    this.cameraBuffer = device.createBuffer({
      size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.timeBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.skyboxSampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'repeat',
      addressModeV: 'clamp-to-edge',
    });

    const vertSrc = makeWaterVertexShader(grid.NX, grid.NZ, grid.DELTA_X, grid.DELTA_Z);
    const vertexModule = device.createShaderModule({ code: vertSrc });
    const fragmentModule = device.createShaderModule({ code: WATER_FRAG });

    this.bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
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

  dispose() {
    this.vertexBuffer.destroy();
    this.indexBuffer.destroy();
    this.cameraBuffer.destroy();
    this.timeBuffer.destroy();
  }

  updateCamera(viewProj: Float32Array, eyeX: number, eyeY: number, eyeZ: number) {
    const data = new Float32Array(20);
    data.set(viewProj, 0);
    data[16] = eyeX;
    data[17] = eyeY;
    data[18] = eyeZ;
    this._device.queue.writeBuffer(this.cameraBuffer, 0, data);
  }

  setSkyboxTexture(texture: GPUTexture) {
    this.skyboxTexture = texture;
  }

  updateTime(time: number, skyIntensity: number) {
    const data = new Float32Array(4);
    data[0] = time;
    data[1] = skyIntensity;
    this._device.queue.writeBuffer(this.timeBuffer, 0, data);
  }

  setStateBuffer(buffer: GPUBuffer) {
    this.bindGroup = this._device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer } },
        { binding: 1, resource: { buffer: this.cameraBuffer } },
        { binding: 2, resource: this.skyboxTexture.createView() },
        { binding: 3, resource: this.skyboxSampler },
        { binding: 4, resource: { buffer: this.timeBuffer } },
      ],
    });
  }

  render(pass: GPURenderPassEncoder) {
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setVertexBuffer(0, this.vertexBuffer);
    pass.setIndexBuffer(this.indexBuffer, this.indexFormat);
    pass.drawIndexed(this.indexCount);
  }
}

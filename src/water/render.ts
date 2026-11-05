import { W, H, WATER_LEVEL, type Grid } from '../types.ts';
import { makeTubMaterialShader } from '../scene/bathtub.ts';
import { SPHERE_SHADING } from '../scene/sphere.ts';
import { POOL_INTERSECTION_WGSL } from './refraction.ts';

const CAMERA = `
@group(0) @binding(1) var<uniform> camera: Camera;
struct Camera {
  viewProj: mat4x4f,
  eye: vec3f,
};
`;

function makeWaterVertexShader(nx: number, nz: number, dx: number, dz: number) {
  return `
@group(0) @binding(0) var<storage, read> heightState: array<vec2f>;
${CAMERA}

struct VertexInput {
  @location(0) position: vec2f,
};

struct VertexOutput {
  @builtin(position) clipPos: vec4f,
  @location(0) worldPos: vec3f,
  @location(1) normal: vec3f,
};

@vertex
fn main(@builtin(vertex_index) idx: u32, input: VertexInput) -> VertexOutput {
  let h = heightState[idx].y;
  let baseWaterLevel = ${WATER_LEVEL}.0;
  let worldPos = vec3f(input.position.x, baseWaterLevel + h, input.position.y);

  let nx = i32(${nx}u);
  let nz = i32(${nz}u);
  let i = i32(idx);

  let ix = i % nx;
  let iz = i / nx;

  // Mirror neighbors at the rigid pool walls, matching the simulation.
  let hL = heightState[iz * nx + abs(ix - 1)].y;
  let hR = heightState[iz * nx + min(ix + 1, 2 * (nx - 1) - ix - 1)].y;
  let hD = heightState[abs(iz - 1) * nx + ix].y;
  let hU = heightState[min(iz + 1, 2 * (nz - 1) - iz - 1) * nx + ix].y;

  let ddx = (hR - hL) / (2.0 * ${dx});
  let ddz = (hU - hD) / (2.0 * ${dz});
  let normal = normalize(vec3f(-ddx, 1.0, -ddz));

  var out: VertexOutput;
  out.clipPos = camera.viewProj * vec4f(worldPos, 1.0);
  out.worldPos = worldPos;
  out.normal = normal;
  return out;
}
`;
}

const WATER_FRAG = `
${CAMERA}
@group(0) @binding(2) var skyboxTexture: texture_2d<f32>;
@group(0) @binding(3) var skyboxSampler: sampler;
struct Lighting { sky: vec4f, sphere: vec4f };
@group(0) @binding(4) var<uniform> lighting: Lighting;
${makeTubMaterialShader(1)}
${SPHERE_SHADING}
${POOL_INTERSECTION_WGSL}

const WATER_IOR = 1.333;
const F0 = ((WATER_IOR - 1.0) / (WATER_IOR + 1.0)) * ((WATER_IOR - 1.0) / (WATER_IOR + 1.0));

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

fn sceneAlongRay(origin: vec3f, direction: vec3f) -> vec3f {
  let hit = intersectPool(origin, direction, lighting.sphere);
  if (hit.kind == 1u) {
    return shadeTub(origin + direction * hit.distance, hit.normal);
  }
  if (hit.kind == 2u) {
    return shadeSphere(hit.normal, -direction);
  }
  // No pool or ball intersection means this ray leaves the open top.
  return textureSampleLevel(skyboxTexture, skyboxSampler, dirToEquirectUV(direction), 0.0).rgb * lighting.sky.x;
}

@fragment
fn main(in: VertexOutput) -> @location(0) vec4f {
  // All visible ripples come from the simulated surface, including dispersion.
  let geometricNormal = normalize(in.normal);
  let v = normalize(camera.eye - in.worldPos);
  let fromAir = camera.eye.y >= in.worldPos.y;
  let n = select(-geometricNormal, geometricNormal, dot(geometricNormal, v) >= 0.0);
  let lightDir = normalize(vec3f(0.2, 0.8, 0.3));
  let refractDir = refract(-v, n, select(WATER_IOR, 1.0 / WATER_IOR, fromAir));
  let totalInternalReflection = dot(refractDir, refractDir) < 0.0001;
  let fresnel = select(fresnel_schlick(max(dot(n, v), 0.0), F0), 1.0, totalInternalReflection);
  
  // Sample skybox reflection via equirect UV, scaled by sky intensity.
  let reflectDir = reflect(-v, n);
  var reflected = textureSample(skyboxTexture, skyboxSampler, dirToEquirectUV(reflectDir)).rgb * lighting.sky.x;
  if (!fromAir) {
    reflected = sceneAlongRay(in.worldPos, reflectDir);
  }
  var transmitted = vec3f(0.0);
  if (!totalInternalReflection) {
    transmitted = sceneAlongRay(in.worldPos, refractDir);
  }
  
  // Specular - sharp highlight (128 exponent for crisp ripple sparkles)
  let h = normalize(lightDir + v);
  let spec = pow(max(dot(n, h), 0.0), 128.0) * fresnel * 1.5;
  
  // Both rasterized and refracted receivers use the same material functions.
  let finalColor = mix(transmitted, reflected, fresnel) + vec3f(spec);
  return vec4f(finalColor, 1.0);
}

struct VertexOutput {
  @builtin(position) clipPos: vec4f,
  @location(0) worldPos: vec3f,
  @location(1) normal: vec3f,
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
  lightingBuffer!: GPUBuffer;
  skyboxTexture!: GPUTexture;
  skyboxSampler!: GPUSampler;

  private _device: GPUDevice;
  private _lighting = new Float32Array(8);
  private _cameraData = new Float32Array(20);

  constructor(device: GPUDevice, format: GPUTextureFormat, depthFormat: GPUTextureFormat, grid: Grid, poolLayout: GPUBindGroupLayout) {
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

    this.lightingBuffer = device.createBuffer({
      size: 32,
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
        { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });

    this.pipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout, poolLayout] }),
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
        targets: [{ format }],
      },
      primitive: { topology: 'triangle-list' },
      depthStencil: {
        format: depthFormat,
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
    });
  }

  dispose() {
    this.vertexBuffer.destroy();
    this.indexBuffer.destroy();
    this.cameraBuffer.destroy();
    this.lightingBuffer.destroy();
  }

  updateCamera(viewProj: Float32Array, eyeX: number, eyeY: number, eyeZ: number) {
    this._cameraData.set(viewProj, 0);
    this._cameraData[16] = eyeX;
    this._cameraData[17] = eyeY;
    this._cameraData[18] = eyeZ;
    this._device.queue.writeBuffer(this.cameraBuffer, 0, this._cameraData);
  }

  setSkyboxTexture(texture: GPUTexture) {
    this.skyboxTexture = texture;
  }

  updateLighting(skyIntensity: number, spherePosition: readonly number[], sphereRadius: number) {
    this._lighting[0] = skyIntensity;
    this._lighting[4] = spherePosition[0];
    this._lighting[5] = spherePosition[1];
    this._lighting[6] = spherePosition[2];
    this._lighting[7] = sphereRadius;
    this._device.queue.writeBuffer(this.lightingBuffer, 0, this._lighting);
  }

  setStateBuffer(buffer: GPUBuffer) {
    this.bindGroup = this._device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer } },
        { binding: 1, resource: { buffer: this.cameraBuffer } },
        { binding: 2, resource: this.skyboxTexture.createView() },
        { binding: 3, resource: this.skyboxSampler },
        { binding: 4, resource: { buffer: this.lightingBuffer } },
      ],
    });
  }

  render(pass: GPURenderPassEncoder, poolBindGroup: GPUBindGroup) {
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setBindGroup(1, poolBindGroup);
    pass.setVertexBuffer(0, this.vertexBuffer);
    pass.setIndexBuffer(this.indexBuffer, this.indexFormat);
    pass.drawIndexed(this.indexCount);
  }
}

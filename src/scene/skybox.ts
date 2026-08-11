import { loadEXR } from '../utils/exr-loader.ts';
import { multiply, inverse } from '../math.ts';

const EQUIRECT_W = 4096;
const EQUIRECT_H = 2048;

// Fullscreen triangle: covers every pixel, so there are no cube edges to clip
// away at the view corners (the old cube skybox left gaps there).
const SKYBOX_VERT = `
struct VertexOutput {
  @builtin(position) clipPos: vec4f,
  @location(0) ndc: vec2f,
};

@vertex
fn main(@builtin(vertex_index) idx: u32) -> VertexOutput {
  var verts = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f( 3.0, -1.0),
    vec2f(-1.0,  3.0),
  );
  let p = verts[idx];
  var out: VertexOutput;
  // z = 1 puts the sky at the far plane; depthCompare less-equal lets it pass
  // against the cleared depth, and it never writes depth.
  out.clipPos = vec4f(p, 1.0, 1.0);
  out.ndc = p;
  return out;
}
`;

const SKYBOX_FRAG = `
struct SkyCamera {
  invViewProjRot: mat4x4f,
  // params.x = intensity (exposure multiplier); yzw reserved.
  params: vec4f,
};

@group(0) @binding(0) var<uniform> camera: SkyCamera;
@group(0) @binding(1) var skyTexture: texture_2d<f32>;
@group(0) @binding(2) var skySampler: sampler;

struct VertexOutput {
  @builtin(position) clipPos: vec4f,
  @location(0) ndc: vec2f,
};

const PI = 3.14159265359;

// Map a unit direction to equirectangular UV (latitude-longitude).
fn dirToEquirectUV(dir: vec3f) -> vec2f {
  let u = atan2(dir.z, dir.x) / (2.0 * PI) + 0.5;
  let v = acos(clamp(dir.y, -1.0, 1.0)) / PI;
  return vec2f(u, v);
}

@fragment
fn main(in: VertexOutput) -> @location(0) vec4f {
  // Camera sits at the origin (translation stripped), so unprojecting any point
  // on the pixel's ray and normalizing yields the world-space view direction.
  let world = camera.invViewProjRot * vec4f(in.ndc, 1.0, 1.0);
  let dir = normalize(world.xyz / world.w);
  let sky = textureSample(skyTexture, skySampler, dirToEquirectUV(dir)).rgb;
  return vec4f(sky * camera.params.x, 1.0);
}
`;

export class Skybox {
  pipeline!: GPURenderPipeline;
  bindGroup!: GPUBindGroup;
  cameraBuffer!: GPUBuffer;
  skyTexture!: GPUTexture;
  skySampler!: GPUSampler;

  private _device: GPUDevice;
  // Sky brightness multiplier (exposure). 1.0 = raw HDR values.
  intensity = 1.0;
  // Scratch buffers reused each frame to avoid per-frame allocation.
  private _viewRot = new Float32Array(16);
  private _pvRot = new Float32Array(16);
  private _inv = new Float32Array(16);
  // Upload staging: 16 floats matrix + 4 floats params = 20.
  private _upload = new Float32Array(20);
  // Current loaded skybox filename
  private _currentSkybox = 'citrus_orchard_puresky_4k.exr';
  // Callback for when initial load completes
  private _onReadyCallback?: () => void;
  constructor(device: GPUDevice, format: GPUTextureFormat, depthFormat: GPUTextureFormat) {
    this._device = device;

    // mat4x4f (64) + params vec4f (16) = 80 bytes.
    this.cameraBuffer = device.createBuffer({
      size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // HDR equirectangular sky texture (float16, uploaded directly from the EXR).
    this.skyTexture = device.createTexture({
      size: [EQUIRECT_W, EQUIRECT_H],
      format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    this.skySampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'repeat',        // wrap the 360-degree longitude seam
      addressModeV: 'clamp-to-edge', // clamp at the poles
    });

    // Load the equirectangular EXR sky asynchronously
    this._loadSky();

    const vertModule = device.createShaderModule({ code: SKYBOX_VERT });
    const fragModule = device.createShaderModule({ code: SKYBOX_FRAG });
    const bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });

    this.pipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      vertex: {
        module: vertModule,
        entryPoint: 'main',
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
        { binding: 1, resource: this.skyTexture.createView() },
        { binding: 2, resource: this.skySampler },
      ],
    });
  }

  updateCamera(view: Float32Array, proj: Float32Array) {
    // Strip translation so the camera sits at the origin: the sky only rotates.
    this._viewRot.set(view);
    this._viewRot[12] = 0;
    this._viewRot[13] = 0;
    this._viewRot[14] = 0;
    // invViewProjRot = inverse(proj * viewRot); fragment shader unprojects with it.
    multiply(proj, this._viewRot, this._pvRot);
    inverse(this._pvRot, this._inv);
    this._upload.set(this._inv, 0);
    this._upload[16] = this.intensity; // params.x
    this._device.queue.writeBuffer(this.cameraBuffer, 0, this._upload);
  }

  render(pass: GPURenderPassEncoder) {
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(3);
  }

  getTexture(): GPUTexture {
    return this.skyTexture;
  }

  async loadSkybox(filename: string) {
    if (this._currentSkybox === filename) return;
    this._currentSkybox = filename;
    await this._loadSky(filename);
  }

  getCurrentSkybox(): string {
    return this._currentSkybox;
  }

  onReady(callback: () => void) {
    this._onReadyCallback = callback;
  }

  private async _loadSky(filename?: string) {
    const skyFile = filename || this._currentSkybox;
    try {
      const { data, width, height } = await loadEXR(`${import.meta.env.BASE_URL}textures/${skyFile}`);
      if (width !== EQUIRECT_W || height !== EQUIRECT_H) {
        console.error(`Sky EXR is ${width}x${height}, expected ${EQUIRECT_W}x${EQUIRECT_H}`);
        return;
      }
      // data is float16 RGBA: 4 channels x 2 bytes per texel.
      this._device.queue.writeTexture(
        { texture: this.skyTexture },
        data,
        { bytesPerRow: width * 4 * 2, rowsPerImage: height },
        [width, height, 1]
      );
      
      // Notify that initial load is complete
      if (this._onReadyCallback) {
        this._onReadyCallback();
        this._onReadyCallback = undefined; // Only call once for initial load
      }
    } catch (err) {
      console.error('Failed to load EXR sky:', err);
    }
  }
}

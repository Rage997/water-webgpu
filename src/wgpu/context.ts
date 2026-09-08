export async function initWebGPU(canvas: HTMLCanvasElement) {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter found');
  const device = await adapter.requestDevice();
  device.lost.then((info) => {
    console.error('WebGPU device lost:', info.message);
  });
  const context = canvas.getContext('webgpu') as GPUCanvasContext;
  const format = navigator.gpu.getPreferredCanvasFormat();

  const depthFormat: GPUTextureFormat = 'depth24plus';

  function configure() {
    const width = Math.max(1, Math.round(canvas.clientWidth * devicePixelRatio));
    const height = Math.max(1, Math.round(canvas.clientHeight * devicePixelRatio));
    canvas.width = width;
    canvas.height = height;

    context.configure({
      device,
      format,
      alphaMode: 'premultiplied',
    });
    return { width, height };
  }

  let { width, height } = configure();

  let depthTexture = createDepthTexture(device, depthFormat, width, height);

  function resize() {
    const w = Math.max(1, Math.round(canvas.clientWidth * devicePixelRatio));
    const h = Math.max(1, Math.round(canvas.clientHeight * devicePixelRatio));
    if (w !== width || h !== height) {
      width = w; height = h;
      canvas.width = width;
      canvas.height = height;
      context.configure({ device, format, alphaMode: 'premultiplied' });
      depthTexture.destroy();
      depthTexture = createDepthTexture(device, depthFormat, width, height);
    }
  }

  return { device, context, format, depthFormat, depthTexture: () => depthTexture, resize, width: () => width, height: () => height };
}

function createDepthTexture(device: GPUDevice, format: GPUTextureFormat, width: number, height: number): GPUTexture {
  const texture = device.createTexture({
    size: { width, height },
    format,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  return texture;
}

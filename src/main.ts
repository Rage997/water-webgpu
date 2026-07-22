import { initWebGPU } from './wgpu/context.ts';
import { OrbitCamera } from './camera/orbit.ts';
import { WaterCompute } from './water/compute.ts';
import { WaterRender } from './water/render.ts';
import { Bathtub } from './scene/bathtub.ts';
import { Floor } from './scene/floor.ts';
import { Skybox } from './scene/skybox.ts';
import { setupInteraction } from './water/interaction.ts';
import type { ClickData } from './types.ts';

async function main() {
  const canvas = document.getElementById('gpu-canvas') as HTMLCanvasElement;
  if (!canvas) throw new Error('Canvas not found');

  const ctx = await initWebGPU(canvas);
  const { device, context, format, depthFormat, depthTexture, resize, width, height } = ctx;

  const camera = new OrbitCamera();
  camera.attach(canvas);
  camera.updateAspect(width() / height());
  camera.updateView();

  const waterCompute = new WaterCompute(device);
  const waterRender = new WaterRender(device, format, depthFormat);
  waterRender.setStateBuffer(waterCompute.currentStateBuffer);

  const bathtub = new Bathtub(device, format, depthFormat);
  const floor = new Floor(device, format, depthFormat);
  const skybox = new Skybox(device, format, depthFormat);

  let clickData: ClickData | null = null;

  setupInteraction(canvas, camera, (x: number, z: number) => {
    clickData = { active: true, x, z };
  });

  let lastTime = performance.now();

  function frame() {
    const now = performance.now();
    const rawDt = now - lastTime;
    lastTime = now;

    resizing();
    const aspect = width() / height();
    camera.update(aspect);

    const commandEncoder = device.createCommandEncoder();

    // Simulate water
    waterCompute.simulate(commandEncoder, rawDt, clickData);
    if (clickData) clickData = null;

    // Rebind state buffer if it swapped
    waterRender.setStateBuffer(waterCompute.currentStateBuffer);

    // Update camera uniforms
    const vp = camera.viewProjMatrix;
    waterRender.updateCamera(vp, camera.eyeX, camera.eyeY, camera.eyeZ);
    bathtub.updateCamera(vp);
    floor.updateCamera(vp);
    skybox.updateCamera(vp);

    // Render
    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: 'clear',
        clearValue: { r: 0.4, g: 0.6, b: 0.8, a: 1.0 },
        storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: depthTexture().createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });

    skybox.render(renderPass);
    bathtub.render(renderPass);
    floor.render(renderPass);
    waterRender.render(renderPass);

    renderPass.end();

    device.queue.submit([commandEncoder.finish()]);

    requestAnimationFrame(frame);
  }

  function resizing() {
    const w = Math.max(1, canvas.clientWidth * devicePixelRatio);
    const h = Math.max(1, canvas.clientHeight * devicePixelRatio);
    if (w !== width() || h !== height()) {
      resize();
    }
  }

  requestAnimationFrame(frame);
}

main().catch(console.error);

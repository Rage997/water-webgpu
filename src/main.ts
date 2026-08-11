import { initWebGPU } from './wgpu/context.ts';
import { OrbitCamera } from './camera/orbit.ts';
import { WaterCompute } from './water/compute.ts';
import { WaterRender } from './water/render.ts';
import { Bathtub } from './scene/bathtub.ts';
import { Floor } from './scene/floor.ts';
import { Skybox } from './scene/skybox.ts';
import { setupInteraction } from './water/interaction.ts';
import type { ClickData } from './types.ts';

declare global {
  interface Window {
    skyControl: Skybox;
    waterControl: WaterCompute;
  }
}

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

  const bathtub = new Bathtub(device, format, depthFormat);
  const floor = new Floor(device, format, depthFormat);
  const skybox = new Skybox(device, format, depthFormat);

  // Hide initial loading panel when skybox is ready
  const loadingPanel = document.getElementById('loading-panel') as HTMLDivElement;
  skybox.onReady(() => {
    loadingPanel.classList.remove('visible');
  });

  // Set skybox texture on water renderer
  waterRender.setSkyboxTexture(skybox.getTexture());
  waterRender.setStateBuffer(waterCompute.currentStateBuffer);

  // Expose controls for UI and console
  window.skyControl = skybox;
  window.waterControl = waterCompute;

  // Wire up UI controls
  const skyboxSelect = document.getElementById('skybox-select') as HTMLSelectElement;
  const skyIntensitySlider = document.getElementById('sky-intensity') as HTMLInputElement;
  const skyIntensityValue = document.getElementById('sky-intensity-value') as HTMLSpanElement;
  const rippleStrengthSlider = document.getElementById('ripple-strength') as HTMLInputElement;
  const rippleStrengthValue = document.getElementById('ripple-strength-value') as HTMLSpanElement;
  const rippleDampingSlider = document.getElementById('ripple-damping') as HTMLInputElement;
  const rippleDampingValue = document.getElementById('ripple-damping-value') as HTMLSpanElement;
  const ambientFrequencySlider = document.getElementById('ambient-frequency') as HTMLInputElement;
  const ambientFrequencyValue = document.getElementById('ambient-frequency-value') as HTMLSpanElement;
  const ambientStrengthSlider = document.getElementById('ambient-strength') as HTMLInputElement;
  const ambientStrengthValue = document.getElementById('ambient-strength-value') as HTMLSpanElement;

  // Skybox selection
  skyboxSelect.addEventListener('change', async () => {
    loadingPanel.classList.add('visible');
    skyboxSelect.disabled = true;
    
    // Show loading panel for minimum 500ms so user sees it
    const minDisplayTime = new Promise(resolve => setTimeout(resolve, 500));
    
    try {
      await Promise.all([
        skybox.loadSkybox(skyboxSelect.value),
        minDisplayTime
      ]);
    } catch (error) {
      console.error('Failed to load skybox:', error);
    } finally {
      loadingPanel.classList.remove('visible');
      skyboxSelect.disabled = false;
    }
  });

  // Sky intensity control
  skyIntensitySlider.addEventListener('input', () => {
    const value = parseFloat(skyIntensitySlider.value);
    skybox.intensity = value;
    skyIntensityValue.textContent = value.toFixed(1);
  });

  // Ripple strength control
  rippleStrengthSlider.addEventListener('input', () => {
    const value = parseFloat(rippleStrengthSlider.value);
    waterCompute.rippleStrength = value;
    rippleStrengthValue.textContent = value.toFixed(1);
  });

  // Ripple damping control
  rippleDampingSlider.addEventListener('input', () => {
    const value = parseFloat(rippleDampingSlider.value);
    waterCompute.rippleDamping = value;
    rippleDampingValue.textContent = value.toFixed(3);
  });

  // Ambient wave frequency control
  ambientFrequencySlider.addEventListener('input', () => {
    const value = parseFloat(ambientFrequencySlider.value);
    waterCompute.ambientFrequency = value;
    ambientFrequencyValue.textContent = value.toFixed(2);
  });

  // Ambient wave strength control
  ambientStrengthSlider.addEventListener('input', () => {
    const value = parseFloat(ambientStrengthSlider.value);
    waterCompute.ambientStrength = value;
    ambientStrengthValue.textContent = value.toFixed(1);
  });

  let clickData: ClickData | null = null;

  setupInteraction(canvas, camera, (x: number, z: number) => {
    clickData = { active: true, x, z };
  });

  let lastTime = performance.now();
  let elapsedTime = 0;

  function frame() {
    const now = performance.now();
    const rawDt = now - lastTime;
    elapsedTime += rawDt * 0.001; // Convert to seconds
    lastTime = now;

    resizing();
    const aspect = width() / height();
    camera.update(aspect);

    const commandEncoder = device.createCommandEncoder();

    // Simulate water
    waterCompute.simulate(commandEncoder, rawDt, clickData, elapsedTime);
    if (clickData) clickData = null;

    // Rebind state buffer if it swapped
    waterRender.setStateBuffer(waterCompute.currentStateBuffer);

    // Update camera uniforms
    const vp = camera.viewProjMatrix;
    waterRender.updateCamera(vp, camera.eyeX, camera.eyeY, camera.eyeZ);
    bathtub.updateCamera(vp);
    floor.updateCamera(vp);
    skybox.updateCamera(camera.viewMatrix, camera.projMatrix);
    waterRender.updateTime(elapsedTime, skybox.intensity);

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

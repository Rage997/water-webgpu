import { initWebGPU } from './wgpu/context.ts';
import { OrbitCamera } from './camera/orbit.ts';
import { WaterCompute } from './water/compute.ts';
import { WaterRender } from './water/render.ts';
import { Bathtub } from './scene/bathtub.ts';
import { Floor } from './scene/floor.ts';
import { Skybox } from './scene/skybox.ts';
import { setupInteraction } from './water/interaction.ts';
import { makeGrid, RESOLUTION_TIERS } from './types.ts';
import type { ClickData, ResolutionTier } from './types.ts';

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

  let grid = makeGrid(RESOLUTION_TIERS.high);
  let waterCompute = new WaterCompute(device, grid);
  let waterRender = new WaterRender(device, format, depthFormat, grid);

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
  const waveSpeedSlider = document.getElementById('wave-speed') as HTMLInputElement;
  const waveSpeedValue = document.getElementById('wave-speed-value') as HTMLSpanElement;
  const dampingSlider = document.getElementById('damping') as HTMLInputElement;
  const dampingValue = document.getElementById('damping-value') as HTMLSpanElement;
  const rippleAmplitudeSlider = document.getElementById('ripple-amplitude') as HTMLInputElement;
  const rippleAmplitudeValue = document.getElementById('ripple-amplitude-value') as HTMLSpanElement;
  const rippleSizeSlider = document.getElementById('ripple-size') as HTMLInputElement;
  const rippleSizeValue = document.getElementById('ripple-size-value') as HTMLSpanElement;
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

  // Resolution selection: rebuilds the water grid (compute + render) in place,
  // preserving camera and current sim settings. Water surface resets (expected).
  const resolutionSelect = document.getElementById('resolution-select') as HTMLSelectElement;
  resolutionSelect.addEventListener('change', () => {
    const nh = RESOLUTION_TIERS[resolutionSelect.value as ResolutionTier];
    if (!nh) return;

    const prevCompute = waterCompute;
    const prevRender = waterRender;

    grid = makeGrid(nh);
    waterCompute = new WaterCompute(device, grid);
    // Carry the live control values over so changing resolution doesn't reset them.
    waterCompute.waveSpeed = prevCompute.waveSpeed;
    waterCompute.damping = prevCompute.damping;
    waterCompute.rippleAmplitude = prevCompute.rippleAmplitude;
    waterCompute.rippleSize = prevCompute.rippleSize;
    waterCompute.ambientFrequency = prevCompute.ambientFrequency;
    waterCompute.ambientStrength = prevCompute.ambientStrength;

    waterRender = new WaterRender(device, format, depthFormat, grid);
    waterRender.setSkyboxTexture(skybox.getTexture());
    waterRender.setStateBuffer(waterCompute.currentStateBuffer);

    prevCompute.dispose();
    prevRender.dispose();
    window.waterControl = waterCompute;
  });

  // Sky intensity control
  skyIntensitySlider.addEventListener('input', () => {
    const value = parseFloat(skyIntensitySlider.value);
    skybox.intensity = value;
    skyIntensityValue.textContent = value.toFixed(1);
  });

  // Wave speed control (propagation speed; dt is CFL-clamped so it stays stable)
  waveSpeedSlider.addEventListener('input', () => {
    const value = parseFloat(waveSpeedSlider.value);
    waterCompute.waveSpeed = value;
    waveSpeedValue.textContent = value.toFixed(2);
  });

  // Damping control (0..1 wave decay, mapped to a stable drag rate inside the sim)
  dampingSlider.addEventListener('input', () => {
    const value = parseFloat(dampingSlider.value);
    waterCompute.damping = value;
    dampingValue.textContent = value.toFixed(2);
  });

  // Ripple amplitude control (click splash height)
  rippleAmplitudeSlider.addEventListener('input', () => {
    const value = parseFloat(rippleAmplitudeSlider.value);
    waterCompute.rippleAmplitude = value;
    rippleAmplitudeValue.textContent = value.toFixed(0);
  });

  // Ripple size control (click splash width)
  rippleSizeSlider.addEventListener('input', () => {
    const value = parseFloat(rippleSizeSlider.value);
    waterCompute.rippleSize = value;
    rippleSizeValue.textContent = value.toFixed(1);
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

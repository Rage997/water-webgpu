import { initWebGPU } from './wgpu/context.ts';
import { OrbitCamera } from './camera/orbit.ts';
import { WaterCompute } from './water/compute.ts';
import { WaterRender } from './water/render.ts';
import { Caustics } from './water/caustics.ts';
import { Bathtub } from './scene/bathtub.ts';
import { Floor } from './scene/floor.ts';
import { Skybox } from './scene/skybox.ts';
import { Rain } from './scene/rain.ts';
import { DragSphere } from './scene/sphere.ts';
import { setupInteraction } from './water/interaction.ts';
import { makeGrid, RESOLUTION_TIERS, SIGMA } from './types.ts';
import type { ClickData, ResolutionTier } from './types.ts';
import { enableSettings, hideLoading, showError, showLoading } from './ui.ts';

declare global {
  interface Window {
    skyControl: Skybox;
    waterControl: WaterCompute;
    causticsControl: Caustics;
  }
}

async function main() {
  const canvas = document.getElementById('gpu-canvas') as HTMLCanvasElement;
  if (!canvas) throw new Error('Canvas not found');

  if (!navigator.gpu) {
    throw new Error('WebGPU is unavailable. Open this page in a WebGPU-capable browser over HTTPS or localhost.');
  }
  const ctx = await initWebGPU(canvas);
  const { device, context, format, depthFormat, depthTexture, resize, width, height } = ctx;

  const camera = new OrbitCamera();
  camera.attach(canvas);
  camera.updateAspect(width() / height());
  camera.updateView();

  let grid = makeGrid(RESOLUTION_TIERS.high);
  let waterCompute = new WaterCompute(device, grid);
  const bathtub = new Bathtub(device, format, depthFormat);
  let waterRender = new WaterRender(device, format, depthFormat, grid, bathtub.pipeline.getBindGroupLayout(0));
  let caustics = new Caustics(device, grid);

  const floor = new Floor(device, format, depthFormat);
  const skybox = new Skybox(device, format, depthFormat);
  const rain = new Rain(device, format, depthFormat);
  const sphere = new DragSphere(device, format, depthFormat);
  showLoading('Loading the environment', 'The water is ready. Preparing the light…');
  skybox.ready.then(enableSettings, (error: unknown) => {
    console.error('Failed to load initial environment:', error);
    showError('Environment could not load', 'Check your connection, then reload the experiment.', true);
  });
  device.lost.then(() => {
    showError('The GPU connection was lost', 'Reload the experiment to start again.', true);
  });

  // Set skybox texture on water renderer
  waterRender.setSkyboxTexture(skybox.getTexture());
  waterRender.setStateBuffer(waterCompute.currentStateBuffer);

  // Expose controls for UI and console
  window.skyControl = skybox;
  window.waterControl = waterCompute;
  window.causticsControl = caustics;
  bathtub.setCausticTextures(caustics.causticTextures);
  (window as any).gpuDevice = device; // for the verification harness

  // Wire up UI controls
  const skyboxSelect = document.getElementById('skybox-select') as HTMLSelectElement;
  const skyIntensitySlider = document.getElementById('sky-intensity') as HTMLInputElement;
  const skyIntensityValue = document.getElementById('sky-intensity-value') as HTMLSpanElement;
  const simulationSpeedSlider = document.getElementById('simulation-speed') as HTMLInputElement;
  const simulationSpeedValue = document.getElementById('simulation-speed-value') as HTMLSpanElement;
  waterCompute.simulationSpeed = parseFloat(simulationSpeedSlider.value);
  const waveSpeedSlider = document.getElementById('wave-speed') as HTMLInputElement;
  const waveSpeedValue = document.getElementById('wave-speed-value') as HTMLSpanElement;
  const dampingSlider = document.getElementById('damping') as HTMLInputElement;
  const dampingValue = document.getElementById('damping-value') as HTMLSpanElement;
  const viscositySlider = document.getElementById('viscosity') as HTMLInputElement;
  const viscosityValue = document.getElementById('viscosity-value') as HTMLSpanElement;
  viscositySlider.addEventListener('input', () => {
    waterCompute.flow.viscosity = parseFloat(viscositySlider.value);
    viscosityValue.textContent = viscositySlider.value;
  });
  const rippleAmplitudeSlider = document.getElementById('ripple-amplitude') as HTMLInputElement;
  const rippleAmplitudeValue = document.getElementById('ripple-amplitude-value') as HTMLSpanElement;
  const rippleSizeSlider = document.getElementById('ripple-size') as HTMLInputElement;
  const rippleSizeValue = document.getElementById('ripple-size-value') as HTMLSpanElement;
  const causticsToggle = document.getElementById('caustics-toggle') as HTMLInputElement;
  const causticsIntensitySlider = document.getElementById('caustics-intensity') as HTMLInputElement;
  const causticsIntensityValue = document.getElementById('caustics-intensity-value') as HTMLSpanElement;
  const rainToggle = document.getElementById('rain-toggle') as HTMLInputElement;
  rain.enabled = rainToggle.checked;
  rainToggle.addEventListener('input', () => {
    rain.enabled = rainToggle.checked;
  });

  const sphereToggle = document.getElementById('sphere-toggle') as HTMLInputElement;
  sphere.enabled = sphereToggle.checked;
  sphereToggle.addEventListener('input', () => {
    sphere.enabled = sphereToggle.checked;
  });

  const rainStrengthSlider = document.getElementById('rain-strength') as HTMLInputElement;
  const rainStrengthValue = document.getElementById('rain-strength-value') as HTMLSpanElement;
  const rainSizeSlider = document.getElementById('rain-size') as HTMLInputElement;
  const rainSizeValue = document.getElementById('rain-size-value') as HTMLSpanElement;
  const rainVarietySlider = document.getElementById('rain-variety') as HTMLInputElement;
  const rainVarietyValue = document.getElementById('rain-variety-value') as HTMLSpanElement;
  rain.strength = parseFloat(rainStrengthSlider.value);
  rain.size = parseFloat(rainSizeSlider.value);
  rain.variety = parseFloat(rainVarietySlider.value);
  rainStrengthSlider.addEventListener('input', () => {
    rain.strength = parseFloat(rainStrengthSlider.value);
    rainStrengthValue.textContent = rain.strength.toFixed(0);
  });
  rainSizeSlider.addEventListener('input', () => {
    rain.size = parseFloat(rainSizeSlider.value);
    rainSizeValue.textContent = rain.size.toFixed(1);
  });
  rainVarietySlider.addEventListener('input', () => {
    rain.variety = parseFloat(rainVarietySlider.value);
    rainVarietyValue.textContent = rain.variety.toFixed(2);
  });

  // Skybox selection
  skyboxSelect.addEventListener('change', async () => {
    showLoading('Changing the light', 'Loading the selected environment…');
    skyboxSelect.disabled = true;
    try {
      await skybox.loadSkybox(skyboxSelect.value);
      hideLoading();
    } catch (error) {
      console.error('Failed to load skybox:', error);
      skyboxSelect.value = skybox.getCurrentSkybox();
      showError('Environment could not load', 'Your previous environment is still active. Try another, or reload.');
    } finally {
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
    waterCompute.simulationSpeed = prevCompute.simulationSpeed;
    waterCompute.waveSpeed = prevCompute.waveSpeed;
    waterCompute.damping = prevCompute.damping;
    waterCompute.rippleAmplitude = prevCompute.rippleAmplitude;
    waterCompute.rippleSize = prevCompute.rippleSize;
    waterCompute.flow.viscosity = prevCompute.flow.viscosity;
    const prevCaustics = caustics;
    waterRender = new WaterRender(device, format, depthFormat, grid, bathtub.pipeline.getBindGroupLayout(0));
    waterRender.setSkyboxTexture(skybox.getTexture());
    waterRender.setStateBuffer(waterCompute.currentStateBuffer);
    caustics = new Caustics(device, grid);
    window.causticsControl = caustics;
    bathtub.setCausticTextures(caustics.causticTextures);

    prevCompute.dispose();
    prevRender.dispose();
    prevCaustics.dispose();
    window.waterControl = waterCompute;
  });

  // Sky intensity control
  skyIntensitySlider.addEventListener('input', () => {
    const value = parseFloat(skyIntensitySlider.value);
    skybox.intensity = value;
    skyIntensityValue.textContent = value.toFixed(1);
  });

  simulationSpeedSlider.addEventListener('input', () => {
    waterCompute.simulationSpeed = parseFloat(simulationSpeedSlider.value);
    simulationSpeedValue.textContent = `${waterCompute.simulationSpeed.toFixed(2)}×`;
  });

  // Wave speed control; fixed substeps are CFL-safe throughout the UI range.
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

  let causticStrength = causticsToggle.checked ? parseFloat(causticsIntensitySlider.value) : 0;

  causticsToggle.addEventListener('input', () => {
    updateCausticStrength();
  });

  // Caustics intensity control
  causticsIntensitySlider.addEventListener('input', () => {
    causticsIntensityValue.textContent = parseFloat(causticsIntensitySlider.value).toFixed(1);
    updateCausticStrength();
  });

  function updateCausticStrength() {
    const slider = parseFloat(causticsIntensitySlider.value);
    causticStrength = causticsToggle.checked ? slider : 0;
  }

  const clicks: ClickData[] = [];

  setupInteraction(canvas, camera, (x: number, z: number, strength: number) => {
    if (!waterCompute.paused && waterCompute.simulationSpeed > 0) clicks.push({ x, z, strength });
  }, () => 3 * Math.sqrt(waterCompute.rippleSize / SIGMA), sphere);

  let lastTime = performance.now();

  function frame() {
    const now = performance.now();
    const rawDt = now - lastTime;
    lastTime = now;

    resizing();
    const aspect = width() / height();
    camera.update(aspect);

    // The ball chases the pointer target even while the sim is paused.
    sphere.update(rawDt);

    const commandEncoder = device.createCommandEncoder();
    rain.update(commandEncoder, rawDt, waterCompute.currentStateBuffer, grid, camera.viewProjMatrix,
      waterCompute.paused ? 0 : waterCompute.simulationSpeed);

    // Simulate water; the ball displaces the surface and stirs the current.
    waterCompute.simulate(commandEncoder, rawDt, clicks, {
      x: sphere.position[0],
      y: sphere.position[1],
      z: sphere.position[2],
      radius: sphere.radius,
      active: sphere.enabled,
    });
    clicks.length = 0;

    // Rebind state buffer if it swapped
    waterRender.setSkyboxTexture(skybox.getTexture());
    waterRender.setStateBuffer(waterCompute.currentStateBuffer);
    caustics.update(commandEncoder, waterCompute.currentStateBuffer);

    // Update camera uniforms
    const vp = camera.viewProjMatrix;
    waterRender.updateCamera(vp, camera.eyeX, camera.eyeY, camera.eyeZ);
    bathtub.updateCamera(vp, causticStrength);
    floor.updateCamera(vp);
    sphere.updateCamera(vp, camera.eyeX, camera.eyeY, camera.eyeZ);
    skybox.updateCamera(camera.viewMatrix, camera.projMatrix);
    waterRender.updateLighting(skybox.intensity, sphere.position, sphere.enabled ? sphere.radius : 0);

    // Exact pool/sphere intersections let water share the ordinary scene pass.
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
    if (sphere.enabled) sphere.render(renderPass);
    waterRender.render(renderPass, bathtub.bindGroup);
    rain.render(renderPass);
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

main().catch((error: unknown) => {
  console.error(error);
  showError('Unable to start the experiment', error instanceof Error ? error.message : 'Check WebGPU support and reload to try again.', true);
});

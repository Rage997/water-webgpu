# Water Wave WebGPU

Real-time water simulation on WebGPU: a 2D Navier–Stokes current field coupled to a dispersive wave surface.

A learning project on how water simulation and rendering work on the GPU. three.js is used only to load the cubemap.

[Live demo](https://rage997.github.io/water-webgpu)

## Requirements

- Modern browser with WebGPU

## Development

```bash
npm install
npm run dev
```

Open `http://localhost:5173`.

## Controls

- **Left-click the water** — splash; **hold and drag** — trail of ripples (camera stays put).
- **Left-drag the ball down/up** — submerge it or lift it out of the water. **Shift-drag** moves it horizontally at its current depth. The grab point stays under the cursor; walls and the floor constrain movement. Release to leave it at the chosen position.
- **Right-drag** — orbit camera; **scroll** — zoom.
- The on-screen panel groups settings into **Water**, **Ripples & rain**, **Light & environment**, and **Quality**. Expand a section to explore its controls.
- Use the panel's chevron to hide or show settings; **Escape** closes it and returns focus to the toggle. On phones, settings start collapsed in a bottom sheet so the water stays reachable.
- The sun/moon switch at the top of the settings panel toggles dark and light UI themes and remains accessible when the panel is collapsed. Dark is the default; your choice is saved locally and restored before the page is painted. This does not change the scene lighting.
- Sliders support keyboard arrow keys and show their current values. Simulation speed at zero pauses the experiment; changing resolution resets the water while retaining parameter values.
- Rain and caustics settings become unavailable when their effect is switched off, without losing their values.
- Loading, unavailable WebGPU, lost-device, and environment-load errors appear on screen. A failed environment selection retains the previous environment; select another or reload.
- On touch screens, drag the ball to submerge or lift it, drag the water to make ripples. Horizontal-only ball dragging, camera orbit, and zoom require a keyboard, mouse, or trackpad.

## How it works

A 2D Stable Fluids solver (Stam's *Real-Time Fluid Dynamics for Games*) advects a dispersive height-field wave model:

```text
∂u/∂t + (u · ∇)u = ν ∇²u − ∇p + f,    ∇ · u = 0
h_tt + γ h_t = c² (L h − ℓ² L² h)
```

- Fixed timestep from the dispersive CFL bound; an accumulator runs all due substeps.
- Rigid walls use zero normal slope with mirrored stencils.
- Pointer and rain impulses conserve volume: they redistribute water rather than filling or draining the pool.
- The ball creates localized ripples with reduced wave strength; the remaining displaced volume changes the pool's mean level. Wet horizontal motion pushes currents, and existing waves keep propagating when the ball stops.
- This is a height-field surface over a 2D current field, not a 3D free-surface solver..

[Full details: solver setup, caustics](docs/how-it-works.md)

## Build

```bash
npm run build
npm run preview
```

## Architecture

- `src/water/` — simulation: flow, waves, splash, caustics
- `src/scene/` — scene components: skybox, bathtub, floor, rain
- `src/camera/` — orbit camera
- `src/wgpu/` — WebGPU context
- `src/shaders/` — WGSL shaders

## Tests

`tests/water-physics.html`

## Planned work

1. **Finite-depth gravity dispersion** — extend the capillary-style height-field model
2. **Refraction** — scene distortion through the water surface
3. **Foam/spray** — particle effects at wave peaks

## Resources

- Sky: [polyhaven](https://polyhaven.com/hdris/skies)

## License

MIT

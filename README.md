# Water Wave WebGPU

Real-time water simulation using WebGPU: a 2D Navier–Stokes current field coupled to a dispersive wave surface.

This is a learning project to learn how water based simulation/rendering works using WebGPU. I am relying on three.js only to load the cubemap.

[Live demo](https://rage997.github.io/water-webgpu)

## Requirements

- Modern browser with WebGPU

## Development

```bash
npm install
npm run dev
```

Open `http://localhost:5173`

## Pointer controls

- **Left-click** adds a ripple immediately. **Hold left and drag** across the water
  to stir horizontal currents and create ripples without moving the camera.
- **Right-drag** orbits the camera; **scroll** zooms.
- **Ripple Amplitude** and **Ripple Size** control both clicks and drag ripples.
  The pointer presses a shallow depression into the water (depth = 20% of amplitude,
  or 8 world units at the default 40), rather than adding height on every movement.
  Repeated overlapping samples do not stack into a tall mound.
  Release or cancel the gesture to stop; positions outside the pool do not disturb it.
- **Flow Viscosity** controls diffusion of horizontal momentum (0–100 world units²/s,
  default 8). Higher values smooth and dissipate currents more quickly. This is
  distinct from **Damping**, which controls vertical wave decay.

## Rain

**Rain** is enabled by default and is the only automatic source of random water disturbances.
It replaces the original ambient-ripple forcing and its frequency/strength controls.
Impacts at the nominal water level produce a depression surrounded by a raised rim,
updating both visible ripples and caustics. The impact is a discrete Laplacian of a
Gaussian with reflected image sources at pool walls. Its area-weighted displacement
sums to zero (up to floating-point roundoff), including near walls and corners:
rain redistributes water instead of steadily lowering the pool. Both height time
levels receive the same displacement, so impacts do not inject net vertical momentum.
No periodic reset or global water-level correction is used. Live controls:

- **Rain Strength**: impact depth scale, 0–5 world units in the UI (default 2). Zero keeps visible rain but removes its water disturbance. The resolved depth depends on grid spacing and wall reflections.
- **Rain Size**: base Gaussian standard deviation, 4–16 world units (default 8).
- **Rain Variety**: 0–1 (default 0.5). Zero makes drops uniform; higher values vary
  streak length/opacity, fall speed, and splash depth/width using fresh samples each spawn.
  Larger drops make wider, deeper splashes. Positions remain random at every setting.

The corresponding public variables are `Rain.strength`, `Rain.size`, and `Rain.variety`
in `src/scene/rain.ts`. Strength and size are base values; variety can produce impacts
above or below them. High strength combined with broad splashes produces exaggerated waves.

Turning rain off removes the drops and stops new impacts; existing waves continue to settle.
The rain toggle and tuning survive water-resolution changes. Pointer splashes remain
independent, controlled by **Ripple Amplitude** and **Ripple Size**. With rain off and
no clicks or drags, no new disturbances are injected. Long frames are capped to avoid catch-up storms.

## Navier–Stokes flow and water surface

`src/water/flow.ts` implements the fixed-domain 2D **Stable Fluids** method in
Jos Stam's *Real-Time Fluid Dynamics for Games* (the supplied `StamFluidforGames.pdf`,
sections “Moving Densities,” “Evolving Velocities,” and “Extensions”):

```text
∂u/∂t + (u · ∇)u = ν ∇²u − ∇p + f
∇ · u = 0
```

Each fixed substep performs implicit velocity diffusion, pressure projection,
bilinear semi-Lagrangian velocity self-advection, and a second projection.
On the GPU, ping-pong Jacobi iterations replace Stam's in-place Gauss–Seidel
iterations: 20 for diffusion and 60 for each pressure solve, without
cross-workgroup read/write races. Diffusion retains a fixed right-hand side;
pressure is reset for each solve. The rectangular-grid operators account for
both cell spacings. Projection is approximate, not an exact zero-divergence solve.

The cell-centred current grid has up to 100 rows along the pool and proportional
columns, independent of the denser rendering mesh. Implicit ghost samples reflect
normal velocity and copy tangential velocity and pressure at solid free-slip walls.
Dragging supplies localized Gaussian momentum impulses proportional to accumulated
pointer travel; a new gesture or crossing outside the pool starts a new segment.
Impulses survive frames without a physics step. Velocity uses world units/second;
flow timesteps convert the existing wave clock from milliseconds to seconds.

The flow transports **both** height time levels with the same backtrace, then the
existing shallow-water, capillary-style dispersive wave model advances them:

```text
h_tt + gamma h_t = c² (L h - length² L² h)
omega² = c² k² (1 + length² k²)
```

`L` is the discrete X–Z Laplacian. `src/water/compute.ts` contains the governing
equation, mirrored-wall stencil, and centered damped-Verlet update. The dispersion
length is 3 world units: shorter ripples travel faster.

**Scope:** this is a 2D horizontal Navier–Stokes flow coupled to a height-field
water surface, not a 3D free-surface liquid solver. As Stam notes in “Extensions,”
a true water/air interface requires additional tracking and interface boundary
conditions. There are no overturning waves, splashes/spray, air phase, or exact
volume conservation here. Surface transport is passive: waves do not feed forces
back into the current field. Rain excites the wave surface, not horizontal currents.

- A fixed timestep is chosen per resolution from the dispersive CFL bound, safe
  across the supported wave-speed range (0–2; larger console values are capped).
  An accumulator runs all due substeps and retains the fractional remainder.
  Frame durations above 100 ms are capped to avoid suspended-tab catch-up.
- Rigid pool walls use zero normal slope rather than pinning the edge height.
  Wave heights, surface normals, and caustic gradients share that boundary.
- Pointer and rain disturbances modify both height time levels equally.
  Rain conserves discrete volume; the pointer's artistic pressure brush does not.
- Surface shading now follows the simulated normals; independent animated
  sine-wave normal perturbations have been removed.

Run `/water-webgpu/tests/water-physics.html` through the development server for
GPU regressions covering frame-rate-independent flow and waves, pressure projection
at interior/wall impulses, reset of currents, surface transport, dispersive mode
propagation, finest-grid stability, clicks on zero-step frames, and sustained rain
volume conservation at coarse/fine resolutions with narrow/broad impacts.

Changing resolution resets both fields and preserves viscosity and other live
controls. `waterControl.zeroState()` also clears currents; `waterControl.paused`
freezes both solvers. GPU resources are released when the simulation is rebuilt.


## Build

```bash
npm run build
npm run preview
```

## Architecture

- `src/water/` - Water simulation (compute + render)
- `src/water/caustics.ts` / `caustic-footprints.ts` - Caustics
- `src/scene/` - 3D scene components (skybox, bathtub, floor)
- `src/scene/rain.ts` - Rain streaks and batched water-impact compute pass
- `src/camera/` - Orbit camera controls
- `src/wgpu/` - WebGPU context management
- `src/shaders/` - WGSL shader code

## Interview preparation

- [Project explanation and numerical-method notes](docs/project-interview-notes.md)
- [Aero Software Developer interview preparation](docs/aero-interview-preparation.md)
- [Motivation letter in LaTeX](docs/motivation-letter.tex)

## Resources

The skymap was taken from [polyhaven](https://polyhaven.com/hdris/skies)

## Planned work

1. **Finite-depth gravity dispersion** - Extend the capillary-style height-field model
2. **Refraction** - Scene distortion through water surface
3. **Foam/spray** - Particle effects at wave peaks

## License

MIT

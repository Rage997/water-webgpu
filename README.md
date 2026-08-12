# Water Wave WebGPU

Real-time water wave simulation using WebGPU compute shaders.

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

## Build

```bash
npm run build
npm run preview
```

## Architecture

- `src/water/` - Water simulation (compute + render)
- `src/scene/` - 3D scene components (skybox, bathtub, floor)
- `src/camera/` - Orbit camera controls
- `src/wgpu/` - WebGPU context management
- `src/shaders/` - WGSL shader code

## Resources

The skymap was taken from [polyhaven](https://polyhaven.com/hdris/skies)

## Planned work

1. **Dispersion** - Frequency-dependent wave speed for more realistic propagation
2. **Refraction** - Scene distortion through water surface
3. **Caustics** - Light patterns on bathtub floor
4. **Foam/spray** - Particle effects at wave peaks

## License

MIT

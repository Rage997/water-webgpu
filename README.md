# Water Wave WebGPU

Real-time water wave simulation using WebGPU compute shaders.

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

## License

MIT

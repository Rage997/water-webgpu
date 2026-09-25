import { W, H } from '../types.ts';

// Physical inside faces of the rendered bathtub. Water spans W x H.
export const TUB_WIDTH = W + 4;
export const TUB_DEPTH = H + 4;
export const TUB_HEIGHT = 100;

// Deterministic tie order at edges: bottom, X walls, then Z walls.
export const CAUSTIC_FACE = { bottom: 0, left: 1, right: 2, front: 3, back: 4 } as const;
export type CausticFace = typeof CAUSTIC_FACE[keyof typeof CAUSTIC_FACE];

// One world unit per texel, with centers half a unit inside each face.
// Bottom: (x + width/2, z + depth/2).
// X walls: (z + depth/2, y). Z walls: (x + width/2, y).
// Not optimal, but simple and works for now. Could be improved with a single 3D texture or a single 2D texture with a more complex mapping.
export const CAUSTIC_RECEIVERS = [
  { name: 'bottom', width: TUB_WIDTH, height: TUB_DEPTH },
  { name: 'left', width: TUB_DEPTH, height: TUB_HEIGHT },
  { name: 'right', width: TUB_DEPTH, height: TUB_HEIGHT },
  { name: 'front', width: TUB_WIDTH, height: TUB_HEIGHT },
  { name: 'back', width: TUB_WIDTH, height: TUB_HEIGHT },
] as const;

export type CausticTextures = readonly [GPUTexture, GPUTexture, GPUTexture, GPUTexture, GPUTexture];

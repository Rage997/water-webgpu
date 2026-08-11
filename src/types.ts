export const W = 290;
export const H = 590;
export interface Grid {
  NH: number;
  NW: number;
  NX: number;
  NZ: number;
  NUM_VERTS: number;
  DELTA_X: number;
  DELTA_Z: number;
  DELTA_X2: number;
  DELTA_Z2: number;
}

// Named resolution tiers surfaced in the UI. NH = grid rows along H; the rest derives.
export const RESOLUTION_TIERS = {
  ultralow: 60,
  low: 130,
  medium: 200,
  high: 300,
  veryhigh: 450,
} as const;

export type ResolutionTier = keyof typeof RESOLUTION_TIERS;

// Build the simulation/render grid for a given NH (number of rows along H).
export function makeGrid(nh: number): Grid {
  const NW = Math.ceil((nh * W) / H);
  const NX = NW + 1;
  const NZ = nh + 1;
  const DELTA_X = W / NW;
  const DELTA_Z = H / nh;
  return {
    NH: nh,
    NW,
    NX,
    NZ,
    NUM_VERTS: NX * NZ,
    DELTA_X,
    DELTA_Z,
    DELTA_X2: DELTA_X * DELTA_X,
    DELTA_Z2: DELTA_Z * DELTA_Z,
  };
}

export const C = 0.05;
export const C2 = C * C;
export const DAMPING = 0.002;
export const SIM_SPEED = 1.3;
export const MAX_DT = 30;
export const MAX_ITERATED_DT = 100;
export const MAX_Y = 40;
export const SIGMA = 0.01;

export interface ClickData {
  active: boolean;
  x: number;
  z: number;
}

export interface CameraData {
  viewProjMatrix: Float32Array;
  eye: Float32Array;
}

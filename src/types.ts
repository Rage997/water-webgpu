export const W = 290;
export const H = 590;
export const NH = 100;
export const NW = Math.ceil(NH * W / H);
export const NX = NW + 1;
export const NZ = NH + 1;
export const NUM_VERTS = NX * NZ;

export const C = 0.05;
export const C2 = C * C;
export const DAMPING = 0.002;
export const SIM_SPEED = 1.3;
export const MAX_DT = 30;
export const MAX_ITERATED_DT = 100;
export const MAX_Y = 40;
export const SIGMA = 0.01;

export const DELTA_X = W / NW;
export const DELTA_Z = H / NH;
export const DELTA_X2 = DELTA_X * DELTA_X;
export const DELTA_Z2 = DELTA_Z * DELTA_Z;

export interface ClickData {
  active: boolean;
  x: number;
  z: number;
}

export interface CameraData {
  viewProjMatrix: Float32Array;
  eye: Float32Array;
}

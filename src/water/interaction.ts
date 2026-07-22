import { OrbitCamera } from '../camera/orbit.ts';

export function setupInteraction(canvas: HTMLCanvasElement, camera: OrbitCamera, onHit: (x: number, z: number) => void) {
  canvas.addEventListener('pointerdown', (e: PointerEvent) => {
    if (e.button !== 0) return;

    const rect = canvas.getBoundingClientRect();

    const ndcX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const ndcY = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    // Unproject: compute ray in world space
    const invViewProj = new Float32Array(16);
    camera.invViewProjMatrix(invViewProj);

    // Near point (NDC z = 0) and far point (NDC z = 1) in world space
    const nearW = transformPoint(invViewProj, ndcX, ndcY, 0);
    const farW = transformPoint(invViewProj, ndcX, ndcY, 1);

    const rayOrig = nearW;
    const rayDir = [
      farW[0] - nearW[0],
      farW[1] - nearW[1],
      farW[2] - nearW[2],
    ];
    const len = Math.sqrt(rayDir[0]*rayDir[0] + rayDir[1]*rayDir[1] + rayDir[2]*rayDir[2]);
    rayDir[0] /= len;
    rayDir[1] /= len;
    rayDir[2] /= len;

    // Intersect with plane Y=0
    if (Math.abs(rayDir[1]) < 1e-6) return;
    const t = -rayOrig[1] / rayDir[1];
    if (t < 0) return;

    const worldX = rayOrig[0] + t * rayDir[0];
    const worldZ = rayOrig[2] + t * rayDir[2];

    onHit(worldX, worldZ);
  });
}

function transformPoint(mat: Float32Array, x: number, y: number, z: number): number[] {
  const w = 1 / (mat[3]*x + mat[7]*y + mat[11]*z + mat[15]);
  return [
    (mat[0]*x + mat[4]*y + mat[8]*z + mat[12]) * w,
    (mat[1]*x + mat[5]*y + mat[9]*z + mat[13]) * w,
    (mat[2]*x + mat[6]*y + mat[10]*z + mat[14]) * w,
  ];
}

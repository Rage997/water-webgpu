import { OrbitCamera } from '../camera/orbit.ts';
import { W, H, WATER_LEVEL } from '../types.ts';

// Replace each abrupt drag splash with smaller overlapping deposits, not more force.
const DRAG_SUBDIVISIONS = 12;

export function setupInteraction(canvas: HTMLCanvasElement, camera: OrbitCamera, onHit: (x: number, z: number, strength: number) => void, getSpacing: () => number) {
  let pointerId: number | null = null;
  const invViewProj = new Float32Array(16);
  let lastHit: [number, number] | null = null;
  let distanceToNext = 0;

  canvas.addEventListener('pointerdown', (e: PointerEvent) => {
    if (e.button !== 0 || pointerId !== null) return;
    pointerId = e.pointerId;
    lastHit = null;
    canvas.setPointerCapture(e.pointerId);
    disturb(e);
  });

  canvas.addEventListener('pointermove', (e: PointerEvent) => {
    if (e.pointerId !== pointerId) return;
    if ((e.buttons & 1) === 0) {
      stop(e);
      return;
    }
    disturb(e);
  });
  canvas.addEventListener('pointerup', stop);
  canvas.addEventListener('pointercancel', stop);
  canvas.addEventListener('lostpointercapture', stop);
  window.addEventListener('blur', () => {
    if (pointerId !== null && canvas.hasPointerCapture(pointerId)) {
      canvas.releasePointerCapture(pointerId);
    }
    pointerId = null;
    lastHit = null;
  });

  function stop(e: PointerEvent) {
    if (e.pointerId !== pointerId) return;
    pointerId = null;
    lastHit = null;
    if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
  }

  function disturb(e: PointerEvent) {
    const previous = lastHit;
    lastHit = null; // Leaving the pool must not connect two unrelated drag segments.
    const rect = canvas.getBoundingClientRect();
    const ndcX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const ndcY = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    // Unproject: compute ray in world space
    if (e.clientX < rect.left || e.clientX > rect.right ||
        e.clientY < rect.top || e.clientY > rect.bottom) return;
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

    // Intersect with the water surface plane (Y = WATER_LEVEL).
    if (Math.abs(rayDir[1]) < 1e-6) return;
    const t = (WATER_LEVEL - rayOrig[1]) / rayDir[1];
    if (t < 0) return;

    const worldX = rayOrig[0] + t * rayDir[0];
    const worldZ = rayOrig[2] + t * rayDir[2];

    if (Math.abs(worldX) <= W / 2 && Math.abs(worldZ) <= H / 2) {
      if (!previous) {
        onHit(worldX, worldZ, 1);
        distanceToNext = getSpacing() / DRAG_SUBDIVISIONS;
      } else {
        const dx = worldX - previous[0];
        const dz = worldZ - previous[1];
        const length = Math.hypot(dx, dz);
        // Normalize strength by distance so denser input does not amplify the wake.
        const spacing = getSpacing() / DRAG_SUBDIVISIONS;
        let distance = Math.min(distanceToNext, spacing);
        while (distance <= length) {
          const t = distance / length;
          onHit(previous[0] + t * dx, previous[1] + t * dz, 1 / DRAG_SUBDIVISIONS);
          distance += spacing;
        }
        distanceToNext = distance - length;
      }
      lastHit = [worldX, worldZ];
    }
  }
}

function transformPoint(mat: Float32Array, x: number, y: number, z: number): number[] {
  const w = 1 / (mat[3]*x + mat[7]*y + mat[11]*z + mat[15]);
  return [
    (mat[0]*x + mat[4]*y + mat[8]*z + mat[12]) * w,
    (mat[1]*x + mat[5]*y + mat[9]*z + mat[13]) * w,
    (mat[2]*x + mat[6]*y + mat[10]*z + mat[14]) * w,
  ];
}

import { OrbitCamera } from '../camera/orbit.ts';
import { W, H, WATER_LEVEL } from '../types.ts';

// Replace each abrupt drag splash with smaller overlapping deposits, not more force.
const DRAG_SUBDIVISIONS = 12;

// The draggable ball, as seen by pointer handling. Structural so the water
// layer does not depend on the scene layer.
export interface GrabSphere {
  enabled: boolean;
  readonly position: readonly [number, number, number];
  raycast(ox: number, oy: number, oz: number, dx: number, dy: number, dz: number): number | null;
  setTarget(x: number, y: number, z: number): void;
}

type Ray = [number, number, number, number, number, number];

export function setupInteraction(canvas: HTMLCanvasElement, camera: OrbitCamera, onHit: (x: number, z: number, strength: number) => void, getSpacing: () => number, sphere?: GrabSphere) {
  let pointerId: number | null = null;
  let mode: 'splash' | 'sphere' | null = null;
  const invViewProj = new Float32Array(16);
  let lastHit: [number, number] | null = null;
  let distanceToNext = 0;
  const grabPoint = [0, 0, 0];
  const grabCenter = [0, 0, 0];
  const dragNormal = [0, 0, 0];

  canvas.addEventListener('pointerdown', (e: PointerEvent) => {
    if (e.button !== 0 || pointerId !== null) return;
    pointerId = e.pointerId;
    lastHit = null;
    canvas.setPointerCapture(e.pointerId);
    const ray = rayFor(e);
    // Grab the ball when the press lands on it; otherwise splash the water.
    const hit = sphere?.enabled && ray ? sphere.raycast(...ray) : null;
    if (sphere && ray && hit !== null) {
      mode = 'sphere';
      canvas.style.cursor = 'grabbing';
      for (let axis = 0; axis < 3; axis++) {
        grabPoint[axis] = ray[axis] + hit * ray[axis + 3];
        grabCenter[axis] = sphere.position[axis];
      }
      // Screen-aligned dragging includes height. Shift locks the grab plane
      // horizontally for sweeping around the pool at a constant depth.
      dragNormal[0] = e.shiftKey ? 0 : camera.eyeX - camera.targetX;
      dragNormal[1] = e.shiftKey ? 1 : camera.eyeY - camera.targetY;
      dragNormal[2] = e.shiftKey ? 0 : camera.eyeZ - camera.targetZ;
      const length = Math.hypot(...dragNormal);
      for (let axis = 0; axis < 3; axis++) dragNormal[axis] /= length;
      sphere.setTarget(grabCenter[0], grabCenter[1], grabCenter[2]);
      return;
    }
    mode = 'splash';
    disturb(ray);
  });

  canvas.addEventListener('pointermove', (e: PointerEvent) => {
    if (e.pointerId !== pointerId) return;
    if ((e.buttons & 1) === 0) {
      stop(e);
      return;
    }
    if (mode === 'sphere') {
      const ray = rayFor(e);
      if (ray) dragSphereTo(ray);
      return;
    }
    disturb(rayFor(e));
  });
  canvas.addEventListener('pointerup', stop);
  canvas.addEventListener('pointercancel', stop);
  canvas.addEventListener('lostpointercapture', stop);
  window.addEventListener('blur', () => {
    if (pointerId !== null && canvas.hasPointerCapture(pointerId)) {
      canvas.releasePointerCapture(pointerId);
    }
    pointerId = null;
    mode = null;
    lastHit = null;
    canvas.style.cursor = '';
  });

  // Hover feedback: grab cursor over the ball, default elsewhere.
  canvas.addEventListener('pointermove', (e: PointerEvent) => {
    if (pointerId !== null) return; // an active drag manages its own cursor
    if (!sphere || !sphere.enabled) {
      canvas.style.cursor = '';
      return;
    }
    const ray = rayFor(e);
    canvas.style.cursor = ray && sphere.raycast(ray[0], ray[1], ray[2], ray[3], ray[4], ray[5]) !== null ? 'grab' : '';
  });

  function stop(e: PointerEvent) {
    if (e.pointerId !== pointerId) return;
    pointerId = null;
    mode = null;
    lastHit = null;
    canvas.style.cursor = '';
    if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
  }

  function dragSphereTo(ray: Ray) {
    const denominator = dragNormal[0] * ray[3] + dragNormal[1] * ray[4] + dragNormal[2] * ray[5];
    if (Math.abs(denominator) < 1e-6) return;
    const t = ((grabPoint[0] - ray[0]) * dragNormal[0]
      + (grabPoint[1] - ray[1]) * dragNormal[1]
      + (grabPoint[2] - ray[2]) * dragNormal[2]) / denominator;
    if (t < 0) return;
    // Preserve the point grabbed on the sphere instead of snapping its center.
    sphere!.setTarget(
      grabCenter[0] + ray[0] + t * ray[3] - grabPoint[0],
      grabCenter[1] + ray[1] + t * ray[4] - grabPoint[1],
      grabCenter[2] + ray[2] + t * ray[5] - grabPoint[2],
    );
  }

  function disturb(ray: Ray | null) {
    const previous = lastHit;
    lastHit = null; // Leaving the pool must not connect two unrelated drag segments.
    if (!ray) return;
    const [ox, oy, oz, dx, dy, dz] = ray;

    // Intersect with the water surface plane (Y = WATER_LEVEL).
    if (Math.abs(dy) < 1e-6) return;
    const t = (WATER_LEVEL - oy) / dy;
    if (t < 0) return;

    const worldX = ox + t * dx;
    const worldZ = oz + t * dz;

    if (Math.abs(worldX) <= W / 2 && Math.abs(worldZ) <= H / 2) {
      if (!previous) {
        onHit(worldX, worldZ, 1);
        distanceToNext = getSpacing() / DRAG_SUBDIVISIONS;
      } else {
        const ddx = worldX - previous[0];
        const ddz = worldZ - previous[1];
        const length = Math.hypot(ddx, ddz);
        // Normalize strength by distance so denser input does not amplify the wake.
        const spacing = getSpacing() / DRAG_SUBDIVISIONS;
        let distance = Math.min(distanceToNext, spacing);
        while (distance <= length) {
          const f = distance / length;
          onHit(previous[0] + f * ddx, previous[1] + f * ddz, 1 / DRAG_SUBDIVISIONS);
          distance += spacing;
        }
        distanceToNext = distance - length;
      }
      lastHit = [worldX, worldZ];
    }
  }

  function rayFor(e: PointerEvent): Ray | null {
    const rect = canvas.getBoundingClientRect();
    if (e.clientX < rect.left || e.clientX > rect.right ||
        e.clientY < rect.top || e.clientY > rect.bottom) return null;
    const ndcX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const ndcY = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    camera.invViewProjMatrix(invViewProj);
    const nearW = transformPoint(invViewProj, ndcX, ndcY, 0);
    const farW = transformPoint(invViewProj, ndcX, ndcY, 1);
    let dx = farW[0] - nearW[0];
    let dy = farW[1] - nearW[1];
    let dz = farW[2] - nearW[2];
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
    dx /= len; dy /= len; dz /= len;
    return [nearW[0], nearW[1], nearW[2], dx, dy, dz];
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

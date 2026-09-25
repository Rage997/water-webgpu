// GPU transport for projected caustic footprints.
//
// The source domain is the simulated water surface. Each unit source cell is
// split into two triangles. A triangle vertex is refracted through the sampled
// water normal and traced to the first bathtub face it exits:
//
//   water cell -> refracted triangle -> receiver-space triangle -> texels
//
// The receiver-space triangle is the important part. Its area and shape encode
// grazing-angle stretching and focusing; this is deliberately not a one-pixel
// splat. The triangle's source area is distributed by exact triangle/texel
// overlap, so the sum of its deposits equals the source power.
//
// Most triangles use the fast main pass. Triangles whose vertices disagree on
// the receiver face enter seamQueue and are subdivided in the seams pass.
// Source-area subdivision prevents corner patches from being duplicated across
// faces. The final tiny ambiguous leaf uses centroid ownership as a bounded
// numerical approximation.
//
// caustics.ts owns buffers and dispatch order. This module only generates WGSL
// and describes the shared packed receiver-buffer layout.

import { W, H } from '../types.ts';
import { CAUSTIC_RECEIVERS, TUB_WIDTH, TUB_DEPTH, TUB_HEIGHT } from './receivers.ts';

// Packed as [bottom, left, right, front, back], matching CAUSTIC_FACE.
// The resolve pass uses these offsets to copy each range into its texture.
let pixelOffset = 0;
export const RECEIVER_REGIONS = CAUSTIC_RECEIVERS.map(({ width, height }) => {
  const region = { width, height, offset: pixelOffset };
  pixelOffset += width * height;
  return region;
});
export const RECEIVER_PIXEL_COUNT = pixelOffset;

// Eight bisections give 1/256 source-unit seam resolution. This only applies
// to ambiguous seam triangles; ordinary triangles do not pay this cost.
const SEAM_DEPTH = 8;

// `common` supplies state/gradient bindings plus bilinear height and gradient
// sampling. Keeping it generated in caustics.ts lets the shader specialize its
// grid dimensions and avoids dynamic indexing overhead for those constants.

export function makeFootprintShader(common: string): string {
  return `
@group(0) @binding(0) var<storage, read> state: array<vec2f>;
@group(0) @binding(1) var<storage, read> grad: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> irradiance: array<atomic<u32>>;
struct SeamQueue { count: atomic<u32>, triangles: array<u32> };
@group(0) @binding(3) var<storage, read_write> seamQueue: SeamQueue;
${common}
const SIZE = array<vec2u, 5>(${RECEIVER_REGIONS.map(r => `vec2u(${r.width}, ${r.height})`).join(', ')});
const OFFSET = array<u32, 5>(${RECEIVER_REGIONS.map(r => `${r.offset}u`).join(', ')});
const HALF_X: f32 = ${TUB_WIDTH / 2};
const HALF_Z: f32 = ${TUB_DEPTH / 2};
const WALL_HEIGHT: f32 = ${TUB_HEIGHT};
const NO_FACE: u32 = 5u;
struct Ray { origin: vec3f, direction: vec3f };
struct Hit { distance: f32, face: u32, point: vec2f };
struct Triangle { a: vec2f, b: vec2f, c: vec2f, depth: u32 };

fn cross2(a: vec2f, b: vec2f) -> f32 { return a.x * b.y - a.y * b.x; }
fn rayAt(p: vec2f) -> Ray {
  let g = sampleGrad(p.x, p.y);
  return Ray(vec3f(p.x, WATER_LEVEL + sampleHeight(p.x, p.y), p.y),
    refract(-L, normalize(vec3f(-g.x, 1.0, -g.y)), ETA));
}
// Convert world positions to receiver-local texel coordinates:
// bottom=XZ, X walls=ZY, and Z walls=XY.
fn coordinates(p: vec3f, face: u32) -> vec2f {
  if (face == 0u) { return p.xz + vec2f(HALF_X, HALF_Z); }
  if (face <= 2u) { return vec2f(p.z + HALF_Z, p.y); }
  return vec2f(p.x + HALF_X, p.y);
}

// Signed distance to a receiver plane; non-positive values are behind or
// parallel to the ray.
fn planeDistance(ray: Ray, face: u32) -> f32 {
  var numerator = -ray.origin.y;
  var denominator = ray.direction.y;
  if (face == 1u || face == 2u) {
    numerator = select(-HALF_X, HALF_X, face == 2u) - ray.origin.x;
    denominator = ray.direction.x;
  } else if (face >= 3u) {
    numerator = select(-HALF_Z, HALF_Z, face == 4u) - ray.origin.z;
    denominator = ray.direction.z;
  }
  if (denominator == 0.0) { return -1.0; }
  return numerator / denominator;
}

// The convex tub's nearest positive plane exit is the first receiver hit.
// Rays that leave through the top rim are treated as escaped light.
fn firstHit(ray: Ray) -> Hit {
  var hit = Hit(3.402823e38, NO_FACE, vec2f(0.0));
  if (!(ray.origin.y > 0.0)) { return hit; }
  // Source XZ lies strictly inside the convex tub. The least positive plane
  // distance is its first exit; testing rounded hit coordinates against every
  // face instead can reject BOTH sides of a seam and create numerical holes.
  if (ray.direction.y > 0.0 && ray.origin.y <= WALL_HEIGHT) {
    hit.distance = (WALL_HEIGHT - ray.origin.y) / ray.direction.y;
  }
  for (var face = 0u; face < 5u; face++) {
    let distance = planeDistance(ray, face);
    if (distance > 0.0 && distance < hit.distance) {
      hit = Hit(distance, face, vec2f(0.0));
    }
  }
  if (hit.face != NO_FACE) {
    let point = coordinates(ray.origin + hit.distance * ray.direction, hit.face);
    if (hit.face != 0u && point.y > WALL_HEIGHT) { hit.face = NO_FACE; }
    else { hit.point = clamp(point, vec2f(0.0), vec2f(SIZE[hit.face])); }
  }
  return hit;
}

// Atomic CAS avoids requiring optional float32 storage-texture atomics and
// allows many source triangles to contribute to one texel without a bin cap.
fn addPower(face: u32, pixel: vec2i, power: f32) {
  if (!(power > 0.0)) { return; }
  let index = OFFSET[face] + u32(pixel.x) + u32(pixel.y) * SIZE[face].x;
  var old = atomicLoad(&irradiance[index]);
  loop {
    let next = bitcast<u32>(bitcast<f32>(old) + power);
    let result = atomicCompareExchangeWeak(&irradiance[index], old, next);
    if (result.exchanged) { break; }
    old = result.old_value;
  }
}

// Green's theorem integrates the clipped triangle area. Each edge is clipped
// against the texel's X slab; no per-texel polygon array is allocated.
fn edgeIntegral(a: vec2f, b: vec2f) -> f32 {
  let delta = b - a;
  if (delta.x == 0.0) { return 0.0; }
  let lo = max(0.0, min(a.x, b.x));
  let hi = min(1.0, max(a.x, b.x));
  if (hi <= lo) { return 0.0; }
  let y0 = a.y + (lo - a.x) * (delta.y / delta.x);
  let y1 = a.y + (hi - a.x) * (delta.y / delta.x);
  let dy = y1 - y0;
  let width = (hi - lo) * sign(delta.x);
  if (abs(dy) < 1e-7) { return width * clamp((y0 + y1) * 0.5, 0.0, 1.0); }
  let at0 = clamp(-y0 / dy, 0.0, 1.0);
  let at1 = clamp((1.0 - y0) / dy, 0.0, 1.0);
  let t0 = min(at0, at1);
  let t1 = max(at0, at1);
  let v0 = clamp(y0, 0.0, 1.0);
  let v1 = clamp(y0 + t0 * dy, 0.0, 1.0);
  let v2 = clamp(y0 + t1 * dy, 0.0, 1.0);
  let v3 = clamp(y1, 0.0, 1.0);
  return 0.5 * width * (t0 * (v0 + v1) + (t1 - t0) * (v1 + v2) + (1.0 - t1) * (v2 + v3));
}
fn pixelArea(a: vec2f, b: vec2f, c: vec2f, pixel: vec2i) -> f32 {
  let p = vec2f(pixel);
  return abs(edgeIntegral(a - p, b - p) + edgeIntegral(b - p, c - p) + edgeIntegral(c - p, a - p));
}
// A projected triangle can collapse under grazing/edge geometry. Its source
// area still needs a destination, so use the exact 1D marginal of the collapsed
// triangle rather than dropping it or assigning all power to one pixel.
// A collapsed triangle pushes uniform source area onto a triangular 1D density,
// not a uniform segment. Integrate its CDF; a fully collapsed patch is a point.
fn lineCdf(t: f32, peak: f32) -> f32 {
  if (t <= 0.0) { return 0.0; }
  if (t >= 1.0) { return 1.0; }
  if (t < peak) { return t * t / peak; }
  return 1.0 - (1.0 - t) * (1.0 - t) / (1.0 - peak);
}
fn depositLine(face: u32, a: vec2f, b: vec2f, c: vec2f, power: f32) {
  let delta = b - a;
  if (dot(delta, delta) < 1e-12) {
    addPower(face, clamp(vec2i(floor(a)), vec2i(0), vec2i(SIZE[face]) - vec2i(1)), power);
    return;
  }
  let peak = clamp(dot(c - a, delta) / dot(delta, delta), 0.0, 1.0);
  let lo = clamp(vec2i(floor(min(a, b))), vec2i(0), vec2i(SIZE[face]) - vec2i(1));
  let hi = clamp(vec2i(floor(max(a, b))), vec2i(0), vec2i(SIZE[face]) - vec2i(1));
  for (var y = lo.y; y <= hi.y; y++) {
    for (var x = lo.x; x <= hi.x; x++) {
      let pixel = vec2i(x, y);
      var enter = 0.0;
      var leave = 1.0;
      for (var axis = 0u; axis < 2u; axis++) {
        if (abs(delta[axis]) < 1e-12) {
          // Half-open ownership prevents doubling a line on a texel edge.
          let owner = clamp(i32(floor(a[axis])), 0, i32(SIZE[face][axis]) - 1);
          if (pixel[axis] != owner) { leave = -1.0; }
        } else {
          let t0 = (f32(pixel[axis]) - a[axis]) / delta[axis];
          let t1 = (f32(pixel[axis] + 1) - a[axis]) / delta[axis];
          enter = max(enter, min(t0, t1));
          leave = min(leave, max(t0, t1));
        }
      }
      addPower(face, pixel, power * max(0.0, lineCdf(leave, peak) - lineCdf(enter, peak)));
    }
  }
}

// Rasterize the projected triangle by exact overlap with each touched texel.
// A collapsed projection uses the line/point path above.
fn depositTriangle(face: u32, a: vec2f, b: vec2f, c: vec2f, power: f32) {
  let area = 0.5 * abs(cross2(b - a, c - a));
  if (area < 1e-8) {
    let ab = dot(b - a, b - a);
    let ac = dot(c - a, c - a);
    let bc = dot(c - b, c - b);
    if (ab >= ac && ab >= bc) { depositLine(face, a, b, c, power); }
    else if (ac >= bc) { depositLine(face, a, c, b, power); }
    else { depositLine(face, b, c, a, power); }
    return;
  }
  let lo = max(vec2i(floor(min(a, min(b, c)))), vec2i(0));
  let hi = min(vec2i(ceil(max(a, max(b, c)))) - vec2i(1), vec2i(SIZE[face]) - vec2i(1));
  for (var y = lo.y; y <= hi.y; y++) {
    for (var x = lo.x; x <= hi.x; x++) {
      let pixel = vec2i(x, y);
      addPower(face, pixel, power * (pixelArea(a, b, c, pixel) / area));
    }
  }
}

// Re-evaluate a nearly seam-aligned ray on a selected face. This is used only
// after source area has been partitioned, so clamping cannot duplicate power.
fn seamProjection(ray: Ray, face: u32, fallback: vec2f) -> vec2f {
  let distance = planeDistance(ray, face);
  if (!(distance > 0.0 && distance < 3.402823e38)) { return fallback; }
  // Roundoff at a solved seam, or the final sub-1/256-unit ambiguous leaf.
  // Source power has already been partitioned before applying this boundary limit.
  return clamp(coordinates(ray.origin + ray.direction * distance, face), vec2f(0.0), vec2f(SIZE[face]));
}

fn sourceTriangle(index: u32) -> Triangle {
  let cell = index / 2u;
  let origin = vec2f(f32(cell % ${W}u) - W_HALF, f32(cell / ${W}u) - H_HALF);
  let middle = select(vec2f(1.0, 0.0), vec2f(0.0, 1.0), index % 2u == 1u);
  return Triangle(origin, origin + middle, origin + vec2f(1.0), 0u);
}
// Locate a face boundary by bisection in source space. If a third face appears
// the caller must subdivide instead of pretending this is a two-face seam.
fn boundary(a: vec2f, b: vec2f, faceA: u32, faceB: u32) -> vec3f {
  var lo = a;
  var hi = b;
  for (var i = 0u; i < 16u; i++) {
    let p = (lo + hi) * 0.5;
    let face = firstHit(rayAt(p)).face;
    if (face == faceA) { lo = p; }
    else if (face == faceB) { hi = p; }
    else { return vec3f(0.0); }
  }
  return vec3f((lo + hi) * 0.5, 1.0);
}

fn depositSourceTriangle(face: u32, a: vec2f, b: vec2f, c: vec2f) {
  if (face == NO_FACE) { return; }
  let power = 0.5 * abs(cross2(b - a, c - a));
  let center = firstHit(rayAt((a + b + c) / 3.0));
  depositTriangle(face, seamProjection(rayAt(a), face, center.point),
    seamProjection(rayAt(b), face, center.point),
    seamProjection(rayAt(c), face, center.point), power);
}

// Split a triangle with two receiver faces into one minority triangle and two
// majority triangles. The three pieces exactly partition source area.
fn splitPair(t: Triangle, fa: u32, fb: u32, fc: u32, center: u32) -> bool {
  var minority = t.c;
  var first = t.a;
  var second = t.b;
  var minorFace = fc;
  var majorFace = fa;
  if (fa == fb && fa != fc) {
    // Already in minority/majority order.
  } else if (fa == fc && fa != fb) {
    minority = t.b; first = t.c; second = t.a;
    minorFace = fb;
  } else if (fb == fc && fb != fa) {
    minority = t.a; first = t.b; second = t.c;
    minorFace = fa; majorFace = fb;
  } else {
    return false;
  }
  if (center != minorFace && center != majorFace) { return false; }
  let edgeA = boundary(minority, first, minorFace, majorFace);
  let edgeB = boundary(minority, second, minorFace, majorFace);
  if (edgeA.z == 0.0 || edgeB.z == 0.0) { return false; }
  depositSourceTriangle(minorFace, minority, edgeA.xy, edgeB.xy);
  depositSourceTriangle(majorFace, first, second, edgeB.xy);
  depositSourceTriangle(majorFace, first, edgeB.xy, edgeA.xy);
  return true;
}

// Keep the common path free of an adaptive traversal stack. Queue capacity is
// exactly the number of source triangles: each can enqueue at most once.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= ${W * H * 2}u) { return; }
  let t = sourceTriangle(id.x);
  let a = firstHit(rayAt(t.a));
  let b = firstHit(rayAt(t.b));
  let c = firstHit(rayAt(t.c));
  let center = firstHit(rayAt((t.a + t.b + t.c) / 3.0));
  if (a.face == b.face && a.face == c.face && a.face == center.face) {
    if (a.face != NO_FACE) { depositTriangle(a.face, a.point, b.point, c.point, 0.5); }
  } else {
    let slot = atomicAdd(&seamQueue.count, 1u);
    seamQueue.triangles[slot] = id.x;
  }
}

// Resolve queued mixed-face triangles. Adaptive subdivision is bounded by
// SEAM_DEPTH; the final fallback uses centroid ownership.
@compute @workgroup_size(64)
fn seams(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= atomicLoad(&seamQueue.count)) { return; }
  var stack: array<Triangle, ${1 + 3 * SEAM_DEPTH}>;
  stack[0] = sourceTriangle(seamQueue.triangles[id.x]);
  var count = 1u;
  loop {
    if (count == 0u) { break; }
    count--;
    let t = stack[count];
    let ra = rayAt(t.a);
    let rb = rayAt(t.b);
    let rc = rayAt(t.c);
    let a = firstHit(ra);
    let b = firstHit(rb);
    let c = firstHit(rc);
    let center = firstHit(rayAt((t.a + t.b + t.c) / 3.0));
    let sameFace = a.face == b.face && a.face == c.face && a.face == center.face;
    let power = 0.5 * abs(cross2(t.b - t.a, t.c - t.a));
    if (sameFace) {
      if (a.face != NO_FACE) { depositTriangle(a.face, a.point, b.point, c.point, power); }
    } else if (splitPair(t, a.face, b.face, c.face, center.face)) {
    } else if (t.depth < ${SEAM_DEPTH}u) {
      let ab = (t.a + t.b) * 0.5;
      let ac = (t.a + t.c) * 0.5;
      let bc = (t.b + t.c) * 0.5;
      let depth = t.depth + 1u;
      stack[count] = Triangle(t.a, ab, ac, depth);
      stack[count + 1u] = Triangle(ab, t.b, bc, depth);
      stack[count + 2u] = Triangle(ac, bc, t.c, depth);
      stack[count + 3u] = Triangle(ab, bc, ac, depth);
      count += 4u;
    } else if (center.face != NO_FACE) {
      depositTriangle(center.face,
        seamProjection(ra, center.face, center.point),
        seamProjection(rb, center.face, center.point),
        seamProjection(rc, center.face, center.point), power);
    }
  }
}
`;
}

export function makeResolveShader({ width, height, offset }: typeof RECEIVER_REGIONS[number]): string {
  return `
@group(0) @binding(0) var<storage, read> irradiance: array<u32>;
@group(0) @binding(1) var outputMap: texture_storage_2d<r32float, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= ${width}u || id.y >= ${height}u) { return; }
  let value = bitcast<f32>(irradiance[${offset}u + id.x + id.y * ${width}u]);
  textureStore(outputMap, vec2i(id.xy), vec4f(value));
}
`;
}

export const SEAM_DISPATCH_SHADER = `
@group(0) @binding(0) var<storage, read> queue: array<u32>;
@group(0) @binding(1) var<storage, read_write> dispatch: array<u32>;
@compute @workgroup_size(1)
fn main() {
  dispatch[0] = (queue[0] + 63u) / 64u;
  dispatch[1] = 1u;
  dispatch[2] = 1u;
}
`;

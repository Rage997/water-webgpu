import { TUB_WIDTH, TUB_DEPTH, TUB_HEIGHT } from './receivers.ts';

// The rendered scene has five finite pool faces and one sphere. Intersect that
// geometry directly: a camera depth image cannot represent hidden receivers.
export const POOL_INTERSECTION_WGSL = `
struct SceneHit {
  distance: f32,
  normal: vec3f,
  kind: u32, // 0: open sky, 1: pool, 2: sphere
};

fn intersectPool(origin: vec3f, direction: vec3f, sphere: vec4f) -> SceneHit {
  let lower = vec3f(${-TUB_WIDTH / 2}, 0.0, ${-TUB_DEPTH / 2});
  let upper = vec3f(${TUB_WIDTH / 2}, ${TUB_HEIGHT}.0, ${TUB_DEPTH / 2});
  let axes = array<u32, 5>(1u, 0u, 0u, 2u, 2u);
  let planes = array<f32, 5>(0.0, lower.x, upper.x, lower.z, upper.z);
  let normals = array<vec3f, 5>(
    vec3f(0, 1, 0), vec3f(1, 0, 0), vec3f(-1, 0, 0),
    vec3f(0, 0, 1), vec3f(0, 0, -1));
  var hit = SceneHit(1e20, vec3f(0.0), 0u);
  for (var face = 0u; face < 5u; face++) {
    let axis = axes[face];
    if (abs(direction[axis]) < 1e-6) { continue; }
    let t = (planes[face] - origin[axis]) / direction[axis];
    if (t <= 1e-4 || t >= hit.distance) { continue; }
    let p = origin + t * direction;
    if (all(p >= lower - 0.001) && all(p <= upper + 0.001)) {
      hit = SceneHit(t, normals[face], 1u);
    }
  }
  if (sphere.w > 0.0) {
    let delta = origin - sphere.xyz;
    let b = dot(delta, direction);
    let c = dot(delta, delta) - sphere.w * sphere.w;
    let discriminant = b * b - c;
    if (discriminant >= 0.0) {
      let root = sqrt(discriminant);
      var t = -b - root;
      if (t <= 1e-4) { t = -b + root; }
      if (t > 1e-4 && t < hit.distance) {
        hit = SceneHit(t, normalize(origin + t * direction - sphere.xyz), 2u);
      }
    }
  }
  return hit;
}
`;

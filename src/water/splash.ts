// Shared pointer/rain potential. Mirrored neighbours make its discrete Laplacian
// sum to zero with half-cell edge and quarter-cell corner weights.
export const CONSERVATIVE_SPLASH_WGSL = `
fn potentialAxis(q: i32, last: i32, extent: f32, center: f32, width: f32) -> vec3f {
  let nodes = vec3i(q-1, q, q+1);
  let mirrored = min(abs(nodes), vec3i(2*last)-nodes);
  let x = vec3f(mirrored) * (extent/f32(last));
  let direct = x-center;
  let lowerImage = x+center;
  let upperImage = x-(2.0*extent-center);
  return exp(-direct*direct/width)
       + exp(-lowerImage*lowerImage/width)
       + exp(-upperImage*upperImage/width);
}

fn splashLaplacian(q: vec2i, last: vec2i, extent: vec2f, center: vec2f,
                   width: f32, inverseSpacing2: vec2f) -> f32 {
  let gx = potentialAxis(q.x, last.x, extent.x, center.x, width);
  let gz = potentialAxis(q.y, last.y, extent.y, center.y, width);
  return (gx.x-2.0*gx.y+gx.z)*gz.y*inverseSpacing2.x
       + (gz.x-2.0*gz.y+gz.z)*gx.y*inverseSpacing2.y;
}
`;

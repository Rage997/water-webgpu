// Procedural seamless detail normal map for velocity-scaled water detail.
//
// A deterministic, seeded sum of periodic sinusoids (integer wave vectors)
// forms a height field; tangent-space normals are derived from its gradient
// and stored in a linear rgba8unorm texture with an averaged mip chain.
// Integer wave vectors make the field periodic, so the map tiles seamlessly.

const SIZE = 128;
const LEVELS = 8; // 128, 64, 32, 16, 8, 4, 2, 1
const WAVE_COUNT = 12;
const SEED = 0x51ce;

// Small deterministic PRNG so the map is identical on every construction.
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface Wave {
  kx: number;
  kz: number;
  amp: number;
  phase: number;
}

function makeWaves(rand: () => number): Wave[] {
  // Integer wave vectors keep the field periodic over the tile. Magnitudes of
  // 4-16 cycles per tile map to 2-8 world-unit wavelengths for a 32-unit tile.
  // Many random directions and phases avoid a dominant stripe orientation.
  const waves: Wave[] = [];
  while (waves.length < WAVE_COUNT) {
    const angle = rand() * Math.PI * 2;
    const mag = 4 + Math.floor(rand() * 13);
    const kx = Math.round(Math.cos(angle) * mag);
    const kz = Math.round(Math.sin(angle) * mag);
    const frequency = Math.hypot(kx, kz);
    if (frequency < 4 || frequency > 16) continue;
    waves.push({ kx, kz, amp: (0.5 + 0.5 * rand()) / frequency, phase: rand() * Math.PI * 2 });
  }
  return waves;
}

function encodeLevel(normals: Float32Array, width: number, height: number): Uint8Array {
  const out = new Uint8Array(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    const o = i * 4;
    out[o] = Math.round((normals[i * 3] * 0.5 + 0.5) * 255);
    out[o + 1] = Math.round((normals[i * 3 + 1] * 0.5 + 0.5) * 255);
    out[o + 2] = Math.round((normals[i * 3 + 2] * 0.5 + 0.5) * 255);
    out[o + 3] = 255;
  }
  return out;
}

export function createDetailNormalTexture(device: GPUDevice): GPUTexture {
  const rand = mulberry32(SEED);
  const waves = makeWaves(rand);
  const TAU = Math.PI * 2;

  // Periodic height field on the base grid.
  const heights = new Float32Array(SIZE * SIZE);
  for (let y = 0; y < SIZE; y++) {
    const v = y / SIZE;
    for (let x = 0; x < SIZE; x++) {
      const u = x / SIZE;
      let h = 0;
      for (const w of waves) {
        h += w.amp * Math.sin(TAU * (w.kx * u + w.kz * v) + w.phase);
      }
      heights[y * SIZE + x] = h;
    }
  }

  // Base-level tangent-space normals from central-difference gradients
  // (per texel, periodic wrap).
  let normals = new Float32Array(SIZE * SIZE * 3);
  let maxSlope = 0;
  for (let y = 0; y < SIZE; y++) {
    const ym = ((y - 1 + SIZE) % SIZE) * SIZE;
    const yp = ((y + 1) % SIZE) * SIZE;
    for (let x = 0; x < SIZE; x++) {
      const xm = (x - 1 + SIZE) % SIZE;
      const xp = (x + 1) % SIZE;
      const gx = (heights[y * SIZE + xp] - heights[y * SIZE + xm]) * 0.5;
      const gz = (heights[yp + x] - heights[ym + x]) * 0.5;
      maxSlope = Math.max(maxSlope, Math.hypot(gx, gz));
      const o = (y * SIZE + x) * 3;
      normals[o] = -gx;
      normals[o + 1] = -gz;
      normals[o + 2] = 1;
    }
  }
  // Normalize the map so its steepest slope is 1; the shader's strength then
  // directly bounds the added tangent slope.
  const scale = 1 / maxSlope;
  for (let i = 0; i < normals.length; i += 3) {
    const nx = normals[i] * scale;
    const ny = normals[i + 1] * scale;
    const inv = 1 / Math.hypot(nx, ny, 1);
    normals[i] = nx * inv;
    normals[i + 1] = ny * inv;
    normals[i + 2] = inv;
  }

  const texture = device.createTexture({
    size: [SIZE, SIZE],
    mipLevelCount: LEVELS,
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });

  let w = SIZE;
  let h = SIZE;
  for (let level = 0; level < LEVELS; level++) {
    device.queue.writeTexture(
      { texture, mipLevel: level },
      encodeLevel(normals, w, h),
      { bytesPerRow: w * 4, rowsPerImage: h },
      [w, h],
    );
    if (w === 1) break;
    // Next mip: average the four parent normal vectors. Deliberately not
    // renormalized, so unresolved detail averages out instead of being
    // re-amplified; the shader decodes slopes as xy/z, which is unaffected
    // by the resulting length shrink.
    const hw = w >> 1;
    const hh = h >> 1;
    const next = new Float32Array(hw * hh * 3);
    for (let y = 0; y < hh; y++) {
      for (let x = 0; x < hw; x++) {
        const i00 = (y * 2 * w + x * 2) * 3;
        const i01 = i00 + 3;
        const i10 = ((y * 2 + 1) * w + x * 2) * 3;
        const i11 = i10 + 3;
        const o = (y * hw + x) * 3;
        next[o] = (normals[i00] + normals[i01] + normals[i10] + normals[i11]) * 0.25;
        next[o + 1] = (normals[i00 + 1] + normals[i01 + 1] + normals[i10 + 1] + normals[i11 + 1]) * 0.25;
        next[o + 2] = (normals[i00 + 2] + normals[i01 + 2] + normals[i10 + 2] + normals[i11 + 2]) * 0.25;
      }
    }
    normals = next;
    w = hw;
    h = hh;
  }

  return texture;
}

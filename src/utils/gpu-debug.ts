// CPU-side readback + stats for verifying GPU compute output.
// Used by the caustics verification harness (and any future debug).

export interface BufferStats {
  min: number;
  max: number;
  mean: number;
  /** sum of (v - mean)^2 / n */
  variance: number;
}

export function bufferStats(data: Float32Array): BufferStats {
  let min = Infinity, max = -Infinity, sum = 0;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
  }
  const mean = data.length ? sum / data.length : 0;
  let sq = 0;
  for (let i = 0; i < data.length; i++) {
    const d = data[i] - mean;
    sq += d * d;
  }
  return { min, max, mean, variance: data.length ? sq / data.length : 0 };
}

// Copy a GPUBuffer's contents to the CPU and resolve. The buffer must have
// COPY_SRC usage. Resolves after the copy lands, so it is safe to call right
// after submitting the command that wrote the buffer.
export async function readBufferAsync(device: GPUDevice, buffer: GPUBuffer): Promise<Float32Array> {
  const size = buffer.size;
  const staging = device.createBuffer({
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return data;
}

// Copy an r32float GPUTexture's contents to the CPU and resolve. Used to read
// back the caustic map (and the pass-1 channel texture) for verification.
// Handles the row-pitch padding WebGPU adds to copy destinations.
export async function readTextureAsync(
  device: GPUDevice,
  texture: GPUTexture,
  width: number,
  height: number,
): Promise<Float32Array> {
  const bytesPerRow = width * 4;
  const rowPitch = Math.ceil(bytesPerRow / 256) * 256;
  const staging = device.createBuffer({
    size: rowPitch * height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer(
    { texture },
    { buffer: staging, bytesPerRow: rowPitch, rowsPerImage: height },
    { width, height, depthOrArrayLayers: 1 },
  );
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(width * height);
  const f32 = new Float32Array(staging.getMappedRange());
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      data[y * width + x] = f32[y * (rowPitch / 4) + x];
    }
  }
  staging.unmap();
  staging.destroy();
  return data;
}

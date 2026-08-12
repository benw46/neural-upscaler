/** GPU texture -> tightly-packed CPU ArrayBuffer readback. WebGPU requires
 * `copyTextureToBuffer` destinations to have bytesPerRow padded to a 256-byte
 * multiple; this strips that padding so files on disk are exactly
 * width*height*bytesPerPixel with no row-alignment gaps for downstream
 * consumers (numpy, PyTorch) to worry about. */
export async function readTextureToArrayBuffer(
  device: GPUDevice,
  texture: GPUTexture,
  width: number,
  height: number,
  bytesPerPixel: number
): Promise<ArrayBuffer> {
  const unpaddedBytesPerRow = width * bytesPerPixel;
  const align = 256;
  const paddedBytesPerRow = Math.ceil(unpaddedBytesPerRow / align) * align;

  const readBuffer = device.createBuffer({
    label: "readback-buffer",
    size: paddedBytesPerRow * height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const encoder = device.createCommandEncoder({ label: "readback-copy" });
  encoder.copyTextureToBuffer(
    { texture },
    { buffer: readBuffer, bytesPerRow: paddedBytesPerRow, rowsPerImage: height },
    { width, height }
  );
  device.queue.submit([encoder.finish()]);

  await readBuffer.mapAsync(GPUMapMode.READ);
  const mapped = readBuffer.getMappedRange();

  const packed = new ArrayBuffer(unpaddedBytesPerRow * height);
  const packedView = new Uint8Array(packed);
  const mappedView = new Uint8Array(mapped);
  for (let row = 0; row < height; row++) {
    const srcStart = row * paddedBytesPerRow;
    const dstStart = row * unpaddedBytesPerRow;
    packedView.set(mappedView.subarray(srcStart, srcStart + unpaddedBytesPerRow), dstStart);
  }

  readBuffer.unmap();
  readBuffer.destroy();
  return packed;
}

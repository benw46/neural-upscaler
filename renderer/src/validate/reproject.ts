import fs from "node:fs/promises";
import path from "node:path";

/**
 * Reprojection validation — Spec 1 Part B step 7.
 *
 * Warps frame N-1's colour into frame N's viewpoint using frame N's stored
 * motion vectors, and compares against frame N's actual colour. Assumes the
 * motion vector convention documented in render/shaders.wgsl:
 *   mv = current_uv - previous_uv  =>  previous_uv = current_uv - mv
 *
 * Disocclusion detection compares reprojected previous-frame depth against
 * current depth at each pixel: a genuinely continuous surface seen from two
 * nearby camera positions 1/30s apart has nearly identical view-space depth
 * along the reprojected ray, so a large relative mismatch (or a reprojected
 * UV that lands off-screen) means the surface wasn't visible last frame —
 * excluded from the error stats rather than counted as an error, per the
 * gate's "excluding a disocclusion mask" requirement.
 */

export interface FrameBuffers {
  width: number;
  height: number;
  color: Float16Array; // RGBA
  depth: Float32Array; // R, linear view-space distance
  motion: Float16Array; // RG, uv-space (current_uv - previous_uv)
}

async function readTyped<T extends Float16Array | Float32Array>(
  filePath: string,
  Ctor: { new (buffer: ArrayBufferLike, byteOffset: number, length: number): T; BYTES_PER_ELEMENT: number }
): Promise<T> {
  const buf = await fs.readFile(filePath);
  const count = buf.byteLength / Ctor.BYTES_PER_ELEMENT;
  return new Ctor(buf.buffer, buf.byteOffset, count);
}

export async function loadFrameBuffers(runDir: string, frameIndex: number, width: number, height: number): Promise<FrameBuffers> {
  const fname = `${String(frameIndex).padStart(6, "0")}.bin`;
  const [color, depth, motion] = await Promise.all([
    readTyped(path.join(runDir, "color", fname), Float16Array),
    readTyped(path.join(runDir, "depth", fname), Float32Array),
    readTyped(path.join(runDir, "motion", fname), Float16Array),
  ]);
  return { width, height, color, depth, motion };
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

/** Bilinear sample of an RGBA buffer at uv (y-down, [0,1]); clamp-to-edge. */
function sampleRGBA(buf: Float16Array, width: number, height: number, u: number, v: number): [number, number, number, number] {
  const fx = u * width - 0.5;
  const fy = v * height - 0.5;
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const tx = fx - x0;
  const ty = fy - y0;

  function texel(xi: number, yi: number): [number, number, number, number] {
    const cx = clamp(xi, 0, width - 1);
    const cy = clamp(yi, 0, height - 1);
    const i = (cy * width + cx) * 4;
    return [buf[i], buf[i + 1], buf[i + 2], buf[i + 3]];
  }

  const c00 = texel(x0, y0);
  const c10 = texel(x0 + 1, y0);
  const c01 = texel(x0, y0 + 1);
  const c11 = texel(x0 + 1, y0 + 1);

  const out: [number, number, number, number] = [0, 0, 0, 0];
  for (let ch = 0; ch < 4; ch++) {
    const top = c00[ch] * (1 - tx) + c10[ch] * tx;
    const bottom = c01[ch] * (1 - tx) + c11[ch] * tx;
    out[ch] = top * (1 - ty) + bottom * ty;
  }
  return out;
}

function sampleDepthBilinear(buf: Float32Array, width: number, height: number, u: number, v: number): number {
  const fx = u * width - 0.5;
  const fy = v * height - 0.5;
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const tx = fx - x0;
  const ty = fy - y0;
  function texel(xi: number, yi: number): number {
    const cx = clamp(xi, 0, width - 1);
    const cy = clamp(yi, 0, height - 1);
    return buf[cy * width + cx];
  }
  const top = texel(x0, y0) * (1 - tx) + texel(x0 + 1, y0) * tx;
  const bottom = texel(x0, y0 + 1) * (1 - tx) + texel(x0 + 1, y0 + 1) * tx;
  return top * (1 - ty) + bottom * ty;
}

export interface ReprojectionResult {
  width: number;
  height: number;
  warped: Float32Array; // RGBA, same layout as input
  absError: Float32Array; // per-pixel scalar (mean abs channel diff), one value per pixel
  disocclusionMask: Uint8Array; // 1 = excluded (disocclusion/off-screen), one value per pixel
  stats: {
    includedFraction: number;
    disocclusionFraction: number;
    meanErrorIncluded: number;
    maxErrorIncluded: number;
    p99ErrorIncluded: number;
  };
}

const DEPTH_REL_THRESHOLD = 0.05;

export function warpAndValidate(prev: FrameBuffers, curr: FrameBuffers): ReprojectionResult {
  const { width, height } = curr;
  const warped = new Float32Array(width * height * 4);
  const absError = new Float32Array(width * height);
  const disocclusionMask = new Uint8Array(width * height);

  const includedErrors: number[] = [];
  let disoccludedCount = 0;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      const currU = (x + 0.5) / width;
      const currV = (y + 0.5) / height;

      const mvx = curr.motion[i * 2];
      const mvy = curr.motion[i * 2 + 1];
      const prevU = currU - mvx;
      const prevV = currV - mvy;

      const offscreen = prevU < 0 || prevU > 1 || prevV < 0 || prevV > 1;

      const [wr, wg, wb, wa] = sampleRGBA(prev.color, width, height, prevU, prevV);
      warped[i * 4] = wr;
      warped[i * 4 + 1] = wg;
      warped[i * 4 + 2] = wb;
      warped[i * 4 + 3] = wa;

      const cr = curr.color[i * 4];
      const cg = curr.color[i * 4 + 1];
      const cb = curr.color[i * 4 + 2];
      const err = (Math.abs(wr - cr) + Math.abs(wg - cg) + Math.abs(wb - cb)) / 3;
      absError[i] = err;

      let disoccluded = offscreen;
      if (!offscreen) {
        const prevDepthSample = sampleDepthBilinear(prev.depth, width, height, prevU, prevV);
        const currDepthSample = curr.depth[i];
        const relDiff = Math.abs(prevDepthSample - currDepthSample) / Math.max(currDepthSample, 1e-4);
        if (relDiff > DEPTH_REL_THRESHOLD) disoccluded = true;
      }

      if (disoccluded) {
        disocclusionMask[i] = 1;
        disoccludedCount++;
      } else {
        includedErrors.push(err);
      }
    }
  }

  includedErrors.sort((a, b) => a - b);
  const n = includedErrors.length;
  const mean = n > 0 ? includedErrors.reduce((a, b) => a + b, 0) / n : 0;
  const max = n > 0 ? includedErrors[n - 1] : 0;
  const p99 = n > 0 ? includedErrors[Math.floor(0.99 * (n - 1))] : 0;

  return {
    width,
    height,
    warped,
    absError,
    disocclusionMask,
    stats: {
      includedFraction: n / (width * height),
      disocclusionFraction: disoccludedCount / (width * height),
      meanErrorIncluded: mean,
      maxErrorIncluded: max,
      p99ErrorIncluded: p99,
    },
  };
}

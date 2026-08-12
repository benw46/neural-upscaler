import { acquireGpu } from "./gpu.ts";
import { WgslUNet } from "./model_wgsl.ts";

/** Live 540p-input vs WGSL-network-1080p-output viewer, on a handful of
 * held-out frames copied from the E:\ dataset (see export/copy_demo_frames.py).
 *
 * Deliberately not a live/interactive camera demo: the hand-written WGSL
 * kernel runs at ~3.3s/frame at this resolution (see docs/PHASE-4-SUMMARY.md)
 * so a per-frame toggle at camera framerate would just be a slideshow. This
 * shows a fixed frame in three states -- the raw 540p input, the network's
 * 1080p reconstruction, and the actual captured 1080p ground truth -- so
 * quality is easy to judge without pretending this runs in real time.
 *
 * Temporal input is cold-started (zeroed warped-previous, fully-invalid
 * disocclusion mask) rather than fed real frame-to-frame history -- exactly
 * frame 0's input in every training sequence (see train_temporal.py), a
 * validated code path, not an approximation. Building a live WGSL warp step
 * so history genuinely accumulates across a sequence of demo frames is real
 * additional work, out of scope here (see the owner's answer in this
 * session's viewer-scoping questions).
 */

const IN_W = 960;
const IN_H = 540;
const IN_H_PAD = 544; // pad_to_multiple(x, 8) on the height only -- 960 is already a multiple of 8
const GT_W = 1920;
const GT_H = 1080; // network output before cropping is (IN_H_PAD*2, GT_W, 3) = (1088, 1920, 3); pixel-shuffle doubles both input dims
const DEPTH_NORM = 50.0; // training/src/dataset.py -- must match exactly, see CLAUDE.md preprocessing-parity note
const IN_CHANNELS = 8;

type Mode = "input" | "network" | "groundtruth";

const statusEl = document.querySelector<HTMLDivElement>("#status")!;
const canvas = document.querySelector<HTMLCanvasElement>("#canvas")!;
const ctx = canvas.getContext("2d")!;
const frameButtonsEl = document.querySelector<HTMLDivElement>("#frameButtons")!;
const modeButtonsEl = document.querySelector<HTMLDivElement>("#modeButtons")!;

function setStatus(line: string) {
  statusEl.textContent = line;
  console.log(line);
}

interface Manifest {
  inputWidth: number;
  inputHeight: number;
  gtWidth: number;
  gtHeight: number;
  frames: number[];
}

interface FrameData {
  color: Float16Array; // (IN_H, IN_W, 4)
  depth: Float32Array; // (IN_H, IN_W, 1)
  gt: Float16Array; // (GT_H, GT_W, 4)
}

async function fetchBin(url: string): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url} failed: ${res.status}`);
  return res.arrayBuffer();
}

async function loadFrame(idx: number): Promise<FrameData> {
  const fname = `${String(idx).padStart(6, "0")}.bin`;
  const [colorBuf, depthBuf, gtBuf] = await Promise.all([
    fetchBin(`/demo_frames/color/${fname}`),
    fetchBin(`/demo_frames/depth/${fname}`),
    fetchBin(`/demo_frames/gt_color/${fname}`),
  ]);
  return {
    color: new Float16Array(colorBuf),
    depth: new Float32Array(depthBuf),
    gt: new Float16Array(gtBuf),
  };
}

/** Builds the 8-channel NHWC input the temporal model expects: colour(3) +
 * depth(1)/DEPTH_NORM + zeroed warped-previous(3) + fully-invalid
 * disocclusion(1), reflect-padded 540->544 rows to match
 * training/src/dataset.py's pad_to_multiple(mode="reflect") exactly --
 * preprocessing parity between training and inference is a named
 * CLAUDE.md fragile-logic item. */
function buildNetworkInput(frame: FrameData): Float32Array {
  const out = new Float32Array(IN_H_PAD * IN_W * IN_CHANNELS);
  for (let y = 0; y < IN_H_PAD; y++) {
    // torch reflect padding: row H+k (k=0..pad-1) mirrors row H-2-k, the
    // edge row (H-1) is not repeated.
    const srcY = y < IN_H ? y : IN_H - 2 - (y - IN_H);
    for (let x = 0; x < IN_W; x++) {
      const colorBase = (srcY * IN_W + x) * 4;
      const depthBase = srcY * IN_W + x;
      const outBase = (y * IN_W + x) * IN_CHANNELS;
      out[outBase + 0] = frame.color[colorBase + 0];
      out[outBase + 1] = frame.color[colorBase + 1];
      out[outBase + 2] = frame.color[colorBase + 2];
      out[outBase + 3] = frame.depth[depthBase] / DEPTH_NORM;
      out[outBase + 4] = 0;
      out[outBase + 5] = 0;
      out[outBase + 6] = 0;
      out[outBase + 7] = 1;
    }
  }
  return out;
}

/** Nearest-neighbour 2x upscale of the raw 540p colour into a 1920x1080
 * ImageData, so the "raw input" view fills the same canvas as the other two
 * modes -- deliberately blocky, that blockiness *is* the point of the
 * comparison. */
function drawInput(frame: FrameData) {
  const img = ctx.createImageData(GT_W, GT_H);
  for (let y = 0; y < GT_H; y++) {
    const srcY = y >> 1;
    for (let x = 0; x < GT_W; x++) {
      const srcX = x >> 1;
      const srcBase = (srcY * IN_W + srcX) * 4;
      const dstBase = (y * GT_W + x) * 4;
      img.data[dstBase + 0] = clamp255(frame.color[srcBase + 0]);
      img.data[dstBase + 1] = clamp255(frame.color[srcBase + 1]);
      img.data[dstBase + 2] = clamp255(frame.color[srcBase + 2]);
      img.data[dstBase + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

function drawGroundTruth(frame: FrameData) {
  const img = ctx.createImageData(GT_W, GT_H);
  for (let i = 0; i < GT_W * GT_H; i++) {
    img.data[i * 4 + 0] = clamp255(frame.gt[i * 4 + 0]);
    img.data[i * 4 + 1] = clamp255(frame.gt[i * 4 + 1]);
    img.data[i * 4 + 2] = clamp255(frame.gt[i * 4 + 2]);
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

/** networkOutput: NHWC Float32Array, (OUT_H_PAD, GT_W, 3) -- cropped
 * top-left to (GT_H, GT_W), matching training/src/dataset.py's
 * crop_to_size (top-left crop undoes the bottom-only reflect pad). */
function drawNetworkOutput(networkOutput: Float32Array) {
  const img = ctx.createImageData(GT_W, GT_H);
  for (let y = 0; y < GT_H; y++) {
    for (let x = 0; x < GT_W; x++) {
      const srcBase = (y * GT_W + x) * 3;
      const dstBase = (y * GT_W + x) * 4;
      img.data[dstBase + 0] = clamp255(networkOutput[srcBase + 0]);
      img.data[dstBase + 1] = clamp255(networkOutput[srcBase + 1]);
      img.data[dstBase + 2] = clamp255(networkOutput[srcBase + 2]);
      img.data[dstBase + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

function clamp255(v: number): number {
  const s = v * 255;
  return s < 0 ? 0 : s > 255 ? 255 : s;
}

async function main() {
  if (!("gpu" in navigator)) {
    throw new Error("navigator.gpu is undefined — WebGPU not available in this browser/context.");
  }

  setStatus("loading manifest…");
  const manifest: Manifest = await (await fetch("/demo_frames/manifest.json")).json();

  setStatus("acquiring GPU + loading network weights…");
  const { device, hasShaderF16, adapter } = await acquireGpu();
  const info = adapter.info;
  const unet = new WgslUNet(device, hasShaderF16);
  await unet.loadWeights("/weights/manifest.json", "/weights/weights.bin");

  const frameCache = new Map<number, FrameData>();
  const networkCache = new Map<number, Float32Array>(); // computed once per frame, toggling is free after that

  let currentFrame = manifest.frames[0];
  let currentMode: Mode = "input";

  function setButtonsEnabled(enabled: boolean) {
    for (const btn of [...frameButtonsEl.children, ...modeButtonsEl.children]) {
      (btn as HTMLButtonElement).disabled = !enabled;
    }
  }

  function highlightActive() {
    for (const btn of frameButtonsEl.children as unknown as HTMLButtonElement[]) {
      btn.classList.toggle("active", Number(btn.dataset.frame) === currentFrame);
    }
    for (const btn of modeButtonsEl.children as unknown as HTMLButtonElement[]) {
      btn.classList.toggle("active", btn.dataset.mode === currentMode);
    }
  }

  async function render() {
    highlightActive();
    const frame = frameCache.get(currentFrame)!;

    if (currentMode === "input") {
      setStatus(`frame ${currentFrame} — raw 540p input (nearest-upscaled 2x for display)`);
      drawInput(frame);
      return;
    }

    if (currentMode === "groundtruth") {
      setStatus(`frame ${currentFrame} — captured 1080p ground truth`);
      drawGroundTruth(frame);
      return;
    }

    // network mode
    let out = networkCache.get(currentFrame);
    if (!out) {
      setButtonsEnabled(false);
      setStatus(`frame ${currentFrame} — running WGSL network (~3.3s at this resolution, see docs/PHASE-4-SUMMARY.md)…`);
      const inputData = buildNetworkInput(frame);
      const t0 = performance.now();
      const inputFm = unet.allocInput(IN_W, IN_H_PAD, IN_CHANNELS, inputData);
      const encoder = device.createCommandEncoder();
      const { output } = unet.forward(encoder, inputFm);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      const full = await unet.readFeatureMap(output); // (OUT_H_PAD, GT_W, 3)
      const t1 = performance.now();
      out = full.subarray(0, GT_H * GT_W * 3); // top-left crop: OUT_H_PAD rows -> GT_H rows, width already matches
      networkCache.set(currentFrame, out);
      setStatus(`frame ${currentFrame} — WGSL network output (computed in ${((t1 - t0) / 1000).toFixed(2)}s, cached)`);
      setButtonsEnabled(true);
    } else {
      setStatus(`frame ${currentFrame} — WGSL network output (cached)`);
    }
    drawNetworkOutput(out);
  }

  for (const idx of manifest.frames) {
    const btn = document.createElement("button");
    btn.textContent = String(idx);
    btn.dataset.frame = String(idx);
    btn.addEventListener("click", async () => {
      currentFrame = idx;
      if (!frameCache.has(idx)) {
        setButtonsEnabled(false);
        setStatus(`loading frame ${idx}…`);
        frameCache.set(idx, await loadFrame(idx));
        setButtonsEnabled(true);
      }
      await render();
    });
    frameButtonsEl.appendChild(btn);
  }

  const modes: { mode: Mode; label: string }[] = [
    { mode: "input", label: "540p input" },
    { mode: "network", label: "network 1080p" },
    { mode: "groundtruth", label: "ground truth 1080p" },
  ];
  for (const { mode, label } of modes) {
    const btn = document.createElement("button");
    btn.textContent = label;
    btn.dataset.mode = mode;
    btn.addEventListener("click", async () => {
      currentMode = mode;
      await render();
    });
    modeButtonsEl.appendChild(btn);
  }

  setStatus(`loading frame ${currentFrame}…`);
  frameCache.set(currentFrame, await loadFrame(currentFrame));
  console.log(`[gpu] adapter: ${info?.vendor ?? "?"} / ${info?.description ?? "?"}, shader-f16: ${hasShaderF16}`);
  await render();
}

main().catch((err) => {
  console.error(err);
  setStatus(`ERROR: ${err instanceof Error ? (err.stack ?? err.message) : String(err)}`);
});

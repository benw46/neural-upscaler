import { acquireGpu } from "./gpu.ts";
import { WgslUNet } from "./model_wgsl.ts";
import { LiveScenePipeline, IN_W, IN_H, GT_W, GT_H, IN_H_PAD, IN_CHANNELS, DEPTH_NORM, type DisplayMode, type LiveFrameResult } from "./live_pipeline.ts";

/** Live 540p-input vs WGSL-network-1080p-output viewer, with two distinct
 * data sources sharing the same three-way display mode (540p input /
 * network 1080p / ground truth 1080p):
 *
 * - **Static** (default): a handful of held-out frames copied from the
 *   E:\ dataset (see export/copy_demo_frames.py). Cold-started (zeroed
 *   warped-previous, fully-invalid disocclusion mask) -- exactly frame 0's
 *   input in every training sequence (see train_temporal.py), a validated
 *   code path, not an approximation.
 * - **Realtime** (the "realtime" toggle): drives LiveScenePipeline (see
 *   that file) -- the project's live scripted-camera-path scene, rendered
 *   and upscaled continuously, with *real* accumulated temporal history
 *   (real motion-vector warping, not cold-started) and a genuine live
 *   unjittered-high-res render for the "ground truth" mode (not
 *   supersampled -- see live_pipeline.ts's docstring). All three display
 *   modes stay live and switchable while the loop keeps running -- clicking
 *   a mode button just changes what the *next* frame draws, it doesn't
 *   restart or stall the loop.
 *
 * The two sources are deliberately kept distinct rather than blended: the
 * static frames are fixed, previously-captured content useful for
 * comparing this project's own held-out gate frames; realtime is the
 * project's own live scene. Switching the "realtime" toggle switches which
 * source the three display-mode buttons are showing.
 *
 * The "pause" button (only enabled during realtime) freezes the scripted
 * camera path exactly where it is -- via LiveScenePipeline.redisplay(),
 * which re-shows the *same* already-rendered frame in a different display
 * mode without advancing anything, rather than stopping the loop and
 * restarting it. Mode buttons stay instantly responsive while paused
 * (no waiting for the next loop tick, unlike the running/unpaused case);
 * resuming continues the camera path from exactly where it was paused.
 */

const statusEl = document.querySelector<HTMLDivElement>("#status")!;
const fpsEl = document.querySelector<HTMLDivElement>("#fpsReadout")!;
const canvas = document.querySelector<HTMLCanvasElement>("#canvas")!;
const ctx = canvas.getContext("2d")!;
const frameButtonsEl = document.querySelector<HTMLDivElement>("#frameButtons")!;
const modeButtonsEl = document.querySelector<HTMLDivElement>("#modeButtons")!;
const realtimeToggleEl = document.querySelector<HTMLButtonElement>("#realtimeToggle")!;
const pauseToggleEl = document.querySelector<HTMLButtonElement>("#pauseToggle")!;

function setStatus(line: string) {
  statusEl.textContent = line;
  console.log(line);
}

function setFps(line: string, idle: boolean) {
  fpsEl.textContent = line;
  fpsEl.classList.toggle("idle", idle);
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
 * CLAUDE.md fragile-logic item. Static-frame path only -- the realtime path
 * builds this on the GPU (see pack_input.wgsl) with real warp channels
 * instead of the zeros/ones below. */
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

/** Nearest-neighbour 2x upscale of raw 540p colour (IN_H, IN_W, 4) into a
 * 1920x1080 ImageData, so the "raw input" view fills the same canvas as the
 * other two modes -- deliberately blocky, that blockiness *is* the point of
 * the comparison. Shared by both the static frames (frame.color) and the
 * realtime pipeline's live-rendered colour (LiveFrameResult.inputColour) --
 * same (H, W, 4) Float16Array layout either way. */
function drawInput(colour: Float16Array) {
  const img = ctx.createImageData(GT_W, GT_H);
  for (let y = 0; y < GT_H; y++) {
    const srcY = y >> 1;
    for (let x = 0; x < GT_W; x++) {
      const srcX = x >> 1;
      const srcBase = (srcY * IN_W + srcX) * 4;
      const dstBase = (y * GT_W + x) * 4;
      img.data[dstBase + 0] = clamp255(colour[srcBase + 0]);
      img.data[dstBase + 1] = clamp255(colour[srcBase + 1]);
      img.data[dstBase + 2] = clamp255(colour[srcBase + 2]);
      img.data[dstBase + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

/** Shared by the static captured ground truth (frame.gt) and the realtime
 * pipeline's live unjittered-high-res render (LiveFrameResult.groundTruthColour)
 * -- same (GT_H, GT_W, 4) Float16Array layout either way. The realtime one is
 * NOT supersampled like the static/offline one is, see live_pipeline.ts. */
function drawGroundTruth(colour: Float16Array) {
  const img = ctx.createImageData(GT_W, GT_H);
  for (let i = 0; i < GT_W * GT_H; i++) {
    img.data[i * 4 + 0] = clamp255(colour[i * 4 + 0]);
    img.data[i * 4 + 1] = clamp255(colour[i * 4 + 1]);
    img.data[i * 4 + 2] = clamp255(colour[i * 4 + 2]);
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

/** networkOutput: NHWC Float32Array, already cropped to (GT_H, GT_W, 3) --
 * matches training/src/dataset.py's crop_to_size (top-left crop undoes the
 * bottom-only reflect pad). Shared by both data sources. */
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

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function modeLabelFor(mode: DisplayMode): string {
  return mode === "input" ? "540p input" : mode === "groundtruth" ? "ground truth 1080p (live, not supersampled)" : "network 1080p";
}

async function main() {
  if (!("gpu" in navigator)) {
    throw new Error("navigator.gpu is undefined — WebGPU not available in this browser/context.");
  }

  setStatus("loading manifest…");
  const manifest: Manifest = await (await fetch("/demo_frames/manifest.json")).json();

  setStatus("acquiring GPU + loading network weights…");
  const { device, hasShaderF16, adapter, maxWorkgroupStorageSize } = await acquireGpu();
  const info = adapter.info;
  const unet = new WgslUNet(device, hasShaderF16, false, maxWorkgroupStorageSize);
  await unet.loadWeights("/weights/manifest.json", "/weights/weights.bin");

  const frameCache = new Map<number, FrameData>();
  const networkCache = new Map<number, Float32Array>(); // computed once per frame, toggling is free after that

  let currentFrame = manifest.frames[0];
  let currentMode: DisplayMode = "input";
  let realtimeActive = false;
  let livePaused = false;
  let pipeline: LiveScenePipeline | null = null; // lazily built on first realtime use -- not every visit needs the live-scene machinery

  /** Shared by runRealtimeLoop (the running case) and the mode-button
   * handler's paused case (via pipeline.redisplay()) -- same draw-dispatch
   * logic either way, just a different source for `result`. */
  function drawLiveResult(mode: DisplayMode, result: LiveFrameResult) {
    for (const btn of modeButtonsEl.children as unknown as HTMLButtonElement[]) {
      btn.classList.toggle("active", btn.dataset.mode === currentMode);
    }
    if (mode === "input" && result.inputColour) {
      drawInput(result.inputColour);
    } else if (mode === "groundtruth" && result.groundTruthColour) {
      drawGroundTruth(result.groundTruthColour);
    } else {
      drawNetworkOutput(result.networkOutput);
    }
  }

  function setFrameButtonsEnabled(enabled: boolean) {
    for (const btn of frameButtonsEl.children) (btn as HTMLButtonElement).disabled = !enabled;
  }
  function setButtonsEnabled(enabled: boolean) {
    setFrameButtonsEnabled(enabled);
    for (const btn of modeButtonsEl.children) (btn as HTMLButtonElement).disabled = !enabled;
  }

  /** Runs one forward pass and returns the cropped (GT_H, GT_W, 3) output
   * plus elapsed ms. `useCache` reuses/populates networkCache for the
   * static-frame click-to-view path only. */
  async function computeNetwork(frame: FrameData, frameIdx: number, useCache: boolean): Promise<{ out: Float32Array; ms: number }> {
    if (useCache) {
      const cached = networkCache.get(frameIdx);
      if (cached) return { out: cached, ms: 0 };
    }
    const inputData = buildNetworkInput(frame);
    const t0 = performance.now();
    const inputFm = unet.allocInput(IN_W, IN_H_PAD, IN_CHANNELS, inputData);
    const encoder = device.createCommandEncoder();
    const unetResult = unet.forward(encoder, inputFm);
    const { output } = unetResult;
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const full = await unet.readFeatureMap(output); // (OUT_H_PAD, GT_W, 3)
    const t1 = performance.now();
    // Nothing downstream holds a GPU reference once readFeatureMap has
    // copied the output into `full` -- safe to release every buffer this
    // forward() pass allocated, including output.buffer and the per-call
    // input buffer.
    WgslUNet.releaseIntermediates(unetResult);
    inputFm.buffer.destroy();
    const out = full.subarray(0, GT_H * GT_W * 3); // top-left crop: OUT_H_PAD rows -> GT_H rows, width already matches
    if (useCache) networkCache.set(frameIdx, out);
    return { out, ms: t1 - t0 };
  }

  /** Drives LiveScenePipeline continuously until realtimeActive is cleared.
   * While livePaused, idles without calling stepFrame() at all -- the
   * camera path genuinely stops advancing, not just "draws the same thing
   * repeatedly." Mode switches while paused go through the mode-button
   * handler's separate pipeline.redisplay() path instead of this loop, for
   * instant response rather than waiting on a poll interval.
   *
   * When *not* paused, reads `currentMode` fresh each iteration -- a
   * mode-button click while a step is in flight takes effect on the *next*
   * iteration, since that step's request (and therefore which of
   * inputColour/groundTruthColour it computed) was already committed when
   * it started; this can show one stale-mode frame in the rare case a click
   * lands mid-step, self-corrects next iteration, not worth extra
   * synchronisation for a one-frame cosmetic gap in a continuously-running
   * loop. */
  async function runRealtimeLoop() {
    let disocclusionLogAccum = 0;
    let disocclusionLogCount = 0;

    while (realtimeActive) {
      if (livePaused) {
        await sleep(100);
        continue;
      }

      const requestedMode = currentMode;
      const result = await pipeline!.stepFrame(requestedMode);
      if (!realtimeActive) break; // stopped while this pass was in flight

      drawLiveResult(requestedMode, result);

      if (result.meanDisocclusion !== null) {
        disocclusionLogAccum += result.meanDisocclusion;
        disocclusionLogCount++;
        if (disocclusionLogCount % 30 === 0) {
          console.log(`[realtime] mean disocclusion fraction over last 30 frames: ${(disocclusionLogAccum / 30).toFixed(4)}`);
          disocclusionLogAccum = 0;
        }
      }

      setStatus(`realtime — live scene, frame ${result.frameIndex}, t=${result.t.toFixed(2)}s, view: ${modeLabelFor(requestedMode)}`);
      // Pause may have been clicked while this step was in flight -- prefer
      // that over showing a real fps number that's about to be stale, so
      // the readout is consistent regardless of exactly when the click landed.
      if (livePaused) {
        setFps("paused", true);
      } else {
        const fps = 1000 / result.ms;
        setFps(`${result.ms.toFixed(1)}ms/frame · ${fps.toFixed(1)}fps`, false);
      }
    }
    setFps("", true);
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
      drawInput(frame.color);
      return;
    }

    if (currentMode === "groundtruth") {
      setStatus(`frame ${currentFrame} — captured 1080p ground truth`);
      drawGroundTruth(frame.gt);
      return;
    }

    // network mode
    const wasCached = networkCache.has(currentFrame);
    if (!wasCached) {
      setButtonsEnabled(false);
      setStatus(`frame ${currentFrame} — running WGSL network (~0.10s at this resolution, see docs/OPTIMISATIONS.md)…`);
    }
    const { out, ms } = await computeNetwork(frame, currentFrame, true);
    if (!wasCached) {
      setStatus(`frame ${currentFrame} — WGSL network output (computed in ${ms.toFixed(1)}ms, cached)`);
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
      if (realtimeActive) return; // frame buttons don't apply during live playback
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

  const modes: { mode: DisplayMode; label: string }[] = [
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
      if (realtimeActive && livePaused) {
        // Paused: redisplay the *same* frozen frame in the new mode right
        // away, via pipeline.redisplay() -- instant, no waiting on
        // runRealtimeLoop's poll interval, and the camera path doesn't move.
        const result = await pipeline!.redisplay(mode);
        drawLiveResult(mode, result);
        setStatus(`realtime — paused, frame ${result.frameIndex}, t=${result.t.toFixed(2)}s, view: ${modeLabelFor(mode)}`);
        return;
      }
      if (realtimeActive) {
        // Running: just update state -- runRealtimeLoop picks it up next
        // iteration (see that function's docstring); don't stop/restart it.
        for (const b of modeButtonsEl.children as unknown as HTMLButtonElement[]) {
          b.classList.toggle("active", b.dataset.mode === currentMode);
        }
        return;
      }
      await render();
    });
    modeButtonsEl.appendChild(btn);
  }

  function setPaused(paused: boolean) {
    livePaused = paused;
    pauseToggleEl.textContent = paused ? "▶ resume" : "⏸ pause";
    pauseToggleEl.classList.toggle("active", paused);
  }

  realtimeToggleEl.addEventListener("click", async () => {
    if (realtimeActive) {
      // Stop: runRealtimeLoop() notices on its next loop check (or the
      // in-flight-pass check right after it) and exits on its own; nothing
      // here needs to await it finishing.
      realtimeActive = false;
      realtimeToggleEl.textContent = "▶ realtime";
      realtimeToggleEl.classList.remove("active");
      setPaused(false);
      pauseToggleEl.disabled = true;
      setFrameButtonsEnabled(true);
      await render(); // redraw whatever the frame/mode buttons currently point at, not a stale realtime frame
      return;
    }

    realtimeToggleEl.textContent = "■ stop";
    realtimeToggleEl.classList.add("active");
    // Frame buttons don't apply to the live scene -- disable just those;
    // mode buttons must stay clickable so the view can switch live.
    setFrameButtonsEnabled(false);
    for (const btn of frameButtonsEl.children as unknown as HTMLButtonElement[]) {
      btn.classList.remove("active"); // no static frame corresponds to the live scene
    }

    if (!pipeline) {
      setStatus("realtime — building live-scene pipeline (first use only)…");
      pipeline = await LiveScenePipeline.create(device, unet, hasShaderF16);
    }
    realtimeActive = true;
    pauseToggleEl.disabled = false;
    runRealtimeLoop(); // intentionally not awaited -- runs in the background until the toggle flips realtimeActive back off
  });

  pauseToggleEl.addEventListener("click", () => {
    if (!realtimeActive) return; // shouldn't be reachable (disabled otherwise), guarded anyway
    setPaused(!livePaused);
    if (livePaused) setFps("paused", true);
  });

  setStatus(`loading frame ${currentFrame}…`);
  frameCache.set(currentFrame, await loadFrame(currentFrame));
  console.log(`[gpu] adapter: ${info?.vendor ?? "?"} / ${info?.description ?? "?"}, shader-f16: ${hasShaderF16}`);
  await render();
}

main().catch((err) => {
  console.error(err);
  setStatus(`ERROR: ${err instanceof Error ? (err.stack ?? err.message) : String(err)}`);
});

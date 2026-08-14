import { acquireGpu } from "./gpu.ts";
import { WgslUNet } from "./model_wgsl.ts";
import { LiveScenePipeline, GT_W, GT_H } from "./live_pipeline.ts";

/** Live renderer + real-time WGSL upscaling + real temporal warping, driving
 * LiveScenePipeline (see that file for the actual mechanism and its
 * validation status) permanently in "network" display mode. viewer.ts's
 * "realtime" toggle drives the same pipeline with all three display modes;
 * this page exists as the focused, single-purpose version of it.
 */

const statusEl = document.querySelector<HTMLDivElement>("#status")!;
const fpsEl = document.querySelector<HTMLDivElement>("#fpsReadout")!;
const canvas = document.querySelector<HTMLCanvasElement>("#canvas")!;
const ctx = canvas.getContext("2d")!;
const liveToggleEl = document.querySelector<HTMLButtonElement>("#liveToggle")!;

function setStatus(line: string) {
  statusEl.textContent = line;
  console.log(line);
}
function setFps(line: string, idle: boolean) {
  fpsEl.textContent = line;
  fpsEl.classList.toggle("idle", idle);
}
function clamp255(v: number): number {
  const s = v * 255;
  return s < 0 ? 0 : s > 255 ? 255 : s;
}

function drawOutput(full: Float32Array) {
  const img = ctx.createImageData(GT_W, GT_H);
  for (let y = 0; y < GT_H; y++) {
    for (let x = 0; x < GT_W; x++) {
      const srcBase = (y * GT_W + x) * 3;
      const dstBase = (y * GT_W + x) * 4;
      img.data[dstBase + 0] = clamp255(full[srcBase + 0]);
      img.data[dstBase + 1] = clamp255(full[srcBase + 1]);
      img.data[dstBase + 2] = clamp255(full[srcBase + 2]);
      img.data[dstBase + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

async function main() {
  if (!("gpu" in navigator)) {
    throw new Error("navigator.gpu is undefined — WebGPU not available in this browser/context.");
  }

  setStatus("acquiring GPU + loading network weights…");
  const { device, hasShaderF16, adapter, maxWorkgroupStorageSize } = await acquireGpu();
  const info = adapter.info;

  const unet = new WgslUNet(device, hasShaderF16, false, maxWorkgroupStorageSize);
  await unet.loadWeights("/weights/manifest.json", "/weights/weights.bin");

  setStatus("compiling live-pipeline kernels…");
  const pipeline = await LiveScenePipeline.create(device, unet, hasShaderF16);

  let liveActive = false;
  let disocclusionLogAccum = 0;
  let disocclusionLogCount = 0;

  async function runLiveLoop() {
    while (liveActive) {
      const result = await pipeline.stepFrame("network");
      if (!liveActive) break; // stopped while this pass was in flight

      drawOutput(result.networkOutput);

      if (result.meanDisocclusion !== null) {
        disocclusionLogAccum += result.meanDisocclusion;
        disocclusionLogCount++;
        if (disocclusionLogCount % 30 === 0) {
          console.log(`[live] mean disocclusion fraction over last 30 frames: ${(disocclusionLogAccum / 30).toFixed(4)}`);
          disocclusionLogAccum = 0;
        }
      }

      const fps = 1000 / result.ms;
      setStatus(`live — frame ${result.frameIndex}, t=${result.t.toFixed(2)}s, jitter [${result.jitter[0].toFixed(3)}, ${result.jitter[1].toFixed(3)}]`);
      setFps(`${result.ms.toFixed(1)}ms/frame · ${fps.toFixed(1)}fps`, false);
    }
    setFps("", true);
  }

  liveToggleEl.addEventListener("click", () => {
    if (liveActive) {
      liveActive = false;
      liveToggleEl.textContent = "▶ start live";
      liveToggleEl.classList.remove("active");
      return;
    }
    liveActive = true;
    liveToggleEl.textContent = "■ stop";
    liveToggleEl.classList.add("active");
    runLiveLoop();
  });

  setStatus("ready — press start to begin the live scripted camera path");
  console.log(`[gpu] adapter: ${info?.vendor ?? "?"} / ${info?.description ?? "?"}, shader-f16: ${hasShaderF16}`);
}

main().catch((err) => {
  console.error(err);
  setStatus(`ERROR: ${err instanceof Error ? (err.stack ?? err.message) : String(err)}`);
});

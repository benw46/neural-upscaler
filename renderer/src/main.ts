import { acquireGpu } from "./gpu/device.ts";
import { SceneRenderer } from "./render/renderer.ts";
import { GBuffer } from "./render/gbuffer.ts";
import { BlitPipeline } from "./render/blit.ts";
import { ScriptedCameraPath, frameState, stateViewProj } from "./camera/sequence.ts";
import mainShaderSource from "./render/shaders.wgsl?raw";

const SEED = 20260812;

const statusEl = document.querySelector<HTMLDivElement>("#status")!;
const canvas = document.querySelector<HTMLCanvasElement>("#gpu-canvas")!;

function setStatus(text: string) {
  statusEl.textContent = text;
}

async function main() {
  setStatus("requesting WebGPU adapter…");
  const { adapter, device, hasShaderF16 } = await acquireGpu();

  let width = 960;
  let height = 540;
  canvas.width = width;
  canvas.height = height;

  const context = canvas.getContext("webgpu");
  if (!context) throw new Error("canvas.getContext('webgpu') returned null.");
  const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format: canvasFormat, alphaMode: "opaque" });

  const renderer = new SceneRenderer(device, mainShaderSource);
  let gbuffer = new GBuffer(device, width, height);
  const blit = new BlitPipeline(device, canvasFormat);
  const cameraPath = new ScriptedCameraPath(SEED);

  let frameIndex = 0;
  let prevViewProj: Float32Array | null = null;

  const info = adapter.info;
  function describe(idx: number, jitter: [number, number]) {
    return [
      `adapter: ${info?.vendor ?? "?"} / ${info?.device ?? "?"}`,
      `shader-f16: ${hasShaderF16}`,
      `resolution: ${width}x${height}  (press R to toggle 960x540 / 1920x1080)`,
      `frame: ${idx}   jitter(texels): [${jitter[0].toFixed(3)}, ${jitter[1].toFixed(3)}]`,
    ].join("\n");
  }

  function frameLoop() {
    const state = frameState(cameraPath, frameIndex, renderer.colliders);
    const viewProj = new Float32Array(stateViewProj(state, width, height, true));
    if (!prevViewProj) prevViewProj = viewProj;

    renderer.renderInto(gbuffer, viewProj, prevViewProj, `frame-${frameIndex}`);

    const encoder = device.createCommandEncoder();
    blit.draw(encoder, gbuffer.colour, context!.getCurrentTexture().createView());
    device.queue.submit([encoder.finish()]);

    setStatus(describe(frameIndex, state.jitter));
    prevViewProj = viewProj;
    frameIndex++;
    requestAnimationFrame(frameLoop);
  }

  window.addEventListener("keydown", (e) => {
    if (e.key.toLowerCase() !== "r") return;
    [width, height] = width === 960 ? [1920, 1080] : [960, 540];
    canvas.width = width;
    canvas.height = height;
    gbuffer.resize(width, height);
    prevViewProj = null; // resolution change invalidates the previous jittered projection
  });

  requestAnimationFrame(frameLoop);
}

main().catch((err) => {
  console.error(err);
  setStatus(`ERROR: ${err instanceof Error ? err.message : String(err)}`);
});

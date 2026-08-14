/**
 * One-off repair tool: re-renders a single frame's ground-truth colour
 * buffer and overwrites it in place. Built specifically to fix
 * gt_color/000946.bin in the seed-20260812-colored dataset, truncated
 * (7.3MB vs the expected 15.8MB) because the interrupted capture process
 * crashed mid-write there and capture.ts's resume logic only checks
 * color/ directory contiguity, not per-file completeness in the other
 * three directories.
 *
 * Safe to do in isolation: the GT pass renders an unjittered view of frame
 * `--frame` alone (frameState/stateViewProj are pure functions of (seed,
 * frameIndex) -- see capture.ts's own comment on this) and never reads a
 * "previous frame" (motion vectors from this pass are never used, so its
 * own matrix stands in for "previous" -- again matching capture.ts). No
 * dependency on neighbouring frames, so this can't introduce any
 * inconsistency with the rest of the dataset.
 *
 * Usage:
 *   node scripts/fix_frame.ts --run E:\neural-upscaler\data\seed-20260812-colored --frame 946 --seed 20260812 --colored true
 */

import { create, globals } from "webgpu";
Object.assign(globalThis, globals);

import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { acquireGpu } from "../src/gpu/device.ts";
import { GBuffer } from "../src/render/gbuffer.ts";
import { SceneRenderer } from "../src/render/renderer.ts";
import { DownsamplePipeline } from "../src/render/downsample.ts";
import { readTextureToArrayBuffer } from "../src/capture/readback.ts";
import { ScriptedCameraPath, frameState, stateViewProj } from "../src/camera/sequence.ts";

const SRC_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "src");

function parseArgs(argv: string[]) {
  const raw: Record<string, string> = {};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg.startsWith("--")) {
      raw[arg.slice(2)] = argv[i + 1];
      i++;
    }
  }
  return {
    run: raw.run,
    frame: Number(raw.frame),
    seed: raw.seed ? Number(raw.seed) : 20260812,
    colored: raw.colored === "true",
    width: raw.width ? Number(raw.width) : 960,
    height: raw.height ? Number(raw.height) : 540,
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.run || Number.isNaN(args.frame)) {
    throw new Error("usage: --run <dir> --frame <index> [--seed 20260812] [--colored true]");
  }
  const gtWidth = args.width * 2;
  const gtHeight = args.height * 2;
  const superWidth = args.width * 4;
  const superHeight = args.height * 4;

  const gpuInstance = create([]);
  const { device } = await acquireGpu(gpuInstance);

  const mainShaderSource = await fs.readFile(path.join(SRC_DIR, "render", "shaders.wgsl"), "utf8");
  const downsampleShaderSource = await fs.readFile(path.join(SRC_DIR, "render", "downsample.wgsl"), "utf8");

  const renderer = new SceneRenderer(device, mainShaderSource, args.colored);
  const superGBuffer = new GBuffer(device, superWidth, superHeight);
  const gtColorTexture = device.createTexture({
    label: "gt-color-fix",
    size: [gtWidth, gtHeight],
    format: "rgba16float",
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
  });
  const downsample = new DownsamplePipeline(device, downsampleShaderSource);
  const cameraPath = new ScriptedCameraPath(args.seed);

  const state = frameState(cameraPath, args.frame, renderer.colliders);
  const gtViewProj = new Float32Array(stateViewProj(state, superWidth, superHeight, false));
  renderer.renderInto(superGBuffer, gtViewProj, gtViewProj, `fix-gt-${args.frame}`);

  const dsEncoder = device.createCommandEncoder({ label: `fix-downsample-${args.frame}` });
  downsample.run(dsEncoder, superGBuffer.colour, gtColorTexture);
  device.queue.submit([dsEncoder.finish()]);

  const gtColorBuf = await readTextureToArrayBuffer(device, gtColorTexture, gtWidth, gtHeight, 8);

  const outPath = path.join(args.run, "gt_color", `${String(args.frame).padStart(6, "0")}.bin`);
  const expectedSize = gtWidth * gtHeight * 4 * 2;
  if (gtColorBuf.byteLength !== expectedSize) {
    throw new Error(`re-rendered buffer is ${gtColorBuf.byteLength} bytes, expected ${expectedSize} -- aborting, not overwriting`);
  }
  await fs.writeFile(outPath, Buffer.from(gtColorBuf));
  console.log(`wrote ${gtColorBuf.byteLength} bytes to ${outPath} (frame ${args.frame}, t=${state.t.toFixed(3)}s)`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

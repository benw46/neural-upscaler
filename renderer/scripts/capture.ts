/**
 * Headless dataset capture — runs outside a browser via Node + Dawn (the
 * `webgpu` npm package). Renders the same procedural scene as the
 * interactive preview (src/main.ts) through the same SceneRenderer, so the
 * two never drift apart. For each output frame this produces:
 *   - a jittered 540p input (colour, linear depth, motion vectors)
 *   - an unjittered 1080p ground-truth colour frame (supersampled 4x at
 *     3840x2160 and box-filtered down)
 *
 * Usage:
 *   node scripts/capture.ts --seed 20260812 --frames 2000 --out E:\neural-upscaler\data
 *
 * Resumable: re-running with the same --run name skips frames already
 * written (detected by contiguous file presence in color/), so an
 * interrupted run can continue rather than restart.
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
import {
  ScriptedCameraPath,
  frameState,
  stateViewProj,
  DT,
  FOV_Y,
  NEAR,
  FAR,
} from "../src/camera/sequence.ts";
import { computeDiskBudget, fitsOnDisk, formatDiskBudgetReport } from "../src/capture/diskBudget.ts";
import { manifestLine, type DatasetHeader, type ManifestRecord } from "../src/capture/manifest.ts";

const SRC_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "src");

interface CliArgs {
  seed: number;
  frames: number;
  out: string;
  run: string;
  width: number;
  height: number;
}

function parseArgs(argv: string[]): CliArgs {
  const raw: Record<string, string> = {};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg.startsWith("--")) {
      raw[arg.slice(2)] = argv[i + 1];
      i++;
    }
  }
  const seed = raw.seed ? Number(raw.seed) : 20260812;
  return {
    seed,
    frames: raw.frames ? Number(raw.frames) : 60,
    out: raw.out ?? "E:\\neural-upscaler\\data",
    run: raw.run ?? `seed-${seed}`,
    width: raw.width ? Number(raw.width) : 960,
    height: raw.height ? Number(raw.height) : 540,
  };
}

function frameFileName(i: number): string {
  return `${String(i).padStart(6, "0")}.bin`;
}

async function ensureDir(p: string) {
  await fs.mkdir(p, { recursive: true });
}

/** Counts contiguous completed frames from 0 — the resumability check. A
 * gap (e.g. a frame that crashed mid-write) is treated as "not done" and
 * everything from there on is re-rendered rather than silently skipped. */
async function countCompleteFrames(colorDir: string): Promise<number> {
  let n = 0;
  for (;;) {
    try {
      await fs.access(path.join(colorDir, frameFileName(n)));
      n++;
    } catch {
      break;
    }
  }
  return n;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const gtWidth = args.width * 2;
  const gtHeight = args.height * 2;
  const superWidth = args.width * 4;
  const superHeight = args.height * 4;

  const runDir = path.join(args.out, args.run);
  const dirs = {
    color: path.join(runDir, "color"),
    depth: path.join(runDir, "depth"),
    motion: path.join(runDir, "motion"),
    gt_color: path.join(runDir, "gt_color"),
  };

  const budget = computeDiskBudget(args.width, args.height, gtWidth, gtHeight, args.frames);
  const driveRoot = path.parse(path.resolve(args.out)).root;
  const statfs = await fs.statfs(driveRoot);
  const freeBytes = statfs.bavail * statfs.bsize;
  console.log(formatDiskBudgetReport(budget, args.frames, freeBytes));
  if (!fitsOnDisk(budget, freeBytes)) {
    console.error("ABORTING: projected dataset would exceed 90% of free space on the target drive.");
    process.exit(1);
  }

  await ensureDir(dirs.color);
  await ensureDir(dirs.depth);
  await ensureDir(dirs.motion);
  await ensureDir(dirs.gt_color);

  const startFrame = await countCompleteFrames(dirs.color);
  if (startFrame > 0) {
    console.log(`Resuming: ${startFrame} frame(s) already present, continuing from frame ${startFrame}.`);
  }
  if (startFrame >= args.frames) {
    console.log("Nothing to do — requested frame count is already complete.");
    return;
  }

  const gpuInstance = create([]);
  const { device } = await acquireGpu(gpuInstance);

  const mainShaderSource = await fs.readFile(path.join(SRC_DIR, "render", "shaders.wgsl"), "utf8");
  const downsampleShaderSource = await fs.readFile(path.join(SRC_DIR, "render", "downsample.wgsl"), "utf8");

  const renderer = new SceneRenderer(device, mainShaderSource);
  const inputGBuffer = new GBuffer(device, args.width, args.height);
  const superGBuffer = new GBuffer(device, superWidth, superHeight);
  const gtColorTexture = device.createTexture({
    label: "gt-color",
    size: [gtWidth, gtHeight],
    format: "rgba16float",
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
  });
  const downsample = new DownsamplePipeline(device, downsampleShaderSource);
  const cameraPath = new ScriptedCameraPath(args.seed);

  const header: DatasetHeader = {
    seed: args.seed,
    frameCount: args.frames,
    inputWidth: args.width,
    inputHeight: args.height,
    gtWidth,
    gtHeight,
    gtSupersampleWidth: superWidth,
    gtSupersampleHeight: superHeight,
    fovYRadians: FOV_Y,
    near: NEAR,
    far: FAR,
    dt: DT,
    buffers: {
      color: { format: "rgba16float", width: args.width, height: args.height },
      depth: { format: "r32float", width: args.width, height: args.height },
      motion: { format: "rg16float", width: args.width, height: args.height },
      gt_color: { format: "rgba16float", width: gtWidth, height: gtHeight },
    },
    createdAt: new Date().toISOString(),
  };
  await fs.writeFile(path.join(runDir, "dataset.json"), JSON.stringify(header, null, 2));

  const manifestHandle = await fs.open(path.join(runDir, "manifest.jsonl"), "a");

  // Seed prevViewProj with frame (startFrame - 1)'s matrix so motion
  // vectors stay continuous across a resume, without needing to persist
  // GPU state between runs — frameState/stateViewProj are pure functions of
  // (seed, frameIndex), so recomputing is exact.
  const seedIndex = Math.max(startFrame - 1, 0);
  const seedState = frameState(cameraPath, seedIndex, renderer.colliders);
  let prevViewProj = new Float32Array(stateViewProj(seedState, args.width, args.height, true));

  const t0 = Date.now();
  for (let frameIndex = startFrame; frameIndex < args.frames; frameIndex++) {
    const state = frameState(cameraPath, frameIndex, renderer.colliders);
    const viewProj = new Float32Array(stateViewProj(state, args.width, args.height, true));

    renderer.renderInto(inputGBuffer, viewProj, prevViewProj, `capture-input-${frameIndex}`);

    // Ground truth: unjittered, supersampled 4x then box-filtered 2x2 twice
    // (3840x2160 -> 1920x1080) per Spec 1 Part B step 1. No real "previous"
    // frame concept applies here — motion vectors from this pass are never
    // read, so its own matrix stands in for "previous".
    const gtViewProj = new Float32Array(stateViewProj(state, superWidth, superHeight, false));
    renderer.renderInto(superGBuffer, gtViewProj, gtViewProj, `capture-gt-${frameIndex}`);

    const dsEncoder = device.createCommandEncoder({ label: `downsample-${frameIndex}` });
    downsample.run(dsEncoder, superGBuffer.colour, gtColorTexture);
    device.queue.submit([dsEncoder.finish()]);

    const [colorBuf, depthBuf, motionBuf, gtColorBuf] = await Promise.all([
      readTextureToArrayBuffer(device, inputGBuffer.colour, args.width, args.height, 8),
      readTextureToArrayBuffer(device, inputGBuffer.depth, args.width, args.height, 4),
      readTextureToArrayBuffer(device, inputGBuffer.motion, args.width, args.height, 4),
      readTextureToArrayBuffer(device, gtColorTexture, gtWidth, gtHeight, 8),
    ]);

    const fname = frameFileName(frameIndex);
    await Promise.all([
      fs.writeFile(path.join(dirs.color, fname), Buffer.from(colorBuf)),
      fs.writeFile(path.join(dirs.depth, fname), Buffer.from(depthBuf)),
      fs.writeFile(path.join(dirs.motion, fname), Buffer.from(motionBuf)),
      fs.writeFile(path.join(dirs.gt_color, fname), Buffer.from(gtColorBuf)),
    ]);

    const record: ManifestRecord = {
      frameIndex,
      t: state.t,
      jitter: state.jitter,
      eye: [state.pose.eye[0], state.pose.eye[1], state.pose.eye[2]],
      target: [state.pose.target[0], state.pose.target[1], state.pose.target[2]],
      up: [state.pose.up[0], state.pose.up[1], state.pose.up[2]],
      viewProj: Array.from(viewProj),
      prevViewProj: Array.from(prevViewProj),
    };
    await manifestHandle.write(manifestLine(record));

    if (frameIndex % 20 === 0 || frameIndex === args.frames - 1) {
      const elapsed = (Date.now() - t0) / 1000;
      const done = frameIndex - startFrame + 1;
      const rate = done / elapsed;
      console.log(`frame ${frameIndex + 1}/${args.frames}  (${rate.toFixed(2)} fps, ${elapsed.toFixed(1)}s elapsed)`);
    }

    prevViewProj = viewProj;
  }

  await manifestHandle.close();
  console.log(`Done. Frames ${startFrame}..${args.frames - 1} written to ${runDir}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

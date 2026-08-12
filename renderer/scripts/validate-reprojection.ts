/**
 * Reprojection validation gate — Spec 1 Part A/B gate criterion.
 *
 * For each consecutive frame pair (N-1, N) in the requested range, warps
 * N-1's colour into N's viewpoint using N's motion vectors and compares
 * against N's actual colour, excluding disoccluded pixels. Writes a warped
 * preview and an error heatmap PNG per pair, plus an aggregate stats JSON.
 *
 * Usage:
 *   node scripts/validate-reprojection.ts --run E:\neural-upscaler\data\seed-20260812 --start 1 --count 25
 */

import fs from "node:fs/promises";
import path from "node:path";

import { loadFrameBuffers, warpAndValidate } from "../src/validate/reproject.ts";
import { heatColor } from "../src/validate/colormap.ts";
import { encodePng } from "../src/validate/png.ts";

interface CliArgs {
  runDir: string;
  start: number;
  count: number;
  outDir: string;
  errorScale: number;
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
  if (!raw.run) throw new Error("--run <capture run dir> is required");
  return {
    runDir: raw.run,
    start: raw.start ? Number(raw.start) : 1,
    count: raw.count ? Number(raw.count) : 20,
    outDir: raw.out ?? path.join(raw.run, "validation"),
    errorScale: raw.errorScale ? Number(raw.errorScale) : 0.1,
  };
}

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

function toRgba8(buf: Float32Array, width: number, height: number): Uint8Array {
  const out = new Uint8Array(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    out[i * 4] = Math.round(clamp01(buf[i * 4]) * 255);
    out[i * 4 + 1] = Math.round(clamp01(buf[i * 4 + 1]) * 255);
    out[i * 4 + 2] = Math.round(clamp01(buf[i * 4 + 2]) * 255);
    out[i * 4 + 3] = 255;
  }
  return out;
}

function heatmapRgba8(absError: Float32Array, mask: Uint8Array, width: number, height: number, scale: number): Uint8Array {
  const out = new Uint8Array(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    let r: number, g: number, b: number;
    if (mask[i]) {
      // Disocclusion/off-screen pixels shown in blue — excluded from stats,
      // visually distinct from real error so the heatmap communicates both
      // at once, per the gate's "show the heatmaps" requirement.
      [r, g, b] = [20, 90, 200];
    } else {
      [r, g, b] = heatColor(absError[i] / scale);
    }
    out[i * 4] = r;
    out[i * 4 + 1] = g;
    out[i * 4 + 2] = b;
    out[i * 4 + 3] = 255;
  }
  return out;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const header = JSON.parse(await fs.readFile(path.join(args.runDir, "dataset.json"), "utf8"));
  const { inputWidth: width, inputHeight: height } = header;

  await fs.mkdir(args.outDir, { recursive: true });

  const perPair: Array<{
    frameIndex: number;
    meanErrorIncluded: number;
    maxErrorIncluded: number;
    p99ErrorIncluded: number;
    disocclusionFraction: number;
  }> = [];

  for (let n = args.start; n < args.start + args.count; n++) {
    const prev = await loadFrameBuffers(args.runDir, n - 1, width, height);
    const curr = await loadFrameBuffers(args.runDir, n, width, height);
    const result = warpAndValidate(prev, curr);

    const fname = `${String(n).padStart(6, "0")}.png`;
    await fs.writeFile(path.join(args.outDir, `warped_${fname}`), encodePng(width, height, toRgba8(result.warped, width, height)));
    await fs.writeFile(
      path.join(args.outDir, `heatmap_${fname}`),
      encodePng(width, height, heatmapRgba8(result.absError, result.disocclusionMask, width, height, args.errorScale))
    );

    perPair.push({ frameIndex: n, ...result.stats } as never);
    console.log(
      `frame ${n}: mean=${result.stats.meanErrorIncluded.toFixed(5)} p99=${result.stats.p99ErrorIncluded.toFixed(
        5
      )} max=${result.stats.maxErrorIncluded.toFixed(5)} disocclusion=${(result.stats.disocclusionFraction * 100).toFixed(2)}%`
    );
  }

  const meanOfMeans = perPair.reduce((a, p) => a + p.meanErrorIncluded, 0) / perPair.length;
  const worstMax = Math.max(...perPair.map((p) => p.maxErrorIncluded));
  const meanDisocclusion = perPair.reduce((a, p) => a + p.disocclusionFraction, 0) / perPair.length;

  const summary = {
    runDir: args.runDir,
    pairsEvaluated: perPair.length,
    frameRange: [args.start, args.start + args.count - 1],
    meanOfMeanErrorIncluded: meanOfMeans,
    worstMaxErrorIncluded: worstMax,
    meanDisocclusionFraction: meanDisocclusion,
    perPair,
  };
  await fs.writeFile(path.join(args.outDir, "summary.json"), JSON.stringify(summary, null, 2));

  console.log("\n--- summary ---");
  console.log(`pairs evaluated       : ${perPair.length}`);
  console.log(`mean-of-mean error    : ${meanOfMeans.toFixed(5)}  (colour channels are ~[0,1])`);
  console.log(`worst-case max error  : ${worstMax.toFixed(5)}`);
  console.log(`mean disocclusion frac: ${(meanDisocclusion * 100).toFixed(2)}%`);
  console.log(`outputs written to    : ${args.outDir}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

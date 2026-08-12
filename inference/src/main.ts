import * as ort from "onnxruntime-web/webgpu";
import { acquireGpu } from "./gpu.ts";
import { WgslUNet } from "./model_wgsl.ts";

const statusEl = document.querySelector<HTMLDivElement>("#status")!;
function log(line: string) {
  statusEl.textContent += (statusEl.textContent === "initialising…" ? "" : "\n") + line;
  if (statusEl.textContent === "initialising…") statusEl.textContent = line;
  console.log(line);
}

ort.env.wasm.wasmPaths = "/ort/";

const PATCH_SIZE = 128;
const IN_CHANNELS = 8; // Spec 3's temporal model: colour(3) + depth(1) + warped-prev(3) + disocclusion(1)
const FP16_TOLERANCE = 0.01; // FP16 has ~3 decimal digits of precision

async function fetchFloat32(url: string): Promise<Float32Array> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url} failed: ${res.status}`);
  const buf = await res.arrayBuffer();
  return new Float32Array(buf);
}

function compare(a: Float32Array, b: Float32Array): { max: number; mean: number } {
  if (a.length !== b.length) throw new Error(`length mismatch: ${a.length} vs ${b.length}`);
  let max = 0;
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const err = Math.abs(a[i] - b[i]);
    if (err > max) max = err;
    sum += err;
  }
  return { max, mean: sum / a.length };
}

async function runOrtWebCheck(): Promise<void> {
  log("--- Step 1: ONNX Runtime Web correctness oracle (Spec 4 step 1) ---");
  // NCHW -- PyTorch/ONNX's native layout, what ORT Web expects. The WGSL
  // path uses a separate NHWC copy of the same values (test_input_temporal_nhwc.bin).
  const inputData = await fetchFloat32("/test_input_temporal.bin");
  const pytorchOutput = await fetchFloat32("/pytorch_output_temporal.bin");

  const session = await ort.InferenceSession.create("/models/temporal_unet.onnx", {
    executionProviders: ["webgpu"],
    graphOptimizationLevel: "all",
  });
  const inputTensor = new ort.Tensor("float32", inputData, [1, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE]);
  const results = await session.run({ [session.inputNames[0]]: inputTensor });
  const ortData = results[session.outputNames[0]].data as Float32Array;

  const { max, mean } = compare(ortData, pytorchOutput);
  log(`ORT Web vs PyTorch: max abs error=${max.toFixed(6)} mean=${mean.toFixed(6)}`);
  log(`${max < FP16_TOLERANCE ? "PASSED" : "FAILED"}: ORT Web still matches PyTorch within FP16 tolerance\n`);
}

const LAYER_CHECKS = [
  { wgslName: "stem.conv", file: "stem_conv.bin" },
  { wgslName: "down1.refine.conv", file: "down1_refine_conv.bin" },
  { wgslName: "down2.refine.conv", file: "down2_refine_conv.bin" },
  { wgslName: "down3.refine.conv", file: "down3_refine_conv.bin" },
  { wgslName: "bottleneck.1.conv", file: "bottleneck_1_conv.bin" },
  { wgslName: "up1.conv2.conv", file: "up1_conv2_conv.bin" },
  { wgslName: "up2.conv2.conv", file: "up2_conv2_conv.bin" },
  { wgslName: "up3.conv2.conv", file: "up3_conv2_conv.bin" },
  { wgslName: "head", file: "head.bin" },
  { wgslName: "pixel_shuffle", file: "pixel_shuffle.bin" },
];

async function runWgslCheck() {
  log("--- Step 2: hand-written WGSL vs PyTorch, per layer (Spec 4 step 6) ---");
  const inputData = await fetchFloat32("/test_input_temporal_nhwc.bin");
  const { device, hasShaderF16, adapter } = await acquireGpu();
  const info = adapter.info;
  log(`adapter: ${info?.vendor ?? "?"} / ${info?.device ?? "?"} / ${info?.description ?? "?"}`);
  log(`shader-f16: ${hasShaderF16}`);

  const unet = new WgslUNet(device, hasShaderF16);
  await unet.loadWeights("/weights/manifest.json", "/weights/weights.bin");

  const inputFm = unet.allocInput(PATCH_SIZE, PATCH_SIZE, IN_CHANNELS, inputData);

  const encoder = device.createCommandEncoder();
  const t0 = performance.now();
  const { output, intermediates } = unet.forward(encoder, inputFm);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const t1 = performance.now();
  log(`forward pass submitted+completed in ${(t1 - t0).toFixed(2)}ms (untimed, no timestamp queries yet)\n`);

  log(`${"layer".padEnd(20)} ${"max err".padStart(10)} ${"mean err".padStart(10)}  status`);
  let allPassed = true;
  for (const check of LAYER_CHECKS) {
    const fm = intermediates.get(check.wgslName);
    if (!fm) throw new Error(`no WGSL intermediate captured for ${check.wgslName}`);
    const wgslData = await unet.readFeatureMap(fm);
    const refData = await fetchFloat32(`/intermediates/${check.file}`);
    const { max, mean } = compare(wgslData, refData);
    const passed = max < FP16_TOLERANCE;
    allPassed = allPassed && passed;
    log(`${check.wgslName.padEnd(20)} ${max.toFixed(6).padStart(10)} ${mean.toFixed(6).padStart(10)}  ${passed ? "OK" : "FAIL"}`);
  }

  // pixel_shuffle.bin (an intermediate capture, NHWC) is the same tensor as
  // pytorch_output_temporal.bin (NCHW) -- reused here instead of adding a
  // third copy of the same values in a third layout.
  const finalWgsl = await unet.readFeatureMap(output);
  const pytorchOutputNhwc = await fetchFloat32("/intermediates/pixel_shuffle.bin");
  const { max: finalMax, mean: finalMean } = compare(finalWgsl, pytorchOutputNhwc);
  log(`\nfinal output vs PyTorch: max=${finalMax.toFixed(6)} mean=${finalMean.toFixed(6)}`);

  log(`\n${allPassed && finalMax < FP16_TOLERANCE ? "PASSED" : "FAILED"}: all layers within FP16 tolerance (${FP16_TOLERANCE})`);
}

// Real deployment resolution per CLAUDE.md hard rule 1 (540p -> 1080p), not
// the 128x128 training patches used for correctness checking above. Padded
// up to a multiple of 8 for the encoder's 3 stride-2 downsamples, same as
// the Python inference-time padding in training/src/dataset.py.
const REAL_WIDTH = 960;
const REAL_HEIGHT = 540;
const PROFILE_WARMUP_RUNS = 5;
const PROFILE_TIMED_RUNS = 20;

function padUp(n: number, multiple: number): number {
  return Math.ceil(n / multiple) * multiple;
}

async function profileAtResolution(unet: WgslUNet, device: GPUDevice, width: number, height: number, warmupRuns: number, timedRuns: number) {
  log(`  size ${width}x${height}: allocating input…`);
  const dummyInput = new Float32Array(width * height * IN_CHANNELS);
  for (let i = 0; i < dummyInput.length; i++) dummyInput[i] = Math.random();

  for (let i = 0; i < warmupRuns; i++) {
    log(`  size ${width}x${height}: warmup ${i + 1}/${warmupRuns}…`);
    const inputFm = unet.allocInput(width, height, IN_CHANNELS, dummyInput);
    const encoder = device.createCommandEncoder();
    unet.forward(encoder, inputFm);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
  }

  const perDispatchTotals = new Map<string, number[]>();
  let totalMsSum = 0;

  for (let i = 0; i < timedRuns; i++) {
    log(`  size ${width}x${height}: timed run ${i + 1}/${timedRuns}…`);
    const inputFm = unet.allocInput(width, height, IN_CHANNELS, dummyInput);
    const encoder = device.createCommandEncoder();
    unet.forward(encoder, inputFm, { profile: true });
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    const profile = await unet.resolveProfile();
    let runTotal = 0;
    for (const { label, ms } of profile) {
      runTotal += ms;
      if (!perDispatchTotals.has(label)) perDispatchTotals.set(label, []);
      perDispatchTotals.get(label)!.push(ms);
    }
    totalMsSum += runTotal;
  }

  const avgTotal = totalMsSum / timedRuns;
  log(`\n${"dispatch".padEnd(20)} ${"avg ms".padStart(10)}`);
  for (const [label, values] of perDispatchTotals) {
    const avg = values.reduce((a, b) => a + b, 0) / values.length;
    log(`${label.padEnd(20)} ${avg.toFixed(4).padStart(10)}`);
  }
  log(`\ntotal (sum of dispatches): ${avgTotal.toFixed(3)}ms/frame, averaged over ${timedRuns} runs at ${width}x${height}`);
  log(`dispatch count: ${perDispatchTotals.size}`);
  return avgTotal;
}

async function runProfiling() {
  log("\n--- Step 3: GPU timestamp-query profiling at deployment resolution (Spec 4 step 7) ---");
  const { device, hasShaderF16, hasTimestampQuery } = await acquireGpu();
  if (!hasTimestampQuery) {
    log("timestamp-query not available on this adapter — skipping GPU profiling.");
    return;
  }

  const unet = new WgslUNet(device, hasShaderF16, hasTimestampQuery);
  await unet.loadWeights("/weights/manifest.json", "/weights/weights.bin");

  // Ramp up in size, with per-iteration progress logging -- an earlier
  // attempt jumping straight to full resolution appeared to hang with no
  // way to tell whether it was genuinely stuck or just slow (a naive,
  // untiled direct-conv kernel at 960x544 might legitimately take a while).
  // Small-to-large ramp + progress logs localise that if it recurs.
  log("small-size sanity check (128x128) before ramping up...");
  await profileAtResolution(unet, device, 128, 128, 2, 3);

  log("\nquarter-size check (240x136)...");
  await profileAtResolution(unet, device, 240, 136, 2, 3);

  const paddedWidth = padUp(REAL_WIDTH, 8);
  const paddedHeight = padUp(REAL_HEIGHT, 8);
  log(`\nfull deployment resolution ${REAL_WIDTH}x${REAL_HEIGHT} (padded to ${paddedWidth}x${paddedHeight})...`);
  const avgTotal = await profileAtResolution(unet, device, paddedWidth, paddedHeight, PROFILE_WARMUP_RUNS, PROFILE_TIMED_RUNS);

  const pssrReference = 2.0;
  log(`\nPSSR reference: ~${pssrReference}ms/frame. This implementation: ${avgTotal.toFixed(2)}ms/frame (${(avgTotal / pssrReference).toFixed(1)}x slower).`);
  log("Not expected to match PSSR (a shipped, heavily-optimised production kernel) -- reporting the honest number and gap, per CLAUDE.md.");
}

async function main() {
  if (!("gpu" in navigator)) {
    throw new Error("navigator.gpu is undefined — WebGPU not available in this browser/context.");
  }

  await runOrtWebCheck();
  await runWgslCheck();
  await runProfiling();
}

main().catch((err) => {
  console.error(err);
  log(`\nERROR: ${err instanceof Error ? (err.stack ?? err.message) : String(err)}`);
});

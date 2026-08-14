import * as ort from "onnxruntime-web/webgpu";
import { acquireGpu } from "./gpu.ts";
import { WgslUNet, scalarPrelude, toScalarBytes } from "./model_wgsl.ts";
import warpSource from "./wgsl/warp_and_disocclude.wgsl?raw";

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
  const { device, hasShaderF16, adapter, maxWorkgroupStorageSize } = await acquireGpu();
  const info = adapter.info;
  log(`adapter: ${info?.vendor ?? "?"} / ${info?.device ?? "?"} / ${info?.description ?? "?"}`);
  log(`shader-f16: ${hasShaderF16}`);
  log(`maxComputeWorkgroupStorageSize: ${maxWorkgroupStorageSize} bytes`);

  const unet = new WgslUNet(device, hasShaderF16, false, maxWorkgroupStorageSize);
  await unet.loadWeights("/weights/manifest.json", "/weights/weights.bin");

  const inputFm = unet.allocInput(PATCH_SIZE, PATCH_SIZE, IN_CHANNELS, inputData);

  const encoder = device.createCommandEncoder();
  const t0 = performance.now();
  const { output, intermediates, allBuffers } = unet.forward(encoder, inputFm);
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

  // Every intermediate has now been read back into a CPU-side Float32Array
  // (readFeatureMap above, per layer) -- safe to release every GPU buffer
  // this forward() pass allocated, plus the per-call input buffer.
  WgslUNet.releaseIntermediates({ allBuffers });
  inputFm.buffer.destroy();
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
    const unetResult = unet.forward(encoder, inputFm);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    // Throwaway pass -- nothing reads the output, release everything this
    // call allocated (including the per-run input buffer) immediately.
    WgslUNet.releaseIntermediates(unetResult);
    inputFm.buffer.destroy();
  }

  const perDispatchTotals = new Map<string, number[]>();
  let totalMsSum = 0;

  for (let i = 0; i < timedRuns; i++) {
    log(`  size ${width}x${height}: timed run ${i + 1}/${timedRuns}…`);
    const inputFm = unet.allocInput(width, height, IN_CHANNELS, dummyInput);
    const encoder = device.createCommandEncoder();
    const unetResult = unet.forward(encoder, inputFm, { profile: true });
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    WgslUNet.releaseIntermediates(unetResult);
    inputFm.buffer.destroy();

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
  const { device, hasShaderF16, hasTimestampQuery, maxWorkgroupStorageSize } = await acquireGpu();
  if (!hasTimestampQuery) {
    log("timestamp-query not available on this adapter — skipping GPU profiling.");
    return;
  }
  log(`maxComputeWorkgroupStorageSize: ${maxWorkgroupStorageSize} bytes`);

  const unet = new WgslUNet(device, hasShaderF16, hasTimestampQuery, maxWorkgroupStorageSize);
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

// Not part of the original Spec 4 sequence -- closes a gap flagged after the
// fact: warp_and_disocclude.wgsl (live_pipeline.ts's reimplementation of
// training/src/warp.py's warp_previous_output + compute_disocclusion_mask)
// was derived and eyeballed for plausibility but never diffed numerically
// against the PyTorch reference the way every other kernel here has been.
// Fixture generated by export/gen_diff_fixtures_warp.py: H=68,W=96
// deliberately unequal so an x/y transposition bug wouldn't pass by luck on
// a square grid, motion large enough that a real fraction of pixels fall
// off-screen (exercises "zeros" colour padding + the offscreen disocclusion
// branch), prev_depth partly perturbed with forced large jumps (exercises
// the depth-mismatch branch), not just the near-zero-motion interior case.
const WARP_H = 68;
const WARP_W = 96;

function makeUniformU32(device: GPUDevice, values: number[]): GPUBuffer {
  const buf = device.createBuffer({ size: values.length * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buf, 0, new Uint32Array(values));
  return buf;
}

async function runWarpCheck(): Promise<void> {
  log("--- Step 4: warp_and_disocclude.wgsl vs PyTorch (training/src/warp.py) -- closing the previously-flagged validation gap ---");
  const prevLowres = await fetchFloat32("/warp_prev_lowres.bin"); // (H,W,3)
  const motion = await fetchFloat32("/warp_motion.bin"); // (H,W,2)
  const currDepth = await fetchFloat32("/warp_curr_depth.bin"); // (H,W)
  const prevDepth = await fetchFloat32("/warp_prev_depth.bin"); // (H,W)
  const pytorchOutput = await fetchFloat32("/pytorch_output_warp.bin"); // (H,W,4) = [warped rgb, mask]

  const { device, hasShaderF16 } = await acquireGpu();
  const prelude = scalarPrelude(hasShaderF16);
  const bytesPerElement = hasShaderF16 ? 2 : 4;

  const layout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    compute: { module: device.createShaderModule({ code: prelude + warpSource }), entryPoint: "main" },
  });

  const prevLowresBuf = device.createBuffer({ size: WARP_H * WARP_W * 3 * bytesPerElement, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(prevLowresBuf, 0, toScalarBytes(prevLowres, hasShaderF16));

  const motionTex = device.createTexture({ size: [WARP_W, WARP_H], format: "rg16float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  device.queue.writeTexture({ texture: motionTex }, new Float16Array(motion).buffer, { bytesPerRow: WARP_W * 4 }, [WARP_W, WARP_H]);

  const currDepthTex = device.createTexture({ size: [WARP_W, WARP_H], format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  device.queue.writeTexture({ texture: currDepthTex }, currDepth.buffer, { bytesPerRow: WARP_W * 4 }, [WARP_W, WARP_H]);

  const prevDepthTex = device.createTexture({ size: [WARP_W, WARP_H], format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  device.queue.writeTexture({ texture: prevDepthTex }, prevDepth.buffer, { bytesPerRow: WARP_W * 4 }, [WARP_W, WARP_H]);

  const outBuf = device.createBuffer({ size: WARP_H * WARP_W * 4 * bytesPerElement, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const uniform = makeUniformU32(device, [WARP_W, WARP_H, 0]); // is_first_frame=0 -- exercise the real math, not the cold-start branch

  const bindGroup = device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: { buffer: uniform } },
      { binding: 1, resource: { buffer: prevLowresBuf } },
      { binding: 2, resource: motionTex.createView() },
      { binding: 3, resource: currDepthTex.createView() },
      { binding: 4, resource: prevDepthTex.createView() },
      { binding: 5, resource: { buffer: outBuf } },
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(WARP_W / 8), Math.ceil(WARP_H / 8), 1);
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();

  const readback = device.createBuffer({ size: outBuf.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const readEncoder = device.createCommandEncoder();
  readEncoder.copyBufferToBuffer(outBuf, 0, readback, 0, outBuf.size);
  device.queue.submit([readEncoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const wgslOutput = hasShaderF16
    ? Float32Array.from(new Float16Array(readback.getMappedRange().slice(0)))
    : new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();

  // The mask channel (index 3 of every 4) is a hard {0,1} threshold
  // decision, not a continuous value -- a near-miss there (a bilinear depth
  // sample landing a hair either side of DEPTH_REL_THRESHOLD) is a
  // different failure mode than FP16 rounding noise in the colour channels,
  // so it's compared separately rather than folded into the same max/mean
  // as the colour channels. Previously this used the shared compare()
  // helper over the *whole* flat array (colour + mask together), which made
  // `max` always ~1.0 whenever even one mask pixel landed on a boundary tie
  // (an expected, documented outcome -- see MASK_MISMATCH_TOLERANCE below),
  // permanently failing the max<FP16_TOLERANCE gate regardless of how
  // accurate the colour channels actually were. Computing colour-channel
  // max/mean directly here (excluding index 3 of every 4) fixes that: the
  // gate now reflects what it's meant to -- colour accuracy and mask
  // agreement, judged by their own, separately-justified tolerances.
  //
  // MASK_MISMATCH_TOLERANCE: independently re-deriving the mismatched
  // pixels on this fixture (H=68,W=96, seed=7) found all of them sit inside
  // a rel_diff band within 0.002 of DEPTH_REL_THRESHOLD=0.05 -- i.e. cases
  // where PyTorch's grid_sample and WGSL's hand-written bilinear agree to
  // ~1e-4 (same order as the colour channels' FP16 rounding) but that tiny
  // disagreement happens to land on opposite sides of a hard threshold.
  // That's an inherent property of comparing two independently-rounded
  // floating point pipelines against a boolean cutoff, not a UV-convention
  // or logic bug -- so a small fraction of boundary flips is accepted
  // rather than gated at zero. 0.5% is a deliberately generous margin
  // against the observed 0.12% (8/6528) on this fixture; a real bug (wrong
  // sign, wrong padding mode, swapped axes) would produce gross,
  // widespread disagreement, not an isolated few-pixel boundary effect.
  const MASK_MISMATCH_TOLERANCE = 0.005;
  let colourMax = 0;
  let colourSum = 0;
  let maskMismatches = 0;
  const maskTotal = wgslOutput.length / 4;
  for (let i = 0; i < wgslOutput.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      const err = Math.abs(wgslOutput[i + c] - pytorchOutput[i + c]);
      if (err > colourMax) colourMax = err;
      colourSum += err;
    }
    if (Math.abs(wgslOutput[i + 3] - pytorchOutput[i + 3]) > 0.5) maskMismatches++;
  }
  const colourMean = colourSum / (maskTotal * 3);
  const maskMismatchFraction = maskMismatches / maskTotal;

  log(`warp+disocclusion vs PyTorch (colour channels only): max abs error=${colourMax.toFixed(6)} mean=${colourMean.toFixed(6)}`);
  log(`disocclusion mask mismatches: ${maskMismatches}/${maskTotal} pixels (${(maskMismatchFraction * 100).toFixed(3)}%, tolerance ${(MASK_MISMATCH_TOLERANCE * 100).toFixed(1)}%) -- expected to be boundary threshold ties, see comment above`);
  log(`${colourMax < FP16_TOLERANCE && maskMismatchFraction <= MASK_MISMATCH_TOLERANCE ? "PASSED" : "FAILED"}: warp kernel matches PyTorch within FP16 tolerance; mask decisions agree within the documented threshold-boundary tolerance\n`);
}

async function main() {
  if (!("gpu" in navigator)) {
    throw new Error("navigator.gpu is undefined — WebGPU not available in this browser/context.");
  }

  await runOrtWebCheck();
  await runWgslCheck();
  await runProfiling();
  await runWarpCheck();
}

main().catch((err) => {
  console.error(err);
  log(`\nERROR: ${err instanceof Error ? (err.stack ?? err.message) : String(err)}`);
});

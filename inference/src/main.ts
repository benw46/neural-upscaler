import * as ort from "onnxruntime-web/webgpu";

const statusEl = document.querySelector<HTMLDivElement>("#status")!;
function log(line: string) {
  statusEl.textContent += (statusEl.textContent === "initialising…" ? "" : "\n") + line;
  if (statusEl.textContent === "initialising…") statusEl.textContent = line;
  console.log(line);
}

ort.env.wasm.wasmPaths = "/ort/";

const PATCH_SIZE = 128;

async function fetchFloat32(url: string): Promise<Float32Array> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url} failed: ${res.status}`);
  const buf = await res.arrayBuffer();
  return new Float32Array(buf);
}

async function main() {
  if (!("gpu" in navigator)) {
    throw new Error("navigator.gpu is undefined — WebGPU not available in this browser/context.");
  }

  log("loading fixed test input and PyTorch reference output…");
  const inputData = await fetchFloat32("/test_input.bin");
  const pytorchOutput = await fetchFloat32("/pytorch_output.bin");
  log(`input: ${inputData.length} floats, pytorch output: ${pytorchOutput.length} floats`);

  log("creating InferenceSession (trained model) with WebGPU execution provider…");
  const session = await ort.InferenceSession.create("/models/spatial_unet_trained.onnx", {
    executionProviders: ["webgpu"],
    graphOptimizationLevel: "all",
  });
  log(`session created. inputNames=${session.inputNames} outputNames=${session.outputNames}`);

  const inputTensor = new ort.Tensor("float32", inputData, [1, 4, PATCH_SIZE, PATCH_SIZE]);

  log("running inference…");
  const t0 = performance.now();
  const results = await session.run({ [session.inputNames[0]]: inputTensor });
  const t1 = performance.now();

  const output = results[session.outputNames[0]];
  const ortData = output.data as Float32Array;
  log(`inference OK in ${(t1 - t0).toFixed(2)}ms`);
  log(`output shape: [${output.dims.join(", ")}]`);

  if (ortData.length !== pytorchOutput.length) {
    throw new Error(`length mismatch: ort=${ortData.length} pytorch=${pytorchOutput.length}`);
  }

  let maxAbsErr = 0;
  let sumAbsErr = 0;
  for (let i = 0; i < ortData.length; i++) {
    const err = Math.abs(ortData[i] - pytorchOutput[i]);
    if (err > maxAbsErr) maxAbsErr = err;
    sumAbsErr += err;
  }
  const meanAbsErr = sumAbsErr / ortData.length;

  log(`\n--- ONNX Runtime Web (WebGPU) vs PyTorch, same fixed input ---`);
  log(`max abs error:  ${maxAbsErr.toFixed(6)}`);
  log(`mean abs error: ${meanAbsErr.toFixed(6)}`);
  // FP16 has ~3 decimal digits of precision; a well-behaved export/runtime
  // round-trip through WebGPU (which may use fp16 internally for some ops)
  // should stay within roughly this tolerance.
  const FP16_TOLERANCE = 0.01;
  const withinTolerance = maxAbsErr < FP16_TOLERANCE;
  log(`\n${withinTolerance ? "PASSED" : "FAILED"}: max abs error ${withinTolerance ? "<" : ">="} ${FP16_TOLERANCE} (FP16 tolerance)`);
}

main().catch((err) => {
  console.error(err);
  log(`\nERROR: ${err instanceof Error ? (err.stack ?? err.message) : String(err)}`);
});

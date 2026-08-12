/** Raw WebGPU device acquisition for the hand-written WGSL inference path
 * (separate from onnxruntime-web's own internal device handling, used only
 * for the ORT Web correctness-oracle checks elsewhere in this project).
 * Same feature-detection discipline as renderer/src/gpu/device.ts: never
 * assume shader-f16, never silently accept a software adapter. */

export interface GpuContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  hasShaderF16: boolean;
  hasTimestampQuery: boolean;
}

export async function acquireGpu(): Promise<GpuContext> {
  if (!("gpu" in navigator)) {
    throw new Error("navigator.gpu is undefined — WebGPU not available.");
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) {
    throw new Error("requestAdapter() returned null.");
  }

  const info = adapter.info;
  const adapterName = `${info?.vendor ?? "?"} / ${info?.architecture ?? "?"} / ${info?.device ?? "?"} / ${info?.description ?? "?"}`;
  const nameForCheck = adapterName.toLowerCase();
  if (nameForCheck.includes("swiftshader") || nameForCheck.includes("warp") || nameForCheck.includes("llvmpipe") || nameForCheck.includes("software")) {
    throw new Error(`Adapter appears to be software (${adapterName}) — refusing to proceed. Check chrome://gpu.`);
  }

  const hasShaderF16 = adapter.features.has("shader-f16");
  const hasTimestampQuery = adapter.features.has("timestamp-query");
  const requiredFeatures: GPUFeatureName[] = [];
  if (hasShaderF16) requiredFeatures.push("shader-f16");
  if (hasTimestampQuery) requiredFeatures.push("timestamp-query");

  const device = await adapter.requestDevice({ requiredFeatures });
  device.lost.then((info) => {
    console.error(`[gpu] device lost: ${info.reason} — ${info.message}`);
  });
  device.addEventListener("uncapturederror", (event) => {
    console.error(`[gpu] uncaptured error: ${(event as GPUUncapturedErrorEvent).error.message}`);
  });

  console.log(`[gpu] adapter: ${adapterName}`);
  console.log(`[gpu] shader-f16: ${hasShaderF16}`);
  console.log(`[gpu] timestamp-query: ${hasTimestampQuery}`);

  return { adapter, device, hasShaderF16, hasTimestampQuery };
}

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
  /** Actually-granted `device.limits.maxComputeWorkgroupStorageSize`, not the
   * adapter's reported maximum -- WebGPU grants only the spec-minimum 16384
   * bytes unless a device explicitly requests more via `requiredLimits`,
   * confirmed on this exact adapter (16384 default vs 32768 available) by
   * probing both paths directly. conv_tiled.wgsl's Cin-chunk sizing reads
   * this rather than assuming the spec minimum. */
  maxWorkgroupStorageSize: number;
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

  // Request the adapter's own reported maximum explicitly -- requestDevice()
  // grants only the spec-minimum 16384 bytes of workgroup storage by
  // default, even when the adapter supports more (confirmed on this exact
  // adapter: 16384 default vs 32768 available). Asking for adapter.limits'
  // value is safe by construction -- it can never exceed what the adapter
  // itself just reported.
  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: { maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize },
  });
  device.lost.then((info) => {
    console.error(`[gpu] device lost: ${info.reason} — ${info.message}`);
  });
  device.addEventListener("uncapturederror", (event) => {
    console.error(`[gpu] uncaptured error: ${(event as GPUUncapturedErrorEvent).error.message}`);
  });

  const maxWorkgroupStorageSize = device.limits.maxComputeWorkgroupStorageSize;

  console.log(`[gpu] adapter: ${adapterName}`);
  console.log(`[gpu] shader-f16: ${hasShaderF16}`);
  console.log(`[gpu] timestamp-query: ${hasTimestampQuery}`);
  console.log(`[gpu] maxComputeWorkgroupStorageSize: ${maxWorkgroupStorageSize} bytes (adapter max: ${adapter.limits.maxComputeWorkgroupStorageSize})`);

  return { adapter, device, hasShaderF16, hasTimestampQuery, maxWorkgroupStorageSize };
}

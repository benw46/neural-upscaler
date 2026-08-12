/**
 * WebGPU adapter/device acquisition with feature detection.
 * Fails loudly rather than silently degrading — see CLAUDE.md hard rule 3.
 */

export interface GpuContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  hasShaderF16: boolean;
}

export class WebGPUUnavailableError extends Error {}
export class SwiftShaderAdapterError extends Error {}

/**
 * `gpu` defaults to `navigator.gpu` for browser use. Node capture mode
 * (Dawn bindings via the `webgpu` package) has no `navigator` global and
 * instead gets a `GPU` instance directly from `create()` — pass it
 * explicitly there so this acquisition/validation logic isn't duplicated.
 */
export async function acquireGpu(gpu?: GPU): Promise<GpuContext> {
  const resolvedGpu = gpu ?? (typeof navigator !== "undefined" ? navigator.gpu : undefined);
  if (!resolvedGpu) {
    throw new WebGPUUnavailableError(
      "No GPU instance available — navigator.gpu is undefined and none was passed explicitly."
    );
  }

  const adapter = await resolvedGpu.requestAdapter({
    powerPreference: "high-performance",
  });

  if (!adapter) {
    throw new WebGPUUnavailableError(
      "requestAdapter() returned null — no compatible WebGPU adapter found."
    );
  }

  const info = adapter.info;
  const adapterName = `${info?.vendor ?? "unknown-vendor"} / ${info?.architecture ?? "unknown-arch"} / ${info?.device ?? "unknown-device"} / ${info?.description ?? "no-description"}`;

  // Software fallback adapters (SwiftShader, WARP, llvmpipe) silently tank
  // performance numbers and invalidate any profiling done against them.
  const nameForCheck = adapterName.toLowerCase();
  const isSoftware =
    nameForCheck.includes("swiftshader") ||
    nameForCheck.includes("warp") ||
    nameForCheck.includes("llvmpipe") ||
    nameForCheck.includes("software");

  if (isSoftware) {
    throw new SwiftShaderAdapterError(
      `Adapter appears to be a software rasteriser (${adapterName}). ` +
        `Refusing to proceed — check chrome://gpu and confirm the D3D12 ` +
        `backend with the real GPU is active (see CLAUDE.md fragile-logic list).`
    );
  }

  const hasShaderF16 = adapter.features.has("shader-f16");

  const requiredFeatures: GPUFeatureName[] = [];
  if (hasShaderF16) requiredFeatures.push("shader-f16");

  const device = await adapter.requestDevice({
    requiredFeatures,
  });

  device.lost.then((info) => {
    console.error(`[gpu] device lost: ${info.reason} — ${info.message}`);
  });

  console.log(`[gpu] adapter: ${adapterName}`);
  console.log(`[gpu] shader-f16: ${hasShaderF16}`);

  return { adapter, device, hasShaderF16 };
}

import convSource from "./wgsl/conv.wgsl?raw";
import convTiledSource from "./wgsl/conv_tiled.wgsl?raw";
import upsampleSource from "./wgsl/upsample.wgsl?raw";
import concatSource from "./wgsl/concat.wgsl?raw";
import pixelShuffleSource from "./wgsl/pixel_shuffle.wgsl?raw";

// The only three layers routed through conv_tiled.wgsl instead of conv.wgsl
// -- the largest-Cin layers, and the only ones (Cin,Cout) specialisation
// alone barely sped up (see docs/PHASE-4-SUMMARY.md's follow-up profile).
// Named explicitly rather than picked by a Cin threshold: several other
// layers share Cin=84 (down3.down, down2.refine, up1.conv2) without being
// anywhere near a bottleneck, since they're much cheaper on resolution --
// an explicit list is auditable and can't accidentally net the wrong layers
// if the architecture changes. conv_tiled.wgsl assumes stride=1, which
// happens to hold for exactly these three (all "same"-padding refine convs
// right after an upsample+concat) -- asserted in conv() below rather than
// assumed silently.
const TILED_CONV_LAYERS = new Set(["up1.conv1.conv", "up2.conv1.conv", "up3.conv1.conv"]);

// Empirical sweep result, not a theoretical pick -- see
// docs/OPTIMISATIONS.md's "further WebGPU exploration" section. Measured
// 32 vs 64 vs ~140 (the full budget this adapter actually grants once
// gpu.ts requests it) at the real deployment resolution: 104.2ms / 106.9ms
// / 107.1ms respectively. 32 -- the original, arbitrarily-chosen-under-a-
// wrong-assumed-16KB-budget value -- was already at or near the practical
// optimum; more headroom made things flat-to-worse, not better. Kept as an
// explicit cap (not derived from the budget) specifically *because* the
// larger budget this device actually grants turned out not to help.
const TILED_CHUNK_CAP = 32;

export interface LayerManifest {
  name: string;
  inChannels: number;
  outChannels: number;
  stride: number;
  activation: "leaky_relu" | "none";
  weightOffset: number;
  weightBytes: number;
  biasOffset: number;
  biasBytes: number;
}

export interface Manifest {
  layers: LayerManifest[];
}

interface FeatureMap {
  buffer: GPUBuffer;
  width: number;
  height: number;
  channels: number;
}

/** Exported for live.ts, which compiles its own small set of WGSL kernels
 * (pack_input/downsample_prev/warp_and_disocclude) against the same
 * Scalar-precision convention as every kernel in this file. */
export function scalarPrelude(hasF16: boolean): string {
  return hasF16 ? "enable f16;\nalias Scalar = f16;\n\n" : "alias Scalar = f32;\n\n";
}

/** Converts an f32 array to the storage representation matching `hasF16` --
 * either a tightly-packed Float16Array or a Float32Array. */
export function toScalarBytes(data: Float32Array, hasF16: boolean): ArrayBuffer {
  if (hasF16) {
    return new Float16Array(data).buffer;
  }
  return data.slice().buffer; // copy into a fresh (non-shared) ArrayBuffer
}

const MAX_PROFILED_DISPATCHES = 32; // this model has 23 (16 conv + 3 upsample + 3 concat + 1 pixel-shuffle)

export class WgslUNet {
  private device: GPUDevice;
  private hasF16: boolean;
  private bytesPerElement: number;
  private hasTimestampQuery: boolean;
  private maxWorkgroupStorageSize: number;

  // Keyed by "${inChannels}x${outChannels}": conv.wgsl's IN_CHANNELS and
  // OUT_CHANNELS are compile-time constants now (see that file's header
  // comment), so each distinct (Cin, Cout) pair in the model (12 of them)
  // gets its own specialised, fully-unrollable shader-module variant instead
  // of one pipeline shared -- and under-specialised -- across all of them.
  // Populated eagerly in loadWeights() once the manifest's actual layer list
  // is known; getConvPipeline() also lazily fills any gap.
  private convPipelines = new Map<string, GPUComputePipeline>();
  private convModuleTemplate!: string; // scalar prelude + conv.wgsl source, {IN,OUT}_CHANNELS_VALUE not yet substituted
  private convLayout!: GPUBindGroupLayout;
  private convPipelineLayout!: GPUPipelineLayout;
  // Same keying/caching scheme as convPipelines, separate map+template since
  // it's a different shader (conv_tiled.wgsl) -- shares convLayout/
  // convPipelineLayout, since the bindings are identical, only the kernel
  // body differs. Only ever holds the 3 TILED_CONV_LAYERS entries.
  private convTiledPipelines = new Map<string, GPUComputePipeline>();
  private convTiledModuleTemplate!: string;
  private upsamplePipeline!: GPUComputePipeline;
  private upsampleLayout!: GPUBindGroupLayout;
  // Keyed by "${channelsA}x${channelsB}", same reasoning as convPipelines --
  // concat.wgsl's copy loops are equally unrollable once channel counts are
  // compile-time. Only 3 distinct pairs ever occur (one per up-block), and
  // unlike conv layers they aren't listed in the weights manifest, so these
  // are compiled lazily on first use rather than precomputed in
  // loadWeights() -- the profiling harness's warmup runs already absorb
  // first-call shader-compile latency before any timed run.
  private concatPipelines = new Map<string, GPUComputePipeline>();
  private concatModuleTemplate!: string;
  private concatLayout!: GPUBindGroupLayout;
  private concatPipelineLayout!: GPUPipelineLayout;
  private pixelShufflePipeline!: GPUComputePipeline;
  private pixelShuffleLayout!: GPUBindGroupLayout;

  private layerBuffers = new Map<string, { weight: GPUBuffer; bias: GPUBuffer }>();
  manifest!: Manifest;

  // Timestamp-query profiling state -- active only between forward(..., {profile:true}) and resolveProfile().
  private querySet?: GPUQuerySet;
  private queryResolveBuffer?: GPUBuffer;
  private queryReadBuffer?: GPUBuffer;
  private profileLabels: string[] = [];
  private profilingActive = false;
  private dispatchIndex = 0;

  /** Per-dispatch timing (ms), populated after a `forward()` call that used
   * `profile: true`. Keyed by a label unique per dispatch in execution order. */
  lastProfile: { label: string; ms: number }[] = [];

  constructor(device: GPUDevice, hasF16: boolean, hasTimestampQuery = false, maxWorkgroupStorageSize = 16384) {
    this.device = device;
    this.hasF16 = hasF16;
    this.bytesPerElement = hasF16 ? 2 : 4;
    this.hasTimestampQuery = hasTimestampQuery;
    // Default 16384 is the WebGPU spec minimum, used only if a caller
    // doesn't pass the real granted value -- always prefer the actual
    // device.limits.maxComputeWorkgroupStorageSize from acquireGpu() (see
    // gpu.ts: the default grant is 16384 even on hardware that supports
    // more, unless explicitly requested).
    this.maxWorkgroupStorageSize = maxWorkgroupStorageSize;
    this.buildPipelines();
    if (hasTimestampQuery) {
      this.querySet = device.createQuerySet({ type: "timestamp", count: MAX_PROFILED_DISPATCHES * 2 });
      this.queryResolveBuffer = device.createBuffer({
        size: MAX_PROFILED_DISPATCHES * 2 * 8, // 8 bytes per u64 timestamp
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      this.queryReadBuffer = device.createBuffer({
        size: MAX_PROFILED_DISPATCHES * 2 * 8,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    }
  }

  /** Returns {beginningOfPassWriteIndex, endOfPassWriteIndex} for the next
   * dispatch if profiling is active, else undefined -- spread directly into
   * a beginComputePass descriptor's `timestampWrites`. */
  private nextTimestampWrites(label: string): GPUComputePassTimestampWrites | undefined {
    if (!this.profilingActive || !this.querySet) return undefined;
    const idx = this.dispatchIndex++;
    this.profileLabels.push(label);
    return { querySet: this.querySet, beginningOfPassWriteIndex: idx * 2, endOfPassWriteIndex: idx * 2 + 1 };
  }

  /** Resolves and reads back GPU timestamp queries recorded during the last
   * `forward(..., {profile: true})` call. Must be called after the
   * submitted work has completed (await device.queue.onSubmittedWorkDone()). */
  async resolveProfile(): Promise<{ label: string; ms: number }[]> {
    if (!this.querySet || !this.queryResolveBuffer || !this.queryReadBuffer) {
      throw new Error("timestamp-query not available on this adapter");
    }
    const n = this.dispatchIndex;
    const encoder = this.device.createCommandEncoder();
    encoder.resolveQuerySet(this.querySet, 0, n * 2, this.queryResolveBuffer, 0);
    encoder.copyBufferToBuffer(this.queryResolveBuffer, 0, this.queryReadBuffer, 0, n * 2 * 8);
    this.device.queue.submit([encoder.finish()]);

    await this.queryReadBuffer.mapAsync(GPUMapMode.READ, 0, n * 2 * 8);
    const raw = new BigUint64Array(this.queryReadBuffer.getMappedRange(0, n * 2 * 8).slice(0));
    this.queryReadBuffer.unmap();

    const results: { label: string; ms: number }[] = [];
    for (let i = 0; i < n; i++) {
      const start = raw[i * 2];
      const end = raw[i * 2 + 1];
      const ns = Number(end - start);
      results.push({ label: this.profileLabels[i], ms: ns / 1e6 });
    }
    this.lastProfile = results;
    this.profilingActive = false;
    return results;
  }

  private buildPipelines() {
    const prelude = scalarPrelude(this.hasF16);
    const device = this.device;

    // --- conv ---
    this.convLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    this.convPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.convLayout] });
    this.convModuleTemplate = prelude + convSource; // OUT_CHANNELS_VALUE substituted per-variant in getConvPipeline
    this.convTiledModuleTemplate = prelude + convTiledSource; // {IN,OUT}_CHANNELS_VALUE substituted per-variant in getConvTiledPipeline

    // --- upsample ---
    this.upsampleLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    this.upsamplePipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.upsampleLayout] }),
      compute: { module: device.createShaderModule({ code: prelude + upsampleSource }), entryPoint: "main" },
    });

    // --- concat ---
    this.concatLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    this.concatPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.concatLayout] });
    this.concatModuleTemplate = prelude + concatSource; // {CHANNELS_A,CHANNELS_B}_VALUE substituted per-variant in getConcatPipeline

    // --- pixel shuffle ---
    this.pixelShuffleLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    this.pixelShufflePipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.pixelShuffleLayout] }),
      compute: { module: device.createShaderModule({ code: prelude + pixelShuffleSource }), entryPoint: "main" },
    });
  }

  async loadWeights(manifestUrl: string, weightsUrl: string) {
    // Safe to call more than once on the same instance (e.g. switching
    // models from the viewer's dropdown) -- destroy any previously loaded
    // weight/bias buffers first, after waiting for any in-flight GPU work
    // that might still reference them to finish. Skipping this would leak
    // a full set of per-layer buffers on every switch, the same class of
    // bug as the original forward()-intermediates leak (see
    // WgslUNet.releaseIntermediates()).
    if (this.layerBuffers.size > 0) {
      await this.device.queue.onSubmittedWorkDone();
      for (const { weight, bias } of this.layerBuffers.values()) {
        weight.destroy();
        bias.destroy();
      }
      this.layerBuffers.clear();
    }

    const manifestRes = await fetch(manifestUrl);
    this.manifest = await manifestRes.json();
    const weightsRes = await fetch(weightsUrl);
    const weightsBlob = await weightsRes.arrayBuffer();

    for (const layer of this.manifest.layers) {
      const weightF32 = new Float32Array(weightsBlob, layer.weightOffset, layer.weightBytes / 4);
      const biasF32 = new Float32Array(weightsBlob, layer.biasOffset, layer.biasBytes / 4);

      const weightBytes = toScalarBytes(weightF32, this.hasF16);
      const biasBytes = toScalarBytes(biasF32, this.hasF16);

      const weightBuf = this.device.createBuffer({
        label: `${layer.name}-weight`,
        size: weightBytes.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(weightBuf, 0, weightBytes);

      const biasBuf = this.device.createBuffer({
        label: `${layer.name}-bias`,
        size: biasBytes.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.device.queue.writeBuffer(biasBuf, 0, biasBytes);

      this.layerBuffers.set(layer.name, { weight: weightBuf, bias: biasBuf });
    }

    // Precompile every distinct (in_channels, out_channels) variant up front
    // (12 of them) rather than paying shader-compile latency lazily on
    // whichever forward() call happens to hit a given pair first. Also
    // precompile the 3 tiled variants, keyed the same way but in a separate
    // cache/pipeline family.
    for (const layer of this.manifest.layers) {
      this.getConvPipeline(layer.inChannels, layer.outChannels);
      if (TILED_CONV_LAYERS.has(layer.name)) {
        if (layer.stride !== 1) {
          throw new Error(`${layer.name}: conv_tiled.wgsl assumes stride=1, got stride=${layer.stride}`);
        }
        this.getConvTiledPipeline(layer.inChannels, layer.outChannels);
      }
    }
  }

  /** Returns the conv pipeline specialised for this exact (inChannels,
   * outChannels) pair (see conv.wgsl's IN_CHANNELS/OUT_CHANNELS), compiling
   * and caching it on first request. */
  private getConvPipeline(inChannels: number, outChannels: number): GPUComputePipeline {
    const key = `${inChannels}x${outChannels}`;
    let pipeline = this.convPipelines.get(key);
    if (pipeline) return pipeline;

    const code = this.convModuleTemplate.replace("IN_CHANNELS_VALUE", String(inChannels)).replace("OUT_CHANNELS_VALUE", String(outChannels));
    pipeline = this.device.createComputePipeline({
      layout: this.convPipelineLayout,
      compute: { module: this.device.createShaderModule({ code }), entryPoint: "main" },
    });
    this.convPipelines.set(key, pipeline);
    return pipeline;
  }

  /** Returns the tiled conv pipeline (conv_tiled.wgsl) specialised for this
   * exact (inChannels, outChannels) pair, compiling and caching it on first
   * request. Only ever called for TILED_CONV_LAYERS. */
  private getConvTiledPipeline(inChannels: number, outChannels: number): GPUComputePipeline {
    const key = `${inChannels}x${outChannels}`;
    let pipeline = this.convTiledPipelines.get(key);
    if (pipeline) return pipeline;

    // TILE_PADDED=10 must match conv_tiled.wgsl's TILE(8)+2 exactly -- two
    // files, one geometry, same fragile-duplication pattern as everywhere
    // else this project names a cross-file constant explicitly rather than
    // letting it drift silently.
    const TILE_PADDED = 10;
    const bytesPerElement = this.hasF16 ? 2 : 4; // tile is now `Scalar`, not always f32 -- see conv_tiled.wgsl
    const tileFootprintPerChannel = TILE_PADDED * TILE_PADDED * bytesPerElement;
    // 90% of the granted limit, not 100% -- headroom for whatever else the
    // driver/runtime accounts against the same budget; the exact number
    // hasn't been pinned down, kept conservative rather than assumed safe.
    const usableBudget = Math.floor(this.maxWorkgroupStorageSize * 0.9);
    const maxChunk = Math.max(1, Math.floor(usableBudget / tileFootprintPerChannel));
    // TILED_CHUNK_CAP: empirical tuning sweep, see docs/OPTIMISATIONS.md --
    // maxing out the budget (chunkSize up to ~140) measured *worse* than the
    // original CHUNK_SIZE=32, pointing at per-workgroup shared-memory
    // footprint limiting occupancy, not barrier count, as the real lever.
    const chunkSize = Math.min(inChannels, maxChunk, TILED_CHUNK_CAP);

    const code = this.convTiledModuleTemplate
      .replace("IN_CHANNELS_VALUE", String(inChannels))
      .replace("OUT_CHANNELS_VALUE", String(outChannels))
      .replace("CHUNK_SIZE_VALUE", String(chunkSize));
    pipeline = this.device.createComputePipeline({
      layout: this.convPipelineLayout, // shared with the plain conv kernel -- identical bindings
      compute: { module: this.device.createShaderModule({ code }), entryPoint: "main" },
    });
    this.convTiledPipelines.set(key, pipeline);
    return pipeline;
  }

  /** Returns the concat pipeline specialised for this exact (channelsA,
   * channelsB) pair (see concat.wgsl's CHANNELS_A/CHANNELS_B), compiling and
   * caching it on first request. */
  private getConcatPipeline(channelsA: number, channelsB: number): GPUComputePipeline {
    const key = `${channelsA}x${channelsB}`;
    let pipeline = this.concatPipelines.get(key);
    if (pipeline) return pipeline;

    const code = this.concatModuleTemplate.replace("CHANNELS_A_VALUE", String(channelsA)).replace("CHANNELS_B_VALUE", String(channelsB));
    pipeline = this.device.createComputePipeline({
      layout: this.concatPipelineLayout,
      compute: { module: this.device.createShaderModule({ code }), entryPoint: "main" },
    });
    this.concatPipelines.set(key, pipeline);
    return pipeline;
  }

  private allocFeatureMap(width: number, height: number, channels: number): GPUBuffer {
    return this.device.createBuffer({
      size: width * height * channels * this.bytesPerElement,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
  }

  private makeUniform(values: number[]): GPUBuffer {
    const buf = this.device.createBuffer({
      size: values.length * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buf, 0, new Uint32Array(values));
    return buf;
  }

  private conv(encoder: GPUCommandEncoder, input: FeatureMap, layerName: string): FeatureMap {
    const layer = this.manifest.layers.find((l) => l.name === layerName);
    if (!layer) throw new Error(`unknown layer ${layerName}`);
    const buffers = this.layerBuffers.get(layerName)!;

    const outWidth = Math.floor((input.width - 1) / layer.stride) + 1;
    const outHeight = Math.floor((input.height - 1) / layer.stride) + 1;
    const output = this.allocFeatureMap(outWidth, outHeight, layer.outChannels);

    const uniformBuf = this.makeUniform([
      input.width,
      input.height,
      layer.inChannels,
      layer.outChannels,
      layer.stride,
      layer.activation === "leaky_relu" ? 1 : 0,
    ]);

    const bindGroup = this.device.createBindGroup({
      layout: this.convLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuf } },
        { binding: 1, resource: { buffer: input.buffer } },
        { binding: 2, resource: { buffer: buffers.weight } },
        { binding: 3, resource: { buffer: buffers.bias } },
        { binding: 4, resource: { buffer: output } },
      ],
    });

    const label = `conv:${layerName}`;
    const pass = encoder.beginComputePass({ label, timestampWrites: this.nextTimestampWrites(label) });
    const pipeline = TILED_CONV_LAYERS.has(layerName) ? this.getConvTiledPipeline(layer.inChannels, layer.outChannels) : this.getConvPipeline(layer.inChannels, layer.outChannels);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    // z=1, not layer.outChannels -- each thread computes every output
    // channel for its (x, y) pixel internally (see conv.wgsl /
    // conv_tiled.wgsl). Both kernels use the same 8x8-output-pixels-per-
    // workgroup convention, so the dispatch math is identical either way.
    pass.dispatchWorkgroups(Math.ceil(outWidth / 8), Math.ceil(outHeight / 8), 1);
    pass.end();

    return { buffer: output, width: outWidth, height: outHeight, channels: layer.outChannels };
  }

  private upsample(encoder: GPUCommandEncoder, input: FeatureMap, label: string): FeatureMap {
    const outWidth = input.width * 2;
    const outHeight = input.height * 2;
    const output = this.allocFeatureMap(outWidth, outHeight, input.channels);
    const uniformBuf = this.makeUniform([input.width, input.height, input.channels]);

    const bindGroup = this.device.createBindGroup({
      layout: this.upsampleLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuf } },
        { binding: 1, resource: { buffer: input.buffer } },
        { binding: 2, resource: { buffer: output } },
      ],
    });

    const pass = encoder.beginComputePass({ label, timestampWrites: this.nextTimestampWrites(label) });
    pass.setPipeline(this.upsamplePipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(outWidth / 8), Math.ceil(outHeight / 8), input.channels);
    pass.end();

    return { buffer: output, width: outWidth, height: outHeight, channels: input.channels };
  }

  private concat(encoder: GPUCommandEncoder, a: FeatureMap, b: FeatureMap, label: string): FeatureMap {
    if (a.width !== b.width || a.height !== b.height) {
      throw new Error(`concat spatial mismatch: a=${a.width}x${a.height} b=${b.width}x${b.height}`);
    }
    const outChannels = a.channels + b.channels;
    const output = this.allocFeatureMap(a.width, a.height, outChannels);
    const uniformBuf = this.makeUniform([a.width, a.height, a.channels, b.channels]);

    const bindGroup = this.device.createBindGroup({
      layout: this.concatLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuf } },
        { binding: 1, resource: { buffer: a.buffer } },
        { binding: 2, resource: { buffer: b.buffer } },
        { binding: 3, resource: { buffer: output } },
      ],
    });

    const pass = encoder.beginComputePass({ label, timestampWrites: this.nextTimestampWrites(label) });
    pass.setPipeline(this.getConcatPipeline(a.channels, b.channels));
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(a.width / 8), Math.ceil(a.height / 8), 1);
    pass.end();

    return { buffer: output, width: a.width, height: a.height, channels: outChannels };
  }

  private pixelShuffle(encoder: GPUCommandEncoder, input: FeatureMap): FeatureMap {
    const outChannels = input.channels / 4;
    const outWidth = input.width * 2;
    const outHeight = input.height * 2;
    const output = this.allocFeatureMap(outWidth, outHeight, outChannels);
    const uniformBuf = this.makeUniform([input.width, input.height, outChannels]);

    const bindGroup = this.device.createBindGroup({
      layout: this.pixelShuffleLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuf } },
        { binding: 1, resource: { buffer: input.buffer } },
        { binding: 2, resource: { buffer: output } },
      ],
    });

    const pass = encoder.beginComputePass({ label: "pixel_shuffle", timestampWrites: this.nextTimestampWrites("pixel_shuffle") });
    pass.setPipeline(this.pixelShufflePipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(outWidth / 8), Math.ceil(outHeight / 8), outChannels);
    pass.end();

    return { buffer: output, width: outWidth, height: outHeight, channels: outChannels };
  }

  /** Runs the full forward pass, matching training/src/model.py's
   * SpatialUNet.forward exactly. Returns the final feature map plus every
   * named intermediate (for per-layer diffing against ORT Web/PyTorch).
   * Pass `profile: true` (requires timestamp-query support) to record
   * per-dispatch GPU timing -- call `resolveProfile()` after the work
   * completes to read it back. */
  forward(encoder: GPUCommandEncoder, input: FeatureMap, options?: { profile?: boolean }): { output: FeatureMap; intermediates: Map<string, FeatureMap>; allBuffers: GPUBuffer[] } {
    if (options?.profile) {
      if (!this.hasTimestampQuery) throw new Error("profile:true requested but timestamp-query is not available on this adapter");
      this.profilingActive = true;
      this.dispatchIndex = 0;
      this.profileLabels = [];
    }
    const intermediates = new Map<string, FeatureMap>();
    // Every buffer allocFeatureMap() creates during this call, named
    // intermediate or not (most aren't -- e.g. d1down/u1cat below are used
    // once as the next layer's input and then unreferenced). forward()
    // itself never destroys these: a still-unsubmitted command encoder may
    // reference them when it returns. Callers release whichever of these
    // they don't need via releaseIntermediates(), after their own submitted
    // commands have completed.
    const allBuffers: GPUBuffer[] = [];
    const track = (fm: FeatureMap): FeatureMap => {
      allBuffers.push(fm.buffer);
      return fm;
    };

    const s0 = track(this.conv(encoder, input, "stem.conv"));
    intermediates.set("stem.conv", s0);

    const d1down = track(this.conv(encoder, s0, "down1.down.conv"));
    const s1 = track(this.conv(encoder, d1down, "down1.refine.conv"));
    intermediates.set("down1.refine.conv", s1);

    const d2down = track(this.conv(encoder, s1, "down2.down.conv"));
    const s2 = track(this.conv(encoder, d2down, "down2.refine.conv"));
    intermediates.set("down2.refine.conv", s2);

    const d3down = track(this.conv(encoder, s2, "down3.down.conv"));
    const s3 = track(this.conv(encoder, d3down, "down3.refine.conv"));
    intermediates.set("down3.refine.conv", s3);

    const bn0 = track(this.conv(encoder, s3, "bottleneck.0.conv"));
    const b = track(this.conv(encoder, bn0, "bottleneck.1.conv"));
    intermediates.set("bottleneck.1.conv", b);

    const u1up = track(this.upsample(encoder, b, "upsample:up1"));
    const u1cat = track(this.concat(encoder, u1up, s2, "concat:up1"));
    const u1c1 = track(this.conv(encoder, u1cat, "up1.conv1.conv"));
    const u1 = track(this.conv(encoder, u1c1, "up1.conv2.conv"));
    intermediates.set("up1.conv2.conv", u1);

    const u2up = track(this.upsample(encoder, u1, "upsample:up2"));
    const u2cat = track(this.concat(encoder, u2up, s1, "concat:up2"));
    const u2c1 = track(this.conv(encoder, u2cat, "up2.conv1.conv"));
    const u2 = track(this.conv(encoder, u2c1, "up2.conv2.conv"));
    intermediates.set("up2.conv2.conv", u2);

    const u3up = track(this.upsample(encoder, u2, "upsample:up3"));
    const u3cat = track(this.concat(encoder, u3up, s0, "concat:up3"));
    const u3c1 = track(this.conv(encoder, u3cat, "up3.conv1.conv"));
    const u3 = track(this.conv(encoder, u3c1, "up3.conv2.conv"));
    intermediates.set("up3.conv2.conv", u3);

    const headOut = track(this.conv(encoder, u3, "head"));
    intermediates.set("head", headOut);

    const output = track(this.pixelShuffle(encoder, headOut));
    intermediates.set("pixel_shuffle", output);

    return { output, intermediates, allBuffers };
  }

  /** Destroys every buffer forward() allocated during the call that
   * produced `result`, except any passed in `keep` (e.g. output.buffer when
   * the caller is carrying it forward, or when intermediates are still
   * needed for per-layer diffing). Must only be called once the command
   * buffer(s) that encoded and consumed these buffers have finished
   * executing on the GPU (await device.queue.onSubmittedWorkDone()) --
   * destroying a buffer still referenced by in-flight or unsubmitted
   * commands is unsafe. */
  static releaseIntermediates(result: { allBuffers: GPUBuffer[] }, keep: Iterable<GPUBuffer> = []): void {
    const keepSet = new Set(keep);
    for (const buf of result.allBuffers) {
      if (!keepSet.has(buf)) buf.destroy();
    }
  }

  allocInput(width: number, height: number, channels: number, data: Float32Array): FeatureMap {
    const buffer = this.allocFeatureMap(width, height, channels);
    this.device.queue.writeBuffer(buffer, 0, toScalarBytes(data, this.hasF16));
    return { buffer, width, height, channels };
  }

  async readFeatureMap(fm: FeatureMap): Promise<Float32Array> {
    const readBuf = this.device.createBuffer({
      size: fm.width * fm.height * fm.channels * this.bytesPerElement,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(fm.buffer, 0, readBuf, 0, readBuf.size);
    this.device.queue.submit([encoder.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const mapped = readBuf.getMappedRange();
    const out = this.hasF16 ? Float32Array.from(new Float16Array(mapped.slice(0))) : new Float32Array(mapped.slice(0));
    readBuf.unmap();
    readBuf.destroy();
    return out;
  }
}

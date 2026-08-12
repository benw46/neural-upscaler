import convSource from "./wgsl/conv.wgsl?raw";
import upsampleSource from "./wgsl/upsample.wgsl?raw";
import concatSource from "./wgsl/concat.wgsl?raw";
import pixelShuffleSource from "./wgsl/pixel_shuffle.wgsl?raw";

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

function scalarPrelude(hasF16: boolean): string {
  return hasF16 ? "enable f16;\nalias Scalar = f16;\n\n" : "alias Scalar = f32;\n\n";
}

/** Converts an f32 array to the storage representation matching `hasF16` --
 * either a tightly-packed Float16Array or a Float32Array. */
function toScalarBytes(data: Float32Array, hasF16: boolean): ArrayBuffer {
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

  private convPipeline!: GPUComputePipeline;
  private convLayout!: GPUBindGroupLayout;
  private upsamplePipeline!: GPUComputePipeline;
  private upsampleLayout!: GPUBindGroupLayout;
  private concatPipeline!: GPUComputePipeline;
  private concatLayout!: GPUBindGroupLayout;
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

  constructor(device: GPUDevice, hasF16: boolean, hasTimestampQuery = false) {
    this.device = device;
    this.hasF16 = hasF16;
    this.bytesPerElement = hasF16 ? 2 : 4;
    this.hasTimestampQuery = hasTimestampQuery;
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
    this.convPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.convLayout] }),
      compute: { module: device.createShaderModule({ code: prelude + convSource }), entryPoint: "main" },
    });

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
    this.concatPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.concatLayout] }),
      compute: { module: device.createShaderModule({ code: prelude + concatSource }), entryPoint: "main" },
    });

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
    pass.setPipeline(this.convPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(outWidth / 8), Math.ceil(outHeight / 8), layer.outChannels);
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
    pass.setPipeline(this.concatPipeline);
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
  forward(encoder: GPUCommandEncoder, input: FeatureMap, options?: { profile?: boolean }): { output: FeatureMap; intermediates: Map<string, FeatureMap> } {
    if (options?.profile) {
      if (!this.hasTimestampQuery) throw new Error("profile:true requested but timestamp-query is not available on this adapter");
      this.profilingActive = true;
      this.dispatchIndex = 0;
      this.profileLabels = [];
    }
    const intermediates = new Map<string, FeatureMap>();

    const s0 = this.conv(encoder, input, "stem.conv");
    intermediates.set("stem.conv", s0);

    const d1down = this.conv(encoder, s0, "down1.down.conv");
    const s1 = this.conv(encoder, d1down, "down1.refine.conv");
    intermediates.set("down1.refine.conv", s1);

    const d2down = this.conv(encoder, s1, "down2.down.conv");
    const s2 = this.conv(encoder, d2down, "down2.refine.conv");
    intermediates.set("down2.refine.conv", s2);

    const d3down = this.conv(encoder, s2, "down3.down.conv");
    const s3 = this.conv(encoder, d3down, "down3.refine.conv");
    intermediates.set("down3.refine.conv", s3);

    const bn0 = this.conv(encoder, s3, "bottleneck.0.conv");
    const b = this.conv(encoder, bn0, "bottleneck.1.conv");
    intermediates.set("bottleneck.1.conv", b);

    const u1up = this.upsample(encoder, b, "upsample:up1");
    const u1cat = this.concat(encoder, u1up, s2, "concat:up1");
    const u1c1 = this.conv(encoder, u1cat, "up1.conv1.conv");
    const u1 = this.conv(encoder, u1c1, "up1.conv2.conv");
    intermediates.set("up1.conv2.conv", u1);

    const u2up = this.upsample(encoder, u1, "upsample:up2");
    const u2cat = this.concat(encoder, u2up, s1, "concat:up2");
    const u2c1 = this.conv(encoder, u2cat, "up2.conv1.conv");
    const u2 = this.conv(encoder, u2c1, "up2.conv2.conv");
    intermediates.set("up2.conv2.conv", u2);

    const u3up = this.upsample(encoder, u2, "upsample:up3");
    const u3cat = this.concat(encoder, u3up, s0, "concat:up3");
    const u3c1 = this.conv(encoder, u3cat, "up3.conv1.conv");
    const u3 = this.conv(encoder, u3c1, "up3.conv2.conv");
    intermediates.set("up3.conv2.conv", u3);

    const headOut = this.conv(encoder, u3, "head");
    intermediates.set("head", headOut);

    const output = this.pixelShuffle(encoder, headOut);
    intermediates.set("pixel_shuffle", output);

    return { output, intermediates };
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

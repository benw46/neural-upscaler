import { WgslUNet, scalarPrelude } from "./model_wgsl.ts";
import { SceneRenderer } from "../../renderer/src/render/renderer.ts";
import { GBuffer } from "../../renderer/src/render/gbuffer.ts";
import { ScriptedCameraPath, frameState, stateViewProj, type FrameCameraState } from "../../renderer/src/camera/sequence.ts";
import mainShaderSource from "../../renderer/src/render/shaders.wgsl?raw";
import packInputSource from "./wgsl/pack_input.wgsl?raw";
import downsamplePrevSource from "./wgsl/downsample_prev.wgsl?raw";
import warpSource from "./wgsl/warp_and_disocclude.wgsl?raw";

/** Shared live-scene machinery: renders the project's scripted deterministic
 * camera path (renderer/src/camera -- not free/user-steerable, the same path
 * Phase 0/1's dataset and every training frame came from) frame by frame,
 * builds the model's 8-channel input with *real* accumulated temporal
 * history (real motion-vector warping + disocclusion detection, not the
 * cold-start every displayed frame in viewer.ts's static 4-frame mode uses),
 * and runs the real WGSL network on it.
 *
 * Single source of truth for this pipeline -- both live.ts (dedicated live
 * page) and viewer.ts (the "realtime" toggle) use this class rather than
 * each carrying their own copy of the warp/pack logic. That logic
 * (warp_and_disocclude.wgsl reimplementing training/src/warp.py) is the
 * single most fragile piece of new code in this project; one copy is one
 * copy to keep correct, not two that can silently drift apart.
 *
 * Gap since closed: the warp kernel now has a formal fixture diff against
 * the PyTorch reference (export/gen_diff_fixtures_warp.py generates the
 * fixture from warp.py's real warp_previous_output/compute_disocclusion_mask;
 * inference/src/main.ts's "Step 4" runs warp_and_disocclude.wgsl on the same
 * input and diffs numerically), matching the standard every other WGSL
 * kernel in this codebase is held to. Result: colour warp matches to ~1e-4
 * (FP16-rounding-level, same as the U-Net's own per-layer checks); the
 * disocclusion mask agrees on 99.88% of pixels on the validation fixture,
 * with 100% of the disagreements landing within a rel_diff band 0.002 wide
 * around DEPTH_REL_THRESHOLD=0.05 -- the expected signature of two
 * independently-rounded bilinear interpolations occasionally tipping a hard
 * boolean threshold in opposite directions, not a UV-convention, padding-
 * mode, or axis bug (see main.ts's MASK_MISMATCH_TOLERANCE comment for the
 * full reasoning). The runtime sanity readout (`meanDisocclusion`, returned
 * every frame) remains in place as an ongoing live smell test on top of the
 * one-time fixture diff, not a replacement for it.
 *
 * The "groundtruth" display mode is a single unjittered high-res render --
 * NOT supersampled, unlike the offline dataset's actual ground truth (see
 * CLAUDE.md's antialiasing fragile-logic note: "a merely-high-resolution GT
 * is not sufficient"). It's a reasonable live reference for eyeballing
 * quality, not the same rigour as the training data; callers should label
 * it as such rather than imply equivalence.
 */

export const SEED = 20260812; // matches renderer/src/main.ts -- same deterministic scene/camera path
export const IN_W = 960;
export const IN_H = 540;
export const IN_H_PAD = 544;
export const GT_W = 1920;
export const GT_H = 1080;
export const OUT_H_PAD = IN_H_PAD * 2; // 1088 -- network's raw output height before the display crop
export const IN_CHANNELS = 8;
export const DEPTH_NORM = 50.0; // training/src/dataset.py -- must match exactly, see CLAUDE.md preprocessing-parity note

export type DisplayMode = "input" | "network" | "groundtruth";

export interface LiveFrameResult {
  frameIndex: number;
  t: number;
  jitter: [number, number];
  ms: number;
  /** Mean disocclusion-mask fraction this frame, or null on frame 0 (no
   * warp/disocclusion computation happens yet -- see warp_and_disocclude.wgsl's
   * is_first_frame branch). See the module docstring's validation-status note. */
  meanDisocclusion: number | null;
  /** (GT_H, GT_W, 3), cropped -- always computed regardless of displayMode,
   * since it's also next frame's real warp input, not just a display value. */
  networkOutput: Float32Array;
  /** (IN_H, IN_W, 4), raw jittered low-res colour -- only read back when
   * displayMode==="input" (the render itself always happens regardless, this
   * is just the CPU readback, skipped when not needed). */
  inputColour?: Float16Array;
  /** (GT_H, GT_W, 4), unjittered high-res reference -- only rendered and
   * read back when displayMode==="groundtruth" (unlike the low-res render,
   * this pass is skippable entirely: it never feeds back into anything). */
  groundTruthColour?: Float16Array;
}

function makeUniform(device: GPUDevice, values: number[]): GPUBuffer {
  const buf = device.createBuffer({ size: values.length * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buf, 0, new Uint32Array(values));
  return buf;
}
// Same uniform buffer layout convention as the pack kernel's Params structs
// that end with an f32 field (depth_norm) -- Uint32Array can't carry a float
// bit-exactly, so this variant writes the raw bytes of a mixed u32/f32 array.
function makeMixedUniform(device: GPUDevice, u32s: number[], f32s: number[]): GPUBuffer {
  const buf = device.createBuffer({ size: (u32s.length + f32s.length) * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const data = new ArrayBuffer((u32s.length + f32s.length) * 4);
  const view = new DataView(data);
  let offset = 0;
  for (const v of u32s) {
    view.setUint32(offset, v, true);
    offset += 4;
  }
  for (const v of f32s) {
    view.setFloat32(offset, v, true);
    offset += 4;
  }
  device.queue.writeBuffer(buf, 0, data);
  return buf;
}

/** Reads an rgba16float texture back to a tightly-packed (H, W, 4)
 * Float16Array, stripping WebGPU's 256-byte row-pitch padding correctly
 * rather than assuming a given width happens to already divide evenly
 * (960 and 1920 both do, at 8 bytes/pixel -- coincidentally convenient, not
 * relied on here). */
async function readColourTexture(device: GPUDevice, texture: GPUTexture, width: number, height: number): Promise<Float16Array> {
  const bytesPerPixel = 8; // rgba16float
  const unpaddedBytesPerRow = width * bytesPerPixel;
  const paddedBytesPerRow = Math.ceil(unpaddedBytesPerRow / 256) * 256;
  const readBuf = device.createBuffer({
    size: paddedBytesPerRow * height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer: readBuf, bytesPerRow: paddedBytesPerRow, rowsPerImage: height }, [width, height]);
  device.queue.submit([encoder.finish()]);
  await readBuf.mapAsync(GPUMapMode.READ);
  const mapped = new Uint8Array(readBuf.getMappedRange().slice(0));
  readBuf.unmap();
  readBuf.destroy();

  if (paddedBytesPerRow === unpaddedBytesPerRow) {
    return new Float16Array(mapped.buffer);
  }
  const out = new Uint8Array(unpaddedBytesPerRow * height);
  for (let y = 0; y < height; y++) {
    out.set(mapped.subarray(y * paddedBytesPerRow, y * paddedBytesPerRow + unpaddedBytesPerRow), y * unpaddedBytesPerRow);
  }
  return new Float16Array(out.buffer);
}

export class LiveScenePipeline {
  private device: GPUDevice;
  private unet: WgslUNet;
  private hasF16: boolean;
  private bytesPerElement: number;

  private renderer: SceneRenderer;
  private gbufferLow: GBuffer;
  private gbufferHigh: GBuffer;
  private cameraPath: ScriptedCameraPath;
  private prevDepthSnapshot: GPUTexture;

  private packPipeline: GPUComputePipeline;
  private packLayout: GPUBindGroupLayout;
  private downsamplePipeline: GPUComputePipeline;
  private downsampleLayout: GPUBindGroupLayout;
  private warpPipeline: GPUComputePipeline;
  private warpLayout: GPUBindGroupLayout;

  private prevOutputBuffer: GPUBuffer;
  private readonly prevLowresBuffer: GPUBuffer;
  private readonly warpedBuffer: GPUBuffer;
  private readonly packedInputBuffer: GPUBuffer;
  private readonly downsampleUniform: GPUBuffer;
  private readonly packBindGroup: GPUBindGroup;
  private downsampleBindGroup: GPUBindGroup;

  frameIndex = 0;
  private prevViewProj: Float32Array | null = null;
  // Cached for redisplay() -- the last frame actually computed by
  // stepFrame(), so pausing can re-show it in a different display mode
  // without advancing the camera path or touching any temporal state.
  private lastState: FrameCameraState | null = null;
  private lastNetworkOutput: Float32Array | null = null;

  private constructor(device: GPUDevice, unet: WgslUNet, hasF16: boolean, colored: boolean) {
    this.device = device;
    this.unet = unet;
    this.hasF16 = hasF16;
    this.bytesPerElement = hasF16 ? 2 : 4;

    const prelude = scalarPrelude(hasF16);
    this.renderer = new SceneRenderer(device, mainShaderSource, colored);
    this.gbufferLow = new GBuffer(device, IN_W, IN_H);
    this.gbufferHigh = new GBuffer(device, GT_W, GT_H);
    this.cameraPath = new ScriptedCameraPath(SEED);

    this.prevDepthSnapshot = device.createTexture({
      label: "prev-depth-snapshot",
      size: [IN_W, IN_H],
      format: "r32float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    this.packLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    this.packPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.packLayout] }),
      compute: { module: device.createShaderModule({ code: prelude + packInputSource }), entryPoint: "main" },
    });

    this.downsampleLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    this.downsamplePipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.downsampleLayout] }),
      compute: { module: device.createShaderModule({ code: prelude + downsamplePrevSource }), entryPoint: "main" },
    });

    this.warpLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    this.warpPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.warpLayout] }),
      compute: { module: device.createShaderModule({ code: prelude + warpSource }), entryPoint: "main" },
    });

    // WebGPU zero-initialises new buffers by spec, so prevOutputBuffer's
    // frame-0 content (never actually read -- see warp_and_disocclude.wgsl's
    // is_first_frame branch) doesn't need explicit clearing.
    this.prevOutputBuffer = device.createBuffer({
      size: OUT_H_PAD * GT_W * 3 * this.bytesPerElement,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.prevLowresBuffer = device.createBuffer({ size: IN_H * IN_W * 3 * this.bytesPerElement, usage: GPUBufferUsage.STORAGE });
    this.warpedBuffer = device.createBuffer({
      size: IN_H * IN_W * 4 * this.bytesPerElement,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, // COPY_SRC for the disocclusion sanity-check readback
    });
    this.packedInputBuffer = device.createBuffer({ size: IN_H_PAD * IN_W * IN_CHANNELS * this.bytesPerElement, usage: GPUBufferUsage.STORAGE });

    this.downsampleUniform = makeUniform(device, [GT_W, GT_H, IN_W, IN_H]);
    this.downsampleBindGroup = this.buildDownsampleBindGroup();

    const packUniform = makeMixedUniform(device, [IN_W, IN_H, IN_H_PAD], [DEPTH_NORM]);
    this.packBindGroup = device.createBindGroup({
      layout: this.packLayout,
      entries: [
        { binding: 0, resource: { buffer: packUniform } },
        { binding: 1, resource: this.gbufferLow.colour.createView() },
        { binding: 2, resource: this.gbufferLow.depth.createView() },
        { binding: 3, resource: { buffer: this.warpedBuffer } },
        { binding: 4, resource: { buffer: this.packedInputBuffer } },
      ],
    });
  }

  static async create(device: GPUDevice, unet: WgslUNet, hasF16: boolean, colored = false): Promise<LiveScenePipeline> {
    return new LiveScenePipeline(device, unet, hasF16, colored);
  }

  private buildDownsampleBindGroup(): GPUBindGroup {
    return this.device.createBindGroup({
      layout: this.downsampleLayout,
      entries: [
        { binding: 0, resource: { buffer: this.downsampleUniform } },
        { binding: 1, resource: { buffer: this.prevOutputBuffer } },
        { binding: 2, resource: { buffer: this.prevLowresBuffer } },
      ],
    });
  }

  private buildWarpBindGroup(isFirstFrame: boolean): GPUBindGroup {
    const warpUniform = makeUniform(this.device, [IN_W, IN_H, isFirstFrame ? 1 : 0]);
    return this.device.createBindGroup({
      layout: this.warpLayout,
      entries: [
        { binding: 0, resource: { buffer: warpUniform } },
        { binding: 1, resource: { buffer: this.prevLowresBuffer } },
        { binding: 2, resource: this.gbufferLow.motion.createView() },
        { binding: 3, resource: this.gbufferLow.depth.createView() },
        { binding: 4, resource: this.prevDepthSnapshot.createView() },
        { binding: 5, resource: { buffer: this.warpedBuffer } },
      ],
    });
  }

  async stepFrame(displayMode: DisplayMode): Promise<LiveFrameResult> {
    const device = this.device;
    const t0 = performance.now();
    const isFirstFrame = this.frameIndex === 0;

    const state = frameState(this.cameraPath, this.frameIndex, this.renderer.colliders);
    const viewProjLow = new Float32Array(stateViewProj(state, IN_W, IN_H, true));
    const prevViewProjForMotion = this.prevViewProj ?? viewProjLow;

    if (!isFirstFrame) {
      const copyEncoder = device.createCommandEncoder();
      copyEncoder.copyTextureToTexture({ texture: this.gbufferLow.depth }, { texture: this.prevDepthSnapshot }, [IN_W, IN_H]);
      device.queue.submit([copyEncoder.finish()]);
    }

    this.renderer.renderInto(this.gbufferLow, viewProjLow, prevViewProjForMotion, `live-frame-${this.frameIndex}`);

    // Ground-truth pass is fully independent of the network/warp path -- it
    // never feeds back into anything -- so it's only rendered when actually
    // being displayed, not every frame regardless of mode.
    if (displayMode === "groundtruth") {
      const viewProjHigh = new Float32Array(stateViewProj(state, GT_W, GT_H, false)); // jittered=false: a real reference render, not the network's jittered input
      this.renderer.renderInto(this.gbufferHigh, viewProjHigh, viewProjHigh, `live-gt-${this.frameIndex}`);
    }

    const encoder = device.createCommandEncoder();
    if (!isFirstFrame) {
      const dsPass = encoder.beginComputePass({ label: "downsample_prev" });
      dsPass.setPipeline(this.downsamplePipeline);
      dsPass.setBindGroup(0, this.downsampleBindGroup);
      dsPass.dispatchWorkgroups(Math.ceil(IN_W / 8), Math.ceil(IN_H / 8), 1);
      dsPass.end();
    }
    const warpPass = encoder.beginComputePass({ label: "warp_and_disocclude" });
    warpPass.setPipeline(this.warpPipeline);
    warpPass.setBindGroup(0, this.buildWarpBindGroup(isFirstFrame));
    warpPass.dispatchWorkgroups(Math.ceil(IN_W / 8), Math.ceil(IN_H / 8), 1);
    warpPass.end();

    const packPass = encoder.beginComputePass({ label: "pack_input" });
    packPass.setPipeline(this.packPipeline);
    packPass.setBindGroup(0, this.packBindGroup);
    packPass.dispatchWorkgroups(Math.ceil(IN_W / 8), Math.ceil(IN_H_PAD / 8), 1);
    packPass.end();

    const unetResult = this.unet.forward(encoder, { buffer: this.packedInputBuffer, width: IN_W, height: IN_H_PAD, channels: IN_CHANNELS });
    const { output } = unetResult;
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    // output.buffer becomes this.prevOutputBuffer below and lives on into
    // next frame -- every other buffer this forward() pass allocated is done
    // being read from now that the submitted work has completed.
    WgslUNet.releaseIntermediates(unetResult, [output.buffer]);

    const full = await this.unet.readFeatureMap(output);
    const networkOutput = full.subarray(0, GT_H * GT_W * 3);

    let inputColour: Float16Array | undefined;
    if (displayMode === "input") {
      inputColour = await readColourTexture(device, this.gbufferLow.colour, IN_W, IN_H);
    }
    let groundTruthColour: Float16Array | undefined;
    if (displayMode === "groundtruth") {
      groundTruthColour = await readColourTexture(device, this.gbufferHigh.colour, GT_W, GT_H);
    }

    // Lightweight ongoing live sanity readout, on top of (not a replacement
    // for) the one-time fixture diff -- see the module docstring.
    let meanDisocclusion: number | null = null;
    if (!isFirstFrame) {
      const warpedReadback = device.createBuffer({ size: this.warpedBuffer.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const readEncoder = device.createCommandEncoder();
      readEncoder.copyBufferToBuffer(this.warpedBuffer, 0, warpedReadback, 0, this.warpedBuffer.size);
      device.queue.submit([readEncoder.finish()]);
      await warpedReadback.mapAsync(GPUMapMode.READ);
      const mapped = this.hasF16 ? Float32Array.from(new Float16Array(warpedReadback.getMappedRange().slice(0))) : new Float32Array(warpedReadback.getMappedRange().slice(0));
      warpedReadback.unmap();
      warpedReadback.destroy();
      let sum = 0;
      for (let i = 3; i < mapped.length; i += 4) sum += mapped[i];
      meanDisocclusion = sum / (mapped.length / 4);
    }

    // Two-frame buffer lifecycle: `output.buffer` becomes next frame's
    // prevOutputBuffer; the buffer that *was* prevOutputBuffer is no longer
    // referenced by anything after this frame's downsample_prev dispatch has
    // completed (guaranteed by the onSubmittedWorkDone above), so it's safe
    // to destroy now rather than let a live session leak one GPU buffer per
    // frame indefinitely.
    if (!isFirstFrame) this.prevOutputBuffer.destroy();
    this.prevOutputBuffer = output.buffer;
    this.downsampleBindGroup = this.buildDownsampleBindGroup();

    const t1 = performance.now();
    this.prevViewProj = viewProjLow;
    this.lastState = state;
    this.lastNetworkOutput = networkOutput;
    const result: LiveFrameResult = {
      frameIndex: this.frameIndex,
      t: state.t,
      jitter: state.jitter,
      ms: t1 - t0,
      meanDisocclusion,
      networkOutput,
      inputColour,
      groundTruthColour,
    };
    this.frameIndex++;
    return result;
  }

  /** Re-displays the *same* already-computed frame in a different display
   * mode, without advancing the scripted camera path or touching any
   * temporal state (prevOutputBuffer, warp, disocclusion) -- for pausing,
   * so switching views doesn't move the scene. Network mode is free (the
   * output from the last real stepFrame() call is simply reused); input
   * mode re-reads the low-res G-buffer's colour texture, which still holds
   * that frame's content since nothing has re-rendered it since; groundtruth
   * mode re-renders the high-res unjittered pass at the *same* cached camera
   * state (frameState/stateViewProj are pure functions of (path,
   * frameIndex), so recomputing the identical viewProj from the cached
   * state reproduces the exact same frame, not an approximation of "the
   * same place"). */
  async redisplay(mode: DisplayMode): Promise<LiveFrameResult> {
    if (!this.lastState || !this.lastNetworkOutput) {
      throw new Error("redisplay() called before any stepFrame() has run");
    }
    const state = this.lastState;
    const device = this.device;

    let inputColour: Float16Array | undefined;
    if (mode === "input") {
      inputColour = await readColourTexture(device, this.gbufferLow.colour, IN_W, IN_H);
    }
    let groundTruthColour: Float16Array | undefined;
    if (mode === "groundtruth") {
      const viewProjHigh = new Float32Array(stateViewProj(state, GT_W, GT_H, false));
      this.renderer.renderInto(this.gbufferHigh, viewProjHigh, viewProjHigh, `live-gt-redisplay-${state.frameIndex}`);
      groundTruthColour = await readColourTexture(device, this.gbufferHigh.colour, GT_W, GT_H);
    }

    return {
      frameIndex: state.frameIndex,
      t: state.t,
      jitter: state.jitter,
      ms: 0,
      meanDisocclusion: null,
      networkOutput: this.lastNetworkOutput,
      inputColour,
      groundTruthColour,
    };
  }
}

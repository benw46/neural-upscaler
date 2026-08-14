// Tiled variant of conv.wgsl's direct 3x3 "same"-padding convolution,
// specialised for the three largest-Cin layers in the model
// (up1/up2/up3.conv1.conv -- see model_wgsl.ts's TILED_CONV_LAYERS), the
// only ones the (Cin,Cout)-specialised plain kernel (conv.wgsl) barely sped
// up despite full loop unrolling helping almost everywhere else. That's the
// diagnostic signal this kernel exists for: a layer unrolling doesn't help
// is memory-bandwidth-bound, not instruction-bound, and the redundancy
// unrolling can't touch is spatial -- neighbouring output pixels' 3x3
// windows overlap on 6 of 9 taps, but every thread still independently
// re-fetches its own copy of that overlap from global memory. This kernel
// caches one input tile per workgroup in workgroup-shared memory, so the
// 8x8 block of output pixels a workgroup covers share one fetch of the
// overlapping region instead of each re-fetching it.
//
// A full-depth tile (10x10xCin, +1 halo each side for the 3x3 kernel) would
// exceed the workgroup-storage budget for every one of these three layers
// (Cin up to 196) at any reasonable precision -- so this chunks over Cin in
// blocks of CHUNK_SIZE, accumulating partial sums across chunks.
//
// CHUNK_SIZE is *not* a fixed guess -- model_wgsl.ts's getConvTiledPipeline
// computes it per variant from the device's actual granted
// maxComputeWorkgroupStorageSize (gpu.ts explicitly requests the adapter's
// real maximum rather than accepting WebGPU's spec-minimum 16KB default;
// confirmed on the development adapter as 32KB granted vs 16KB default).
// The tile is also `Scalar`-typed (f16 when available), not always f32 --
// halves the per-channel footprint on top of the larger budget. Together
// these let `up2.conv1` and `up3.conv1` (Cin=140/84) fit their *entire*
// depth in a single chunk on this adapter -- no chunking loop at all for
// those two -- while `up1.conv1` (Cin=196) still needs 2. This directly
// targets the barrier-count hypothesis from the first (CHUNK_SIZE=32,
// hardcoded, f32 tile, 16KB-budget-assumed) version of this kernel: see
// docs/OPTIMISATIONS.md stage 4 for why that version underperformed
// prediction and what wasn't confirmed.
//
// Assumes stride=1 -- true for all three target layers (they're all
// "same"-padding refine convs right after an upsample+concat, never a
// downsampling conv); model_wgsl.ts asserts this before dispatch. This
// kernel does not handle stride>1 tiling.
//
// IN_CHANNELS/OUT_CHANNELS/CHUNK_SIZE are compile-time constants substituted
// per layer/adapter; IN_CHANNELS/OUT_CHANNELS follow conv.wgsl's mechanism.
//
// Zero-padding semantics: the cooperative load zero-fills any tile position
// outside [0, in_width) x [0, in_height) before the accumulate phase reads
// it -- exactly equivalent to conv.wgsl's per-tap bounds-check-and-skip (a
// tap that would've been skipped there contributes a v=0.0 term here
// instead, same numeric result). Preserving this exactly was flagged as the
// specific risk of adding tiling; verified against the existing per-layer
// diff harness, not just reasoned about.

const TILE: u32 = 8u;
const TILE_PADDED: u32 = TILE + 2u; // +1 halo each side for the 3x3 kernel
const CHUNK_SIZE: u32 = CHUNK_SIZE_VALUEu;
const IN_CHANNELS: u32 = IN_CHANNELS_VALUEu;
const OUT_CHANNELS: u32 = OUT_CHANNELS_VALUEu;
const NUM_CHUNKS: u32 = (IN_CHANNELS + CHUNK_SIZE - 1u) / CHUNK_SIZE;

struct Params {
  in_width: u32,
  in_height: u32,
  in_channels: u32, // unused below -- IN_CHANNELS is the compile-time-equal value actually used; kept for debug visibility
  out_channels: u32, // unused below -- OUT_CHANNELS is the compile-time-equal value actually used; kept for debug visibility
  stride: u32, // unused below -- this kernel assumes stride=1, asserted by model_wgsl.ts before dispatch
  apply_activation: u32, // 1 = LeakyReLU(0.2), 0 = none
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_tex: array<Scalar>;   // (H, W, Cin)
@group(0) @binding(2) var<storage, read> weights: array<Scalar>;     // (Cout, 3, 3, Cin)
@group(0) @binding(3) var<storage, read> bias_buf: array<Scalar>;    // (Cout)
@group(0) @binding(4) var<storage, read_write> output_tex: array<Scalar>; // (H, W, Cout)

// (ly, lx, lc) row-major, ly/lx in [0, TILE_PADDED), lc in [0, CHUNK_SIZE) --
// only the first `chunk_len` lc-slots are valid on a chunk whose remainder
// is smaller than CHUNK_SIZE (Cin not an exact multiple of it). `Scalar`
// (f16 when available), matching input_tex's own storage precision exactly
// -- storing an already-f16 value as f16 here introduces no additional
// rounding versus the previous always-f32 tile.
var<workgroup> tile: array<Scalar, TILE_PADDED * TILE_PADDED * CHUNK_SIZE>;

@compute @workgroup_size(TILE, TILE, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wgid: vec3<u32>) {
  let ox = gid.x;
  let oy = gid.y;
  let tx = lid.x;
  let ty = lid.y;
  let local_flat = ty * TILE + tx; // 0..63, this thread's index within the workgroup

  var acc: array<f32, OUT_CHANNELS>; // zero-initialised; accumulate in f32 regardless of storage precision

  for (var chunk_idx: u32 = 0u; chunk_idx < NUM_CHUNKS; chunk_idx = chunk_idx + 1u) {
    let chunk_start = chunk_idx * CHUNK_SIZE;
    let chunk_len = min(CHUNK_SIZE, IN_CHANNELS - chunk_start);
    let tile_elems = TILE_PADDED * TILE_PADDED * chunk_len;

    // Cooperative load: every thread fills a strided slice of the tile,
    // regardless of whether its own output pixel is in bounds -- this loop
    // and both barriers below must execute uniformly across the whole
    // workgroup; only the accumulate step further down is allowed to be
    // conditional per-thread.
    for (var i: u32 = local_flat; i < tile_elems; i = i + TILE * TILE) {
      let lc = i % chunk_len;
      let rest = i / chunk_len;
      let lx = rest % TILE_PADDED;
      let ly = rest / TILE_PADDED;
      let gx = i32(wgid.x * TILE) + i32(lx) - 1;
      let gy = i32(wgid.y * TILE) + i32(ly) - 1;
      var v: Scalar = Scalar(0.0);
      if (gx >= 0 && gx < i32(params.in_width) && gy >= 0 && gy < i32(params.in_height)) {
        let src_idx = (u32(gy) * params.in_width + u32(gx)) * IN_CHANNELS + (chunk_start + lc);
        v = input_tex[src_idx];
      }
      tile[i] = v;
    }
    workgroupBarrier();

    if (ox < params.in_width && oy < params.in_height) {
      for (var ky: u32 = 0u; ky < 3u; ky = ky + 1u) {
        let tly = ty + ky;
        for (var kx: u32 = 0u; kx < 3u; kx = kx + 1u) {
          let tlx = tx + kx;
          let w_tap_base = (ky * 3u + kx) * IN_CHANNELS;
          let tile_row_base = (tly * TILE_PADDED + tlx) * chunk_len;
          for (var lc: u32 = 0u; lc < chunk_len; lc = lc + 1u) {
            let v = f32(tile[tile_row_base + lc]);
            let w_base = w_tap_base + (chunk_start + lc);
            for (var co: u32 = 0u; co < OUT_CHANNELS; co = co + 1u) {
              acc[co] = acc[co] + v * f32(weights[co * 9u * IN_CHANNELS + w_base]);
            }
          }
        }
      }
    }
    workgroupBarrier(); // every thread must finish reading `tile` before any thread overwrites it for the next chunk
  }

  if (ox >= params.in_width || oy >= params.in_height) {
    return;
  }
  let out_base = (oy * params.in_width + ox) * OUT_CHANNELS;
  for (var co: u32 = 0u; co < OUT_CHANNELS; co = co + 1u) {
    var val = acc[co] + f32(bias_buf[co]);
    if (params.apply_activation == 1u) {
      val = select(val * 0.2, val, val >= 0.0);
    }
    output_tex[out_base + co] = Scalar(val);
  }
}

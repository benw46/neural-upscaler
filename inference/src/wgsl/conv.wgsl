// Direct convolution (3x3, "same" zero-padding) fused with LeakyReLU(0.2).
// NHWC layout throughout. `Scalar` (f16 or f32) is prepended by the caller
// based on shader-f16 feature detection -- see src/gpu.ts.
//
// Direct convolution chosen over im2col+GEMM for every layer in this model:
// channel counts here are small (max 196) and spatial resolutions modest
// (16x16 to 128x128) -- genuinely tiny compute workloads by GPU standards.
// im2col needs its own materialisation dispatch before the GEMM pass, which
// roughly doubles dispatch count per conv layer; per CLAUDE.md's fragile-
// logic note, WebGPU's per-dispatch overhead compounds across many small
// passes and is expected to dominate over intra-kernel efficiency at this
// scale, so avoiding the extra dispatch outweighs any GEMM tiling gains
// here. See docs/PHASE-4-SUMMARY.md for the reasoning and measurements.
//
// One thread computes *every* output channel for one (x, y) pixel, instead
// of one thread per (x, y, outChannel). The previous version had all
// `out_channels` threads at a given pixel independently re-fetch the
// identical 3x3xCin input window from `input_tex` -- confirmed as the
// dominant term of the kernel's redundant memory traffic (up to 9*Cout
// re-reads per input element, e.g. 756x for up1.conv1's Cin=196/Cout=84;
// see docs/PHASE-4-SUMMARY.md's fusion evaluation). Restructuring the loop
// so `ci` is outermost relative to `co` means each input tap is read once
// per thread and reused across every output-channel accumulator, cutting
// that redundancy from 9*Cout down to 9 (still un-shared across
// *neighbouring* output pixels -- that's the separate, larger follow-up:
// real workgroup-shared-memory tiling with input-channel chunking, not
// done here).
//
// OUT_CHANNELS and IN_CHANNELS are compile-time constants, not read from
// `params`. First cut of this only specialised OUT_CHANNELS (right-sizing
// the per-thread accumulator array, previously a single fixed
// array<f32, 112> shared -- and over-provisioned -- by every layer). That
// measured as a big win, well beyond the register-sizing story alone: with
// out_channels a *runtime* uniform, the compiler couldn't know the inner
// `co` loop's trip count and had to emit a real loop; as a compile-time
// constant it can fully unroll it. The `ci` loop was still runtime-bounded
// by `params.in_channels`, and the follow-up profile confirmed it as the
// next bottleneck: the three widest-Cin layers (up1/up2/up3.conv1, Cin =
// 196/140/84) became the two or three most expensive dispatches in the
// network by a clear margin, exactly where an un-unrolled `ci` loop hurts
// most. IN_CHANNELS closes that the same way. model_wgsl.ts now compiles one
// shader-module variant per distinct (in_channels, out_channels) *pair*
// actually present in the model (12 of them, up from 5) -- see
// getConvPipeline. A layer is only ever dispatched against the pipeline
// variant compiled for its own (Cin, Cout), so `params.in_channels` /
// `params.out_channels` (still sent, for debugging/clarity) and these
// constants are always equal by construction -- every loop below uses the
// compile-time constants, so the compiler can fully unroll the entire
// 3x3xCin block, not just the Cout dimension. WGSL zero-initialises `var`
// declarations with no initialiser, so `acc` starts at all-zero without an
// explicit clear loop.
const OUT_CHANNELS: u32 = OUT_CHANNELS_VALUEu;
const IN_CHANNELS: u32 = IN_CHANNELS_VALUEu;

struct Params {
  in_width: u32,
  in_height: u32,
  in_channels: u32, // unused below -- IN_CHANNELS is the compile-time-equal value actually used; kept for debug visibility
  out_channels: u32, // unused below -- OUT_CHANNELS is the compile-time-equal value actually used; kept for debug visibility
  stride: u32,
  apply_activation: u32, // 1 = LeakyReLU(0.2), 0 = none
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_tex: array<Scalar>;   // (H, W, Cin)
@group(0) @binding(2) var<storage, read> weights: array<Scalar>;     // (Cout, 3, 3, Cin)
@group(0) @binding(3) var<storage, read> bias_buf: array<Scalar>;    // (Cout)
@group(0) @binding(4) var<storage, read_write> output_tex: array<Scalar>; // (Hout, Wout, Cout)

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let out_width = (params.in_width - 1u) / params.stride + 1u;
  let out_height = (params.in_height - 1u) / params.stride + 1u;
  let ox = gid.x;
  let oy = gid.y;
  if (ox >= out_width || oy >= out_height) {
    return;
  }

  var acc: array<f32, OUT_CHANNELS>; // accumulate in f32 regardless of storage precision; zero-initialised

  for (var ky: i32 = 0; ky < 3; ky = ky + 1) {
    let iy = i32(oy * params.stride) + ky - 1;
    if (iy < 0 || iy >= i32(params.in_height)) {
      continue;
    }
    for (var kx: i32 = 0; kx < 3; kx = kx + 1) {
      let ix = i32(ox * params.stride) + kx - 1;
      if (ix < 0 || ix >= i32(params.in_width)) {
        continue;
      }
      let in_base = (u32(iy) * params.in_width + u32(ix)) * IN_CHANNELS;
      let w_tap_base = (u32(ky) * 3u + u32(kx)) * IN_CHANNELS; // offset within each out-channel's (3,3,Cin) block
      for (var ci: u32 = 0u; ci < IN_CHANNELS; ci = ci + 1u) {
        let v = f32(input_tex[in_base + ci]); // read once, reused across every out_channel below
        for (var co: u32 = 0u; co < OUT_CHANNELS; co = co + 1u) {
          let w_idx = co * 9u * IN_CHANNELS + w_tap_base + ci;
          acc[co] = acc[co] + v * f32(weights[w_idx]);
        }
      }
    }
  }

  let out_base = (oy * out_width + ox) * OUT_CHANNELS;
  for (var co: u32 = 0u; co < OUT_CHANNELS; co = co + 1u) {
    var val = acc[co] + f32(bias_buf[co]);
    if (params.apply_activation == 1u) {
      val = select(val * 0.2, val, val >= 0.0);
    }
    output_tex[out_base + co] = Scalar(val);
  }
}

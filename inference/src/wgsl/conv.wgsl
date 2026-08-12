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
// One thread computes one (x, y, outChannel) output element -- gives
// channel-dimension parallelism even at the 16x16 bottleneck resolution,
// where spatial-only parallelism (256 threads) would leave most of the GPU
// idle.

struct Params {
  in_width: u32,
  in_height: u32,
  in_channels: u32,
  out_channels: u32,
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
  let co = gid.z;
  if (ox >= out_width || oy >= out_height || co >= params.out_channels) {
    return;
  }

  var acc: f32 = 0.0; // accumulate in f32 regardless of storage precision
  let in_ch = params.in_channels;

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
      let in_base = (u32(iy) * params.in_width + u32(ix)) * in_ch;
      let w_base = ((co * 3u + u32(ky)) * 3u + u32(kx)) * in_ch;
      for (var ci: u32 = 0u; ci < in_ch; ci = ci + 1u) {
        acc = acc + f32(input_tex[in_base + ci]) * f32(weights[w_base + ci]);
      }
    }
  }

  acc = acc + f32(bias_buf[co]);
  if (params.apply_activation == 1u) {
    acc = select(acc * 0.2, acc, acc >= 0.0);
  }

  let out_base = (oy * out_width + ox) * params.out_channels + co;
  output_tex[out_base] = Scalar(acc);
}

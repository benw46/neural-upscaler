// Channel concatenation, NHWC: output = [a's channels, b's channels] per
// pixel. Order matters -- must match model.py's UpBlock.forward:
// `torch.cat([x, skip], dim=1)` with x (upsampled) first, skip second, so
// `a` here must be the upsampled tensor and `b` the skip connection.
//
// CHANNELS_A/CHANNELS_B are compile-time constants, not read from `params`
// -- same reasoning as conv.wgsl's IN_CHANNELS/OUT_CHANNELS: a runtime-
// uniform-bounded copy loop can't be unrolled by the compiler, and this
// dispatch's share of total frame time went from negligible to ~20% once
// conv.wgsl got fast (see docs/PHASE-4-SUMMARY.md's follow-up profile) --
// worth the same fix. Only 3 distinct (channels_a, channels_b) pairs exist
// (one per up-block), so model_wgsl.ts's getConcatPipeline compiles 3
// variants total.

const CHANNELS_A: u32 = CHANNELS_A_VALUEu;
const CHANNELS_B: u32 = CHANNELS_B_VALUEu;

struct Params {
  width: u32,
  height: u32,
  channels_a: u32, // unused below -- CHANNELS_A is the compile-time-equal value actually used; kept for debug visibility
  channels_b: u32, // unused below -- CHANNELS_B is the compile-time-equal value actually used; kept for debug visibility
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<Scalar>;
@group(0) @binding(2) var<storage, read> b: array<Scalar>;
@group(0) @binding(3) var<storage, read_write> output_tex: array<Scalar>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) {
    return;
  }
  let out_base = (y * params.width + x) * (CHANNELS_A + CHANNELS_B);
  let a_base = (y * params.width + x) * CHANNELS_A;
  let b_base = (y * params.width + x) * CHANNELS_B;

  for (var c: u32 = 0u; c < CHANNELS_A; c = c + 1u) {
    output_tex[out_base + c] = a[a_base + c];
  }
  for (var c: u32 = 0u; c < CHANNELS_B; c = c + 1u) {
    output_tex[out_base + CHANNELS_A + c] = b[b_base + c];
  }
}

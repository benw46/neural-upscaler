// Channel concatenation, NHWC: output = [a's channels, b's channels] per
// pixel. Order matters -- must match model.py's UpBlock.forward:
// `torch.cat([x, skip], dim=1)` with x (upsampled) first, skip second, so
// `a` here must be the upsampled tensor and `b` the skip connection.

struct Params {
  width: u32,
  height: u32,
  channels_a: u32,
  channels_b: u32,
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
  let total_c = params.channels_a + params.channels_b;
  let out_base = (y * params.width + x) * total_c;
  let a_base = (y * params.width + x) * params.channels_a;
  let b_base = (y * params.width + x) * params.channels_b;

  for (var c: u32 = 0u; c < params.channels_a; c = c + 1u) {
    output_tex[out_base + c] = a[a_base + c];
  }
  for (var c: u32 = 0u; c < params.channels_b; c = c + 1u) {
    output_tex[out_base + params.channels_a + c] = b[b_base + c];
  }
}

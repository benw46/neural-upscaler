// Nearest-neighbour 2x upsample, NHWC. Matches nn.Upsample(scale_factor=2,
// mode="nearest") from model.py's UpBlock -- see model.py's docstring for
// why nearest-then-conv was chosen over transposed convolution (avoids
// checkerboard artifacts).

struct Params {
  in_width: u32,
  in_height: u32,
  channels: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_tex: array<Scalar>;
@group(0) @binding(2) var<storage, read_write> output_tex: array<Scalar>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let out_width = params.in_width * 2u;
  let out_height = params.in_height * 2u;
  let ox = gid.x;
  let oy = gid.y;
  let c = gid.z;
  if (ox >= out_width || oy >= out_height || c >= params.channels) {
    return;
  }
  let ix = ox / 2u;
  let iy = oy / 2u;
  let in_idx = (iy * params.in_width + ix) * params.channels + c;
  let out_idx = (oy * out_width + ox) * params.channels + c;
  output_tex[out_idx] = input_tex[in_idx];
}

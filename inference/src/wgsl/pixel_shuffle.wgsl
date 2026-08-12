// Pixel-shuffle / depth-to-space, upscale factor 2, NHWC. Matches PyTorch's
// nn.PixelShuffle(2): output[2h+i, 2w+j, c] = input[h, w, c*4 + i*2 + j].
// The head conv's output channel ordering was preserved exactly from
// PyTorch during weight extraction (see extract_weights.py), so this
// mapping lines up with the trained weights with no extra permutation.

struct Params {
  in_width: u32,
  in_height: u32,
  out_channels: u32, // final channel count (3); input has out_channels * 4
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
  if (ox >= out_width || oy >= out_height || c >= params.out_channels) {
    return;
  }
  let ix = ox / 2u;
  let iy = oy / 2u;
  let i = oy % 2u;
  let j = ox % 2u;
  let in_channels = params.out_channels * 4u;
  let in_c = c * 4u + i * 2u + j;
  let in_idx = (iy * params.in_width + ix) * in_channels + in_c;
  let out_idx = (oy * out_width + ox) * params.out_channels + c;
  output_tex[out_idx] = input_tex[in_idx];
}

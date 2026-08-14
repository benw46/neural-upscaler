// Bilinear downsample of the network's previous raw output down to the true
// 540x960 low-res grid, matching PyTorch's F.interpolate(mode="bilinear",
// align_corners=False) exactly -- this is the first step of
// train_temporal.py's unroll_sequence (`prev_output_lowres =
// F.interpolate(prev_output_highres, size=(PATCH_SIZE, PATCH_SIZE))`),
// reimplemented in WGSL for the live viewer since nothing in the codebase
// needed a GPU version of it before now.
//
// The source buffer physically has 1088 rows (OUT_H_PAD, the network's own
// padded output height) but only the first 1080 are real image content --
// the rest is reflect-pad artifact from the *input* side's pad_to_multiple,
// carried through to the output. `high_height` below is passed as 1080, not
// 1088, so those rows are never sampled -- same "top-left crop undoes the
// reflect pad" rule viewer.ts's drawNetworkOutput already follows for
// display.
//
// Edge-replicated (clamped) at the borders -- standard image-resize
// behaviour, not zero-padded. Zero-padding is warp_and_disocclude.wgsl's
// job for the reprojection step that follows this one, a different
// operation with different padding semantics -- see that file.

struct Params {
  high_width: u32,  // 1920
  high_height: u32, // 1080 -- true, cropped height (see header comment)
  low_width: u32,   // 960
  low_height: u32,  // 540
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> prev_output: array<Scalar>;       // (1088, 1920, 3) -- only rows [0, high_height) read
@group(0) @binding(2) var<storage, read_write> prev_lowres: array<Scalar>; // (low_height, low_width, 3)

fn sample_channel(ix: u32, iy: u32, c: u32) -> f32 {
  let cx = min(ix, params.high_width - 1u);
  let cy = min(iy, params.high_height - 1u);
  let idx = (cy * params.high_width + cx) * 3u + c;
  return f32(prev_output[idx]);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ox = gid.x;
  let oy = gid.y;
  if (ox >= params.low_width || oy >= params.low_height) {
    return;
  }

  let scale_x = f32(params.high_width) / f32(params.low_width);
  let scale_y = f32(params.high_height) / f32(params.low_height);
  let in_x = (f32(ox) + 0.5) * scale_x - 0.5;
  let in_y = (f32(oy) + 0.5) * scale_y - 0.5;

  let x0f = floor(in_x);
  let y0f = floor(in_y);
  let fx = in_x - x0f;
  let fy = in_y - y0f;
  let x0 = u32(max(x0f, 0.0));
  let y0 = u32(max(y0f, 0.0));
  let x1 = x0 + 1u;
  let y1 = y0 + 1u;

  let out_base = (oy * params.low_width + ox) * 3u;
  for (var c: u32 = 0u; c < 3u; c = c + 1u) {
    let v00 = sample_channel(x0, y0, c);
    let v10 = sample_channel(x1, y0, c);
    let v01 = sample_channel(x0, y1, c);
    let v11 = sample_channel(x1, y1, c);
    let top = mix(v00, v10, fx);
    let bottom = mix(v01, v11, fx);
    prev_lowres[out_base + c] = Scalar(mix(top, bottom, fy));
  }
}

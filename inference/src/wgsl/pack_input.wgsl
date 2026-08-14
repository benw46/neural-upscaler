// Assembles the model's 8-channel NHWC input buffer for one live frame:
// colour(3) + depth(1)/DEPTH_NORM from the G-buffer textures, plus
// warped-previous-output(3) + disocclusion(1) from warp_and_disocclude.wgsl's
// output -- reflect-padding both true-540-row sources up to the network's
// required 544 rows identically, the same "row H+k mirrors row H-2-k" rule
// viewer.ts's buildNetworkInput uses for the static demo (training/src/
// dataset.py's pad_to_multiple(mode="reflect")). Padding is applied
// uniformly across *all eight* channels here -- a deliberate, principled
// choice (not something any earlier part of this codebase had to decide,
// since the static demo never computed real warp/disocclusion channels to
// begin with): a padded row is treated as an exact copy of its mirrored true
// row's data for every channel alike, extending dataset.py's existing
// colour+depth-only convention rather than inventing a second one.

struct Params {
  in_width: u32,         // 960
  in_height: u32,        // 540 -- true, unpadded height
  in_height_padded: u32, // 544
  depth_norm: f32,       // training/src/dataset.py's DEPTH_NORM = 50.0
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var gb_colour: texture_2d<f32>;                      // rgba16float, true res
@group(0) @binding(2) var gb_depth: texture_2d<f32>;                       // r32float, true res
@group(0) @binding(3) var<storage, read> warped: array<Scalar>;            // (in_height, in_width, 4), true res -- warp_and_disocclude.wgsl's output
@group(0) @binding(4) var<storage, read_write> packed_input: array<Scalar>; // (in_height_padded, in_width, 8)

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y_padded = gid.y;
  if (x >= params.in_width || y_padded >= params.in_height_padded) {
    return;
  }

  // torch reflect padding: row H+k (k=0..pad-1) mirrors row H-2-k; the edge
  // row (H-1) is not repeated -- identical to buildNetworkInput's srcY rule.
  let src_y = select(params.in_height - 2u - (y_padded - params.in_height), y_padded, y_padded < params.in_height);

  let colour = textureLoad(gb_colour, vec2<i32>(i32(x), i32(src_y)), 0);
  let depth = textureLoad(gb_depth, vec2<i32>(i32(x), i32(src_y)), 0).r;
  let warp_base = (src_y * params.in_width + x) * 4u;

  let out_base = (y_padded * params.in_width + x) * 8u;
  packed_input[out_base + 0u] = Scalar(colour.r);
  packed_input[out_base + 1u] = Scalar(colour.g);
  packed_input[out_base + 2u] = Scalar(colour.b);
  packed_input[out_base + 3u] = Scalar(depth / params.depth_norm);
  packed_input[out_base + 4u] = warped[warp_base + 0u];
  packed_input[out_base + 5u] = warped[warp_base + 1u];
  packed_input[out_base + 6u] = warped[warp_base + 2u];
  packed_input[out_base + 7u] = warped[warp_base + 3u];
}

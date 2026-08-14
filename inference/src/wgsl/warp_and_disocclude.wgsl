// Reimplements training/src/warp.py's warp_previous_output (grid_sample,
// bilinear, "zeros" padding) and compute_disocclusion_mask (grid_sample,
// bilinear, "border" padding, DEPTH_REL_THRESHOLD=0.05) at the true 540x960
// resolution -- the single most fragile piece of new code in the live
// pipeline; get the UV convention wrong here and the temporal behaviour
// silently diverges from what the model was actually trained on, exactly
// the failure mode CLAUDE.md's fragile-logic list warns about hardest.
// Deliberately NOT re-derived from first principles: reuses the exact
// convention documented in warp.py's own header and independently verified
// against renderer/src/render/shaders.wgsl's actual motion-vector output
// (`motionVector = currUv - prevUv`, UV origin top-left/V-down, channel
// order (du, dv)) rather than assumed.
//
// align_corners=False grid_sample unnormalisation, algebraically simplified:
// warp.py's uv_to_sample_grid computes grid = uv*2-1, then PyTorch
// unnormalises via pixel = ((grid+1)*size-1)/2. Substituting: pixel =
// ((uv*2-1+1)*size-1)/2 = (2*uv*size-1)/2 = uv*size-0.5. Used directly below
// instead of computing the intermediate [-1,1] grid at all -- fewer steps,
// same result, verified by substitution not just asserted.
//
// Frame 0 (is_first_frame=1) skips all of this and writes the cold-start
// values directly -- zeroed warped-previous, fully-invalid disocclusion --
// exactly train_temporal.py's t==0 special case (and viewer.ts's existing
// static-demo cold start), not an approximation of it.

const DEPTH_REL_THRESHOLD: f32 = 0.05;

struct Params {
  width: u32,   // 960
  height: u32,  // 540
  is_first_frame: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> prev_lowres: array<Scalar>;      // (height, width, 3) -- downsample_prev.wgsl's output
@group(0) @binding(2) var curr_motion: texture_2d<f32>;                   // rg16float, (du, dv) = current_uv - previous_uv
@group(0) @binding(3) var curr_depth: texture_2d<f32>;                    // r32float, linear view-space depth, this frame
@group(0) @binding(4) var prev_depth: texture_2d<f32>;                    // r32float, snapshot of *last* frame's depth
@group(0) @binding(5) var<storage, read_write> out_warped: array<Scalar>; // (height, width, 4) -- [0..2] warped colour, [3] disocclusion mask

fn sample_prev_lowres_zeros(ix: i32, iy: i32, c: u32) -> f32 {
  if (ix < 0 || ix >= i32(params.width) || iy < 0 || iy >= i32(params.height)) {
    return 0.0; // "zeros" padding_mode -- an out-of-bounds corner contributes 0 to the bilinear sum, not a clamp
  }
  let idx = (u32(iy) * params.width + u32(ix)) * 3u + c;
  return f32(prev_lowres[idx]);
}

fn sample_depth_border(tex: texture_2d<f32>, ix: i32, iy: i32) -> f32 {
  let cx = clamp(ix, 0, i32(params.width) - 1);
  let cy = clamp(iy, 0, i32(params.height) - 1);
  return textureLoad(tex, vec2<i32>(cx, cy), 0).r; // "border" padding_mode -- clamp to the edge texel
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) {
    return;
  }
  let out_base = (y * params.width + x) * 4u;

  if (params.is_first_frame == 1u) {
    out_warped[out_base + 0u] = Scalar(0.0);
    out_warped[out_base + 1u] = Scalar(0.0);
    out_warped[out_base + 2u] = Scalar(0.0);
    out_warped[out_base + 3u] = Scalar(1.0); // fully invalid -- nothing to trust yet
    return;
  }

  let motion = textureLoad(curr_motion, vec2<i32>(i32(x), i32(y)), 0).rg; // (du, dv)
  let curr_uv = vec2<f32>((f32(x) + 0.5) / f32(params.width), (f32(y) + 0.5) / f32(params.height));
  let prev_uv = curr_uv - motion; // mv = curr - prev  =>  prev = curr - mv

  let offscreen = prev_uv.x < 0.0 || prev_uv.x > 1.0 || prev_uv.y < 0.0 || prev_uv.y > 1.0;

  // Shared fractional pixel coordinate for both bilinear samples below --
  // the same prev_uv drives both the colour warp and the depth
  // reprojection, exactly as warp.py computes prev_uv once and reuses it.
  let px = prev_uv.x * f32(params.width) - 0.5;
  let py = prev_uv.y * f32(params.height) - 0.5;
  let x0 = i32(floor(px));
  let y0 = i32(floor(py));
  let x1 = x0 + 1;
  let y1 = y0 + 1;
  let fx = px - floor(px);
  let fy = py - floor(py);

  // Colour warp: bilinear, "zeros" padding.
  for (var c: u32 = 0u; c < 3u; c = c + 1u) {
    let v00 = sample_prev_lowres_zeros(x0, y0, c);
    let v10 = sample_prev_lowres_zeros(x1, y0, c);
    let v01 = sample_prev_lowres_zeros(x0, y1, c);
    let v11 = sample_prev_lowres_zeros(x1, y1, c);
    let top = mix(v00, v10, fx);
    let bottom = mix(v01, v11, fx);
    out_warped[out_base + c] = Scalar(mix(top, bottom, fy));
  }

  // Disocclusion: reproject *previous* depth through the same prev_uv,
  // "border" padding, compare against this frame's own depth at (x, y).
  let d00 = sample_depth_border(prev_depth, x0, y0);
  let d10 = sample_depth_border(prev_depth, x1, y0);
  let d01 = sample_depth_border(prev_depth, x0, y1);
  let d11 = sample_depth_border(prev_depth, x1, y1);
  let dtop = mix(d00, d10, fx);
  let dbottom = mix(d01, d11, fx);
  let reprojected_prev_depth = mix(dtop, dbottom, fy);

  let curr_d = textureLoad(curr_depth, vec2<i32>(i32(x), i32(y)), 0).r;
  let rel_diff = abs(reprojected_prev_depth - curr_d) / max(curr_d, 1e-4);
  let depth_mismatch = rel_diff > DEPTH_REL_THRESHOLD;

  out_warped[out_base + 3u] = Scalar(select(0.0, 1.0, offscreen || depth_mismatch));
}

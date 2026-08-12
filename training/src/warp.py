"""Differentiable reprojection (warp) + disocclusion masking for temporal
training (Spec 3 steps 1-2).

Reuses the exact motion-vector convention validated in Spec 1
(renderer/src/render/shaders.wgsl, renderer/src/validate/reproject.ts):

    mv = current_uv - previous_uv   (UV space, origin top-left, V down)
    previous_uv = current_uv - mv

Deliberately does NOT re-derive this from first principles here -- Spec 1's
reprojection validation gate already proved this convention correct via an
independent (non-differentiable, NumPy-based) implementation. Re-deriving it
independently here would risk arriving at a *different* but self-consistent
convention that silently disagrees with the dataset's actual motion vectors.

Both the warp and the disocclusion mask run at *low* (input) resolution --
the model's previous *high-res* output is downsampled first, then warped
using the motion field at its native resolution. This is simpler and cheaper
than upsampling the motion field to high-res and warping there (which
Spec 1's own capture pipeline never needed, since it only ever warped
same-resolution frames for validation) -- the two approaches aren't
perfectly equivalent (resampling doesn't commute), but the difference is
well within the noise the network already has to be robust to.
"""

import torch
import torch.nn.functional as F

# Relative depth-mismatch threshold for disocclusion detection -- same value
# used in Spec 1's validate/reproject.ts, empirical (see docs/PHASE-0-1-SUMMARY.md).
DEPTH_REL_THRESHOLD = 0.05


def _current_uv_grid(height: int, width: int, device, dtype) -> torch.Tensor:
    """Pixel-centre UV coordinates, shape (H, W, 2), last dim (u, v)."""
    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2) -- (u, v)


def uv_to_sample_grid(prev_uv: torch.Tensor) -> torch.Tensor:
    """(B, H, W, 2) UV in [0,1] -> normalised grid_sample coordinates in
    [-1,1]. UV's (u, v) already matches grid_sample's expected (x, y) order
    for an (N, C, H, W) tensor (x horizontal, y vertical, y-down)."""
    return prev_uv * 2 - 1


def upsample_motion(motion: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Upsamples a motion field to a finer spatial grid. Valid because
    motion vectors are stored in UV space (dimensionless, resolution-
    independent values) -- upsampling just resamples the same continuous
    field onto more sample points, no magnitude rescaling needed (unlike
    pixel-space motion vectors, which would need scaling by the resolution
    ratio)."""
    return F.interpolate(motion, size=size, mode="bilinear", align_corners=False)


def warp_previous_output(prev_output: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
    """Warps `prev_output` (B, C, H, W) into the current frame's viewpoint
    using `motion` (B, 2, H, W) at the *same* resolution as `prev_output` --
    upsample with `upsample_motion` first if warping at a resolution other
    than motion's native (stored) one. Returns (B, C, H, W). Zero-padded
    where the sample falls off-screen (a real disocclusion case -- the
    disocclusion mask below should also flag these so the network doesn't
    trust the resulting zeros as real history)."""
    b, _, h, w = prev_output.shape
    curr_uv = _current_uv_grid(h, w, prev_output.device, prev_output.dtype)
    curr_uv = curr_uv.unsqueeze(0).expand(b, -1, -1, -1)  # (B, H, W, 2)

    motion_uv = motion.permute(0, 2, 3, 1)  # (B, H, W, 2)
    prev_uv = curr_uv - motion_uv  # mv = curr - prev  =>  prev = curr - mv

    grid = uv_to_sample_grid(prev_uv)
    return F.grid_sample(prev_output, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def compute_disocclusion_mask(prev_depth: torch.Tensor, curr_depth: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
    """1.0 where history is invalid (disoccluded or reprojects off-screen),
    0.0 where it's valid to trust -- same criterion as Spec 1's
    validate/reproject.ts. Returns (B, 1, H, W).

    `prev_depth`/`curr_depth`: (B, 1, H, W) linear view-space depth (see
    dataset.py's DEPTH_NORM -- either raw or normalised is fine here since
    only the *relative* difference matters, but both must use the same
    scale)."""
    b, _, h, w = curr_depth.shape
    curr_uv = _current_uv_grid(h, w, curr_depth.device, curr_depth.dtype)
    curr_uv = curr_uv.unsqueeze(0).expand(b, -1, -1, -1)
    motion_uv = motion.permute(0, 2, 3, 1)
    prev_uv = curr_uv - motion_uv

    offscreen = (prev_uv[..., 0] < 0) | (prev_uv[..., 0] > 1) | (prev_uv[..., 1] < 0) | (prev_uv[..., 1] > 1)

    grid = uv_to_sample_grid(prev_uv)
    reprojected_prev_depth = F.grid_sample(prev_depth, grid, mode="bilinear", padding_mode="border", align_corners=False)

    rel_diff = (reprojected_prev_depth - curr_depth).abs() / curr_depth.clamp(min=1e-4)
    depth_mismatch = rel_diff.squeeze(1) > DEPTH_REL_THRESHOLD

    disoccluded = (offscreen | depth_mismatch).to(curr_depth.dtype)
    return disoccluded.unsqueeze(1)  # (B, 1, H, W)


if __name__ == "__main__":
    # Sanity check: a static scene (motion vectors near zero) should warp a
    # frame into itself with minimal difference wherever depth is
    # continuous, and a large synthetic depth jump should trigger disocclusion.
    torch.manual_seed(0)
    b, h, w = 1, 32, 32
    frame = torch.rand(b, 3, h, w)
    zero_motion = torch.zeros(b, 2, h, w)

    warped = warp_previous_output(frame, zero_motion)
    err = (warped - frame).abs().mean().item()
    print(f"zero-motion self-warp mean abs error: {err:.5f} (should be near 0)")
    assert err < 0.01

    depth = torch.ones(b, 1, h, w) * 10.0
    mask_static = compute_disocclusion_mask(depth, depth, zero_motion)
    print(f"static-depth disocclusion fraction: {mask_static.mean().item():.4f} (should be ~0)")
    assert mask_static.mean().item() < 0.05

    depth_jump = depth.clone()
    depth_jump[:, :, h // 2 :, :] = 1.0  # sudden near-camera object in the bottom half
    mask_jump = compute_disocclusion_mask(depth, depth_jump, zero_motion)
    print(f"depth-jump disocclusion fraction: {mask_jump.mean().item():.4f} (should be ~0.5)")
    assert mask_jump.mean().item() > 0.4

    print("all checks passed")

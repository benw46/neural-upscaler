"""Diff fixture for warp_and_disocclude.wgsl against training/src/warp.py's
warp_previous_output + compute_disocclusion_mask -- the one WGSL kernel in
the live pipeline that has never been run through this project's per-layer
PyTorch-diff validation chain (see live_pipeline.ts's module docstring,
"known gap"). Same purpose and pattern as gen_diff_fixtures_temporal.py, for
this kernel instead of the U-Net.

H/W deliberately unequal (not the U-Net's square PATCH_SIZE) so a stray
x/y or width/height transposition in the WGSL port would show up as a
diff rather than passing by accident on a square grid.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training" / "src"))
from warp import compute_disocclusion_mask, warp_previous_output  # noqa: E402

H, W = 68, 96
SEED = 7
OUT_DIR = Path(__file__).resolve().parent


def main():
    torch.manual_seed(SEED)

    prev_lowres = torch.rand(1, 3, H, W)

    # Motion magnitude large enough that pixels near the UV border land
    # outside [0,1] a meaningful fraction of the time -- exercises the
    # "zeros" padding path in warp_previous_output and the offscreen branch
    # in compute_disocclusion_mask, not just the interior bilinear case.
    motion = (torch.rand(1, 2, H, W) - 0.5) * 0.3

    curr_depth = 1.0 + torch.rand(1, 1, H, W) * 49.0  # plausible linear depth, ~[1, 50]
    prev_depth = curr_depth + torch.randn(1, 1, H, W) * 0.1  # continuous case: small noise, should stay under DEPTH_REL_THRESHOLD
    jump_mask = torch.rand(1, 1, H, W) < 0.25
    prev_depth = torch.where(jump_mask, curr_depth * 4.0, prev_depth)  # forced depth-mismatch case, real coverage of that branch

    with torch.no_grad():
        warped = warp_previous_output(prev_lowres, motion)  # (1, 3, H, W)
        mask = compute_disocclusion_mask(prev_depth, curr_depth, motion)  # (1, 1, H, W)

    # (H, W, 4) = [warped rgb, mask] -- matches warp_and_disocclude.wgsl's
    # out_warped storage-buffer layout exactly, so the TS harness can diff
    # against this file with no further reshaping.
    combined = torch.cat([warped, mask], dim=1).permute(0, 2, 3, 1).squeeze(0).contiguous()

    prev_lowres_nhwc = prev_lowres.permute(0, 2, 3, 1).squeeze(0).contiguous().numpy().astype(np.float32)
    motion_hw2 = motion.permute(0, 2, 3, 1).squeeze(0).contiguous().numpy().astype(np.float32)
    curr_depth_hw = curr_depth.squeeze(0).squeeze(0).contiguous().numpy().astype(np.float32)
    prev_depth_hw = prev_depth.squeeze(0).squeeze(0).contiguous().numpy().astype(np.float32)
    combined_np = combined.numpy().astype(np.float32)

    prev_lowres_nhwc.tofile(OUT_DIR / "warp_prev_lowres.bin")
    motion_hw2.tofile(OUT_DIR / "warp_motion.bin")
    curr_depth_hw.tofile(OUT_DIR / "warp_curr_depth.bin")
    prev_depth_hw.tofile(OUT_DIR / "warp_prev_depth.bin")
    combined_np.tofile(OUT_DIR / "pytorch_output_warp.bin")

    print(f"H={H} W={W}")
    print(f"prev_lowres: {prev_lowres_nhwc.shape} -> warp_prev_lowres.bin")
    print(f"motion:      {motion_hw2.shape} -> warp_motion.bin ({(motion_hw2 < 0).mean():.3f} negative fraction)")
    print(f"curr_depth:  {curr_depth_hw.shape} -> warp_curr_depth.bin")
    print(f"prev_depth:  {prev_depth_hw.shape} -> warp_prev_depth.bin (forced-jump fraction={jump_mask.float().mean().item():.3f})")
    print(f"output:      {combined_np.shape} -> pytorch_output_warp.bin")
    print(f"reference disocclusion fraction: {mask.mean().item():.4f}")


if __name__ == "__main__":
    main()

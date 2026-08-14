"""Combined L1 + LPIPS loss.

Why L1 alone is insufficient (Spec 2 step 2): for any given low-res input,
several different high-res reconstructions of the aliased/ambiguous fine
detail are equally plausible -- there's no single correct answer for exactly
where a sub-pixel edge "really" was. A pixelwise loss like L1 is minimised by
predicting something close to the *average* of those plausible
reconstructions, not any single sharp one -- so a network trained on L1
alone learns to hedge, producing soft/blurry output even when it has
correctly localised the underlying structure. This is well documented in the
super-resolution literature (SRGAN/ESRGAN and others motivate exactly this
with perceptual/adversarial losses). LPIPS evaluates similarity in a
pretrained deep feature space that correlates much better with human
judgements of sharpness and texture than per-pixel distance does, so it
pulls the network away from the "safe blurry average" and toward plausible
sharp detail. L1 is kept as the dominant term regardless -- LPIPS alone can
hallucinate plausible-looking but structurally wrong detail; L1 anchors the
prediction to the actual target.
"""

import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    def __init__(self, l1_weight: float = 1.0, lpips_weight: float = 0.1, lpips_net: str = "vgg", lpips_scale: float = 1.0):
        """`lpips_scale`: downsamples pred/target by this factor before
        computing LPIPS specifically -- L1 always stays at full resolution.
        1.0 = no downsampling (the original behaviour). LPIPS's VGG16
        backbone is ~150x this model's own parameter count and dominates
        training step time regardless of everything else done to speed up
        training: measured at ~89% of a full step, even under bf16
        autocast. lpips_scale=0.5 measured a 2.8x full-step speedup, at the
        cost of one octave of LPIPS's own multi-scale pyramid -- it loses
        the single finest detail level LPIPS would otherwise see, every
        coarser scale is unchanged. This IS a real change to what the loss
        can supervise (LPIPS's whole purpose is pushing the network toward
        sharp detail -- see this class's module docstring), not a free win
        like mixed-precision training was -- validate finished-model
        quality (sharpness, not just speed) before trusting a value other
        than 1.0 for real training, don't assume the timing win is free."""
        super().__init__()
        self.l1 = nn.L1Loss()
        self.lpips = lpips.LPIPS(net=lpips_net)
        for p in self.lpips.parameters():
            p.requires_grad = False
        self.l1_weight = l1_weight
        self.lpips_weight = lpips_weight
        self.lpips_scale = lpips_scale

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        l1_val = self.l1(pred, target)
        # LPIPS expects inputs in [-1, 1]; our images are [0, 1].
        if self.lpips_scale != 1.0:
            size = (max(1, round(pred.shape[-2] * self.lpips_scale)), max(1, round(pred.shape[-1] * self.lpips_scale)))
            pred_lp = F.interpolate(pred, size=size, mode="bilinear", align_corners=False)
            target_lp = F.interpolate(target, size=size, mode="bilinear", align_corners=False)
        else:
            pred_lp, target_lp = pred, target
        lpips_val = self.lpips(pred_lp * 2 - 1, target_lp * 2 - 1).mean()
        total = self.l1_weight * l1_val + self.lpips_weight * lpips_val
        return total, {"l1": l1_val.item(), "lpips": lpips_val.item(), "total": total.item()}


def temporal_consistency_loss(pred: torch.Tensor, warped_prev: torch.Tensor, disocclusion_mask: torch.Tensor) -> torch.Tensor:
    """Spec 3 step 4: penalise frame-to-frame difference in regions motion
    vectors mark as static.

    Read literally, "static" would mean near-zero motion vectors -- but with
    a moving camera almost nothing has *exactly* zero screen-space motion,
    so that reading would apply this term almost nowhere. Interpreted
    instead (a judgement call, documented rather than silently assumed) as
    "regions where the motion vectors correctly describe correspondence
    with the previous frame" -- i.e. wherever `disocclusion_mask` says
    history is valid, whether the underlying surface is truly static or
    just tracked correctly through camera-induced motion. This is the
    standard target for temporal-consistency terms in the TAA/temporal-SR
    literature: penalise the network for changing its answer about a pixel
    whose correct value it already established, without penalising it at
    real disocclusions where the old answer is no longer valid at all.

    `pred`, `warped_prev`: (B, 3, H, W), same resolution (typically full
    output resolution, not the downsampled version fed as network input --
    flicker is most visible at the resolution actually displayed).
    `disocclusion_mask`: (B, 1, h, w) at any resolution; upsampled
    (nearest, since it's a boolean-ish mask) to match if needed.
    """
    if disocclusion_mask.shape[-2:] != pred.shape[-2:]:
        disocclusion_mask = F.interpolate(disocclusion_mask, size=pred.shape[-2:], mode="nearest")
    valid = (1.0 - disocclusion_mask).expand_as(pred)  # broadcast to match pred's channel count
    diff = (pred - warped_prev).abs()
    return (diff * valid).sum() / valid.sum().clamp(min=1.0)


if __name__ == "__main__":
    loss_fn = CombinedLoss()
    pred = torch.rand(2, 3, 64, 64)
    target = torch.rand(2, 3, 64, 64)
    total, parts = loss_fn(pred, target)
    print(f"total={total.item():.4f}  parts={parts}")

    # Loss against itself should be ~0 (L1 exactly 0; LPIPS near 0 but not
    # exactly, since its internal normalisation isn't perfectly idempotent).
    total_self, parts_self = loss_fn(pred, pred)
    print(f"self-comparison: total={total_self.item():.6f}  parts={parts_self}")
    assert parts_self["l1"] < 1e-6, "L1 against itself must be exactly zero"

    # Temporal consistency: fully-valid mask should equal plain L1; a mask
    # that's all-disoccluded should return 0 (nothing to penalise).
    mask_valid = torch.zeros(2, 1, 64, 64)
    mask_disoccluded = torch.ones(2, 1, 64, 64)
    t_valid = temporal_consistency_loss(pred, target, mask_valid).item()
    t_disoccluded = temporal_consistency_loss(pred, target, mask_disoccluded).item()
    plain_l1 = (pred - target).abs().mean().item()
    print(f"temporal loss (fully valid mask): {t_valid:.5f}  (plain L1: {plain_l1:.5f})")
    print(f"temporal loss (fully disoccluded mask): {t_disoccluded:.5f} (should be 0)")
    assert abs(t_valid - plain_l1) < 1e-5
    assert t_disoccluded < 1e-6

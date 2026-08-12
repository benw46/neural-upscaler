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


class CombinedLoss(nn.Module):
    def __init__(self, l1_weight: float = 1.0, lpips_weight: float = 0.1, lpips_net: str = "vgg"):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.lpips = lpips.LPIPS(net=lpips_net)
        for p in self.lpips.parameters():
            p.requires_grad = False
        self.l1_weight = l1_weight
        self.lpips_weight = lpips_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        l1_val = self.l1(pred, target)
        # LPIPS expects inputs in [-1, 1]; our images are [0, 1].
        lpips_val = self.lpips(pred * 2 - 1, target * 2 - 1).mean()
        total = self.l1_weight * l1_val + self.lpips_weight * lpips_val
        return total, {"l1": l1_val.item(), "lpips": lpips_val.item(), "total": total.item()}


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

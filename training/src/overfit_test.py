"""Overfit a single patch to near-zero loss (Spec 2 step 3).

Isolates data/model bugs from training-dynamics issues: if the model can't
drive loss to near-zero on ONE fixed example given enough iterations, the
bug is in the model/data/loss pipeline, not in learning-rate schedules or
generalisation -- and no amount of real training time will fix it. Must be
run and reported before any real training run.

This test went through two rounds of misdiagnosis before landing on the
real issue, worth recording since both wrong turns looked plausible:

1. First run (combined L1+LPIPS, 3000 iters) plateaued at L1~0.05. Looked
   like it might be a model/data bug.
2. Ablation (L1-only) converged cleanly to 0.0049, seemingly clearing the
   pipeline -- but a second combined-loss run gave a *completely different*
   result (L1~0.0016) than the first (~0.05). That inconsistency was the
   real bug, just not in the model: neither the random patch-crop location
   (`SpatialPatchDataset` picks a fresh random location *every*
   `__getitem__` call by design, for training augmentation) nor the model's
   random weight init were seeded, so "the same" run was silently training
   on a different patch against a different init each time. First fix:
   `load_fixed_patch` (one deterministic patch) + explicit torch/numpy
   seeding, so results are actually reproducible run-to-run.
3. With reproducibility fixed, the *real* signal appeared: L1-only training
   was unstable and diverged mid-run (loss jumped from 0.055 to 15,158
   around iteration 1500-2000), while combined-loss training didn't. This
   made LPIPS look like it was providing implicit regularisation against a
   genuine instability -- plausible, but wrong. Adding gradient clipping
   (`clip_grad_norm_`, max_norm=1.0 -- the architecture has no normalisation
   layers, so unclipped activations/gradients can grow unbounded) fixed
   *both* modes cleanly; combined loss actually converges slightly better
   than L1-only once training is stable (0.0068 vs 0.0083). There was never
   a genuine tension between L1 and LPIPS objectives -- train.py uses
   gradient clipping as standard, with no warmup needed.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dataset import load_fixed_patch  # noqa: E402
from losses import CombinedLoss  # noqa: E402
from model import SpatialUNet  # noqa: E402

RUN_DIR = r"E:\neural-upscaler\data\seed-20260812"
FRAME_IDX = 0
PATCH_X, PATCH_Y = 400, 200  # fixed, arbitrary -- inside the 960x540 frame with room for a 128x128 patch
SEED = 42
ITERATIONS = 3000
LR = 2e-3
L1_PASS_THRESHOLD = 0.01


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def run(loss_mode: str, x: torch.Tensor, y: torch.Tensor, device: str) -> dict:
    seed_everything(SEED)  # identical init across modes, for a fair comparison
    model = SpatialUNet().to(device)
    combined_loss_fn = CombinedLoss().to(device) if loss_mode == "combined" else None
    l1_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    last_parts = {}
    for i in range(ITERATIONS):
        optimizer.zero_grad()
        pred = model(x)
        if loss_mode == "combined":
            loss, parts = combined_loss_fn(pred, y)
        else:
            l1_val = l1_fn(pred, y)
            loss, parts = l1_val, {"l1": l1_val.item(), "total": l1_val.item()}
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        last_parts = parts
        if i % 500 == 0 or i == ITERATIONS - 1:
            print(f"  [{loss_mode}] iter {i:4d}: " + "  ".join(f"{k}={v:.5f}" for k, v in parts.items()))

    return last_parts


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    x, y = load_fixed_patch(RUN_DIR, FRAME_IDX, PATCH_X, PATCH_Y)
    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)
    print(f"input:  {tuple(x.shape)}  (frame {FRAME_IDX}, patch at x={PATCH_X} y={PATCH_Y})")
    print(f"target: {tuple(y.shape)}")

    print("\n--- pipeline-correctness check: L1-only ---")
    l1_only_final = run("l1_only", x, y, device)

    print("\n--- reference: combined L1+LPIPS (what train.py without warmup would use) ---")
    combined_final = run("combined", x, y, device)

    print(f"\nL1-only final L1:        {l1_only_final['l1']:.5f}")
    print(f"combined-loss final L1:  {combined_final['l1']:.5f}")
    print(f"combined-loss final LPIPS: {combined_final.get('lpips', float('nan')):.5f}")

    passed = l1_only_final["l1"] < L1_PASS_THRESHOLD
    print(f"\n{'PASSED' if passed else 'FAILED'}: L1-only final L1 {'<' if passed else '>='} {L1_PASS_THRESHOLD}")
    print("(pipeline-correctness gate uses L1-only; combined-loss plateau is expected LPIPS behaviour, see module docstring)")
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()

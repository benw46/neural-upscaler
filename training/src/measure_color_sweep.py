"""One-off comparison: does raising lpips_weight reduce the L1-driven colour
desaturation / highlight-darkening diagnosed on the deployed model? Runs the
baseline (l1=1.0, lpips=0.1) and both sweep checkpoints (lpips02: lpips=0.2,
lpips04: lpips=0.4) on genuinely held-out frames (1700-1999, never seen in
training -- see dataset.py's train_val_split) and bins chroma/luma error
against ground-truth chroma/luma, same methodology as the original frame
1650 diagnosis.

Not part of the training pipeline -- a throwaway analysis script, not
imported anywhere.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dataset import FullFrameDataset, pad_to_multiple, crop_to_size, train_val_split  # noqa: E402
from model import SpatialUNet  # noqa: E402

RUN_DIR = r"E:\neural-upscaler\data\seed-20260812"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
N_SAMPLE_FRAMES = 10
BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def chroma(img: np.ndarray) -> np.ndarray:
    return img.max(axis=-1) - img.min(axis=-1)


def luma(img: np.ndarray) -> np.ndarray:
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


@torch.no_grad()
def run_checkpoint(ckpt_path: Path, val_ds: FullFrameDataset, frame_idxs: list[int], device: str):
    model = SpatialUNet().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    pred_chroma, gt_chroma, pred_luma, gt_luma = [], [], [], []
    for i in frame_idxs:
        x, y = val_ds[i]
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)
        x_padded, orig_size = pad_to_multiple(x)
        pred = model(x_padded)
        pred = crop_to_size(pred, (orig_size[0] * 2, orig_size[1] * 2)).clamp(0, 1)

        pred_np = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
        gt_np = y.squeeze(0).permute(1, 2, 0).cpu().numpy()

        pred_chroma.append(chroma(pred_np).ravel())
        gt_chroma.append(chroma(gt_np).ravel())
        pred_luma.append(luma(pred_np).ravel())
        gt_luma.append(luma(gt_np).ravel())

    return (np.concatenate(pred_chroma), np.concatenate(gt_chroma), np.concatenate(pred_luma), np.concatenate(gt_luma))


def binned_gap(pred: np.ndarray, gt: np.ndarray, bins: list[float]) -> list[str]:
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (gt >= lo) & (gt < hi)
        if mask.sum() == 0:
            out.append(f"{lo:.1f}-{hi:.1f}: (empty)")
            continue
        gap = (pred[mask] - gt[mask]).mean()
        out.append(f"{lo:.1f}-{hi:.1f}: {gap:+.4f} (n={mask.sum()})")
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    _, val_idx = train_val_split(RUN_DIR)
    frame_idxs = list(np.linspace(0, len(val_idx) - 1, N_SAMPLE_FRAMES, dtype=int))
    real_frame_numbers = [val_idx[i] for i in frame_idxs]
    print(f"held-out val block: frames {val_idx[0]}-{val_idx[-1]} ({len(val_idx)} total)")
    print(f"sampling {N_SAMPLE_FRAMES} frames: {real_frame_numbers}\n")

    val_ds = FullFrameDataset(RUN_DIR, val_idx)

    checkpoints = {
        "baseline (l1=1.0, lpips=0.1)": CHECKPOINT_DIR / "best.pt",
        "lpips02  (l1=1.0, lpips=0.2)": CHECKPOINT_DIR / "lpips02_best.pt",
        "lpips04  (l1=1.0, lpips=0.4)": CHECKPOINT_DIR / "lpips04_best.pt",
    }

    results = {}
    for label, path in checkpoints.items():
        if not path.exists():
            print(f"SKIP {label}: {path} not found")
            continue
        print(f"running {label}...")
        results[label] = run_checkpoint(path, val_ds, frame_idxs, device)

    print("\n=== CHROMA gap (pred - gt), binned by gt chroma ===")
    for label, (pred_c, gt_c, _, _) in results.items():
        print(f"\n{label}:")
        for line in binned_gap(pred_c, gt_c, BINS):
            print(f"  {line}")

    print("\n=== LUMA gap (pred - gt), binned by gt luma ===")
    for label, (_, _, pred_l, gt_l) in results.items():
        print(f"\n{label}:")
        for line in binned_gap(pred_l, gt_l, BINS):
            print(f"  {line}")


if __name__ == "__main__":
    main()

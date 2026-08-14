"""Temporal-model counterpart to measure_color_sweep.py -- that script only
ever ran the chroma/luma desaturation diagnostic against the Phase 2
spatial checkpoints (checkpoints/*.pt), never against checkpoints_temporal/,
even though the whole reason the deployed temporal model uses
lpips_weight=0.2 instead of the original 0.1 was to address this same
diagnosed colour issue (see docs/PHASE-3-SUMMARY.md, notebook/). That
premise was never actually checked on the model it was meant to fix.

Unlike the spatial version's 10 single-frame cold-start predictions, this
runs a genuine recurrent rollout (same mechanics as test_long_sequence.py:
no ground-truth teacher-forcing, no BPTT, real warped-previous-output
feedback) over the full held-out calm window, and bins chroma/luma error
across every frame in that rollout -- a much larger, and more
deployment-realistic, sample than the spatial script's approach.

Round 2 (2026-08-14): evaluates all four temporal checkpoints -- the three
originally trained on the grayscale scene, plus the new colour_tw0.75
(trained on the seed-20260812-colored dataset) -- against the SAME
held-out COLOURED frames, not each on its own training distribution. That
isolates the actual question: does training on colour reduce desaturation,
evaluated on real colour content, rather than each model's own (possibly
easier or harder) held-out split. The original seed-20260812 (grayscale)
dataset this script first ran against has since been deleted (its job was
done -- the three grayscale-trained checkpoints already exist and don't
need the source data to run inference); re-pointed at the colour dataset
for this round.

Not part of the training pipeline -- a throwaway analysis script, not
imported anywhere.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from dataset import DEPTH_NORM, crop_to_size, pad_to_multiple, train_val_split  # noqa: E402
from model import SpatialUNet  # noqa: E402
from warp import compute_disocclusion_mask, warp_previous_output  # noqa: E402

RUN_DIR = Path(r"E:\neural-upscaler\data\seed-20260812-colored")
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints_temporal"
SEQUENCE_LENGTH = 350  # same calm held-out window test_long_sequence.py's default gate uses
BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def chroma(img: np.ndarray) -> np.ndarray:
    return img.max(axis=-1) - img.min(axis=-1)


def luma(img: np.ndarray) -> np.ndarray:
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


class BinnedAccumulator:
    """Running per-bin sum/count of (pred-gt), instead of concatenating
    every frame's full-resolution pixel array and computing the mean at
    the end -- holding 349 frames x 4 arrays x 1920x1080 pixels x 3
    checkpoints simultaneously (the original version of this script) is
    ~35GB, more than this machine's 32GB RAM. Mathematically identical
    result (mean = sum/count either way); each frame's temporary arrays
    go out of scope and get freed immediately after updating the running
    totals, so peak memory stays a handful of floats regardless of how
    many frames or checkpoints are processed."""

    def __init__(self, bins: list[float]):
        self.bins = bins
        n = len(bins) - 1
        self.sum = np.zeros(n, dtype=np.float64)
        self.count = np.zeros(n, dtype=np.int64)

    def update(self, pred: np.ndarray, gt: np.ndarray):
        for idx, (lo, hi) in enumerate(zip(self.bins[:-1], self.bins[1:])):
            mask = (gt >= lo) & (gt < hi)
            n = int(mask.sum())
            if n == 0:
                continue
            self.sum[idx] += (pred[mask] - gt[mask]).sum()
            self.count[idx] += n

    def report(self) -> list[str]:
        out = []
        for idx, (lo, hi) in enumerate(zip(self.bins[:-1], self.bins[1:])):
            if self.count[idx] == 0:
                out.append(f"{lo:.1f}-{hi:.1f}: (empty)")
                continue
            gap = self.sum[idx] / self.count[idx]
            out.append(f"{lo:.1f}-{hi:.1f}: {gap:+.4f} (n={self.count[idx]})")
        return out


def load_frame(frame_idx: int, input_w: int, input_h: int, gt_w: int, gt_h: int):
    fname = f"{frame_idx:06d}.bin"
    color = np.fromfile(RUN_DIR / "color" / fname, dtype=np.float16).reshape(input_h, input_w, 4)[:, :, :3].astype(np.float32)
    depth = np.fromfile(RUN_DIR / "depth" / fname, dtype=np.float32).reshape(input_h, input_w, 1) / DEPTH_NORM
    motion = np.fromfile(RUN_DIR / "motion" / fname, dtype=np.float16).reshape(input_h, input_w, 2).astype(np.float32)
    gt = np.fromfile(RUN_DIR / "gt_color" / fname, dtype=np.float16).reshape(gt_h, gt_w, 4)[:, :, :3].astype(np.float32)
    return color, depth, motion, gt


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


@torch.no_grad()
def run_checkpoint(ckpt_path: Path, frame_range: range, header: dict, device: str):
    input_w, input_h = header["inputWidth"], header["inputHeight"]
    gt_w, gt_h = header["gtWidth"], header["gtHeight"]

    model = SpatialUNet(in_channels=8).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    prev_output_highres = None
    prev_depth = None
    chroma_acc = BinnedAccumulator(BINS)
    luma_acc = BinnedAccumulator(BINS)

    for i, frame_idx in enumerate(frame_range):
        color, depth, motion, gt = load_frame(frame_idx, input_w, input_h, gt_w, gt_h)
        color_t = to_tensor(color).to(device)
        depth_t = to_tensor(depth).to(device)
        motion_t = to_tensor(motion).to(device)
        gt_t = to_tensor(gt).to(device)

        input_padded, orig_size = pad_to_multiple(color_t)
        depth_padded, _ = pad_to_multiple(depth_t)
        motion_padded, _ = pad_to_multiple(motion_t)
        padded_h, padded_w = input_padded.shape[-2:]

        if i == 0:
            disocclusion_mask = torch.ones((1, 1, padded_h, padded_w), device=device)
            warped_prev_lowres = torch.zeros((1, 3, padded_h, padded_w), device=device)
        else:
            disocclusion_mask = compute_disocclusion_mask(prev_depth, depth_padded, motion_padded)
            prev_lowres = F.interpolate(prev_output_highres, size=(padded_h, padded_w), mode="bilinear", align_corners=False)
            warped_prev_lowres = warp_previous_output(prev_lowres, motion_padded)

        model_input = torch.cat([input_padded, depth_padded, warped_prev_lowres, disocclusion_mask], dim=1)
        pred_padded = model(model_input)
        pred = crop_to_size(pred_padded, (orig_size[0] * 2, orig_size[1] * 2)).clamp(0, 1)

        # Skip the cold-start frame (i==0) for the diagnostic -- it has no
        # real temporal history yet, same reasoning test_long_sequence.py's
        # own metrics apply from frame 1 onward.
        if i > 0:
            pred_np = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
            gt_np = gt_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
            chroma_acc.update(chroma(pred_np).ravel(), chroma(gt_np).ravel())
            luma_acc.update(luma(pred_np).ravel(), luma(gt_np).ravel())

        prev_output_highres = pred_padded.clamp(0, 1)  # matches training/test_long_sequence.py feedback convention
        prev_depth = depth_padded

    return chroma_acc, luma_acc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    header = json.loads((RUN_DIR / "dataset.json").read_text())
    _, val_idx = train_val_split(str(RUN_DIR), val_fraction=0.25)
    start_frame = val_idx[0]
    frame_range = range(start_frame, start_frame + SEQUENCE_LENGTH)
    print(f"rolling out frames {frame_range.start}..{frame_range.stop - 1} ({SEQUENCE_LENGTH} frames, held-out calm window)\n")

    checkpoints = {
        # temporal_w1.0_final_best.pt (the original lpips=0.1 baseline) was
        # accidentally overwritten by an unrelated smoke test earlier this
        # session -- colorbaseline_tw1.0_lpips01 is a freshly-retrained
        # equivalent (same recipe: tw=1.0, lpips=0.1) standing in for it.
        # All three below were trained on the grayscale scene -- this run
        # evaluates them out-of-distribution, on real colour content, which
        # is exactly the comparison that matters here.
        "colorbaseline (grayscale-trained, tw=1.0, lpips=0.1)": CHECKPOINT_DIR / "colorbaseline_tw1.0_lpips01_best.pt",
        "temporal_w1.0_lpips02 (grayscale-trained, tw=1.0, lpips=0.2)": CHECKPOINT_DIR / "temporal_w1.0_lpips02_best.pt",
        "sweep_tw0.75 (grayscale-trained, DEPLOYED, tw=0.75, lpips=0.2)": CHECKPOINT_DIR / "sweep_tw0.75_best.pt",
        "colour_tw0.75 (COLOUR-trained, tw=0.75, lpips=0.2)": CHECKPOINT_DIR / "colour_tw0.75_best.pt",
        "colour_tw0.75_sat0.5 (COLOUR-trained, tw=0.75, lpips=0.2, sat=0.5)": CHECKPOINT_DIR / "colour_tw0.75_sat0.5_best.pt",
        "colour_tw0.75_sat1.0 (COLOUR-trained, tw=0.75, lpips=0.2, sat=1.0)": CHECKPOINT_DIR / "colour_tw0.75_sat1.0_best.pt",
    }

    results = {}
    for label, path in checkpoints.items():
        if not path.exists():
            print(f"SKIP {label}: {path} not found")
            continue
        print(f"running {label}...")
        results[label] = run_checkpoint(path, frame_range, header, device)

    print("\n=== CHROMA gap (pred - gt), binned by gt chroma -- negative = desaturated ===")
    for label, (chroma_acc, _) in results.items():
        print(f"\n{label}:")
        for line in chroma_acc.report():
            print(f"  {line}")

    print("\n=== LUMA gap (pred - gt), binned by gt luma -- negative = darkened ===")
    for label, (_, luma_acc) in results.items():
        print(f"\n{label}:")
        for line in luma_acc.report():
            print(f"  {line}")


if __name__ == "__main__":
    main()

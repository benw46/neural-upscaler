"""Full-convergence old-vs-new dataloader comparison for the frame-grouping
+ pool-shuffle change to SpatialPatchDataset/train.py (see MEMORY.md's
frame-grouping-full-verification note). verify_frame_grouping.py already
proved this doesn't cause a *gross* problem (3 epochs, 200-frame subset);
this is the stronger, more expensive check that was explicitly deferred:
two complete BATCH_SIZE=16, EPOCHS=40 runs over the *full* train/val split,
same seed, same model init, same loss weights, same LR schedule as real
production training (train.py) -- the only difference is which dataset
class supplies patches. `--variant old` reimplements SpatialPatchDataset's
pre-change per-crop __getitem__ behaviour (plain, unpooled batches);
`--variant new` uses today's frame-grouped + pool-shuffled dataset/train.py
path directly, reusing train.py's own `validate()` so the val metric is
computed identically to production.

Run one variant at a time (each is a full ~2h training run) via:
    python verify_frame_grouping_full.py --variant old
    python verify_frame_grouping_full.py --variant new
then:
    python verify_frame_grouping_full.py --compare

Not part of the training pipeline -- throwaway, not imported anywhere.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from dataset import DEPTH_NORM, FullFrameDataset, SpatialPatchDataset, train_val_split  # noqa: E402
from losses import CombinedLoss  # noqa: E402
from model import SpatialUNet  # noqa: E402
from train import validate  # reuse production's exact val_l1 computation  # noqa: E402

RUN_DIR = r"E:\neural-upscaler\data\seed-20260812"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "frame_grouping_convergence_results"

SEED = 42
BATCH_SIZE = 16
EPOCHS = 40
FRAME_REPEAT = 8
FRAMES_PER_FETCH = 64
LR = 1e-3
GRAD_CLIP_NORM = 1.0
VAL_EVERY = 4
NUM_WORKERS = 4


class OldSpatialPatchDataset(Dataset):
    """Reimplements SpatialPatchDataset's *pre-change* behaviour: one
    __getitem__ call = one crop (called len(frame_indices) times per epoch,
    where frame_indices already has each frame repeated FRAME_REPEAT times
    by the caller) -- for an honest side-by-side comparison against the new
    frame-grouped version. Uses memmap (today's already-verified, separate
    fix) so this isolates the frame-grouping change specifically."""

    def __init__(self, run_dir: str, frame_indices: list[int], patch_size: int = 128):
        self.run_dir = Path(run_dir)
        header = json.loads((self.run_dir / "dataset.json").read_text())
        self.input_w, self.input_h = header["inputWidth"], header["inputHeight"]
        self.gt_w, self.gt_h = header["gtWidth"], header["gtHeight"]
        self.frame_indices = list(frame_indices)
        self.patch_size = patch_size

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        frame_idx = self.frame_indices[idx]
        fname = f"{frame_idx:06d}.bin"
        color = np.memmap(self.run_dir / "color" / fname, dtype=np.float16, mode="r", shape=(self.input_h, self.input_w, 4))
        depth = np.memmap(self.run_dir / "depth" / fname, dtype=np.float32, mode="r", shape=(self.input_h, self.input_w, 1))
        gt = np.memmap(self.run_dir / "gt_color" / fname, dtype=np.float16, mode="r", shape=(self.gt_h, self.gt_w, 4))

        ps = self.patch_size
        x = int(np.random.randint(0, self.input_w - ps + 1))
        y = int(np.random.randint(0, self.input_h - ps + 1))
        color_patch = color[y : y + ps, x : x + ps, :3].astype(np.float32)
        depth_patch = depth[y : y + ps, x : x + ps, :].astype(np.float32) / DEPTH_NORM
        input_patch = np.concatenate([color_patch, depth_patch], axis=-1)
        gt_x, gt_y, gt_ps = x * 2, y * 2, ps * 2
        gt_patch = gt[gt_y : gt_y + gt_ps, gt_x : gt_x + gt_ps, :3].astype(np.float32)
        return torch.from_numpy(input_patch).permute(2, 0, 1).contiguous(), torch.from_numpy(gt_patch).permute(2, 0, 1).contiguous()


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run(variant: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{variant}] device: {device}")
    seed_everything(SEED)

    train_idx, val_idx = train_val_split(RUN_DIR)
    val_ds = FullFrameDataset(RUN_DIR, val_idx)
    print(f"[{variant}] train frames: {len(train_idx)}  val frames: {len(val_idx)}")

    if variant == "old":
        train_ds = OldSpatialPatchDataset(RUN_DIR, train_idx * FRAME_REPEAT)
        loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True)
    else:
        train_ds = SpatialPatchDataset(RUN_DIR, train_idx, patches_per_frame=FRAME_REPEAT)
        loader = DataLoader(train_ds, batch_size=FRAMES_PER_FETCH, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True)

    model = SpatialUNet().to(device)
    loss_fn = CombinedLoss().to(device)  # defaults l1_weight=1.0, lpips_weight=0.1 -- matches train.py's defaults
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = []
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = {"l1": 0.0, "lpips": 0.0, "total": 0.0}
        n_steps = 0
        for batch in loader:
            if variant == "old":
                batches = [batch]
            else:
                gx, gy = batch
                pool_x = gx.reshape(-1, *gx.shape[2:])
                pool_y = gy.reshape(-1, *gy.shape[2:])
                perm = torch.randperm(pool_x.shape[0])
                pool_x, pool_y = pool_x[perm], pool_y[perm]
                batches = [(pool_x[i : i + BATCH_SIZE], pool_y[i : i + BATCH_SIZE]) for i in range(0, pool_x.shape[0], BATCH_SIZE)]

            for x, y in batches:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad()
                pred = model(x)
                loss, parts = loss_fn(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
                optimizer.step()
                for k in epoch_losses:
                    epoch_losses[k] += parts[k]
                n_steps += 1

        scheduler.step()
        for k in epoch_losses:
            epoch_losses[k] /= n_steps
        record = {"epoch": epoch, "elapsed_s": time.time() - t0, **{f"train_{k}": v for k, v in epoch_losses.items()}}

        if (epoch + 1) % VAL_EVERY == 0 or epoch == EPOCHS - 1:
            val_l1 = validate(model, val_ds, device)
            record["val_l1"] = val_l1
            print(f"[{variant}] epoch {epoch + 1}/{EPOCHS}: train_l1={epoch_losses['l1']:.5f} train_lpips={epoch_losses['lpips']:.5f} val_l1={val_l1:.5f}  elapsed={record['elapsed_s']:.0f}s")
        else:
            print(f"[{variant}] epoch {epoch + 1}/{EPOCHS}: train_l1={epoch_losses['l1']:.5f} train_lpips={epoch_losses['lpips']:.5f}  elapsed={record['elapsed_s']:.0f}s")

        history.append(record)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{variant}_history.json"
    out_path.write_text(json.dumps(history, indent=2))
    print(f"\n[{variant}] done in {time.time() - t0:.1f}s, history written to {out_path}")


def compare():
    old = json.loads((RESULTS_DIR / "old_history.json").read_text())
    new = json.loads((RESULTS_DIR / "new_history.json").read_text())
    print(f"{'epoch':>6} {'old_train_l1':>14} {'new_train_l1':>14} {'diff':>10}   {'old_val_l1':>12} {'new_val_l1':>12} {'val_diff':>10}")
    for o, n in zip(old, new):
        val_str = ""
        if "val_l1" in o and "val_l1" in n:
            val_str = f"{o['val_l1']:12.5f} {n['val_l1']:12.5f} {n['val_l1'] - o['val_l1']:+10.5f}"
        print(f"{o['epoch'] + 1:6d} {o['train_l1']:14.5f} {n['train_l1']:14.5f} {n['train_l1'] - o['train_l1']:+10.5f}   {val_str}")

    final_old_val = next(r["val_l1"] for r in reversed(old) if "val_l1" in r)
    final_new_val = next(r["val_l1"] for r in reversed(new) if "val_l1" in r)
    print(f"\nfinal val_l1: old={final_old_val:.5f} new={final_new_val:.5f} diff={final_new_val - final_old_val:+.5f} ({(final_new_val - final_old_val) / final_old_val * 100:+.2f}%)")
    print(f"old total time: {old[-1]['elapsed_s']:.0f}s   new total time: {new[-1]['elapsed_s']:.0f}s   speedup: {old[-1]['elapsed_s'] / new[-1]['elapsed_s']:.2f}x")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["old", "new"])
    p.add_argument("--compare", action="store_true")
    args = p.parse_args()
    if args.compare:
        compare()
    elif args.variant:
        run(args.variant)
    else:
        raise SystemExit("pass --variant old|new or --compare")

"""Verification for the frame-grouping + pool-shuffle change to
SpatialPatchDataset/train.py (see MEMORY.md's dataloader-redundant-reads-fix
note): (1) real DataLoader throughput, old (13600 individual __getitem__
calls) vs new (1700 frame-opens + pool-shuffle), (2) a short side-by-side
training comparison -- same seed, same model init, few epochs -- to
sanity-check the pool-shuffle doesn't visibly disrupt the loss trajectory.
Not a proof of identical final quality (too expensive to run to
convergence twice just to verify an I/O change); a bounded, honest check
for gross problems (instability, divergence, wildly different descent
rate), which is what a batching-diversity regression would actually look
like if it mattered.

Not part of the training pipeline -- throwaway, not imported anywhere.
"""

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

RUN_DIR = r"E:\neural-upscaler\data\seed-20260812"
PATCH_SIZE = 128
FRAME_REPEAT = 8
FRAMES_PER_FETCH = 64
BATCH_SIZE = 16
N_COMPARE_EPOCHS = 3
SEED = 42


class OldSpatialPatchDataset(Dataset):
    """Reimplements SpatialPatchDataset's *pre-change* behaviour inline (one
    __getitem__ call = one crop, called FRAME_REPEAT times per frame via an
    index list multiplied by FRAME_REPEAT) -- for an honest side-by-side
    comparison against the new frame-grouped version. Uses memmap (today's
    already-verified, separate fix) so this isolates the frame-grouping
    change specifically, not re-testing the memmap change too."""

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


def time_loader(loader, n_items, label):
    t0 = time.perf_counter()
    seen = 0
    for batch in loader:
        seen += batch[0].shape[0] if batch[0].dim() == 4 else batch[0].shape[0] * batch[0].shape[1]
        if seen >= n_items:
            break
    elapsed = time.perf_counter() - t0
    print(f"  {label}: {elapsed:.2f}s for ~{seen} patches ({elapsed / seen * 1000:.3f}ms/patch)")
    return elapsed


def run_short_training(train_ds, is_grouped: bool, device: str, epochs: int, label: str) -> list[float]:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = SpatialUNet().to(device)
    loss_fn = CombinedLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size_for_loader = FRAMES_PER_FETCH if is_grouped else BATCH_SIZE
    loader = DataLoader(train_ds, batch_size=batch_size_for_loader, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)

    epoch_l1_history = []
    for epoch in range(epochs):
        total_l1, n_steps = 0.0, 0
        for gx, gy in loader:
            if is_grouped:
                pool_x = gx.reshape(-1, *gx.shape[2:])
                pool_y = gy.reshape(-1, *gy.shape[2:])
                perm = torch.randperm(pool_x.shape[0])
                pool_x, pool_y = pool_x[perm], pool_y[perm]
                batches = [(pool_x[i : i + BATCH_SIZE], pool_y[i : i + BATCH_SIZE]) for i in range(0, pool_x.shape[0], BATCH_SIZE)]
            else:
                batches = [(gx, gy)]

            for x, y in batches:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad()
                pred = model(x)
                loss, parts = loss_fn(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_l1 += parts["l1"]
                n_steps += 1

        mean_l1 = total_l1 / n_steps
        epoch_l1_history.append(mean_l1)
        print(f"  [{label}] epoch {epoch + 1}/{epochs}: train_l1={mean_l1:.5f} ({n_steps} steps)")

    return epoch_l1_history


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}\n")

    train_idx, _ = train_val_split(RUN_DIR)
    # Small subset for a fast comparison -- this is a sanity check on loss
    # trajectory *shape*, not a real training run.
    subset = train_idx[:200]

    print("=== throughput: old (per-crop __getitem__) vs new (frame-grouped) ===")
    old_ds = OldSpatialPatchDataset(RUN_DIR, subset * FRAME_REPEAT)
    new_ds = SpatialPatchDataset(RUN_DIR, subset, patches_per_frame=FRAME_REPEAT)
    old_loader = DataLoader(old_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)
    new_loader = DataLoader(new_ds, batch_size=FRAMES_PER_FETCH, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)
    N_ITEMS = 1600
    t_old = time_loader(old_loader, N_ITEMS, "old (13600 file-opens/epoch pattern)")
    t_new = time_loader(new_loader, N_ITEMS, "new (1700 file-opens/epoch pattern)")
    print(f"  speedup: {t_old / t_new:.2f}x\n")
    del old_loader, new_loader

    print(f"=== {N_COMPARE_EPOCHS}-epoch loss-trajectory comparison (subset: {len(subset)} frames) ===")
    old_history = run_short_training(OldSpatialPatchDataset(RUN_DIR, subset * FRAME_REPEAT), is_grouped=False, device=device, epochs=N_COMPARE_EPOCHS, label="old")
    new_history = run_short_training(SpatialPatchDataset(RUN_DIR, subset, patches_per_frame=FRAME_REPEAT), is_grouped=True, device=device, epochs=N_COMPARE_EPOCHS, label="new")

    print("\n=== summary ===")
    for i, (o, n) in enumerate(zip(old_history, new_history)):
        print(f"  epoch {i + 1}: old={o:.5f}  new={n:.5f}  diff={n - o:+.5f}")


if __name__ == "__main__":
    main()

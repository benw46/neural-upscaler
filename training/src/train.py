"""Train SpatialUNet at 128x128 patches (Spec 2 step 4).

Windows uses spawn (not fork) for DataLoader workers -- the entry point must
be guarded with `if __name__ == "__main__":` or workers recursively relaunch
this whole script. See CLAUDE.md.

Gradient clipping is standard here, not optional -- the overfit sanity check
(overfit_test.py) found real training instability without it (L1-only loss
diverged mid-run from 0.055 to 15,158) purely because this architecture has
no normalisation layers to keep activations bounded.

Live progress: this logs to TensorBoard (scalars every batch, sample images
every validation pass). While training runs, view it with:
    tensorboard --logdir "training/runs"
then open the printed http://localhost:6006 URL -- it updates live as new
data is written, no need to wait for the run to finish.
"""

import argparse
import contextlib
import datetime
import sys
import time
from pathlib import Path

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from dataset import FullFrameDataset, SpatialPatchDataset, pad_to_multiple, crop_to_size, train_val_split  # noqa: E402
from losses import CombinedLoss  # noqa: E402
from model import SpatialUNet  # noqa: E402

RUN_DIR = r"E:\neural-upscaler\data\seed-20260812"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
TENSORBOARD_DIR = Path(__file__).resolve().parent.parent / "runs"

SEED = 42
BATCH_SIZE = 16
EPOCHS = 40
FRAME_REPEAT = 8  # oversample so each epoch draws several random crops per training frame, not just one
# How many frames' worth of crops (FRAME_REPEAT each) to pull from the
# DataLoader per raw fetch, before splitting into real training batches --
# see MEMORY.md's dataloader-redundant-reads-fix note. SpatialPatchDataset
# now opens each frame once and returns all FRAME_REPEAT crops from that
# open (8x fewer file-opens/epoch than before), but naively using those 8
# crops as-is would mean every training batch is dominated by just one or
# two source frames -- a real drop in per-step content diversity versus
# today's fully-shuffled-across-13600-crops batches. Pooling
# FRAMES_PER_FETCH frames and shuffling that pool before slicing into
# BATCH_SIZE=16 chunks restores most of that diversity, at zero extra I/O
# cost (the shuffle happens on already-loaded tensors, not on disk --
# file-opens/epoch is always len(train_idx) regardless of this value).
# Measured with a standalone simulation (mean distinct frames/batch,
# BATCH_SIZE=16, FRAME_REPEAT=8): today's full shuffle over all 13600 crops
# averages ~15.94; pooling only 16 frames/fetch (128 crops) averaged just
# ~10.7 -- a real, non-trivial diversity loss. Diminishing returns past
# that: 32->13.08, 48->13.98, 64->14.37, 128->15.23. Picked 64 as the point
# where doubling further only bought +0.86 more -- ~90% of full-shuffle
# diversity restored, comfortably worth the ~524MB one raw fetch holds in
# CPU memory before its shuffle-and-slice (negligible against 32GB RAM).
FRAMES_PER_FETCH = 64
LR = 1e-3
GRAD_CLIP_NORM = 1.0
VAL_EVERY = 4  # epochs
NUM_WORKERS = 4
NUM_IMAGE_SAMPLES = 3  # fixed held-out frames whose prediction gets logged as an image every validation pass


def parse_args() -> argparse.Namespace:
    """CLI overrides for the loss-weight sweep (see docs/OPTIMISATIONS.md's
    colour-loss investigation) -- defaults reproduce Phase 2's original run
    exactly (same weights, same 40 epochs, same checkpoints/ output), so
    calling this script with no arguments is unchanged from before. A
    non-default --run-label writes to checkpoints/<label>_{best,final}.pt
    and runs/<label>_<timestamp> instead of the original bare filenames,
    specifically so a sweep run can never overwrite Phase 2's
    already-gate-passed checkpoint."""
    p = argparse.ArgumentParser()
    p.add_argument("--l1-weight", type=float, default=1.0)
    p.add_argument("--lpips-weight", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--run-label", type=str, default=None)
    return p.parse_args()


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def amp_autocast(device: str):
    """bf16 autocast on CUDA, no-op on CPU. bf16 over classic fp16+GradScaler
    deliberately: this architecture has no normalisation layers and is
    documented (see module docstring) to have diverged to a loss of 15,158
    without gradient clipping -- bf16 has fp32's exponent range, so it can't
    overflow-to-inf the way fp16 can on a blow-up, and needs no loss scaler
    at all. Ampere (RTX 3060) supports bf16 tensor cores natively."""
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def validate(model: torch.nn.Module, val_ds: FullFrameDataset, device: str) -> float:
    model.eval()
    total_l1 = 0.0
    for i in range(len(val_ds)):
        x, y = val_ds[i]
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)
        x_padded, orig_size = pad_to_multiple(x)
        with amp_autocast(device):
            pred = model(x_padded)
        pred = crop_to_size(pred.float(), (orig_size[0] * 2, orig_size[1] * 2))
        total_l1 += torch.nn.functional.l1_loss(pred, y).item()
    model.train()
    return total_l1 / len(val_ds)


@torch.no_grad()
def log_sample_images(writer: SummaryWriter, model: torch.nn.Module, val_ds: FullFrameDataset, device: str, sample_indices: list[int], step: int):
    """Logs [nearest-upsampled input | model prediction | ground truth] side
    by side for a few fixed held-out frames -- seeing the actual image
    improve is far more informative than watching a loss number decrease."""
    model.eval()
    for idx in sample_indices:
        x, y = val_ds[idx]
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)
        x_padded, orig_size = pad_to_multiple(x)
        with amp_autocast(device):
            pred = model(x_padded)
        pred = crop_to_size(pred.float(), (orig_size[0] * 2, orig_size[1] * 2)).clamp(0, 1)

        input_upsampled = torch.nn.functional.interpolate(x[:, :3], scale_factor=2, mode="nearest")
        grid = vutils.make_grid(torch.cat([input_upsampled, pred, y], dim=0), nrow=3)
        writer.add_image(f"val_sample_{idx}/input_pred_target", grid, global_step=step)
    model.train()


def main():
    args = parse_args()
    epochs = args.epochs
    prefix = f"{args.run_label}_" if args.run_label else ""

    seed_everything(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"loss weights: l1={args.l1_weight} lpips={args.lpips_weight}  epochs={epochs}  run_label={args.run_label!r}")

    train_idx, val_idx = train_val_split(RUN_DIR)
    print(f"train frames: {len(train_idx)}  val frames: {len(val_idx)}")

    train_ds = SpatialPatchDataset(RUN_DIR, train_idx, patches_per_frame=FRAME_REPEAT)
    val_ds = FullFrameDataset(RUN_DIR, val_idx)
    train_loader = DataLoader(
        train_ds,
        batch_size=FRAMES_PER_FETCH,
        shuffle=True,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        pin_memory=True,
    )
    approx_steps_per_epoch = (len(train_idx) * FRAME_REPEAT) // BATCH_SIZE
    print(f"raw fetches/epoch: {len(train_loader)} (file-opens: {len(train_idx)})  approx optimizer steps/epoch: {approx_steps_per_epoch}  total planned steps: {approx_steps_per_epoch * epochs}")

    model = SpatialUNet().to(device)
    loss_fn = CombinedLoss(l1_weight=args.l1_weight, lpips_weight=args.lpips_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    run_name = prefix + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=str(TENSORBOARD_DIR / run_name))
    print(f"tensorboard logdir: {TENSORBOARD_DIR / run_name}")
    image_sample_indices = list(range(0, len(val_ds), max(1, len(val_ds) // NUM_IMAGE_SAMPLES)))[:NUM_IMAGE_SAMPLES]

    best_val_l1 = float("inf")
    history = []
    global_step = 0

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_losses = {"l1": 0.0, "lpips": 0.0, "total": 0.0}
        n_steps = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}")
        for group_x, group_y in pbar:
            # group_x/group_y: (FRAMES_PER_FETCH, FRAME_REPEAT, C, H, W) --
            # one DataLoader fetch = FRAMES_PER_FETCH file-opens, each
            # already containing FRAME_REPEAT crops from SpatialPatchDataset.
            # Flatten to a pool and shuffle before slicing into real training
            # batches, so a batch draws from many distinct source frames
            # rather than being dominated by the 1-2 frames in the raw fetch
            # that happen to land in it (see FRAMES_PER_FETCH's comment above).
            pool_x = group_x.reshape(-1, *group_x.shape[2:])
            pool_y = group_y.reshape(-1, *group_y.shape[2:])
            perm = torch.randperm(pool_x.shape[0])
            pool_x, pool_y = pool_x[perm], pool_y[perm]

            for i in range(0, pool_x.shape[0], BATCH_SIZE):
                x = pool_x[i : i + BATCH_SIZE].to(device, non_blocking=True)
                y = pool_y[i : i + BATCH_SIZE].to(device, non_blocking=True)

                optimizer.zero_grad()
                with amp_autocast(device):
                    pred = model(x)
                    loss, parts = loss_fn(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
                optimizer.step()
                for k in epoch_losses:
                    epoch_losses[k] += parts[k]
                n_steps += 1
                pbar.set_postfix(l1=f"{parts['l1']:.4f}", lpips=f"{parts['lpips']:.4f}", step=n_steps)

                writer.add_scalar("train/l1_batch", parts["l1"], global_step)
                writer.add_scalar("train/lpips_batch", parts["lpips"], global_step)
                writer.add_scalar("train/total_batch", parts["total"], global_step)
                global_step += 1

        scheduler.step()
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        for k in epoch_losses:
            epoch_losses[k] /= n_steps
        record = {"epoch": epoch, "elapsed_s": time.time() - t0, **{f"train_{k}": v for k, v in epoch_losses.items()}}
        for k, v in epoch_losses.items():
            writer.add_scalar(f"train_epoch/{k}", v, epoch)

        if (epoch + 1) % VAL_EVERY == 0 or epoch == epochs - 1:
            val_l1 = validate(model, val_ds, device)
            record["val_l1"] = val_l1
            writer.add_scalar("val/l1", val_l1, epoch)
            log_sample_images(writer, model, val_ds, device, image_sample_indices, epoch)
            print(f"epoch {epoch + 1}: train_l1={epoch_losses['l1']:.5f} train_lpips={epoch_losses['lpips']:.5f} val_l1={val_l1:.5f}")
            if val_l1 < best_val_l1:
                best_val_l1 = val_l1
                torch.save(model.state_dict(), CHECKPOINT_DIR / f"{prefix}best.pt")
                print(f"  new best val_l1={val_l1:.5f}, saved checkpoint")
        else:
            print(f"epoch {epoch + 1}: train_l1={epoch_losses['l1']:.5f} train_lpips={epoch_losses['lpips']:.5f}")

        history.append(record)

    torch.save(model.state_dict(), CHECKPOINT_DIR / f"{prefix}final.pt")
    writer.close()
    print(f"\ntraining complete in {time.time() - t0:.1f}s")
    print(f"best val_l1: {best_val_l1:.5f}")
    print(f"checkpoints saved to {CHECKPOINT_DIR}")

    import json

    (CHECKPOINT_DIR / f"{prefix}history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()

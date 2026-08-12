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
LR = 1e-3
GRAD_CLIP_NORM = 1.0
VAL_EVERY = 4  # epochs
NUM_WORKERS = 4
NUM_IMAGE_SAMPLES = 3  # fixed held-out frames whose prediction gets logged as an image every validation pass


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validate(model: torch.nn.Module, val_ds: FullFrameDataset, device: str) -> float:
    model.eval()
    total_l1 = 0.0
    for i in range(len(val_ds)):
        x, y = val_ds[i]
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)
        x_padded, orig_size = pad_to_multiple(x)
        pred = model(x_padded)
        pred = crop_to_size(pred, (orig_size[0] * 2, orig_size[1] * 2))
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
        pred = model(x_padded)
        pred = crop_to_size(pred, (orig_size[0] * 2, orig_size[1] * 2)).clamp(0, 1)

        input_upsampled = torch.nn.functional.interpolate(x[:, :3], scale_factor=2, mode="nearest")
        grid = vutils.make_grid(torch.cat([input_upsampled, pred, y], dim=0), nrow=3)
        writer.add_image(f"val_sample_{idx}/input_pred_target", grid, global_step=step)
    model.train()


def main():
    seed_everything(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_idx, val_idx = train_val_split(RUN_DIR)
    print(f"train frames: {len(train_idx)}  val frames: {len(val_idx)}")

    train_ds = SpatialPatchDataset(RUN_DIR, train_idx * FRAME_REPEAT)
    val_ds = FullFrameDataset(RUN_DIR, val_idx)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        pin_memory=True,
    )
    print(f"batches/epoch: {len(train_loader)}  total planned iterations: {len(train_loader) * EPOCHS}")

    model = SpatialUNet().to(device)
    loss_fn = CombinedLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=str(TENSORBOARD_DIR / run_name))
    print(f"tensorboard logdir: {TENSORBOARD_DIR / run_name}")
    image_sample_indices = list(range(0, len(val_ds), max(1, len(val_ds) // NUM_IMAGE_SAMPLES)))[:NUM_IMAGE_SAMPLES]

    best_val_l1 = float("inf")
    history = []
    global_step = 0

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = {"l1": 0.0, "lpips": 0.0, "total": 0.0}
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{EPOCHS}")
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(x)
            loss, parts = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()
            for k in epoch_losses:
                epoch_losses[k] += parts[k]
            pbar.set_postfix(l1=f"{parts['l1']:.4f}", lpips=f"{parts['lpips']:.4f}")

            writer.add_scalar("train/l1_batch", parts["l1"], global_step)
            writer.add_scalar("train/lpips_batch", parts["lpips"], global_step)
            writer.add_scalar("train/total_batch", parts["total"], global_step)
            global_step += 1

        scheduler.step()
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        n_batches = len(train_loader)
        for k in epoch_losses:
            epoch_losses[k] /= n_batches
        record = {"epoch": epoch, "elapsed_s": time.time() - t0, **{f"train_{k}": v for k, v in epoch_losses.items()}}
        for k, v in epoch_losses.items():
            writer.add_scalar(f"train_epoch/{k}", v, epoch)

        if (epoch + 1) % VAL_EVERY == 0 or epoch == EPOCHS - 1:
            val_l1 = validate(model, val_ds, device)
            record["val_l1"] = val_l1
            writer.add_scalar("val/l1", val_l1, epoch)
            log_sample_images(writer, model, val_ds, device, image_sample_indices, epoch)
            print(f"epoch {epoch + 1}: train_l1={epoch_losses['l1']:.5f} train_lpips={epoch_losses['lpips']:.5f} val_l1={val_l1:.5f}")
            if val_l1 < best_val_l1:
                best_val_l1 = val_l1
                torch.save(model.state_dict(), CHECKPOINT_DIR / "best.pt")
                print(f"  new best val_l1={val_l1:.5f}, saved checkpoint")
        else:
            print(f"epoch {epoch + 1}: train_l1={epoch_losses['l1']:.5f} train_lpips={epoch_losses['lpips']:.5f}")

        history.append(record)

    torch.save(model.state_dict(), CHECKPOINT_DIR / "final.pt")
    writer.close()
    print(f"\ntraining complete in {time.time() - t0:.1f}s")
    print(f"best val_l1: {best_val_l1:.5f}")
    print(f"checkpoints saved to {CHECKPOINT_DIR}")

    import json

    (CHECKPOINT_DIR / "history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()

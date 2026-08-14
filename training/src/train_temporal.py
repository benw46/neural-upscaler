"""Recurrent temporal training (Spec 3 steps 1-4): feeds the previous
frame's *own predicted output* (warped by motion vectors, not ground truth)
back in as history, backpropagating through the whole unrolled sequence.

Frame 0 of every training sequence has no real predecessor -- treated as a
full disocclusion (all-ones mask, zeroed history), which both handles the
"cold start" case and reuses the same code path real mid-sequence
disocclusions need anyway, rather than a special case.

See notebook/PHASE-3-EXPERIMENTS.md for the loss-weighting search this
script was run under, and CLAUDE.md for why: "Loss weighting for temporal
consistency... is empirical, not derivable."
"""

import argparse
import contextlib
import datetime
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from dataset import train_val_split  # noqa: E402
from dataset_sequence import SequenceDataset  # noqa: E402
from losses import CombinedLoss, temporal_consistency_loss  # noqa: E402
from model import SpatialUNet  # noqa: E402
from warp import compute_disocclusion_mask, upsample_motion, warp_previous_output  # noqa: E402

RUN_DIR = r"E:\neural-upscaler\data\seed-20260812"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints_temporal"
TENSORBOARD_DIR = Path(__file__).resolve().parent.parent / "runs_temporal"

SEED = 42
PATCH_SIZE = 128
SEQ_LEN = 6
BATCH_SIZE = 4  # smaller than Phase 2's 16 -- BPTT over SEQ_LEN unrolled steps, no detaching, costs much more memory
EPOCHS = 20
VAL_FRACTION = 0.25  # 1500 train / 500 held-out on the 2000-frame Spec 3 dataset
LR = 1e-3
GRAD_CLIP_NORM = 1.0
VAL_EVERY = 2

# The knob this whole phase is about tuning empirically -- see
# notebook/PHASE-3-EXPERIMENTS.md for the full search (0.0, 0.5, 4.0, 1.0
# tried). 1.0 chosen: a real, measurable reduction in temporal
# inconsistency (val_temporal ~half of the 0.0/0.5 baselines) without the
# spatial-accuracy collapse seen at 4.0 (the "just copy history" degenerate
# solution Spec 3 warns about -- confirmed and reproduced there).
TEMPORAL_WEIGHT = 1.0
RUN_LABEL = "temporal_w1.0_final"  # short tag identifying this config, used in the tensorboard/checkpoint dir name


def parse_args() -> argparse.Namespace:
    """CLI overrides -- defaults reproduce the exact recipe that produced
    the original deployed checkpoint (lpips_weight=0.1, temporal_weight=1.0,
    run_label="temporal_w1.0_final"), so calling this script with no
    arguments is unchanged from before. `--temporal-weight` was deliberately
    NOT a CLI arg originally (Spec 3's own coarse sweep -- 0.0/0.5/1.0/4.0,
    4 epochs each -- had already chosen 1.0, see the comment above) but that
    sweep is explicitly being re-run finer now that training is ~6x faster
    (see MEMORY.md), so it's exposed here too. Always pass --run-label
    explicitly alongside a non-default --temporal-weight, so a sweep run
    can't collide with (and overwrite) any existing checkpoint family."""
    p = argparse.ArgumentParser()
    p.add_argument("--lpips-weight", type=float, default=0.1)
    p.add_argument("--lpips-scale", type=float, default=1.0)
    p.add_argument("--run-label", type=str, default=RUN_LABEL)
    p.add_argument("--temporal-weight", type=float, default=TEMPORAL_WEIGHT)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def amp_autocast(device: str):
    """bf16 autocast on CUDA, no-op on CPU. bf16 over classic fp16+GradScaler
    deliberately: this architecture has no normalisation layers, and the
    recurrent feedback path specifically has documented history of an
    exponential-blowup cascade (see the `prev_output_highres` clamp comment
    below) -- bf16 has fp32's exponent range, so it can't overflow-to-inf
    the way fp16 can on a blow-up, and needs no loss scaler. Ampere (RTX
    3060) supports bf16 tensor cores natively. Scoped tightly in
    unroll_sequence() to just the model forward + loss (see there) --
    everything else in the recurrence (warp, disocclusion, the clamp) stays
    fp32, unchanged from before this was added."""
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def unroll_sequence(model, batch, loss_fn, device, temporal_weight: float):
    """Runs one full recurrent unroll over a (B, seq_len, ...) batch,
    returns the total loss (summed over steps, still attached to the graph
    for BPTT) plus per-step diagnostics for logging."""
    color = batch["color"].to(device, non_blocking=True)  # (B, T, 3, ps, ps)
    depth = batch["depth"].to(device, non_blocking=True)  # (B, T, 1, ps, ps)
    motion = batch["motion"].to(device, non_blocking=True)  # (B, T, 2, ps, ps)
    gt = batch["gt"].to(device, non_blocking=True)  # (B, T, 3, ps*2, ps*2)

    b, seq_len = color.shape[0], color.shape[1]
    out_size = gt.shape[-2:]

    prev_output_highres = None
    prev_depth = None
    total_loss = torch.zeros((), device=device)
    step_records = []
    predictions = []

    for t in range(seq_len):
        curr_color = color[:, t]
        curr_depth = depth[:, t]
        curr_motion = motion[:, t]
        curr_gt = gt[:, t]

        if t == 0:
            disocclusion_mask = torch.ones((b, 1, PATCH_SIZE, PATCH_SIZE), device=device)
            warped_prev_lowres = torch.zeros((b, 3, PATCH_SIZE, PATCH_SIZE), device=device)
            warped_prev_highres = torch.zeros((b, 3, *out_size), device=device)
        else:
            disocclusion_mask = compute_disocclusion_mask(prev_depth, curr_depth, curr_motion)
            prev_output_lowres = F.interpolate(prev_output_highres, size=(PATCH_SIZE, PATCH_SIZE), mode="bilinear", align_corners=False)
            warped_prev_lowres = warp_previous_output(prev_output_lowres, curr_motion)
            motion_highres = upsample_motion(curr_motion, out_size)
            warped_prev_highres = warp_previous_output(prev_output_highres, motion_highres)

        model_input = torch.cat([curr_color, curr_depth, warped_prev_lowres, disocclusion_mask], dim=1)
        # Autocast scoped to just the model forward + loss -- the heaviest
        # compute (16 conv layers x this loss's own VGG forward) -- not the
        # surrounding warp/disocclusion/feedback logic, which stays exactly
        # fp32 as it always was (see amp_autocast's docstring). `.float()`
        # immediately after re-establishes fp32 before `pred` is used
        # anywhere outside the loss, so nothing downstream (the temporal
        # loss, the clamped feedback into next step's warp) changes numeric
        # behaviour from before this was added.
        with amp_autocast(device):
            pred = model(model_input)
            spatial_loss, spatial_parts = loss_fn(pred, curr_gt)
        pred = pred.float()
        if t == 0:
            temporal_loss = torch.zeros((), device=device)
        else:
            temporal_loss = temporal_consistency_loss(pred, warped_prev_highres, disocclusion_mask)

        step_loss = spatial_loss + temporal_weight * temporal_loss
        total_loss = total_loss + step_loss

        step_records.append(
            {
                "l1": spatial_parts["l1"],
                "lpips": spatial_parts["lpips"],
                "temporal": temporal_loss.item(),
                "disocclusion_frac": disocclusion_mask.mean().item(),
            }
        )
        predictions.append(pred.detach())

        # NOT detached -- BPTT through the recurrence, per Spec 3 step 3.
        # Clamped (feedback path only -- `pred` itself, used above for the
        # loss, stays unclamped so L1/LPIPS still see and penalise
        # out-of-range predictions directly) to prevent the forward-pass
        # exponential-blowup cascade found during the first full training
        # run: an unclamped prediction feeding back through warp/downsample
        # can amplify step-to-step and run away numerically before gradient
        # clipping (which only bounds the optimizer step, not the forward
        # pass) ever gets a chance to intervene. See notebook/PHASE-3-EXPERIMENTS.md.
        prev_output_highres = pred.clamp(0, 1)
        prev_depth = curr_depth
        prev_depth = curr_depth

    return total_loss, step_records, predictions


@torch.no_grad()
def validate(model, val_ds: SequenceDataset, loss_fn, device: str, n_sequences: int = 20) -> dict:
    model.eval()
    totals = {"l1": 0.0, "lpips": 0.0, "temporal": 0.0}
    n = min(n_sequences, len(val_ds))
    loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    for i, batch in enumerate(loader):
        if i >= n:
            break
        _, step_records, _ = unroll_sequence(model, batch, loss_fn, device, TEMPORAL_WEIGHT)
        for k in totals:
            totals[k] += sum(r[k] for r in step_records) / len(step_records)
    model.train()
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def log_rollout_images(writer: SummaryWriter, model, val_ds: SequenceDataset, loss_fn, device: str, step: int):
    """Logs a full recurrent rollout [prediction | ground truth] for one
    fixed held-out sequence -- seeing frame-to-frame drift/ghosting directly
    is the point, not just a scalar."""
    model.eval()
    batch = val_ds[0]
    batch = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    _, _, predictions = unroll_sequence(model, batch, loss_fn, device, TEMPORAL_WEIGHT)
    gt = batch["gt"][0]  # (T, 3, H, W)
    preds = torch.cat(predictions, dim=0).clamp(0, 1).cpu()  # (T, 3, H, W)
    grid = vutils.make_grid(torch.cat([preds, gt], dim=0), nrow=preds.shape[0])
    writer.add_image("val_rollout/pred_top_gt_bottom", grid, global_step=step)
    model.train()


def main():
    global TEMPORAL_WEIGHT
    args = parse_args()
    run_label = args.run_label
    TEMPORAL_WEIGHT = args.temporal_weight

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  run: {run_label}  temporal_weight: {TEMPORAL_WEIGHT}  lpips_weight: {args.lpips_weight}")

    train_idx, val_idx = train_val_split(RUN_DIR, val_fraction=VAL_FRACTION)
    print(f"train frames: {len(train_idx)}  val frames: {len(val_idx)}")

    val_ds = SequenceDataset(RUN_DIR, val_idx, seq_len=SEQ_LEN, patch_size=PATCH_SIZE)
    print(f"val sequences: {len(val_ds)}")

    model = SpatialUNet(in_channels=8).to(device)
    loss_fn = CombinedLoss(l1_weight=1.0, lpips_weight=args.lpips_weight, lpips_scale=args.lpips_scale).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    run_name = f"{run_label}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=str(TENSORBOARD_DIR / run_name))
    print(f"tensorboard logdir: {TENSORBOARD_DIR / run_name}")

    best_val_l1 = float("inf")
    history = []
    global_step = 0

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        epoch_totals = {"l1": 0.0, "lpips": 0.0, "temporal": 0.0}
        # Fresh dataset + loader every epoch, offset by `epoch` -- see
        # SequenceDataset's docstring. Cuts the ~6x sliding-window read
        # redundancy down to ~1x within an epoch by using non-overlapping
        # (stride=seq_len) windows, while still covering every possible
        # window phase over the course of the run as the offset cycles
        # 0..SEQ_LEN-1. Not persistent_workers -- a new loader (and new
        # workers) every epoch is exactly what varying the offset needs;
        # the one-off spawn cost per epoch is small next to the I/O saved.
        train_ds = SequenceDataset(RUN_DIR, train_idx, seq_len=SEQ_LEN, patch_size=PATCH_SIZE, epoch=epoch)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{EPOCHS} ({len(train_ds)} sequences)")
        for batch in pbar:
            optimizer.zero_grad()
            total_loss, step_records, _ = unroll_sequence(model, batch, loss_fn, device, TEMPORAL_WEIGHT)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()

            avg = {k: sum(r[k] for r in step_records) / len(step_records) for k in ("l1", "lpips", "temporal")}
            for k in epoch_totals:
                epoch_totals[k] += avg[k]
            pbar.set_postfix(l1=f"{avg['l1']:.4f}", temporal=f"{avg['temporal']:.4f}")

            writer.add_scalar("train/l1_batch", avg["l1"], global_step)
            writer.add_scalar("train/lpips_batch", avg["lpips"], global_step)
            writer.add_scalar("train/temporal_batch", avg["temporal"], global_step)
            global_step += 1

        scheduler.step()
        n_batches = len(train_loader)
        for k in epoch_totals:
            epoch_totals[k] /= n_batches
        record = {"epoch": epoch, "elapsed_s": time.time() - t0, **{f"train_{k}": v for k, v in epoch_totals.items()}}

        if (epoch + 1) % VAL_EVERY == 0 or epoch == EPOCHS - 1:
            val_metrics = validate(model, val_ds, loss_fn, device)
            record.update({f"val_{k}": v for k, v in val_metrics.items()})
            writer.add_scalar("val/l1", val_metrics["l1"], epoch)
            writer.add_scalar("val/temporal", val_metrics["temporal"], epoch)
            log_rollout_images(writer, model, val_ds, loss_fn, device, epoch)
            print(f"epoch {epoch + 1}: train_l1={epoch_totals['l1']:.5f} val_l1={val_metrics['l1']:.5f} val_temporal={val_metrics['temporal']:.5f}")
            if val_metrics["l1"] < best_val_l1:
                best_val_l1 = val_metrics["l1"]
                torch.save(model.state_dict(), CHECKPOINT_DIR / f"{run_label}_best.pt")
                print(f"  new best val_l1={val_metrics['l1']:.5f}, saved checkpoint")
        else:
            print(f"epoch {epoch + 1}: train_l1={epoch_totals['l1']:.5f} train_temporal={epoch_totals['temporal']:.5f}")

        history.append(record)

    torch.save(model.state_dict(), CHECKPOINT_DIR / f"{run_label}_final.pt")
    writer.close()
    print(f"\ntraining complete in {time.time() - t0:.1f}s")
    print(f"best val_l1: {best_val_l1:.5f}")

    import json

    (CHECKPOINT_DIR / f"{run_label}_history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()

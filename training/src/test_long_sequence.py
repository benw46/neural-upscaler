"""Long-sequence stability test (Spec 3 gate) + explicit degenerate-solution
checks (Spec 3 step 6): brightness drift, accumulating artifacts, "just copy
history", and ghosting at disocclusions.

Runs a genuine recurrent rollout (no ground-truth teacher-forcing, no BPTT --
just forward inference, exactly how it would run in deployment) over the
held-out validation block at *full frame* resolution (960x540 -> 1920x1080),
not the 128x128 training patches -- full frames are what the gate cares
about and avoid patch-boundary edge effects patches would introduce.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from dataset import DEPTH_NORM, crop_to_size, pad_to_multiple, train_val_split  # noqa: E402
from model import SpatialUNet  # noqa: E402
from warp import compute_disocclusion_mask, upsample_motion, warp_previous_output  # noqa: E402

RUN_DIR = Path(r"E:\neural-upscaler\data\seed-20260812")
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "checkpoints_temporal" / "temporal_w1.0_final_best.pt"
OUT_DIR = Path(__file__).resolve().parent.parent / "long_sequence_results"
SEQUENCE_LENGTH = 350  # gate requires 300+; some margin


def load_frame(frame_idx: int, input_w: int, input_h: int, gt_w: int, gt_h: int):
    fname = f"{frame_idx:06d}.bin"
    color = np.fromfile(RUN_DIR / "color" / fname, dtype=np.float16).reshape(input_h, input_w, 4)[:, :, :3].astype(np.float32)
    depth = np.fromfile(RUN_DIR / "depth" / fname, dtype=np.float32).reshape(input_h, input_w, 1) / DEPTH_NORM
    motion = np.fromfile(RUN_DIR / "motion" / fname, dtype=np.float16).reshape(input_h, input_w, 2).astype(np.float32)
    gt = np.fromfile(RUN_DIR / "gt_color" / fname, dtype=np.float16).reshape(gt_h, gt_w, 4)[:, :, :3].astype(np.float32)
    return color, depth, motion, gt


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def save_png(t: torch.Tensor, path: Path):
    arr = (t.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def luminance(t: torch.Tensor) -> float:
    # Rec. 601 luma weights -- fine for a relative brightness-drift check.
    return (0.299 * t[:, 0] + 0.587 * t[:, 1] + 0.114 * t[:, 2]).mean().item()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    header = json.loads((RUN_DIR / "dataset.json").read_text())
    input_w, input_h = header["inputWidth"], header["inputHeight"]
    gt_w, gt_h = header["gtWidth"], header["gtHeight"]

    _, val_idx = train_val_split(str(RUN_DIR), val_fraction=0.25)
    start_frame = val_idx[0]
    assert len(val_idx) >= SEQUENCE_LENGTH, f"held-out block ({len(val_idx)} frames) shorter than requested sequence ({SEQUENCE_LENGTH})"
    frame_range = range(start_frame, start_frame + SEQUENCE_LENGTH)
    print(f"rolling out frames {frame_range.start}..{frame_range.stop - 1} ({SEQUENCE_LENGTH} frames)")

    model = SpatialUNet(in_channels=8).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    print(f"loaded checkpoint: {CHECKPOINT_PATH}")

    OUT_DIR.mkdir(exist_ok=True)
    sample_frames_dir = OUT_DIR / "sample_frames"
    sample_frames_dir.mkdir(exist_ok=True)

    prev_output_highres = None
    prev_depth = None

    records = []
    sample_every = 25

    with torch.no_grad():
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
            out_size = (padded_h * 2, padded_w * 2)

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

            # Metrics against ground truth
            l1_vs_gt = (pred - gt_t).abs().mean().item()
            bright = luminance(pred)
            bright_gt = luminance(gt_t)

            # Degenerate "just copy history" check: how close is pred to the
            # warped previous frame vs to ground truth? A healthy model
            # should track GT much more closely than a stale/wrong history.
            if i == 0:
                l1_vs_warped_prev = float("nan")
                warped_prev_vs_gt = float("nan")
                ghosting_disocc_l1 = float("nan")
            else:
                motion_highres = upsample_motion(motion_padded, out_size)
                warped_prev_highres = warp_previous_output(prev_output_highres, motion_highres)
                warped_prev_highres_cropped = crop_to_size(warped_prev_highres, (orig_size[0] * 2, orig_size[1] * 2))
                l1_vs_warped_prev = (pred - warped_prev_highres_cropped).abs().mean().item()
                warped_prev_vs_gt = (warped_prev_highres_cropped - gt_t).abs().mean().item()

                # Ghosting check: within disoccluded regions specifically,
                # is pred closer to the (wrong, stale) warped history than
                # to ground truth? Upsample mask to output res for this.
                mask_highres = F.interpolate(disocclusion_mask, size=pred.shape[-2:], mode="nearest")
                mask_highres = crop_to_size(mask_highres, (orig_size[0] * 2, orig_size[1] * 2))
                disocc_pixels = mask_highres > 0.5
                if disocc_pixels.any():
                    diff_pred_gt = (pred - gt_t).abs().mean(dim=1, keepdim=True)
                    ghosting_disocc_l1 = diff_pred_gt[disocc_pixels.expand_as(diff_pred_gt)].mean().item()
                else:
                    ghosting_disocc_l1 = float("nan")

            records.append(
                {
                    "frame": frame_idx,
                    "l1_vs_gt": l1_vs_gt,
                    "brightness": bright,
                    "brightness_gt": bright_gt,
                    "l1_vs_warped_prev": l1_vs_warped_prev,
                    "warped_prev_vs_gt": warped_prev_vs_gt,
                    "disocclusion_frac": disocclusion_mask.mean().item(),
                    "ghosting_disocc_l1": ghosting_disocc_l1,
                }
            )

            if i % sample_every == 0 or i == SEQUENCE_LENGTH - 1:
                save_png(pred.cpu(), sample_frames_dir / f"pred_{i:04d}_frame{frame_idx}.png")
                save_png(gt_t.cpu(), sample_frames_dir / f"gt_{i:04d}_frame{frame_idx}.png")
                print(f"frame {i}/{SEQUENCE_LENGTH} (idx {frame_idx}): l1_vs_gt={l1_vs_gt:.5f} brightness={bright:.5f} disocc={disocclusion_mask.mean().item():.4f}")

            prev_output_highres = pred_padded.clamp(0, 1)  # keep feeding padded resolution forward, consistent with training
            prev_depth = depth_padded

    (OUT_DIR / "records.json").write_text(json.dumps(records, indent=2))

    # --- summary ---
    brightness = [r["brightness"] for r in records]
    brightness_gt = [r["brightness_gt"] for r in records]
    l1s = [r["l1_vs_gt"] for r in records]
    first_half_brightness = np.mean(brightness[: len(brightness) // 2])
    second_half_brightness = np.mean(brightness[len(brightness) // 2 :])
    first_half_brightness_gt = np.mean(brightness_gt[: len(brightness_gt) // 2])
    second_half_brightness_gt = np.mean(brightness_gt[len(brightness_gt) // 2 :])
    first_half_l1 = np.mean(l1s[: len(l1s) // 2])
    second_half_l1 = np.mean(l1s[len(l1s) // 2 :])

    pred_drift = second_half_brightness - first_half_brightness
    gt_drift = second_half_brightness_gt - first_half_brightness_gt
    excess_drift = pred_drift - gt_drift  # the part NOT explained by the scene's own content

    print("\n--- summary ---")
    print(f"frames: {len(records)}")
    print(f"brightness (pred): first half={first_half_brightness:.5f}  second half={second_half_brightness:.5f}  drift={pred_drift:+.5f}")
    print(f"brightness (GT):   first half={first_half_brightness_gt:.5f}  second half={second_half_brightness_gt:.5f}  drift={gt_drift:+.5f}")
    print(f"excess brightness drift (pred beyond what GT's own content explains): {excess_drift:+.5f}")
    print(f"l1_vs_gt:   first half mean={first_half_l1:.5f}  second half mean={second_half_l1:.5f}  change={second_half_l1 - first_half_l1:+.5f}")

    valid_ghost = [r["ghosting_disocc_l1"] for r in records if not np.isnan(r["ghosting_disocc_l1"])]
    valid_l1 = [r["l1_vs_gt"] for r in records[1:]]
    print(f"mean l1_vs_gt (all frames): {np.mean(l1s):.5f}")
    if valid_ghost:
        print(f"mean l1 within disoccluded regions: {np.mean(valid_ghost):.5f}  (vs overall mean l1_vs_gt: {np.mean(valid_l1):.5f})")

    valid_copy = [(r["l1_vs_warped_prev"], r["warped_prev_vs_gt"], r["l1_vs_gt"]) for r in records[1:]]
    mean_pred_vs_warped = np.mean([v[0] for v in valid_copy])
    mean_warped_vs_gt = np.mean([v[1] for v in valid_copy])
    mean_pred_vs_gt = np.mean([v[2] for v in valid_copy])
    print(f"mean l1(pred, warped_prev)={mean_pred_vs_warped:.5f}  mean l1(warped_prev, gt)={mean_warped_vs_gt:.5f}  mean l1(pred, gt)={mean_pred_vs_gt:.5f}")
    print("(degenerate copying would show l1(pred,gt) close to l1(warped_prev,gt) and l1(pred,warped_prev) near zero)")

    print(f"\nsample frames + full records written to {OUT_DIR}")


if __name__ == "__main__":
    main()

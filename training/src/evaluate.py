"""Baseline comparison + gate metrics (Spec 2 step 4 / gate).

Compares the trained model against bilinear, bicubic, and Lanczos upsampling
on every held-out frame, reporting PSNR/SSIM/LPIPS for each -- the gate
requires beating bicubic with these numbers reported directly, not just
"better."
"""

import sys
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, str(Path(__file__).parent))
from dataset import FullFrameDataset, crop_to_size, pad_to_multiple, train_val_split  # noqa: E402
from model import SpatialUNet  # noqa: E402

RUN_DIR = r"E:\neural-upscaler\data\seed-20260812"
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "best.pt"


def upsample_torch(x: torch.Tensor, mode: str) -> torch.Tensor:
    kwargs = {"scale_factor": 2, "mode": mode}
    if mode in ("bilinear", "bicubic"):
        kwargs["align_corners"] = False
    return F.interpolate(x, **kwargs)


def upsample_lanczos(x: torch.Tensor) -> torch.Tensor:
    """PIL's Lanczos resize -- torch has no built-in Lanczos filter."""
    arr = (x.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    h, w = arr.shape[:2]
    img = Image.fromarray(arr).resize((w * 2, h * 2), Image.LANCZOS)
    out = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)


@torch.no_grad()
def model_predict(model: torch.nn.Module, x: torch.Tensor, device: str) -> torch.Tensor:
    x_padded, orig_size = pad_to_multiple(x.to(device))
    pred = model(x_padded)
    pred = crop_to_size(pred, (orig_size[0] * 2, orig_size[1] * 2))
    return pred.clamp(0, 1).cpu()


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, lpips_fn: lpips.LPIPS, device: str) -> dict[str, float]:
    pred_np = pred.squeeze(0).permute(1, 2, 0).numpy()
    target_np = target.squeeze(0).permute(1, 2, 0).numpy()

    psnr = peak_signal_noise_ratio(target_np, pred_np, data_range=1.0)
    ssim = structural_similarity(target_np, pred_np, data_range=1.0, channel_axis=2)
    with torch.no_grad():
        # Full 1920x1080 VGG forward passes are heavy -- run on GPU, not CPU
        # (an earlier CPU-bound version of this script was still running
        # after 7+ minutes of wall-clock time across ~300 forward passes).
        lpips_val = lpips_fn(pred.to(device) * 2 - 1, target.to(device) * 2 - 1).item()

    return {"psnr": psnr, "ssim": ssim, "lpips": lpips_val}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    _, val_idx = train_val_split(RUN_DIR)
    val_ds = FullFrameDataset(RUN_DIR, val_idx)
    print(f"held-out frames: {len(val_ds)}")

    model = SpatialUNet().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    print(f"loaded checkpoint: {CHECKPOINT_PATH}")

    lpips_fn = lpips.LPIPS(net="vgg").to(device)

    methods = ["bilinear", "bicubic", "lanczos", "model"]
    results = {m: {"psnr": [], "ssim": [], "lpips": []} for m in methods}

    for i in range(len(val_ds)):
        x, y = val_ds[i]
        x_in = x.unsqueeze(0)  # (1, 4, H, W) -- colour+depth
        x_color = x_in[:, :3]  # baselines only use colour, not depth
        y_target = y.unsqueeze(0)

        preds = {
            "bilinear": upsample_torch(x_color, "bilinear"),
            "bicubic": upsample_torch(x_color, "bicubic"),
            "lanczos": upsample_lanczos(x_color),
            "model": model_predict(model, x_in, device),
        }

        for method, pred in preds.items():
            m = compute_metrics(pred, y_target, lpips_fn, device)
            for k, v in m.items():
                results[method][k].append(v)

        print(f"frame {val_idx[i]}: " + "  ".join(f"{m}_psnr={results[m]['psnr'][-1]:.2f}" for m in methods))

    print("\n--- averages over held-out frames ---")
    print(f"{'method':<10} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
    summary = {}
    for m in methods:
        avg_psnr = float(np.mean(results[m]["psnr"]))
        avg_ssim = float(np.mean(results[m]["ssim"]))
        avg_lpips = float(np.mean(results[m]["lpips"]))
        summary[m] = {"psnr": avg_psnr, "ssim": avg_ssim, "lpips": avg_lpips}
        print(f"{m:<10} {avg_psnr:>8.3f} {avg_ssim:>8.4f} {avg_lpips:>8.4f}")

    beats_bicubic = summary["model"]["psnr"] > summary["bicubic"]["psnr"] and summary["model"]["ssim"] > summary["bicubic"]["ssim"]
    print(f"\n{'PASSED' if beats_bicubic else 'FAILED'}: model beats bicubic on PSNR and SSIM")

    import json

    out_path = Path(__file__).resolve().parent.parent / "checkpoints" / "eval_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"summary written to {out_path}")


if __name__ == "__main__":
    main()

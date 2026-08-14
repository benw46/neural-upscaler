"""Captures PyTorch's intermediate activations at each checkpoint layer the
WGSL harness also exposes, on the exact same fixed input used for the
whole-model ONNX diff -- Spec 4 step 6: diff per layer, not just at the
final output, so a future bug can be localised quickly.

Saved in NHWC layout (permuted from PyTorch's native NCHW) to match the
WGSL pipeline's feature-map layout directly, avoiding a transpose in the
comparison code on either side.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training" / "src"))
from model import SpatialUNet  # noqa: E402

PATCH_SIZE = 128
IN_CHANNELS = 8
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "training" / "checkpoints_temporal" / "temporal_w1.0_lpips02_best.pt"
INPUT_PATH = Path(__file__).resolve().parent / "test_input_temporal.bin"
OUT_DIR = Path(__file__).resolve().parent / "intermediates"

# Module names to hook -- must match model_wgsl.ts's `forward()` checkpoints exactly.
HOOK_NAMES = [
    "stem.conv",
    "down1.refine.conv",
    "down2.refine.conv",
    "down3.refine.conv",
    "bottleneck.1.conv",
    "up1.conv2.conv",
    "up2.conv2.conv",
    "up3.conv2.conv",
    "head",
    "pixel_shuffle",
]


def main():
    model = SpatialUNet(in_channels=IN_CHANNELS)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()

    modules_by_name = dict(model.named_modules())
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name):
        def hook(_module, _input, output):
            captured[name] = output.detach()

        return hook

    for name in HOOK_NAMES:
        if name not in modules_by_name:
            raise KeyError(f"module '{name}' not found -- available: {list(modules_by_name.keys())}")
        handles.append(modules_by_name[name].register_forward_hook(make_hook(name)))

    x_np = np.fromfile(INPUT_PATH, dtype=np.float32).reshape(1, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE)
    x = torch.from_numpy(x_np)

    with torch.no_grad():
        model(x)

    for h in handles:
        h.remove()

    OUT_DIR.mkdir(exist_ok=True)
    for name in HOOK_NAMES:
        tensor = captured[name]  # (1, C, H, W)
        nhwc = tensor.permute(0, 2, 3, 1).contiguous().numpy().astype(np.float32)
        safe_name = name.replace(".", "_")
        out_path = OUT_DIR / f"{safe_name}.bin"
        nhwc.tofile(out_path)
        print(f"{name:20s} shape(NHWC)={tuple(nhwc.shape)[1:]}  -> {out_path.name}")


if __name__ == "__main__":
    main()

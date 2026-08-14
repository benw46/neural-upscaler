"""Same purpose as gen_diff_fixtures.py, for the final trained *temporal*
model (in_channels=8) rather than Phase 2's spatial-only model. Spec 4 step
1: confirm ORT Web still works end to end with the Spec 3 model before
touching any WGSL.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training" / "src"))
from model import SpatialUNet  # noqa: E402

PATCH_SIZE = 128
IN_CHANNELS = 8
SEED = 123
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "training" / "checkpoints_temporal" / "temporal_w1.0_lpips02_best.pt"
OUT_DIR = Path(__file__).resolve().parent


def main():
    torch.manual_seed(SEED)
    model = SpatialUNet(in_channels=IN_CHANNELS)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()

    # Channels: [0:3]=colour [0,1], [3:4]=depth (normalised, small positive
    # values), [4:7]=warped-previous colour [0,1]-ish, [7:8]=disocclusion
    # mask {0,1} -- roughly matching real value ranges rather than pure
    # uniform noise, closer to what the network actually sees.
    x = torch.rand(1, IN_CHANNELS, PATCH_SIZE, PATCH_SIZE)
    x[:, 3:4] *= 0.5  # depth: smaller positive range
    x[:, 7:8] = (torch.rand(1, 1, PATCH_SIZE, PATCH_SIZE) > 0.5).float()  # disocclusion: binary-ish

    with torch.no_grad():
        y = model(x)

    x_np = x.numpy().astype(np.float32)  # NCHW -- what ORT Web expects (PyTorch/ONNX native layout)
    y_np = y.numpy().astype(np.float32)
    # NHWC -- what the hand-written WGSL harness expects (see model_wgsl.ts).
    # A separate file, not a replacement: the ORT Web fixture must stay NCHW.
    x_np_nhwc = x.permute(0, 2, 3, 1).contiguous().numpy().astype(np.float32)

    x_np.tofile(OUT_DIR / "test_input_temporal.bin")
    x_np_nhwc.tofile(OUT_DIR / "test_input_temporal_nhwc.bin")
    y_np.tofile(OUT_DIR / "pytorch_output_temporal.bin")

    print(f"input (NCHW): shape={tuple(x_np.shape)} -> {OUT_DIR / 'test_input_temporal.bin'} ({x_np.nbytes} bytes)")
    print(f"input (NHWC): shape={tuple(x_np_nhwc.shape)} -> {OUT_DIR / 'test_input_temporal_nhwc.bin'} ({x_np_nhwc.nbytes} bytes)")
    print(f"output: shape={tuple(y_np.shape)} -> {OUT_DIR / 'pytorch_output_temporal.bin'} ({y_np.nbytes} bytes)")
    print(f"output range: min={y_np.min():.5f} max={y_np.max():.5f} mean={y_np.mean():.5f}")


if __name__ == "__main__":
    main()

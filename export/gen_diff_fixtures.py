"""Generates a fixed test input and the trained PyTorch model's output on it,
as raw float32 binaries -- used to diff ONNX Runtime Web's output against
PyTorch's on the *same* input (Spec 2 step 6). The actual comparison runs in
the browser (inference/src/main.ts) so ~200K floats of model output don't
need to round-trip back out of the browser -- only the two summary numbers
(max/mean abs error) do.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training" / "src"))
from model import SpatialUNet  # noqa: E402

PATCH_SIZE = 128
SEED = 123
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "training" / "checkpoints" / "best.pt"
OUT_DIR = Path(__file__).resolve().parent


def main():
    torch.manual_seed(SEED)
    model = SpatialUNet()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()

    x = torch.rand(1, 4, PATCH_SIZE, PATCH_SIZE)  # colour channels in [0,1]-like range; depth already normalised in real data
    with torch.no_grad():
        y = model(x)

    x_np = x.numpy().astype(np.float32)
    y_np = y.numpy().astype(np.float32)

    x_np.tofile(OUT_DIR / "test_input.bin")
    y_np.tofile(OUT_DIR / "pytorch_output.bin")

    print(f"input:  shape={tuple(x_np.shape)} -> {OUT_DIR / 'test_input.bin'} ({x_np.nbytes} bytes)")
    print(f"output: shape={tuple(y_np.shape)} -> {OUT_DIR / 'pytorch_output.bin'} ({y_np.nbytes} bytes)")
    print(f"output range: min={y_np.min():.5f} max={y_np.max():.5f} mean={y_np.mean():.5f}")


if __name__ == "__main__":
    main()

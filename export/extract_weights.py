"""Extracts trained conv weights/biases into a flat binary blob + JSON
manifest, loadable directly by the hand-written WGSL inference harness
(Spec 4) without needing PyTorch or ONNX at runtime.

Weight layout per layer: (Cout, 3, 3, Cin) row-major -- i.e. for a fixed
output channel and kernel tap (ky,kx), the Cin input-channel weights are
contiguous. This matches the NHWC feature-map layout the WGSL kernels use:
at a given tap, the shader reads Cin contiguous input values and Cin
contiguous weight values together. PyTorch's native Conv2d weight layout is
(Cout, Cin, 3, 3) -- transposed here, once, offline, rather than making the
shader do strided reads every dispatch.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training" / "src"))
from model import SpatialUNet  # noqa: E402

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "training" / "checkpoints_temporal" / "temporal_w1.0_final_best.pt"
OUT_DIR = Path(__file__).resolve().parent.parent / "inference" / "public" / "weights"

# (layer name, has LeakyReLU activation) -- every conv has one except the
# final head, which feeds straight into pixel-shuffle.
LAYERS = [
    ("stem.conv", True),
    ("down1.down.conv", True),
    ("down1.refine.conv", True),
    ("down2.down.conv", True),
    ("down2.refine.conv", True),
    ("down3.down.conv", True),
    ("down3.refine.conv", True),
    ("bottleneck.0.conv", True),
    ("bottleneck.1.conv", True),
    ("up1.conv1.conv", True),
    ("up1.conv2.conv", True),
    ("up2.conv1.conv", True),
    ("up2.conv2.conv", True),
    ("up3.conv1.conv", True),
    ("up3.conv2.conv", True),
    ("head", False),
]


def main():
    model = SpatialUNet(in_channels=8)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()
    state = model.state_dict()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    blob_parts = []
    offset = 0
    manifest = {"layers": []}

    for name, has_activation in LAYERS:
        weight = state[f"{name}.weight"].detach().numpy()  # (Cout, Cin, 3, 3)
        bias = state[f"{name}.bias"].detach().numpy()  # (Cout,)
        cout, cin, kh, kw = weight.shape
        assert (kh, kw) == (3, 3), f"{name}: expected 3x3 kernel, got {kh}x{kw}"

        stride = 2 if ".down.conv" in name else 1

        # (Cout, Cin, 3, 3) -> (Cout, 3, 3, Cin)
        weight_reordered = np.transpose(weight, (0, 2, 3, 1)).astype(np.float32).copy()

        weight_bytes = weight_reordered.tobytes()
        bias_bytes = bias.astype(np.float32).tobytes()
        blob_parts.append(weight_bytes)
        blob_parts.append(bias_bytes)

        manifest["layers"].append(
            {
                "name": name,
                "inChannels": int(cin),
                "outChannels": int(cout),
                "stride": stride,
                "activation": "leaky_relu" if has_activation else "none",
                "weightOffset": offset,
                "weightBytes": len(weight_bytes),
                "biasOffset": offset + len(weight_bytes),
                "biasBytes": len(bias_bytes),
            }
        )
        offset += len(weight_bytes) + len(bias_bytes)

    blob = b"".join(blob_parts)
    (OUT_DIR / "weights.bin").write_bytes(blob)
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"wrote {len(blob):,} bytes across {len(LAYERS)} layers to {OUT_DIR / 'weights.bin'}")
    print(f"manifest: {OUT_DIR / 'manifest.json'}")
    for layer in manifest["layers"]:
        print(f"  {layer['name']:20s} in={layer['inChannels']:4d} out={layer['outChannels']:4d} stride={layer['stride']} act={layer['activation']}")


if __name__ == "__main__":
    main()

"""Exports SpatialUNet to ONNX.

Run standalone to export the *untrained* (randomly-initialised) model, per
CLAUDE.md: "Check ONNX export compatibility during Phase 2, not Phase 4."
This must pass before any real training time is spent -- an export/operator
problem discovered after training would waste the training run.

Usage:
    python export_onnx.py --out spatial_unet.onnx [--weights path/to/state_dict.pt]
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training" / "src"))
from model import SpatialUNet  # noqa: E402

PATCH_SIZE = 128


def export(out_path: str, weights_path: str | None = None):
    model = SpatialUNet()
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        print(f"loaded weights from {weights_path}")
    else:
        print("exporting untrained (randomly-initialised) model")
    model.eval()

    dummy_input = torch.randn(1, 4, PATCH_SIZE, PATCH_SIZE)

    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height2x", 3: "width2x"},
        },
        opset_version=18,
    )
    print(f"exported to {out_path}")

    # The exporter defaults to writing weights as an external ".onnx.data"
    # file. ONNX Runtime Web can't resolve that relative-path reference on
    # its own in a browser (no arbitrary filesystem access) -- it needs the
    # external data passed explicitly as session-creation options, which
    # complicates the harness for no benefit at this model size (~3.6MB).
    # Consolidate into one self-contained file instead.
    import onnx

    onnx_model = onnx.load(out_path, load_external_data=True)
    onnx.save_model(onnx_model, out_path, save_as_external_data=False)
    external_data_path = Path(out_path).parent / (Path(out_path).name + ".data")
    if external_data_path.exists():
        external_data_path.unlink()
    print(f"consolidated into a single self-contained file (removed {external_data_path.name})")

    onnx.checker.check_model(onnx.load(out_path))
    print("onnx.checker.check_model passed")

    ops = sorted({node.op_type for node in onnx_model.graph.node})
    print(f"operators used ({len(ops)}): {', '.join(ops)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(Path(__file__).parent / "spatial_unet.onnx"))
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()
    export(args.out, args.weights)

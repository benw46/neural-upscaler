"""Copies a handful of frames' raw buffers from the E:\\ dataset into
inference/public/demo_frames/ for the live 540p-vs-network-1080p viewer
(inference/viewer.html). Not the dataset itself -- just enough frames for
the demo to work without the browser needing filesystem access to E:\\.
Regenerable; inference/public/demo_frames/ is gitignored like every other
generated asset under inference/public/.

Frames chosen from the Spec 3 gate's held-out block (val_fraction=0.25 ->
frames 1500-1999, see test_long_sequence.py) -- genuinely held-out, not
frames the model trained on, spread across that block for variety.
"""

import json
import shutil
from pathlib import Path

RUN_DIR = Path(r"E:\neural-upscaler\data\seed-20260812")
OUT_DIR = Path(__file__).resolve().parent.parent / "inference" / "public" / "demo_frames"
DEMO_FRAMES = [1500, 1650, 1800, 1949]
BUFFERS = ["color", "depth", "gt_color"]


def main():
    header = json.loads((RUN_DIR / "dataset.json").read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for buf in BUFFERS:
        (OUT_DIR / buf).mkdir(exist_ok=True)

    for idx in DEMO_FRAMES:
        fname = f"{idx:06d}.bin"
        for buf in BUFFERS:
            src = RUN_DIR / buf / fname
            if not src.exists():
                raise FileNotFoundError(src)
            shutil.copy2(src, OUT_DIR / buf / fname)
        print(f"frame {idx}: copied {', '.join(BUFFERS)}")

    manifest = {
        "inputWidth": header["inputWidth"],
        "inputHeight": header["inputHeight"],
        "gtWidth": header["gtWidth"],
        "gtHeight": header["gtHeight"],
        "frames": DEMO_FRAMES,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest -> {OUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()

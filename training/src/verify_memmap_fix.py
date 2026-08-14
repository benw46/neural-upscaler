"""One-off verification for the memmap I/O fix (see MEMORY.md's
dataloader-memmap-io-fix note): (1) bit-exact correctness against the old
np.fromfile-based read, (2) an empirical before/after timing comparison on
this actual Windows/SATA machine, not just the theoretical ~4x estimate.

Not part of the training pipeline -- throwaway, not imported anywhere.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

RUN_DIR = Path(r"E:\neural-upscaler\data\seed-20260812")
PATCH_SIZE = 128
N_TIMING_DRAWS = 400


def load_fromfile(frame_idx: int, w: int, h: int, gw: int, gh: int):
    fname = f"{frame_idx:06d}.bin"
    color = np.fromfile(RUN_DIR / "color" / fname, dtype=np.float16).reshape(h, w, 4)
    depth = np.fromfile(RUN_DIR / "depth" / fname, dtype=np.float32).reshape(h, w, 1)
    gt = np.fromfile(RUN_DIR / "gt_color" / fname, dtype=np.float16).reshape(gh, gw, 4)
    return color, depth, gt


def load_memmap(frame_idx: int, w: int, h: int, gw: int, gh: int):
    fname = f"{frame_idx:06d}.bin"
    color = np.memmap(RUN_DIR / "color" / fname, dtype=np.float16, mode="r", shape=(h, w, 4))
    depth = np.memmap(RUN_DIR / "depth" / fname, dtype=np.float32, mode="r", shape=(h, w, 1))
    gt = np.memmap(RUN_DIR / "gt_color" / fname, dtype=np.float16, mode="r", shape=(gh, gw, 4))
    return color, depth, gt


def crop(color, depth, gt, x, y, ps):
    color_patch = color[y : y + ps, x : x + ps, :3].astype(np.float32)
    depth_patch = depth[y : y + ps, x : x + ps, :].astype(np.float32)
    gt_patch = gt[y * 2 : y * 2 + ps * 2, x * 2 : x * 2 + ps * 2, :3].astype(np.float32)
    return color_patch, depth_patch, gt_patch


def main():
    header = json.loads((RUN_DIR / "dataset.json").read_text())
    w, h, gw, gh = header["inputWidth"], header["inputHeight"], header["gtWidth"], header["gtHeight"]

    # --- correctness: bit-exact check across several frames/offsets ---
    rng = np.random.default_rng(0)
    print("=== correctness check ===")
    all_match = True
    for _ in range(8):
        frame_idx = int(rng.integers(0, 1700))
        x = int(rng.integers(0, w - PATCH_SIZE))
        y = int(rng.integers(0, h - PATCH_SIZE))

        c1, d1, g1 = crop(*load_fromfile(frame_idx, w, h, gw, gh), x, y, PATCH_SIZE)
        c2, d2, g2 = crop(*load_memmap(frame_idx, w, h, gw, gh), x, y, PATCH_SIZE)

        match = np.array_equal(c1, c2) and np.array_equal(d1, d2) and np.array_equal(g1, g2)
        all_match &= match
        print(f"  frame={frame_idx} x={x} y={y}: {'MATCH' if match else 'MISMATCH'}")

    print(f"\n{'ALL MATCH -- bit-exact' if all_match else 'MISMATCH FOUND -- do not trust the fix'}\n")
    if not all_match:
        sys.exit(1)

    # --- timing: N random patch draws, fromfile vs memmap ---
    # Two DISJOINT sets of (frame, x, y) draws, not the same set reused --
    # reusing one list for both timed passes would let whichever method runs
    # second unfairly benefit from the first pass's OS file-cache warming,
    # inflating its apparent speedup. Separate draws (same distribution,
    # same count) avoid that cross-contamination.
    print(f"=== timing ({N_TIMING_DRAWS} random patch draws each, disjoint draw sets) ===")
    draws_a = [(int(rng.integers(0, 1700)), int(rng.integers(0, w - PATCH_SIZE)), int(rng.integers(0, h - PATCH_SIZE))) for _ in range(N_TIMING_DRAWS)]
    draws_b = [(int(rng.integers(0, 1700)), int(rng.integers(0, w - PATCH_SIZE)), int(rng.integers(0, h - PATCH_SIZE))) for _ in range(N_TIMING_DRAWS)]

    t0 = time.perf_counter()
    for frame_idx, x, y in draws_a:
        crop(*load_fromfile(frame_idx, w, h, gw, gh), x, y, PATCH_SIZE)
    t_fromfile = time.perf_counter() - t0

    t0 = time.perf_counter()
    for frame_idx, x, y in draws_b:
        crop(*load_memmap(frame_idx, w, h, gw, gh), x, y, PATCH_SIZE)
    t_memmap = time.perf_counter() - t0

    print(f"  np.fromfile: {t_fromfile:.3f}s ({t_fromfile / N_TIMING_DRAWS * 1000:.2f}ms/draw)")
    print(f"  np.memmap:   {t_memmap:.3f}s ({t_memmap / N_TIMING_DRAWS * 1000:.2f}ms/draw)")
    print(f"  speedup: {t_fromfile / t_memmap:.2f}x")


if __name__ == "__main__":
    main()

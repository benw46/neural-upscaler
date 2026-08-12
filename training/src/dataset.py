"""PyTorch Dataset over the Spec 1 capture output.

Reads raw .bin buffers directly (no intermediate format) per the manifest
written by renderer/scripts/capture.ts. Random 128x128 low-res patches with
the matching 256x256 ground-truth patch.

Picklable by design (stores only paths/plain data, opens files fresh per
__getitem__) -- required for Windows' spawn-based DataLoader workers per
CLAUDE.md: fork isn't available, so anything captured in worker closures
must survive being pickled and re-imported in a fresh process.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Depth normalisation: divide raw linear view-space depth by this constant.
# Empirical, tied to this specific scene's extent (observed max ~44 world
# units against a far plane of 200; see docs/PHASE-0-1-SUMMARY.md) -- not
# derived from first principles. CLAUDE.md fragile-logic list: preprocessing
# parity between training and inference matters, so this exact constant must
# be replicated by any future inference path (ONNX export, then Spec 4's
# hand-written WGSL) rather than re-derived independently.
DEPTH_NORM = 50.0


class SpatialPatchDataset(Dataset):
    """Random-crop training dataset -- a fresh random patch location each
    call, like standard crop augmentation. For deterministic/reproducible
    evaluation (baseline comparison, gate metrics), use `FullFrameDataset`
    instead, not this class with a fixed index."""

    def __init__(self, run_dir: str, frame_indices: list[int], patch_size: int = 128):
        self.run_dir = Path(run_dir)
        header = json.loads((self.run_dir / "dataset.json").read_text())
        self.input_w = header["inputWidth"]
        self.input_h = header["inputHeight"]
        self.gt_w = header["gtWidth"]
        self.gt_h = header["gtHeight"]
        self.frame_indices = list(frame_indices)
        self.patch_size = patch_size

    def __len__(self) -> int:
        return len(self.frame_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        frame_idx = self.frame_indices[idx]
        fname = f"{frame_idx:06d}.bin"

        color = np.fromfile(self.run_dir / "color" / fname, dtype=np.float16).reshape(self.input_h, self.input_w, 4)
        depth = np.fromfile(self.run_dir / "depth" / fname, dtype=np.float32).reshape(self.input_h, self.input_w, 1)
        gt = np.fromfile(self.run_dir / "gt_color" / fname, dtype=np.float16).reshape(self.gt_h, self.gt_w, 4)

        max_x = self.input_w - self.patch_size
        max_y = self.input_h - self.patch_size
        x = int(np.random.randint(0, max_x + 1))
        y = int(np.random.randint(0, max_y + 1))
        ps = self.patch_size

        color_patch = color[y : y + ps, x : x + ps, :3].astype(np.float32)
        depth_patch = depth[y : y + ps, x : x + ps, :].astype(np.float32) / DEPTH_NORM
        input_patch = np.concatenate([color_patch, depth_patch], axis=-1)  # (ps, ps, 4)

        gt_x, gt_y, gt_ps = x * 2, y * 2, ps * 2
        gt_patch = gt[gt_y : gt_y + gt_ps, gt_x : gt_x + gt_ps, :3].astype(np.float32)

        input_tensor = torch.from_numpy(input_patch).permute(2, 0, 1).contiguous()
        target_tensor = torch.from_numpy(gt_patch).permute(2, 0, 1).contiguous()
        return input_tensor, target_tensor


def load_fixed_patch(run_dir: str, frame_idx: int, x: int, y: int, patch_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic single-patch loader -- no randomness, same patch every
    call. For the overfit-single-patch sanity check (Spec 2 step 3), which
    needs one fixed reproducible example, not `SpatialPatchDataset`'s
    fresh-random-crop-per-call behaviour."""
    run_dir = Path(run_dir)
    header = json.loads((run_dir / "dataset.json").read_text())
    input_w, input_h = header["inputWidth"], header["inputHeight"]
    gt_w, gt_h = header["gtWidth"], header["gtHeight"]
    fname = f"{frame_idx:06d}.bin"

    color = np.fromfile(run_dir / "color" / fname, dtype=np.float16).reshape(input_h, input_w, 4)
    depth = np.fromfile(run_dir / "depth" / fname, dtype=np.float32).reshape(input_h, input_w, 1)
    gt = np.fromfile(run_dir / "gt_color" / fname, dtype=np.float16).reshape(gt_h, gt_w, 4)

    ps = patch_size
    color_patch = color[y : y + ps, x : x + ps, :3].astype(np.float32)
    depth_patch = depth[y : y + ps, x : x + ps, :].astype(np.float32) / DEPTH_NORM
    input_patch = np.concatenate([color_patch, depth_patch], axis=-1)

    gt_x, gt_y, gt_ps = x * 2, y * 2, ps * 2
    gt_patch = gt[gt_y : gt_y + gt_ps, gt_x : gt_x + gt_ps, :3].astype(np.float32)

    input_tensor = torch.from_numpy(input_patch).permute(2, 0, 1).contiguous()
    target_tensor = torch.from_numpy(gt_patch).permute(2, 0, 1).contiguous()
    return input_tensor, target_tensor


class FullFrameDataset(Dataset):
    """Whole-frame loader (no cropping) for deterministic evaluation --
    baseline comparison and gate metrics need the same held-out content
    every time, not a fresh random crop per call."""

    def __init__(self, run_dir: str, frame_indices: list[int]):
        self.run_dir = Path(run_dir)
        header = json.loads((self.run_dir / "dataset.json").read_text())
        self.input_w = header["inputWidth"]
        self.input_h = header["inputHeight"]
        self.gt_w = header["gtWidth"]
        self.gt_h = header["gtHeight"]
        self.frame_indices = list(frame_indices)

    def __len__(self) -> int:
        return len(self.frame_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        frame_idx = self.frame_indices[idx]
        fname = f"{frame_idx:06d}.bin"

        color = np.fromfile(self.run_dir / "color" / fname, dtype=np.float16).reshape(self.input_h, self.input_w, 4)
        depth = np.fromfile(self.run_dir / "depth" / fname, dtype=np.float32).reshape(self.input_h, self.input_w, 1)
        gt = np.fromfile(self.run_dir / "gt_color" / fname, dtype=np.float16).reshape(self.gt_h, self.gt_w, 4)

        color_f = color[:, :, :3].astype(np.float32)
        depth_f = depth.astype(np.float32) / DEPTH_NORM
        input_full = np.concatenate([color_f, depth_f], axis=-1)
        gt_full = gt[:, :, :3].astype(np.float32)

        input_tensor = torch.from_numpy(input_full).permute(2, 0, 1).contiguous()
        target_tensor = torch.from_numpy(gt_full).permute(2, 0, 1).contiguous()
        return input_tensor, target_tensor


# Encoder has 3 stride-2 downsamples (see model.py) -> spatial dims must be
# divisible by 8, but the dataset's native resolution (960x540) isn't
# (540/8 = 67.5). Reflect-pad up to the nearest multiple of 8 before
# inference, then crop the (2x) output back down to the exact un-padded
# target size -- standard technique, needed for any full-frame (as opposed
# to 128x128 patch) inference.
NET_STRIDE = 8


def pad_to_multiple(x: torch.Tensor, multiple: int = NET_STRIDE) -> tuple[torch.Tensor, tuple[int, int]]:
    h, w = x.shape[-2], x.shape[-1]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (h, w)
    x_padded = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x_padded, (h, w)


def crop_to_size(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    h, w = size
    return x[..., :h, :w]


def train_val_split(run_dir: str, val_fraction: float = 0.15) -> tuple[list[int], list[int]]:
    """Held-out frames are a contiguous trailing block, not randomly
    interspersed -- adjacent frames in this dataset are highly correlated
    (slow, smooth camera motion), so random interspersion would leak most of
    a held-out frame's content via its near-identical neighbours in the
    training set. A held-out block the model never saw anything adjacent to
    is a meaningfully harder, more honest test."""
    header = json.loads((Path(run_dir) / "dataset.json").read_text())
    n = header["frameCount"]
    n_val = max(1, int(n * val_fraction))
    train_indices = list(range(0, n - n_val))
    val_indices = list(range(n - n_val, n))
    return train_indices, val_indices


if __name__ == "__main__":
    RUN_DIR = r"E:\neural-upscaler\data\seed-20260812"
    train_idx, val_idx = train_val_split(RUN_DIR)
    print(f"train frames: {len(train_idx)} ({train_idx[0]}..{train_idx[-1]})")
    print(f"val frames:   {len(val_idx)} ({val_idx[0]}..{val_idx[-1]})")

    ds = SpatialPatchDataset(RUN_DIR, train_idx)
    x, y = ds[0]
    print(f"input patch:  {tuple(x.shape)} dtype={x.dtype} min={x.min():.4f} max={x.max():.4f}")
    print(f"target patch: {tuple(y.shape)} dtype={y.dtype} min={y.min():.4f} max={y.max():.4f}")

    # Two draws of the same index should differ (random patch location per call).
    x2, _ = ds[0]
    print(f"same index, different patch each call: {not torch.equal(x, x2)}")

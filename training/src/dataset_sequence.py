"""Sequence dataset for recurrent temporal training (Spec 3 step 3).

Returns short (4-8 frame) windows of *consecutive* frames sharing a single
random patch location across the whole window -- cropping a different
random region per frame would make the stored motion vectors meaningless
(they describe how content moves *within a fixed screen-space window* over
time, which is only true if the window itself doesn't move between frames).
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset import DEPTH_NORM


class SequenceDataset(Dataset):
    def __init__(self, run_dir: str, frame_indices: list[int], seq_len: int = 6, patch_size: int = 128):
        self.run_dir = Path(run_dir)
        header = json.loads((self.run_dir / "dataset.json").read_text())
        self.input_w = header["inputWidth"]
        self.input_h = header["inputHeight"]
        self.gt_w = header["gtWidth"]
        self.gt_h = header["gtHeight"]
        self.seq_len = seq_len
        self.patch_size = patch_size

        # frame_indices is assumed a contiguous sorted range (true for the
        # train/val splits this project uses) -- valid sequence starts are
        # any index with `seq_len` consecutive frames still inside it.
        frame_indices = sorted(frame_indices)
        assert frame_indices == list(range(frame_indices[0], frame_indices[-1] + 1)), "frame_indices must be contiguous"
        self.first, self.last = frame_indices[0], frame_indices[-1]
        self.valid_starts = list(range(self.first, self.last - seq_len + 2))
        if not self.valid_starts:
            raise ValueError(f"no valid sequence starts: {len(frame_indices)} frames, seq_len={seq_len}")

    def __len__(self) -> int:
        return len(self.valid_starts)

    def _load_frame(self, frame_idx: int):
        # memmap, not fromfile -- see MEMORY.md's dataloader-memmap-io-fix
        # note. Only the rows touched by __getitem__'s crop below get paged
        # in from disk, instead of all 4 full channels being read per frame
        # just to harvest one shared small patch across the sequence window
        # (this was the ~223GB/epoch source for Phase 3). The .astype calls
        # in __getitem__ already force a real copy at the point each patch
        # is sliced out, so correctness is unaffected.
        fname = f"{frame_idx:06d}.bin"
        color = np.memmap(self.run_dir / "color" / fname, dtype=np.float16, mode="r", shape=(self.input_h, self.input_w, 4))
        depth = np.memmap(self.run_dir / "depth" / fname, dtype=np.float32, mode="r", shape=(self.input_h, self.input_w, 1))
        motion = np.memmap(self.run_dir / "motion" / fname, dtype=np.float16, mode="r", shape=(self.input_h, self.input_w, 2))
        gt = np.memmap(self.run_dir / "gt_color" / fname, dtype=np.float16, mode="r", shape=(self.gt_h, self.gt_w, 4))
        return color, depth, motion, gt

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = self.valid_starts[idx]
        ps = self.patch_size

        max_x = self.input_w - ps
        max_y = self.input_h - ps
        x = int(np.random.randint(0, max_x + 1))
        y = int(np.random.randint(0, max_y + 1))
        gt_x, gt_y, gt_ps = x * 2, y * 2, ps * 2

        colors, depths, motions, gts = [], [], [], []
        for t in range(self.seq_len):
            color, depth, motion, gt = self._load_frame(start + t)

            color_patch = color[y : y + ps, x : x + ps, :3].astype(np.float32)
            depth_patch = depth[y : y + ps, x : x + ps, :].astype(np.float32) / DEPTH_NORM
            motion_patch = motion[y : y + ps, x : x + ps, :].astype(np.float32)
            gt_patch = gt[gt_y : gt_y + gt_ps, gt_x : gt_x + gt_ps, :3].astype(np.float32)

            colors.append(color_patch)
            depths.append(depth_patch)
            motions.append(motion_patch)
            gts.append(gt_patch)

        # (seq_len, C, ps, ps) / (seq_len, 3, ps*2, ps*2)
        color_t = torch.from_numpy(np.stack(colors)).permute(0, 3, 1, 2).contiguous()
        depth_t = torch.from_numpy(np.stack(depths)).permute(0, 3, 1, 2).contiguous()
        motion_t = torch.from_numpy(np.stack(motions)).permute(0, 3, 1, 2).contiguous()
        gt_t = torch.from_numpy(np.stack(gts)).permute(0, 3, 1, 2).contiguous()

        return {"color": color_t, "depth": depth_t, "motion": motion_t, "gt": gt_t, "start_frame": start}


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from dataset import train_val_split

    RUN_DIR = r"E:\neural-upscaler\data\seed-20260812"
    train_idx, val_idx = train_val_split(RUN_DIR, val_fraction=0.25)
    print(f"train frames: {len(train_idx)} ({train_idx[0]}..{train_idx[-1]})")
    print(f"val frames:   {len(val_idx)} ({val_idx[0]}..{val_idx[-1]})")

    ds = SequenceDataset(RUN_DIR, train_idx, seq_len=6)
    print(f"valid sequence starts: {len(ds)}")
    sample = ds[0]
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)} dtype={v.dtype} min={v.min():.4f} max={v.max():.4f}")
        else:
            print(f"{k}: {v}")

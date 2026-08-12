# Phase 2 Summary — Spatial Model

Covers Spec 2 in full: a PyTorch spatial (non-temporal) 2x upscaler, proven
against baselines and proven exportable to ONNX/WebGPU.

## Environment setup

Python wasn't actually installed at the start of this phase (`python` on
PATH was just the Windows Store alias stub). Set up: Python 3.12 (winget),
a venv at `training/.venv` (PowerShell's default execution policy blocks
`Activate.ps1`, so tooling always invokes the venv's `python.exe` by full
path rather than activating — also more robust for this workflow generally,
since each tool invocation is a fresh shell that wouldn't retain activation
state anyway), PyTorch 2.11.0+cu128, confirmed against the RTX 3060
(`torch.cuda.is_available()` — CLAUDE.md's explicit instruction not to
assume this).

**A real gotcha worth flagging for future work:** installing `lpips` pulled
in `torchvision`, which pulled in a *CPU-only* `torch` from PyPI's default
index and silently replaced the CUDA build. Any future package that depends
on `torch`/`torchvision` needs `--index-url https://download.pytorch.org/whl/cu128`
pinned explicitly, or `torch.cuda.is_available()` should be re-checked
immediately after installing it.

## What was built

- `training/src/model.py` — `SpatialUNet`, 915,360 params (target was
  0.5-1M). 3 encoder downsamples / 3 decoder upsamples with skip
  connections, all at input resolution, followed by a dedicated
  pixel-shuffle head for the actual 2x upscale. No normalisation layers
  (common for SR networks — norm can suppress fine detail; also avoids
  running-stats/eval-mode complications for ONNX export and Spec 4's
  hand-written WGSL). **Architecture-reading judgement call:** Spec 2's
  "input res -> down -> down -> down -> bottleneck -> up -> up -> up ->
  output res" is ambiguous about whether the down/up chain itself reaches
  2x resolution or returns to input resolution with a separate upsample
  head. Read as the latter (standard, parameter-efficient SR pattern,
  keeps skip connections at matching resolutions) — documented in the
  module docstring rather than silently assumed.
- `training/src/dataset.py` — reads the Spec 1 capture output directly (raw
  `.bin` buffers, no intermediate format). `SpatialPatchDataset` for
  training (random 128x128 crops, fresh location every call — standard
  crop augmentation). `FullFrameDataset` for deterministic evaluation.
  `pad_to_multiple`/`crop_to_size` handle full-frame inference: the encoder
  needs dimensions divisible by 8, but the dataset's native 960x540 isn't
  (540/8 = 67.5) — reflect-pad up, run, crop back down.
- `training/src/losses.py` — `CombinedLoss` (L1 + LPIPS/VGG). L1 alone
  pushes toward the pixelwise average of several plausible high-res
  reconstructions of ambiguous/aliased fine detail — i.e. blurry output
  even when the network has correctly localised structure. LPIPS evaluates
  similarity in a pretrained deep feature space that correlates much better
  with human judgements of sharpness, pulling the network away from that
  "safe blurry average." L1 stays dominant (weight 1.0 vs LPIPS's 0.1) so
  LPIPS refines rather than dominates.
- `training/src/overfit_test.py`, `train.py`, `evaluate.py` — see below.
- `export/export_onnx.py`, `export/gen_diff_fixtures.py` — ONNX export
  (consolidated to one self-contained file, not the exporter's default
  external-data layout — see "ONNX export" below) and fixed-input diff
  fixture generation.
- `inference/` — new Vite+TS project, `onnxruntime-web` with the WebGPU
  execution provider, used both for the early operator-compatibility check
  and the final PyTorch-vs-ONNX-Runtime-Web diff.

## The overfit-single-patch test: two rounds of misdiagnosis

Worth recording in full since both wrong turns looked plausible at the
time — this is exactly the kind of thing CLAUDE.md's fragile-logic
philosophy warns about (diagnose systematically, don't guess):

1. **First run** (combined L1+LPIPS, 3000 iterations) plateaued at L1~0.05
   instead of near-zero. Looked like a possible model/data bug.
2. **Ablation**: an L1-only run against "the same" example converged
   cleanly to 0.0049 — seemed to clear the pipeline. But a *second*
   combined-loss run gave a completely different result (L1~0.0016) than
   the first (~0.05). That inconsistency was the actual bug, just not in
   the model: `SpatialPatchDataset` picks a fresh random crop location on
   *every* call by design (training augmentation), and neither that nor the
   model's random weight init were seeded — "the same" overfit run was
   silently training on a different patch against a different init each
   time. Fixed with `load_fixed_patch` (one deterministic patch, no
   randomness) and explicit seeding.
3. With reproducibility fixed, the *real* signal appeared: L1-only training
   diverged mid-run (loss jumped from 0.055 to 15,158 around iteration
   1500-2000), while combined-loss training didn't — which looked like
   LPIPS providing implicit regularisation against a genuine instability.
   Also plausible, also wrong: adding gradient clipping (`clip_grad_norm_`,
   max_norm=1.0 — this architecture has no normalisation layers, so
   unclipped activations/gradients can grow unbounded) fixed *both* modes
   cleanly, with combined loss converging slightly *better* than L1-only
   once training is stable (0.0068 vs 0.0083 final L1). There was never a
   genuine tension between the two loss terms. `train.py` uses gradient
   clipping as standard; no warmup was needed.

## Training

`training/src/train.py`: 40 epochs, batch size 16, Adam (LR 1e-3, cosine
schedule), gradient clipping (max_norm=1.0), 425 training frames / 75
held-out (contiguous trailing block — adjacent frames are highly correlated
given the slow scripted camera motion, so a random interspersed split would
leak most of a held-out frame's content via its near-identical neighbours).
Completed in 3143s (~52 min) on the RTX 3060.

Validation L1 trajectory shows genuine convergence, not premature stopping:
0.0238 (epoch 4) -> 0.0197 (epoch 12) -> 0.0183 (epoch 32) -> 0.0182 (epoch
36) -> 0.0182 (epoch 40) — under 1% relative change over the last 12 epochs.

**Live training visualisation** (added after this run, ready for the next
one): `train.py` now logs to TensorBoard — scalar loss curves every batch,
plus [input | prediction | ground truth] sample-image grids for 3 fixed
held-out frames every validation pass, so actual output quality is visible
improving over epochs, not just a loss number. View with
`tensorboard --logdir training/runs` while training runs; updates live.

## Baseline comparison (gate)

75 held-out frames, `training/src/evaluate.py`:

| method | PSNR | SSIM | LPIPS |
|---|---|---|---|
| bilinear | 23.245 | 0.8970 | 0.1381 |
| bicubic | 22.556 | 0.8698 | 0.1588 |
| lanczos | 22.505 | 0.8574 | 0.1862 |
| **model** | **22.365** | **0.9058** | **0.0768** |

Model beats bicubic decisively on SSIM (+0.036) and LPIPS (0.077 vs 0.159 —
nearly 2x better), but is marginally worse on PSNR (22.365 vs 22.556, a
0.19dB gap). This is the well-documented PSNR-vs-perceptual-quality
trade-off in the SR literature: PSNR is minimised by predicting a blurry
pixelwise average, which is exactly what the LPIPS loss term steers away
from — note bilinear (the blurriest baseline) has the *highest* PSNR of all
four methods here, a signature of the same effect, not a coincidence. This
was flagged as a genuine judgement call rather than silently declared a
pass or fail; **decision (owner): accept as passing** — 2 of 3 metrics win
decisively, and the PSNR gap is small and mechanistically understood rather
than symptomatic of a problem.

## ONNX export and WebGPU diff (gate)

`export/export_onnx.py` exports via `torch.onnx.export` (opset 18). The
exporter's default output is an external-data layout (a separate
`.onnx.data` weights file) — ONNX Runtime Web can't resolve that relative
reference on its own in a browser sandbox (no arbitrary filesystem access;
it needs external data passed explicitly as session-creation options), so
the export step consolidates into one self-contained `.onnx` file instead
(`onnx.save_model(..., save_as_external_data=False)`), which is simpler
than teaching the harness to supply external data at this model size
(~3.6MB either way).

5 operators used, all with confirmed working WebGPU kernels (verified by
actually running inference in `onnxruntime-web`, not just checking the
graph): `Conv`, `LeakyRelu`, `Concat`, `Resize` (the nearest-upsample), and
`DepthToSpace` (the pixel-shuffle head).

Diff against PyTorch on an identical fixed input (`export/gen_diff_fixtures.py`
generates the input + PyTorch's reference output as raw binaries;
`inference/src/main.ts` runs the same input through ONNX Runtime Web's
WebGPU backend and diffs client-side, so ~200K floats of output never need
to round-trip back out of the browser — only the two summary numbers do):

```
max abs error:  0.000001
mean abs error: 0.000000
```

Essentially floating-point noise, an order of magnitude inside the FP16
tolerance (0.01) the gate asks for.

## Gate results

- **Beats bicubic on held-out frames**: SSIM and LPIPS yes (LPIPS by a wide
  margin), PSNR marginally no — accepted per the judgement call above, full
  numbers in the table.
- **Trained model round-trips through ONNX Runtime Web within FP16
  tolerance**: yes, max abs error 0.000001, mean 0.000000.

## Where the fragile logic lives

- `training/src/dataset.py` — `DEPTH_NORM = 50.0`, empirical, tied to this
  scene's extent. Any future inference path (WGSL, Spec 4) must replicate
  this exact constant — preprocessing parity between training and inference
  is a named CLAUDE.md fragile-logic item.
- `training/src/model.py` — the down/up-chain-vs-2x-output ambiguity
  (documented in the module docstring); `NET_STRIDE = 8` in dataset.py
  depends on the encoder's exact depth (3 downsamples) and must be updated
  together if the architecture changes.
- `training/src/train.py` — gradient clipping is load-bearing, not
  optional, given the no-normalisation architecture; see the overfit-test
  saga above for why.
- `export/export_onnx.py` — the external-data consolidation step. If a
  future, much larger model makes single-file export impractical, the
  inference harness will need to actually handle ORT Web's `externalData`
  session option instead.

## Known gaps / notes for later phases

- The environment-setup CUDA-breaking `lpips`/`torchvision` install gotcha
  (above) — worth a quick `torch.cuda.is_available()` check after any new
  Python package install in Phase 3, not just an assumption it's still fine.
- PSNR trade-off: if held-out PSNR needs to beat bicubic outright in a
  later phase (e.g. if Phase 3's temporal metrics are sensitive to this),
  reducing the LPIPS loss weight and/or training longer are the levers to
  pull — not attempted here since the owner accepted the current trade-off.

# Neural Upscaler — Project Rules

Persistent context for Claude Code. Read this before any work in this repo.

---

## What this is

A temporal neural upscaler (DLSS/PSSR-style) built from scratch: a WebGPU
rasteriser generates training data, a small U-Net is trained in PyTorch, and
inference runs in the browser via WebGPU.

**Claude Code builds this end to end.** The owner will review, learn from, and
extend the result afterward — but for now, optimise for a correct, working
system across all phases. No task is off-limits to write, including WGSL
compute kernels, motion vector math, and loss design.

This does not relax the quality bar. A fast wrong answer is worse than a
slower correct one — bugs in motion vectors or jitter are invisible until
much later and quietly cap quality forever. Move quickly through mechanical
work; be careful and explicit about anything in the "fragile" list below.

---

## Environment

- **Windows 11, entirely native — no WSL2.** Training, dataset generation,
  and browser/WebGPU work all run directly on Windows.
- Python + PyTorch installed natively with a CUDA-enabled build. Verify
  `torch.cuda.is_available()` returns `True` before assuming GPU training
  works — native Windows uses the Windows NVIDIA driver directly, which is a
  different path from WSL2's CUDA passthrough.
- GPU: RTX 3060 12 GB · CPU: Ryzen 5600X (6C/12T) · RAM: 32 GB DDR4-3000
- Ample disk available
- **PyTorch `DataLoader` on Windows uses `spawn`, not `fork`, for worker
  processes.** This means: (a) the training entry point must be guarded with
  `if __name__ == "__main__":`, or workers will recursively relaunch the
  whole script; (b) dataset/collate code must be picklable; (c) worker
  startup is slower than on Linux — don't mistake this for a training
  slowdown when profiling.

---

## Hard rules

1. **Work at 540p → 1080p** (2x upscale) throughout. Do not use 1080p → 4K
   unless explicitly told to.
2. **Store datasets uncompressed, on the `E:` drive** (e.g. `E:\neural-upscaler\data`),
   not on the system drive. Confirm the path exists and has enough free space
   before generating; report the projected size before writing.
3. **Never advance a phase until its gate passes.** If a gate fails, stop and
   report clearly what failed and why — do not silently work around it or
   quietly loosen the gate's criteria to make it pass.
4. **No new dependency without asking**, especially anything that ends up in
   the browser bundle.
5. **Commit at every gate.**
6. **Deterministic camera paths.** Dataset generation reproducible from a seed.
7. **Explain what you built, in plain terms, at the end of each phase** — a
   short written summary of the approach, the key decisions made, and where
   the fragile logic lives. The owner will read this before reviewing code.
   This is not optional: it's the main thing that makes later learning
   possible.

---

## The validation chain — do not skip links

Three implementations of the same network exist deliberately:

```
PyTorch  (ground truth — trains the model)
   ↓  export
ONNX Runtime Web  (proves it runs in-browser; diff vs PyTorch)
   ↓  reimplement
Hand-written WGSL  (the version being optimised; diff vs ORT Web)
```

When WGSL output is wrong, diff against ORT Web first to localise whether the
bug is in the kernels, the export, or the model — don't guess.

**Check ONNX export compatibility during Phase 2, not Phase 4.** Export an
untrained, randomly-initialised model and confirm every operator has a
working WebGPU kernel in ONNX Runtime Web before investing in training.

---

## Fragile logic — get these right, and say clearly when you're uncertain

These are the specific places where a plausible-looking bug silently caps
quality rather than crashing. Write them carefully, test them explicitly, and
flag any assumption you're making rather than asserting confidence you don't
have.

- **Motion vector sign and coordinate space.** Several conventions are
  plausible; only one is consistent with the projection setup used elsewhere
  in the codebase. State which convention you chose and why.
- **Jitter must be applied to the projection matrix**, not by translating the
  camera (translating changes parallax, which is wrong).
- **WebGPU NDC conventions** — Y is down in framebuffer space, depth is 0..1.
  Verify depth visually (near = dark, far = light) rather than assuming.
- **Ground truth antialiasing.** Supersample and downfilter, or accumulate
  jittered samples — a merely-high-resolution GT is not sufficient.
- **Preprocessing parity between training and inference.** Any mismatch here
  produces subtly-wrong output that's hard to attribute.
- **Loss weighting for temporal consistency.** This is empirical, not
  derivable — expect to iterate, log what was tried and why a config was
  kept or discarded.
- **Kernel fusion / dispatch count in WGSL inference.** WebGPU's per-dispatch
  validation overhead compounds across many small compute passes; a naive
  15-layer network is 15+ dispatches per frame. Don't assume CUDA-style
  per-kernel optimisation intuitions transfer — measure dispatch overhead
  explicitly, not just kernel-internal throughput.
- **`shader-f16` availability** — feature-detect, never assume.
- **Verify the adapter in `chrome://gpu` before trusting any profiling
  number.** Chrome should be using the D3D12 backend with the RTX 3060 as
  the active adapter — confirm this rather than assuming it, especially
  after any driver update.

---

## Phase gates

Report results against these explicitly; don't just say "done."

| Phase | Gate |
|---|---|
| 0 — Rasteriser | Renders the scene at arbitrary resolution; dumps colour + depth + motion vectors to disk |
| 1 — Data pipeline | Warping frame N−1 by motion vectors into frame N gives near-zero error except at disocclusions — show the error heatmap |
| 2 — Spatial model | Beats bicubic on held-out frames, with PSNR/SSIM/LPIPS numbers reported |
| 3 — Temporal | Stable over 300+ frame sequences: no brightness drift, no accumulating artifacts — show a long-sequence result, not just a short clip |
| 4 — WGSL inference | WGSL output matches ORT Web within FP16 tolerance (report max/mean error); per-layer profile captured and reported |

---

## Repo layout

```
/renderer        WebGPU rasteriser + dataset generation (JS/TS + WGSL)
/training        PyTorch: model, training loop, losses, metrics
/export          ONNX export + compatibility checks
/inference       ORT Web harness, then hand-written WGSL kernels
/profiling       Timestamp query harness, benchmark scripts
/data            (gitignored) datasets — stored on E:\, uncompressed
                 (repo holds only a pointer/config to the E:\ path, not the data itself)
/notebook        Experiment log, especially Phase 3 — keep this current
/docs            End-of-phase summaries (see hard rule 7)
/specs           Per-phase specs. Current phase only; archive the rest.
```

---

## Working style

- Proceed phase by phase without re-confirming each one, but stop hard at
  gates and report clearly rather than continuing into the next phase.
- When something is a genuine judgement call (loss weighting, architecture
  size, jitter sequence choice) rather than a fact, say so and state the
  reasoning — don't present a guess as settled.
- Do not implement future phases speculatively.

# Project Summary — Temporal Neural Upscaler, End to End

All four specs are complete and gated. This document is the one Spec 4 asks
for after its own gate passes: the whole pipeline in one place, the key
decisions, and what would most reward the owner's own review. Each phase has
its own detailed summary (`docs/PHASE-0-1-SUMMARY.md`,
`docs/PHASE-2-SUMMARY.md`, `docs/PHASE-3-SUMMARY.md`,
`docs/PHASE-4-SUMMARY.md`) and Phase 3 additionally has a full experiment
log (`notebook/PHASE-3-EXPERIMENTS.md`) — this is a map of those, not a
replacement.

## The pipeline, in one pass

1. **Rasteriser** (`/renderer`, WebGPU + TypeScript) renders a deterministic
   procedural scene from a seed, at 540p with sub-pixel jitter, dumping
   colour, linear depth, and motion vectors per frame — plus an unjittered,
   supersampled 1080p ground-truth colour frame for training targets.
2. **Dataset** (`E:\neural-upscaler\data\seed-20260812`, 2000 frames,
   uncompressed) is captured headlessly via the same renderer code running
   under Node+Dawn, verified reproducible (byte-identical on resume) and
   validated by reprojecting frame N−1 into frame N with the stored motion
   vectors and confirming near-zero error outside disocclusions.
3. **Spatial model** (`/training`, PyTorch) — a ~915K-parameter U-Net,
   trained with L1+LPIPS, beats bicubic on SSIM and LPIPS (marginally loses
   on PSNR, an understood and accepted trade-off — see below).
4. **Temporal model** — the same architecture, `in_channels` extended from 4
   to 8 (colour, depth, warped-previous-output, disocclusion mask), trained
   with BPTT over 6-frame windows and a temporal-consistency loss. Stable
   over a 350-frame held-out rollout: no brightness drift beyond what the
   scene's own content explains, no accumulating artifacts.
5. **ONNX export + ORT Web** (`/export`, `/inference`) — the trained temporal
   model exported to a single self-contained ONNX file, confirmed running
   correctly in-browser via WebGPU (max abs error 0.000001 vs PyTorch) —
   this is the correctness oracle Phase 4's WGSL work is diffed against.
6. **Hand-written WGSL inference** (`/inference/src/wgsl`, `model_wgsl.ts`) —
   the same 16-conv-layer network reimplemented as direct-convolution
   compute kernels, NHWC throughout, matching ORT Web within FP16 tolerance
   on every layer (max error 0.000687). Profiled with GPU timestamp queries
   at deployment resolution: 3342ms/frame (PSSR's own reference is ~2ms —
   reported honestly, not hidden; see Phase 4's summary for exactly where
   that time goes and why).

## Key decisions, by phase

- **Procedural scene, not a sourced asset** (Phase 0/1) — avoided an
  asset-licensing dependency before any rendering work could start.
- **Motion vectors in UV space** (`mv = current_uv − previous_uv`), **jitter
  applied to the projection matrix**, **depth as linear view-space distance
  via the `clip.w` interpolation identity** — the three CLAUDE.md-flagged
  fragile conventions, each verified (against gl-matrix's actual layout, by
  derivation, and via the reprojection heatmap gate) rather than assumed.
  Documented once in Phase 0/1 and *reused*, not re-derived, in Phase 3's
  differentiable warp — the single most important consistency point in the
  whole project, since a silent re-derivation drifting from the original
  convention would have been invisible until much later.
- **Value-noise textures, not white noise** (Phase 0/1) — the first
  reprojection gate attempt failed at ~14% error; root-caused to texture
  aliasing (zero spatial correlation between neighbouring texels defeats
  even perfect motion vectors under point sampling), not motion-vector math.
  The scene was fixed, not the gate's threshold — CLAUDE.md hard rule 3.
- **U-Net stays at input resolution; a dedicated pixel-shuffle head does the
  actual 2x upscale** (Phase 2) — one of two valid readings of Spec 2's
  architecture description; documented as a judgement call, not asserted as
  the only correct one.
- **No normalisation layers** (Phase 2) — standard for SR networks (avoids
  suppressing fine detail) but makes gradient clipping load-bearing, not
  optional; discovered via a genuine mid-training divergence, not
  anticipated in advance.
- **PSNR trade-off accepted** (Phase 2) — LPIPS pulls the model away from
  the blurry pixelwise-average solution that PSNR itself rewards; model
  wins SSIM/LPIPS decisively, loses PSNR by 0.19dB. Flagged as a judgement
  call, owner accepted explicitly rather than the gate silently passing.
- **Feedback-path clamp, not gradient clipping, fixed the real instability**
  (Phase 3) — gradient clipping only bounds the optimizer step; an
  undetached recurrent feedback loop can blow up on the *forward* pass
  regardless. This is the single most important debugging lesson of the
  project: two failure modes that look identical from the loss curve
  (numbers exploding) had different root causes and needed different fixes.
- **Direct convolution over im2col+GEMM** (Phase 4) — reasoned a priori from
  small channel counts and an assumption that dispatch overhead would
  dominate. The full profile **partially contradicts that assumption**:
  three specific large-channel decoder layers account for 79.5% of total
  time, and it's compute/memory cost inside those kernels, not dispatch
  overhead, that dominates. The *decision* (direct conv) still looks right
  given dispatch overhead genuinely is what the small ops (upsample/concat)
  are bound by — but the *reasoning offered for it* needed correcting once
  measured. See Phase 4's summary for the full evaluation.

## What's most fragile, or would most reward review

Ranked by how silently a mistake there would cap quality or produce
plausible-looking-but-wrong output:

1. **The motion-vector convention and its reuse across phases**
   (`renderer/src/render/shaders.wgsl`, `renderer/src/validate/reproject.ts`,
   `training/src/warp.py`). Everything downstream — the reprojection gate,
   the temporal model's warped-previous-frame input, the temporal
   consistency loss — depends on this one sign/space convention staying
   consistent. It's documented in three places specifically so it can't
   drift; worth checking those three descriptions actually still agree
   before trusting any temporal result.
2. **The NCHW/NHWC layout boundary** (`export/extract_weights.py`'s weight
   transpose, `export/gen_diff_fixtures_temporal.py`'s dual-layout fixtures,
   every WGSL kernel's implicit NHWC assumption). This produced Phase 4's
   one real bug — wrong output with no error thrown, only caught via the
   per-layer diff's specific error *pattern* (wrong from the first layer,
   not localised to one later layer). Any future change touching tensor
   layout on either side of this boundary should re-run the per-layer diff,
   not just the final-output check.
3. **The temporal feedback clamp**
   (`training/src/train_temporal.py`'s `prev_output_highres = pred.clamp(0, 1)`).
   Load-bearing, not cosmetic — removing it reintroduces the forward-pass
   blowup risk that produced a loss of 68 million mid-run before this fix.
   Easy to mistake for a redundant safety check since gradient clipping is
   already present elsewhere in the same file, doing a genuinely different
   job.
4. **`conv.wgsl`'s zero-data-reuse kernel** (every thread independently
   re-fetches its full `3×3×Cin` input window from global memory, with no
   sharing across neighbouring output pixels or across the `Cout` threads
   that read the identical window). Not a correctness risk — the per-layer
   diff confirms it's numerically right — but it's the concrete, measured
   reason inference is 1671x slower than PSSR's reference, and the natural
   starting point if performance work continues past this project's current
   scope.
5. **Empirical constants that would silently stop matching reality if the
   scene or architecture changes**: `DEPTH_NORM = 50.0`
   (`training/src/dataset.py`, tied to this specific scene's extent),
   `DEPTH_REL_THRESHOLD = 0.05` (disocclusion detection, both in the
   reprojection gate and the temporal warp), `NET_STRIDE = 8` (tied to the
   encoder's exact depth). None of these are derived from first principles;
   all would need to be revisited by hand if the scene or model architecture
   changes materially.

## Gate scorecard

| Phase | Gate | Result |
|---|---|---|
| 0 — Rasteriser | Renders at arbitrary resolution, dumps colour+depth+motion | **PASSED** |
| 1 — Data pipeline | Reprojection near-zero error outside disocclusions | **PASSED** (mean 0.039, heatmaps reviewed) |
| 2 — Spatial model | Beats bicubic, PSNR/SSIM/LPIPS reported | **PASSED** (2/3 metrics, PSNR trade-off accepted) |
| 3 — Temporal | Stable over 300+ frames, no drift/accumulation | **PASSED** (350-frame rollout) |
| 4 — WGSL inference | Matches ORT Web within FP16 tolerance, profiled, live in browser | **PASSED** (max error 0.000687, 3342ms/frame reported honestly) |

## Not done, on purpose

Per Spec 4's own closing instruction: no further feature work (higher
resolution, dynamic scenes, larger models, the shared-memory-tiling
optimisation identified in Phase 4) was started without being asked. The
project as specced is complete at this document.

## Follow-on work after this document (2026-08-14)

The owner asked for further optimisation work after the project as specced
was already complete — the numbers above (3342ms/frame, `temporal_weight=1.0`)
are the historical record of what Spec 4's gate actually passed with, kept
intact rather than edited in place. What's actually deployed now is
different; full detail is in `docs/OPTIMISATIONS.md`, which grew a
"Training Pipeline Optimisations" section alongside its original WGSL one.
Headline changes:

- **WGSL inference**: 3342ms/frame → **96.0ms/frame**, a **32x+ cumulative
  speedup**, via four rounds of kernel work plus a GPU buffer-leak fix that
  was quietly inflating every profiling number until found and fixed.
- **Training speed**: mixed-precision (bf16) training and a fix to the
  temporal dataloader's ~6x redundant reads combined for a **~6x** faster
  full training run (73min → 12min for 20 epochs).
- **The deployed model changed**: a finer temporal-weight sweep — made
  affordable by the training-speed work above — found `temporal_weight=0.75`
  beats the original `1.0` on both the real long-sequence gate and a new
  disocclusion stress test. **`sweep_tw0.75_best.pt` is the deployed
  checkpoint now**, not the model this document's numbers describe.
- **Two things were tested and honestly rejected**, not silently dropped:
  computing LPIPS at reduced resolution (real speedup, real quality cost,
  not adopted) and a couple of training-log anomalies that turned out to
  be benign, self-correcting instability rather than bugs worth fixing.
- **Now hosted on GitHub**: `https://github.com/benw46/neural-upscaler`.

This section exists so this document stays a trustworthy entry point
rather than a frozen snapshot that quietly stops matching reality —
`docs/OPTIMISATIONS.md` and `docs/PHASE-3-SUMMARY.md`'s own follow-on
section have the full detail and numbers.

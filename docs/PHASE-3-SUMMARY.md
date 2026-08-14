# Phase 3 Summary — Temporal Model

Covers Spec 3 in full: integrating frame history so output is stable and
detailed across long sequences. Per Spec 3's framing, this phase's real
output is as much `notebook/PHASE-3-EXPERIMENTS.md` (the full empirical
log, including rejected configs) as the model itself — this document
summarises it, but the notebook has the detailed narrative.

## What was built

- **Dataset**: regenerated to 2000 frames (up from Phase 1/2's 500) so a
  300+ frame *held-out* sequence exists for the stability gate — the old
  500-frame set's 75-frame validation block was too short. 1500 train /
  500 held-out. Generation hit a real, unresolved gap: the Node/Dawn
  WebGPU bindings segfault reproducibly after ~900-960 frames rendered
  within a single process (confirmed twice, at different absolute frame
  ranges — process-relative, not scene-content-dependent). Capture's
  existing resumability handled it, but see "Known gaps" below.
- **`training/src/warp.py`** — differentiable reprojection + disocclusion
  masking, reusing Spec 1's exact validated motion-vector convention
  (`mv = current_uv - previous_uv`) rather than re-deriving it. Verified
  against real dataset frames: 0.043 mean warp error, 0.73% disocclusion
  fraction, matching Phase 0/1's independently-validated numbers.
- **`training/src/dataset_sequence.py`** — sequence loader for recurrent
  training: consecutive-frame windows sharing one random patch location
  (a per-frame random crop would make the stored motion vectors
  meaningless).
- **Model**: no architecture changes needed — `SpatialUNet` from Phase 2
  already parameterises `in_channels`, so temporal training just
  instantiates it with `in_channels=8` (colour 3 + depth 1 + downsampled
  warped-previous-output 3 + disocclusion mask 1) instead of 4.
- **`training/src/losses.py`** — added `temporal_consistency_loss`,
  penalising frame-to-frame difference in valid (non-disoccluded) regions.
  A real bug was caught immediately by its own unit test: the initial
  version divided by a 1-channel mask sum while summing a 3-channel diff,
  overcounting error 3x. Fixed and reverified before use.
- **`training/src/train_temporal.py`** — recurrent BPTT training loop:
  frame 0 of every sequence gets zeroed history + a full-invalid
  disocclusion mask (no special-casing needed at inference either, since
  real disocclusions use the same code path), later frames warp the
  model's own previous prediction (not ground truth) and backprop flows
  through the whole unrolled sequence, undetached.
- **`training/src/test_long_sequence.py`** — the actual gate test: genuine
  recurrent inference (no BPTT, no teacher-forcing) over a 350-frame
  held-out block at full resolution, tracking brightness drift (against a
  ground-truth baseline, not in isolation), error accumulation, and a
  ghosting/disocclusion-region check.
- **TensorBoard** wired up for temporal training too (scalars + full
  rollout image grids), same pattern as Phase 2.

## What failed along the way (the narrative Spec 3 asks for)

Full details in `notebook/PHASE-3-EXPERIMENTS.md`; summary here.

**Loss-weighting search** (Spec 3 step 5 — empirical, not derivable): four
4-epoch comparative runs.
- `temporal_weight=0.0` (baseline) and `=0.5`: statistically indistinguishable
  from each other at this training length — inconclusive, not evidence that
  0.5 does nothing.
- `temporal_weight=4.0`: **reproduced the exact "just copy history"
  degenerate solution Spec 3 warns about**, on purpose, as a diagnostic
  probe. val_temporal collapsed to ~0.005 (10x better than baseline) while
  val_l1 got 2-5x *worse* (0.099-0.102 vs baseline's 0.019-0.056) — clean,
  unambiguous evidence the tradeoff is real and reachable.
- `temporal_weight=1.0`: the chosen config. Real (not noise-level)
  improvement in temporal consistency over baseline, without the
  spatial-accuracy collapse seen at 4.0.

**Forward-pass exponential blowup** (found during the first full 20-epoch
run, not the short probes): epoch 2's logged loss was 186,545 against a
normal ~0.05-0.09 elsewhere. Per-batch values showed a genuine cascade —
roughly 68 → 72 → 238 → 485 → 5,264 → 13,078 → 24,081 → 92,718 → 1,145,145
→ 68,486,897 — before self-recovering by epoch 3. Root cause: gradient
clipping bounds the *optimizer step*, not the *forward pass* — an
unclamped prediction fed back as history can amplify step-to-step within
and across unrolls, entirely on the forward side, before any gradient
(clipped or not) is computed. This recovery looked like luck, not a
property of the training recipe, so it wasn't accepted as-is. **Fix**:
clamp the prediction to `[0,1]` specifically on the feedback path (not on
the value the loss sees, which still needs to see and penalise real
out-of-range predictions). Retrained from scratch with the fix — the
blowup did not recur (worst single-batch loss across the whole retrained
run: 15.9). This is the model used for everything downstream.

**An honest tradeoff, not hidden**: the retrained (stable) model's best
held-out L1 (0.0514) is worse than the unstable run's best (0.0394), and
plateaus rather than continuing to improve. Not resolved — could be the
clamp discarding information, or ordinary run-to-run variance from a
different optimisation trajectory. Kept anyway: the gate is about
stability, not minimising L1, and a slightly-worse-but-provably-stable
model is the right choice over one that happened to avoid catastrophic
divergence once.

## Degenerate-solution testing (Spec 3 step 6)

- **"Just copy history"**: the *aggregate numeric proxy* alone
  (`l1(pred,gt)` vs `l1(warped_prev,gt)`) was ambiguous on the final model
  and initially looked concerning — but that's expected in a very
  slow-camera scene, where a correctly-functioning model's output
  *should* stay close to a well-warped previous frame. Resolved by
  checking for the actual failure signature instead: a degenerate
  copy-model progressively loses fine detail (repeated warp+resample
  softens texture). Sample frames at t=0, t=175, and t=349 of the 350-frame
  rollout show equally sharp detail throughout, closely matching ground
  truth — not degenerate.
- **Brightness drift**: initially looked real (-0.0175 over the sequence)
  until checked against ground truth's own brightness over the same
  window (-0.0176, nearly identical) — excess drift beyond what the
  scene's actual content explains: +0.00004, i.e. none.
- **Accumulating artifacts**: error vs ground truth *decreased* across the
  sequence (first-half mean 0.0426 → second-half mean 0.0355) — the
  opposite of accumulation.
- **Ghosting at disocclusions**: real but bounded — mean error inside
  disoccluded regions is ~54% higher than the overall mean, expected given
  the network has no valid history there at all, but this particular
  350-frame test window has consistently low disocclusion fractions
  (0.8-1.3%) so it doesn't exercise a dramatic disocclusion event. Flagged
  as a gap for a targeted follow-up test on a higher-disocclusion window,
  not resolved here.

## Gate results

- **Stable over 300+ frame sequences**: 350-frame held-out rollout, no
  brightness drift beyond what the scene's own content explains (+0.00004
  excess), no accumulating artifacts (error trend improving, not
  worsening). **PASSED.**
- **`/notebook` log of attempted configurations**: `notebook/PHASE-3-EXPERIMENTS.md`,
  6 entries covering the loss-weighting search (including the rejected
  4.0 config) and the forward-pass-blowup investigation and fix.
  **PASSED.**

## Where the fragile logic lives

- `training/src/warp.py` — the motion-vector convention (reused, not
  re-derived, from Spec 1) and the low-res-vs-high-res warp resolution
  choice (input feed warps at low-res for cost; the temporal loss warps at
  full output res since flicker is most visible there).
- `training/src/train_temporal.py` — the feedback clamp
  (`prev_output_highres = pred.clamp(0, 1)`) is load-bearing, not
  cosmetic; removing it reintroduces the forward-pass blowup risk. Frame-0
  cold-start handling (zeroed history, full-invalid mask) is what makes
  real mid-sequence disocclusions and sequence-start handling share one
  code path rather than needing a special case.
- `training/src/losses.py` — `temporal_consistency_loss`'s channel-count
  bug (fixed, but a reminder: any future loss combining a 1-channel mask
  with a multi-channel diff needs the mask broadcast to match before
  summing, not after).

## Post-gate follow-on work (2026-08-14)

Everything above is the historical record of how Phase 3's gate was
originally passed — kept intact, not rewritten. Since then, follow-on work
(prompted by the owner, same pattern as Phase 4's `docs/OPTIMISATIONS.md`)
changed which checkpoint is actually deployed and closed two of the "Known
gaps" below. Full detail lives in `docs/OPTIMISATIONS.md`'s "Training
Pipeline Optimisations" section; summary here so this document doesn't go
stale in the reader's hands:

- **`temporal_weight=1.0` is no longer the deployed config.** Two training
  speed fixes (bf16 mixed precision, and fixing the sequence dataloader's
  ~6x redundant reads) cut a full 20-epoch run from ~73min to ~12min,
  making a proper finer sweep affordable. `temporal_weight=0.75` won
  consistently on spatial accuracy (val_l1) with no meaningful temporal
  cost, confirmed across two seeds and on the real long-sequence gate (not
  just training-time metrics) — **`sweep_tw0.75_best.pt` is the deployed
  checkpoint now**, referenced by `extract_weights.py`,
  `gen_diff_fixtures_temporal.py`, `capture_intermediates.py`, and
  `test_long_sequence.py`'s defaults.
- **The "disocclusion/ghosting under heavier load" gap (below) has a first
  answer.** The held-out validation block turns out to have almost no
  disocclusion pressure anywhere (max 1.6% across all 500 frames) — a
  structural property of this dataset, not fixable by picking a different
  held-out window. Found and used a training-range window instead (frames
  318-667, real disocclusion event) as a stress/diagnostic test (not a
  generalisation test, since the model has seen that data) — the deployed
  model degrades under it (mean L1 17% higher than the calm window,
  ghosting present) but not catastrophically (no brightness drift, no
  degenerate copying). A genuine held-out disocclusion test would still
  need new captured data extending the camera path — not done.
- **The "loss-weighting search was coarse" gap is addressed** by the finer
  sweep above.
- **One further speed idea was tested and rejected**: computing LPIPS at
  half input resolution gave a real 1.38x training speedup but a
  consistent, if modest, quality cost across three independent metrics —
  not adopted, though the capability (`--lpips-scale`) is kept, off by
  default.
- **The colour-desaturation question (why `lpips_weight` was raised to
  0.2) was checked directly against this model for the first time.** See
  `docs/OPTIMISATIONS.md`'s colour-desaturation section for the result.

## Known gaps / notes for later phases

- **Capture segfault** (this phase's dataset regeneration): the Node/Dawn
  WebGPU bindings crash reproducibly after ~900-960 frames rendered within
  one process — confirmed at two different absolute frame ranges,
  strongly suggesting per-process GPU resource accumulation (leaked bind
  groups/textures/command buffers) rather than anything scene-specific.
  Capture's resumability absorbed it without data loss, but if a future
  phase needs a much larger single capture run, this is worth root-causing
  (likely fix: explicit resource cleanup per frame, or periodic process
  restart within the capture script itself) rather than relying on retry.
- **Disocclusion/ghosting under heavier load**: the 350-frame gate window
  happened to have light disocclusion throughout (slow camera). A
  dedicated test on a window with a larger disocclusion event (e.g.
  reusing one of Phase 0/1's higher-disocclusion frame ranges) would give
  a more demanding read on ghosting behaviour.
- **Second full training run took ~2x longer** than the first (10828s vs
  5483s) with no code change that should affect runtime — not
  investigated, possibly background system load during a long session
  rather than anything about the fix itself.
- **Loss-weighting search was coarse** (4 points: 0.0, 0.5, 1.0, 4.0), each
  only 4 epochs. A finer sweep or longer probes might find a better
  spatial/temporal tradeoff point than 1.0, particularly to recover some
  of the L1 gap against the (unstable, rejected) first run.

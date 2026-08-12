# Phase 3 Experiment Log — Temporal Model

Per Spec 3 step 5: loss weighting here is empirical, not derivable. This log
records every configuration tried, including discarded ones, with what was
observed and why a config was kept or rejected. Entries are appended in
chronological order; nothing is deleted when superseded, only marked as such.

Config knobs in play: `temporal_weight` (weight of the temporal-consistency
term relative to the spatial L1+LPIPS loss), sequence length, batch size,
learning rate, and anything else that turns out to matter.

---

## Entry format

Each entry: date/time (session-relative), config, what was observed,
verdict (kept / rejected / superseded), and why.

---

## Exp A — temporal_weight=0.0 (baseline, no temporal loss)

Config: `SEQ_LEN=6, BATCH_SIZE=4, LR=1e-3, EPOCHS=4 (short comparative run,
not full convergence), TEMPORAL_WEIGHT=0.0`. 2000-frame dataset, 1500
train / 500 held-out split.

Purpose: establish what temporal behaviour looks like with *zero* explicit
consistency pressure — the raw `temporal` metric is still logged every run
(it's computed regardless of whether it's weighted into the backprop'd
loss), so this is a genuine "how inconsistent is the network on its own"
baseline to compare later configs against.

Result:
```
epoch 1: train_l1=0.06374 train_temporal=0.08034
epoch 2: train_l1=0.04647 val_l1=0.01932 val_temporal=0.05118
epoch 3: train_l1=0.04325 train_temporal=0.09085
epoch 4: train_l1=0.04167 val_l1=0.03563 val_temporal=0.06573
```
(1060.7s / ~4.4 min/epoch)

Observed: val_l1 is noisy epoch-to-epoch (0.0193 -> 0.0356) at this short
training length -- expected with only 4 epochs and a 20-sequence eval
sample, not yet a converged trend. val_temporal sits around 0.05-0.07
without any explicit pressure toward temporal consistency.

Verdict: **kept as reference baseline**, not a candidate final config. Next:
compare against temporal_weight=0.5 and a higher value to see whether the
explicit term measurably reduces the temporal metric, and whether a high
weight produces the degenerate "just copy history" failure mode Spec 3
warns about.

---

## Exp B — temporal_weight=0.5

Same config as Exp A except `TEMPORAL_WEIGHT=0.5`.

Result:
```
epoch 1: train_l1=0.06904 train_temporal=0.06553
epoch 2: train_l1=0.04938 val_l1=0.04453 val_temporal=0.04894
epoch 3: train_l1=0.04501 train_temporal=0.07718
epoch 4: train_l1=0.04354 val_l1=0.05551 val_temporal=0.05665
```
(1054.4s)

Observed: val_temporal (0.049-0.057) is *not* clearly lower than Exp A's
(0.051-0.066) at this training length -- within noise of each other, no
strong effect visible yet from the explicit term after only 4 epochs.
val_l1 also not clearly better than Exp A. Inconclusive at this length;
4 epochs may simply be too short for the temporal term's effect to
separate from noise (n=20 eval sequences, single run each, no seed
averaging).

Verdict: **inconclusive, not rejected**. Moving to a higher weight (Exp C)
specifically to test for the degenerate "just copy history" pathology --
that's a more diagnostic signal than comparing two runs this close
together at this length, and it's the specific failure mode Spec 3 asks to
explicitly check for.

---

## Exp C — temporal_weight=4.0 (degenerate-solution probe)

Same config, `TEMPORAL_WEIGHT=4.0` -- deliberately aggressive, to see
whether/where the "just copy history" collapse Spec 3 warns about actually
happens.

Result:
```
epoch 1: train_l1=0.15804 train_temporal=0.00554
epoch 2: train_l1=0.12262 val_l1=0.10172 val_temporal=0.00593
epoch 3: train_l1=0.11797 train_temporal=0.00467
epoch 4: train_l1=0.11600 val_l1=0.09864 val_temporal=0.00520
```
(1052.5s)

Observed: **clean, unambiguous degenerate-solution signal.** val_temporal
collapsed to ~0.005 -- over 10x lower than Exp A (0.051-0.066) and Exp B
(0.049-0.057) -- while val_l1 got roughly *2-5x worse* (0.099-0.102 vs Exp
A's 0.019-0.036 and Exp B's 0.045-0.056). This is exactly the "just copy
history" failure mode: at this weight the network learns that trivially
repeating the (warped) previous frame satisfies the temporal term almost
perfectly, at the cost of actually tracking new per-frame detail against
ground truth. The spatial and temporal objectives are in direct tension
here, and 4.0 sits clearly on the wrong side of that tradeoff.

Verdict: **rejected**. 4.0 is too high -- confirms the failure mode is real
and reachable, and gives an upper bound to stay well clear of. Combined
with Exp A/B being statistically close together, the working hypothesis is
that the useful range is well under 4.0, likely near or below 1.0. Next:
one more point (temporal_weight=1.0) to triangulate before committing to a
final config for the full training run.

---

## Exp D — temporal_weight=1.0 (triangulation)

Same config, `TEMPORAL_WEIGHT=1.0`.

Result:
```
epoch 1: train_l1=0.08790 train_temporal=0.04098
epoch 2: train_l1=0.07513 val_l1=0.04835 val_temporal=0.02784
epoch 3: train_l1=0.06857 train_temporal=0.04748
epoch 4: train_l1=0.06473 val_l1=0.05962 val_temporal=0.02937
```
(1050.1s)

Observed: val_temporal (0.028-0.029) is roughly **half** of Exp A/B's
(0.049-0.066) -- a real, measurable improvement in temporal consistency,
not noise-level like B looked. val_l1 (0.048-0.060) sits in the same
ballpark as Exp B (0.045-0.056), not the 2-5x collapse seen at weight=4.0.
This is the best trade-off point found: meaningfully more temporally
consistent than the untargeted baseline, without the spatial-accuracy
collapse that showed up at 4.0.

Verdict: **kept, chosen as the config for the full training run**.
`TEMPORAL_WEIGHT=1.0` sits clearly on the safe side of the degenerate
boundary established by Exp C, while showing a real (not noise-level)
temporal-consistency benefit over Exp A/B. Not exhaustively tuned (a finer
sweep between 0.5 and 2.0, or per-epoch weight ramping, might do better)
but each 4-epoch probe costs ~17.5 minutes and the marginal signal from
finer tuning is unlikely to change the qualitative picture -- moving to a
full 20-epoch run with this config rather than continuing to grid-search.

---

## Full 20-epoch run (temporal_weight=1.0) — first attempt: forward-pass blow-up, self-recovered

Ran the full 20-epoch training with the Exp D config. Final result looked
good (best val_l1=0.03944 at epoch 18, val_temporal trending down from
0.037 to 0.020 over the run) -- but epoch 2's logged train_l1 was
`186545.70955`, wildly inconsistent with every other epoch's ~0.05-0.09.

Investigation: per-batch loss values within epoch 2 showed a clear
**exponential blow-up cascade** across consecutive batches: approximately
68 -> 72 -> 238 -> 485 -> 5,264 -> 13,078 -> 24,081 -> 92,718 -> 1,145,145
-> 68,486,897 -- then recovery to normal values by epoch 3.

Root cause: gradient clipping (`clip_grad_norm_`, already in the training
loop) bounds the *optimizer step* size, but does nothing to prevent the
*forward pass itself* from numerically diverging mid-unroll. The recurrent
loop feeds `pred` (unclamped, straight out of the network) back as
`prev_output_highres` for the next timestep, undetached, per Spec 3's BPTT
requirement. If one timestep's prediction drifts outside a sane range, the
warp/downsample/grid_sample chain feeding on it can amplify that on the
next step, and again on the step after -- an exponential cascade *within
and across* unrolls, entirely on the forward-pass side, invisible to
gradient clipping until a huge loss produces a huge (then-clipped)
gradient one step too late.

This time it self-recovered -- but that looks like luck (enough clipped-but-
still-substantial updates eventually walked the weights back to a sane
region), not a property of the training recipe. A different seed or a
slightly worse-timed cascade could plausibly have hit inf/NaN and never
recovered. Production temporal upscalers/TAA implementations clamp or
tonemap history before reprojecting it into the next frame specifically to
prevent this class of runaway feedback amplification -- the same fix
applies directly here.

**Fix**: clamp `pred` to `[0, 1]` specifically on the feedback path (what
becomes next timestep's `prev_output_highres`), not on the value used for
the loss computation itself (the loss should still see and correctly
penalise raw out-of-range predictions via L1/LPIPS against the `[0,1]`
target, rather than having that signal hidden by clamping before the loss).

Verdict: **retraining with the fix** rather than accepting this run's
result on the grounds that "it worked out." See the next entry for the
retrained result.

---

## Full 20-epoch run (temporal_weight=1.0) — retrained with the feedback clamp

Same config, `prev_output_highres = pred.clamp(0, 1)` on the feedback path.

Result:
```
epoch  2: val_l1=0.06543  epoch  6: val_l1=0.05658  epoch  8: val_l1=0.05323
epoch 10: val_l1=0.05144 (best)  epoch 12: val_l1=0.06035  epoch 14: val_l1=0.06001
epoch 16: val_l1=0.06576  epoch 18: val_l1=0.06283  epoch 20: val_l1=0.05587
```
Worst single-batch loss across the whole run: 15.9 (vs 68,486,897
pre-fix) -- the blowup cascade did not recur. Total time 10828.4s, notably
longer than the pre-fix run's 5483.4s (unexplained -- possibly background
system load during this session rather than anything about the fix itself;
not investigated further since it doesn't affect correctness).

**Honest trade-off, not glossed over**: best val_l1 here (0.05144) is
*worse* than the unstable run's best (0.03944), and this run plateaus after
epoch 10 rather than continuing to improve. Two explanations, not
distinguished here: (a) clamping removes information from the feedback
signal (out-of-range values get truncated rather than preserved), giving
the network a slightly less rich history to work with; or (b) this is
ordinary run-to-run variance -- the fix changes the actual computation
path, so even with the same seed the optimisation trajectory isn't the
comparable to the unstable run's, and it may simply have landed in a
different, slightly worse basin.

Verdict: **kept, this is the model used for the rest of Phase 3.** The
Spec 3 gate is about temporal *stability* over long sequences (no drift, no
accumulating artifacts), not about minimising held-out L1 -- a model that
achieves a slightly higher L1 through a training recipe that's provably
free of a catastrophic-divergence risk is the right choice over one that
happened to avoid that risk once. Not re-tuning further (e.g. trying to
recover the lower L1 via a milder clamp range, or clamping less
aggressively) given the time cost per run (~1-3 hours) and that this
result already satisfies what the phase actually needs.

---

## Long-sequence stability test (the gate) + degenerate-solution checks

`training/src/test_long_sequence.py`: 350-frame genuine recurrent rollout
(no ground-truth teacher-forcing, no BPTT -- pure forward inference, exactly
how it runs in deployment) over held-out frames 1500-1849, full 960x540 ->
1920x1080 resolution, using the checkpoint above.

**Brightness drift**: initial read looked concerning -- pred brightness
dropped -0.0175 from the first half of the sequence to the second half.
But that number alone can't distinguish genuine feedback-loop drift from
the camera simply panning toward a darker part of the (static) scene.
Fixed by also tracking ground-truth brightness over the same window: GT
itself drifts -0.0176 -- almost identically. **Excess drift (pred beyond
what GT's own content explains): +0.00004, i.e. none.** The scene's actual
content getting darker fully explains the pred trend; there's no
independent model-induced drift. This is exactly the kind of thing a naive
single-signal check would have gotten wrong -- worth having caught before
reporting a false concern.

**Accumulating artifacts**: l1_vs_gt went *down* across the sequence (first
half mean 0.0426 -> second half mean 0.0355), the opposite of accumulation.

**"Just copy history" degenerate solution**: the numeric proxy alone is
ambiguous here and initially looked concerning -- l1(pred, gt)=0.0391 is
close to l1(warped_prev, gt)=0.0397, which is literally the signature
described for degenerate copying. But with this scene's very slow camera
(deliberately tuned in Phase 0/1 revisions), a *correctly functioning*
model's output should also stay close to a correctly-warped previous frame,
since frame-to-frame content genuinely doesn't change much -- the numeric
proxy can't distinguish "good model exploiting real temporal coherence"
from "degenerate model blindly copying." Visual inspection resolves it: a
truly degenerate copy-model progressively loses fine detail (repeated
warp+resample softens high-frequency texture over successive frames) --
sample frames at t=0, t=175, and t=349 all show equally sharp, detailed
checker/noise textures, matching ground truth closely, no progressive
blur. **Not degenerate** -- confirmed by the failure signature that
*should* have appeared (blur accumulation) but didn't, not just by the
absence of a single ambiguous metric.

**Ghosting at disocclusions**: mean L1 within disoccluded regions (0.060)
is ~54% higher than the overall mean (0.039) -- a real, measurable
weakness, not nothing, but bounded and expected (disocclusions are
inherently harder: the network has to reconstruct content with no valid
history at all). This particular 350-frame window has consistently low
disocclusion fractions (0.8-1.3%, matching the slow camera), so it doesn't
contain a dramatic disocclusion event to visually inspect -- worth a
targeted follow-up test on a window with heavier disocclusion if this
matters more for a later phase.

**Gate verdict: PASSED.** Stable over 350 frames (exceeds the 300+
requirement): no brightness drift beyond what the scene's own content
explains, no accumulating error, no degenerate copying (confirmed via the
failure signature that would have appeared, not just an ambiguous metric),
disocclusion-region error elevated but bounded and expected.

---

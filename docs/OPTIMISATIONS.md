# WGSL Inference Optimisations

Four rounds of performance work on the hand-written WGSL inference kernel, done
after Phase 4's own gate had already passed (`docs/PHASE-4-SUMMARY.md`) — this
is optional follow-on work, not a phase in its own right, undertaken at the
owner's request once the gate's correctness/profiling requirements were
already satisfied. Starting point: **3342.3ms/frame, 1671x slower than
PSSR's ~2ms reference**, at the real deployment resolution (960×544 input,
960×544→1920×1088 with a 1080-row crop). Ending point after four rounds:
**104.5ms/frame, 52.3x slower — a 32.0x cumulative speedup.**

A companion version of this document with rendered charts is available as a
published artifact; this file is the permanent, versioned record.

## Method: validated at every step, not just measured

Every change below was checked against the same two-step correctness harness
*before* being accepted, per the project's validation-chain rule (never trust
a kernel change on speed alone):

1. **ORT Web vs PyTorch** — the correctness oracle, unaffected by any of this
   work (it doesn't touch the WGSL path) but re-run every time as a sanity
   check that the harness itself is intact.
2. **Hand-written WGSL vs PyTorch, per intermediate layer** — 10 checkpointed
   layers plus the final output, each compared against a fixed PyTorch
   reference, FP16 tolerance 0.01.

All four rounds passed both checks, with per-layer error consistently in the
0.00002–0.0005 range (well inside tolerance) throughout — none of the
speedups below came at the cost of correctness, and none were accepted on
projected/theoretical grounds alone. All timing figures are from GPU
timestamp-query profiling at the real deployment resolution, 5 warmup runs
followed by 20 timed runs, averaged.

## Summary

| stage | change | total ms/frame | ≈fps | vs PSSR (~2ms) | vs previous stage | vs baseline |
|---|---|---:|---:|---:|---:|---:|
| 0 | Baseline (Phase 4 gate) | 3342.3 | 0.30 | 1671x | — | 1.00x |
| 1 | Output-channel reuse | 1209.0 | 0.83 | 604.5x | 2.76x | 2.76x |
| 2 | Per-`Cout` pipeline specialisation | 142.78 | 7.00 | 71.4x | 8.47x | 23.4x |
| 3 | Per-`(Cin,Cout)` specialisation + concat fix | 111.893 | 8.94 | 55.9x | 1.28x | 29.9x |
| 4 | Spatial shared-memory tiling | 104.504 | 9.57 | 52.3x | 1.07x | **32.0x** |

The four rounds are not four equal steps — the first two account for nearly
all of the total gain (30.9x of the 32.0x); the last two delivered real,
correctness-verified but comparatively small further wins, and the reasoning
for why is as informative as the numbers themselves. Read on for the
mechanism behind each.

---

## Stage 0 — Baseline

`inference/src/wgsl/conv.wgsl` as it stood at the Phase 4 gate: one GPU
thread per `(x, y, out_channel)` output element. Every one of a layer's
`Cout` threads at a given pixel independently re-fetched the identical
3×3×`Cin` input window from global memory — no sharing across output
channels, none across neighbouring pixels either.

| layer | ms/frame | share of total |
|---|---:|---:|
| `up2.conv1.conv` | 1170.8 | 35.0% |
| `up3.conv1.conv` | 1095.2 | 32.8% |
| `up1.conv1.conv` | 391.9 | 11.7% |
| `up1.conv2.conv` | 87.3 | 2.6% |
| `down2.down.conv` | 123.8 | 3.7% |
| `down2.refine.conv` | 88.5 | 2.6% |
| `down1.down.conv` | 78.6 | 2.4% |
| `down1.refine.conv` | 72.1 | 2.2% |
| `up2.conv2.conv` | 72.9 | 2.2% |
| all `upsample`/`concat`/`pixel_shuffle` combined | 52.4 | 1.6% |
| everything else (7 layers) | ~108.8 | 3.3% |

Three layers — the `.conv1` convolutions immediately following each
up-block's upsample+concat, and therefore the ones with the largest `Cin`
(196, 140, 84) — accounted for **79.5% of the entire frame**. Each of a
layer's `Cout` threads re-reading an identical window works out to every
input element being re-fetched from global memory up to `9 × Cout` times
(756x for `up1.conv1`'s Cout=84, derived exactly, not estimated: `9 × 84 =
756`, matching the measured MACs-to-input-element ratio). This was the
starting diagnosis for all four rounds that follow.

## Stage 1 — Output-channel reuse

**Problem.** The 9×`Cout` redundancy above is dominated by the `Cout` term
for these layers — removing that alone should recover most of the available
memory-traffic saving without needing true spatial sharing at all.

**Fix.** Restructured `conv.wgsl` to dispatch one thread per output *pixel*
instead of per `(pixel, channel)`: each thread holds a per-channel
accumulator array and reads every input tap exactly once, reusing it across
all `Cout` channels via an inner loop instead of relying on `Cout` separate
threads to each fetch their own copy. Dispatch's `z` dimension dropped from
`layer.outChannels` to `1`.

**Result.**

| layer | before | after | speedup |
|---|---:|---:|---:|
| `up1.conv1.conv` | 391.9 | 136.9 | 2.86x |
| `up2.conv1.conv` | 1170.8 | 257.1 | 4.56x |
| `up3.conv1.conv` | 1095.2 | 166.7 | 6.57x |
| **total** | **3342.3** | **1209.0** | **2.76x** |

**This undershot the naive prediction badly, and that shortfall is the
interesting part.** Removing the `Cout`-fold memory redundancy alone
predicts something close to an 84x/56x/28x reduction on these three layers
respectively — instead the actual gains were 2.86x/4.56x/6.57x. Worse,
**nine other layers got slower**, not faster — `stem.conv`, `down1.refine`,
`down3.down`, `down3.refine`, both `bottleneck` convs, `up2.conv2`,
`up3.conv2`, and `head`, totalling **~142.5ms of regression**, all on
layers with smaller `Cout` that were never memory-bound to begin with. The
cause: collapsing from many threads-per-pixel down to one much "fatter"
thread per pixel trades away thread-level parallelism that had been hiding
memory latency, and a fixed `array<f32, 112>` per-thread accumulator
(sized to the *widest* layer in the model, `Cout=112` at `down3`/
`bottleneck`) cost every narrower layer real register pressure for no
benefit — `head` (`Cout=12`) paid for a 112-slot array it used 12 of.

## Stage 2 — Per-`Cout` pipeline specialisation

**Problem.** Two compounding costs from Stage 1's fixed-size design: (a) the
oversized shared accumulator array, and — the bigger one — (b) `out_channels`
being a *runtime* uniform value meant the shader compiler could not know the
inner channel loop's trip count at compile time, so it had to emit a real
loop with real loop-control overhead on every layer, every iteration.

**Fix.** `model_wgsl.ts` now compiles one shader-module variant per distinct
`out_channels` value actually present in the model (5 of them: 12, 28, 56,
84, 112), substituting `OUT_CHANNELS` as a WGSL `const` into the source text
before creating each pipeline, and looks up the right variant per layer at
dispatch time. The per-thread accumulator is now exactly the right size for
its own layer, and — because the loop bound is compile-time — the compiler
can fully unroll it into straight-line code.

**Result.**

| layer | before | after | speedup |
|---|---:|---:|---:|
| `up1.conv1.conv` | 136.9 | 8.10 | 15.9x |
| `up2.conv1.conv` | 257.1 | 19.84 | 12.1x |
| `up3.conv1.conv` | 166.7 | 19.98 | 7.85x |
| **total** | **1209.0** | **142.78** | **8.47x** |

**This overshot expectations, in the opposite direction from Stage 1, and for
the opposite reason.** Every single layer improved this time, including ones
whose accumulator only shrank modestly (`up1.conv1`'s array went from 112
slots to 84 — not a large change — yet it sped up 15.9x). The register-sizing
story alone doesn't explain a win this large; the dominant effect was almost
certainly the compiler being able to unroll the channel loop once its bound
became a compile-time constant, eliminating loop-control overhead and
enabling better instruction scheduling across the board.

## Stage 3 — Per-`(Cin,Cout)` specialisation + concat fix

**Problem, part one.** Re-profiling after Stage 2 showed `up1/2/3.conv1`
were — again — the two or three most expensive dispatches in the network,
now by a clear margin over everything else. These are exactly the
widest-`Cin` layers (196, 140, 84); the *input*-channel loop was still bound
by a runtime uniform (`params.in_channels`), unable to benefit from the same
unrolling that had just worked so well on the output side.

**Problem, part two — a genuine surprise.** `concat`'s share of total frame
time had grown from 1.6% to 21.2%. Its *absolute* cost had barely moved
(~52ms → ~43ms across `concat`+`upsample`+`pixel_shuffle` combined — none of
those kernels had been touched) — it simply hadn't shrunk while everything
else around it shrank by 8.5x, a plain Amdahl's-law effect. Looking at
`concat.wgsl` confirmed the same root cause as conv's original problem: its
per-thread copy loops were bounded by runtime uniforms (`channels_a`,
`channels_b`), unable to be unrolled.

**Fix.** Extended conv's specialisation from `(Cout)` alone to the full
`(Cin, Cout)` pair — 12 distinct pairs across the 16 conv layers, each
compiled as its own variant with both `IN_CHANNELS` and `OUT_CHANNELS` as
compile-time constants. Applied the identical technique to `concat.wgsl`
(3 distinct `(channels_a, channels_b)` pairs, one per up-block — compiled
lazily since concat pairs aren't listed in the weights manifest the way conv
layers are, with the harness's existing warmup runs absorbing first-call
compile latency).

**Result.**

| component | before | after | speedup |
|---|---:|---:|---:|
| `concat` (3 dispatches) | 30.2ms | 7.93ms | **3.8x** |
| `up2.conv1.conv` | 19.84 | 20.08 | ~flat |
| `up3.conv1.conv` | 19.98 | 20.59 | ~flat |
| **total** | **142.78** | **111.893** | **1.28x** |

The `concat` fix delivered exactly as predicted. The conv `Cin` extension,
however, was a genuinely **mixed result**: `bottleneck.0`/`bottleneck.1`/
`down3.refine` (all `Cin=112`) roughly halved, but `up2.conv1` and
`up3.conv1` — the two layers this fix was specifically aimed at — barely
moved at all, and together now account for **36.3% of the entire frame**.
Working hypothesis, **not confirmed**: fully unrolling the channel loop at
very large depths (`Cin × Cout` fully flattened is ≈7,840 iterations for
`up2.conv1`, ≈2,352 for `up3.conv1`) may hit a compiler unrolling cap or
generate enough code to cause instruction-cache pressure that offsets the
scheduling win seen at smaller depths (`112×112 ≈ 12,544` also large, yet
*that* pair improved sharply — so this isn't a clean threshold effect,
more a "returns become inconsistent past some point" one). This is flagged
honestly as an open question, not a settled explanation.

## Stage 4 — Spatial shared-memory tiling (the deferred "real" fix)

**Diagnosis.** The signal from Stage 3 is itself informative: a layer that
doesn't respond to full loop unrolling is memory-bandwidth-bound, not
instruction-bound — unrolling only changes how efficiently generated code
runs once data has arrived, not how much data has to be fetched. The
redundancy unrolling can never touch is the *spatial* one: neighbouring
output pixels' 3×3 windows overlap on 6 of 9 taps, and every thread was still
independently re-fetching its own copy of that overlap from global memory —
the `9x` of redundancy left over after Stage 1 removed the `Cout`-fold
portion.

**Fix.** A new kernel, `inference/src/wgsl/conv_tiled.wgsl`, applied only to
`up1/2/3.conv1.conv` (the rest keep the Stage 3 kernel — they don't have this
problem and adding tiling to them would only add risk for no benefit). Each
workgroup cooperatively loads a 10×10 input tile (an 8×8 output block plus a
1-pixel halo for the 3×3 kernel) into `var<workgroup,...>` shared memory once,
shared by all 64 threads in that workgroup instead of each re-fetching it.
Because a full-depth tile (10×10×`Cin`) would exceed WebGPU's
guaranteed-minimum 16KB workgroup-storage budget for all three target layers
(`Cin` up to 196), the tile is chunked over input channels in blocks of 32
(a chunk's tile is 10×10×32×4 bytes = 12.8KB), with partial sums accumulated
across chunks and a `workgroupBarrier()` between the cooperative load and the
accumulate phase of each chunk. The zero-padding boundary semantics — flagged
in `docs/PHASE-4-SUMMARY.md` as the specific risk of adding tiling — are
preserved by zero-filling any tile position outside the true feature-map
bounds during the load, exactly equivalent to the plain kernel's per-tap
bounds check, and verified against the same per-layer diff harness rather
than just reasoned about.

**Result.**

| layer | before | after | change |
|---|---:|---:|---:|
| `up1.conv1.conv` | 7.52 | 8.55 | **14% worse** |
| `up2.conv1.conv` | 20.08 | 13.28 | 1.51x better |
| `up3.conv1.conv` | 20.59 | 16.76 | 1.23x better |
| **total** | **111.893** | **104.504** | **1.07x** |

**Correct, but well short of what the theory predicted**, and this is the
first round in the whole sequence where the result doesn't clearly justify
its added complexity. `up1.conv1` regressed outright. Candidate causes —
**none confirmed**, and confirming them would need lower-level GPU profiling
(e.g. Nsight Compute) that isn't reachable from a WebGPU/browser context,
only wall-clock-per-dispatch timing:

- **Barrier cost.** Each `Cin` chunk costs two `workgroupBarrier()` calls —
  14 total for `up1.conv1` (7 chunks of 32 from `Cin=196`), 10 for
  `up2.conv1`, 6 for `up3.conv1`. `up1.conv1` has both the most chunks and
  the only regression, at least consistent with barrier cost being real,
  though it doesn't fully explain why `up3.conv1` (fewest chunks) improved
  less than `up2.conv1`.
- **Address-decode overhead.** Unpacking a flat cooperative-load index into
  `(row, col, channel)` needs modulo/division by `chunk_len`, up to ~50 times
  per thread per chunk — real division instructions if the compiler didn't
  fully constant-fold `chunk_len` at each unrolled chunk-loop site.
- **Shared-memory bank conflicts.** `CHUNK_SIZE=32` was chosen for capacity
  against the 16KB budget, not for the GPU's shared-memory bank layout —
  never checked for conflicts.
- **Occupancy.** 12.8KB of workgroup shared memory is a new resource
  competing (alongside the already-substantial per-thread accumulator
  registers) for how many workgroups can run concurrently per SM — the same
  class of cost that caused Stage 1's first cut to regress on small layers.

---

## Composition shift across stages (Amdahl's law, visibly)

| stage | conv | concat | upsample | pixel_shuffle | total |
|---|---:|---:|---:|---:|---:|
| 0 — baseline | 3289.9ms (98.4%) | 36.9ms (1.1%) | 15.1ms (0.5%) | 0.4ms (0.0%) | 3342.3ms |
| 2 — per-`Cout` | 99.64ms (69.8%) | 30.2ms (21.2%) | 12.5ms (8.8%) | 0.44ms (0.3%) | 142.78ms |
| 3 — `(Cin,Cout)`+concat | 91.68ms (81.9%) | 7.93ms (7.1%) | 11.95ms (10.7%) | 0.33ms (0.3%) | 111.893ms |
| 4 — tiling | 83.20ms (79.6%) | 8.75ms (8.4%) | 12.17ms (11.6%) | 0.38ms (0.4%) | 104.504ms |

`concat` moving from 1.1% to 21.2% of the frame between stages 0 and 2 —
without its own cost changing by more than a few ms — is the clearest
illustration in this whole log of why profiling has to be re-run after every
change rather than trusted from an earlier baseline: optimising the dominant
cost doesn't just shrink the total, it silently promotes whatever's left.

---

## Stage 5 — Further WebGPU exploration (a validated null result, and one real bug fix)

Prompted by a direct question — "are we at an impasse on WebGPU?" — this
round tested every concrete unexplored lever from stage 4's write-up rather
than accepting 104.5ms as a ceiling. It found one genuine, worth-keeping fix
that didn't move the speed number, and a clean empirical answer that
overturns stage 4's own barrier-cost hypothesis.

**A real bug, found by querying rather than assuming.** `gpu.ts` requested a
device with `requiredFeatures` but no `requiredLimits` — under WebGPU's
spec, that grants only the spec-*minimum* `maxComputeWorkgroupStorageSize`
(16384 bytes), never the adapter's actual maximum, even when the hardware
supports more. Probing this adapter directly confirmed exactly that gap:
`adapter.limits` reported 32768 bytes available; the device stage 4's kernel
actually ran against only had 16384 granted. `conv_tiled.wgsl`'s
`CHUNK_SIZE=32` had been sized against an assumed 16KB ceiling that was
never actually queried — a `feature-detect, never assume` lapse in exactly
the spirit CLAUDE.md already asks for elsewhere in this project (`shader-f16`
was always feature-detected correctly; workgroup storage never was, until
now). **Fixed**: `gpu.ts` now explicitly requests
`requiredLimits: { maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize }`,
and `model_wgsl.ts` sizes each tiled-kernel variant's `Cin` chunk from the
device's actually-granted limit instead of a hardcoded assumption. Worth
keeping regardless of the speed result below — it makes the kernel portable
to whatever a given adapter actually grants instead of silently
under-using hardware that supports more, or (worse, on stricter hardware)
assuming a budget that was never actually confirmed.

**The tiled kernel's shared tile is now also `Scalar`-typed** (f16 when
available) instead of always `f32`, matching `input_tex`'s own storage
precision exactly — free to add, halves the tile's footprint per channel,
no new rounding introduced (an already-f16 value stored as f16 loses
nothing it hadn't already lost).

**The experiment this unlocked**: with the real 32KB budget and an f16 tile,
`up2.conv1` (Cin=140) and `up3.conv1` (Cin=84) can each fit their *entire*
input-channel depth in a single chunk — no chunking loop, no repeated
`workgroupBarrier()` calls at all, just one load-barrier-compute-barrier
cycle instead of the original 10 and 6 respectively. Stage 4's own write-up
named barrier count as its leading unconfirmed hypothesis for that stage's
underperformance. This was the direct test of it.

| config | `up1.conv1` chunks | `up2.conv1` chunks | `up3.conv1` chunks | total ms/frame |
|---|---:|---:|---:|---:|
| Stage 4 (f32 tile, `CHUNK_SIZE=32`, 16KB assumed) | 7 | 5 | 3 | 104.504 |
| f16 tile, `CHUNK_SIZE=32` (footprint halved, chunk count unchanged) | 7 | 5 | 3 | ~98–104 (run-to-run noise) |
| f16 tile, `CHUNK_SIZE=64` | 4 | 3 | 2 | 106.912 |
| f16 tile, full 32KB budget (`CHUNK_SIZE≈140`) | 2 | 1 | 1 | 107.115 |

**The result overturns the hypothesis it was built to test.** Going from 10
barriers to 2 on `up2.conv1` made it *slower* (13.28ms → 17.24ms in the
full-budget config), not faster. Halving the shared-memory footprint at a
*fixed* chunk count (the f16-only row) changed nothing measurable either.
Tested at three points (32/64/~140), the relationship between chunk size and
speed on this kernel is flat-to-negative past 32 — the opposite of what more
chunking headroom was expected to buy. `CHUNK_SIZE=32` — the original value,
picked for capacity reasons under an assumption that turned out to be
wrong — was already at or near the practical optimum anyway. Kept as an
explicit, empirically-chosen constant rather than derived from the (now
correctly-queried) budget, specifically *because* using more of that budget
measured worse.

**Why, still not confirmed** — same limitation as stage 4: WebGPU
timestamp-query profiling reports wall-clock time per dispatch, not
occupancy, register pressure, or bank-conflict counts, so there's no way
from inside a browser to see *which* GPU-internal resource is actually
binding here. A plausible guess (larger per-workgroup shared-memory
reservations limiting how many workgroups run concurrently per SM, trading
fewer barriers for worse occupancy) is offered, not claimed.

**Subgroups: available, deliberately not attempted.** `adapter.features`
confirmed `"subgroups"` is supported on this hardware — a real, concrete
unexplored lever. Not pursued here on a specific technical judgment, not
because it's unexplored: subgroup operations (`subgroupBroadcast`,
`subgroupShuffle`, `subgroupAdd`) are built for "every lane wants the
identical value" or "reduce a value across the group." This kernel's
redundancy is spatial — each thread needs a *different but overlapping*
window — which is exactly the pattern `var<workgroup>` shared memory exists
for and subgroup intrinsics don't naturally express. A subgroups-based
rewrite here would be a speculative fit search, not a targeted fix; flagged
as a real option if a future pass wants to pursue it, not ruled out by
laziness.

**Net effect of this whole round**: no measurable speed win over the
published 104.5ms (the best re-measurement, ~98.5ms, is within the ~6%
run-to-run noise observed across repeated runs of the *identical* config) —
a validated null result, not a failure to find one. What it did produce is a
real correctness/portability fix (the workgroup-storage limit request) and a
clean overturn of stage 4's leading hypothesis, which is exactly the kind of
finding worth keeping even though the frame-time number didn't move.

---

## Where this leaves the network

104.504ms/frame is **≈9.6fps** — a real 32.0x cumulative speedup from the
Phase 4 gate, but not "real-time" by the usual graphics bar of 30–60fps.

| target | frame budget | further speedup needed from here |
|---|---:|---:|
| 20fps | 50.0ms | 2.1x |
| 30fps | 33.3ms | 3.1x |
| 60fps | 16.7ms | 6.3x |

Two structural points worth carrying forward, not just the numbers:

- **The easy, low-risk wins are behind us.** Stages 1–2 (removing redundant
  memory traffic, then letting the compiler exploit compile-time-constant
  loop bounds) delivered 23.4x of the total 32.0x gain, cleanly and with
  correctness holding at every step. Stages 3–4 pushed the same techniques
  further and returned real but small, sometimes inconsistent, sometimes
  negative results — a sign the cheap structural improvements available to
  this kernel design are largely exhausted.
- **Tensor Cores are unreachable from WebGPU entirely**, independent of any
  further kernel work here — WGSL has no cooperative-matrix / matrix-multiply
  instruction (unlike Vulkan's `VK_KHR_cooperative_matrix` or CUDA's WMMA),
  so this kernel is permanently confined to the GPU's general-purpose ALUs.
  Reaching the hardware DLSS/PSSR-class techniques likely use would mean
  leaving WebGPU for a native API (Vulkan + cooperative_matrix keeps the
  hand-written-kernel approach; CUDA/TensorRT would be fastest but furthest
  from "runs in a browser tab") — a different project shape, not a next
  optimisation.

## Post-fix re-profile (2026-08-14)

The 104.5ms/frame figure above (and everything derived from it in this
document) was captured by a profiling harness (`main.ts`'s
`profileAtResolution()`) that was itself leaking GPU buffers throughout
every run — `WgslUNet.forward()` allocated ~23 buffers per call and never
freed the ~13 purely-transient ones (see the model-code comment on
`WgslUNet.releaseIntermediates()`), so a 25-call profiling run (5 warmup +
20 timed) accumulated roughly 1800 dead buffers before finishing. That's a
live caveat this document carried for a while: the numbers were real
measurements, but taken under mounting memory pressure the deployed
realtime path would never actually experience frame-to-frame, since nothing
in it disposed of those buffers either.

Fixed the leak at all four real call sites (`live_pipeline.ts`,
`viewer.ts`, and both loops in `main.ts`'s profiler), then re-ran the full
correctness + profiling harness:

- Both correctness checks (ORT Web vs PyTorch, WGSL vs PyTorch per-layer)
  **PASSED**, errors in the same 0.00002–0.0005 range as every prior round —
  the leak fix touched buffer lifetime only, not any dispatch or kernel.
- Full deployment resolution (960×544), 20 timed runs, two independent
  re-runs for consistency: **95.8ms/frame** and **96.1ms/frame** (~0.3%
  apart, within ordinary run-to-run noise).

That's **~96.0ms/frame, ≈10.4fps — a further ~8% faster than the
leak-tainted 104.5ms figure**, purely from removing the profiling harness's
own memory pressure rather than any kernel change. The 32.0x cumulative
speedup claim above understates the real number very slightly as a result;
treat **96.0ms/frame** as the current, leak-free baseline for any future
optimisation work from here, not 104.5ms.

## File map

| file | role |
|---|---|
| `inference/src/wgsl/conv.wgsl` | Per-`(Cin,Cout)`-specialised plain conv kernel (stages 1–3) |
| `inference/src/wgsl/conv_tiled.wgsl` | Spatial-tiled kernel, `up1/2/3.conv1.conv` only (stage 4) |
| `inference/src/wgsl/concat.wgsl` | Per-`(channels_a,channels_b)`-specialised (stage 3) |
| `inference/src/model_wgsl.ts` | Pipeline specialisation/caching, per-layer dispatch routing |
| `inference/src/main.ts` | The correctness + profiling harness every stage above was checked against |
| `docs/PHASE-4-SUMMARY.md` | The original gate this work continues from (baseline numbers) |

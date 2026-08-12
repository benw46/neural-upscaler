# Phase 4 Summary — WGSL Inference

Covers Spec 4: exporting the Phase 3 temporal model out of PyTorch and
running it in-browser two ways — once through ONNX Runtime Web (the
correctness oracle) and once through hand-written WGSL compute kernels (the
version being optimised) — validated against each other per the project's
validation-chain rule (never trust WGSL output without diffing it against a
known-good reference first).

## What was built

- **`export/extract_weights.py`** — pulls all 16 conv layers
  (`stem.conv` → `down{1,2,3}.{down,refine}.conv` → `bottleneck.{0,1}.conv`
  → `up{1,2,3}.{conv1,conv2}.conv` → `head`) out of
  `training/checkpoints_temporal/temporal_w1.0_final_best.pt`, transposing
  PyTorch's native `(Cout, Cin, 3, 3)` weight layout to `(Cout, 3, 3, Cin)`
  once, offline, so the WGSL kernel can read contiguous per-tap input and
  weight values without strided access at runtime. Writes a flat
  `weights.bin` + `manifest.json` under `inference/public/weights/`.
- **`export/gen_diff_fixtures_temporal.py`** — a fixed-seed 128x128 input
  fixture in *both* layouts: NCHW for ORT Web (its native layout) and NHWC
  for the WGSL harness. Same underlying values, two files — this is what
  Phase 4's earlier layout bug (below) was ultimately traced back to
  confusing.
- **`export/capture_intermediates.py`** — hooks 10 checkpoint layers in the
  real PyTorch model and dumps each one's activation (NHWC) on the same
  fixed input, so a WGSL bug can be localised to a specific layer instead of
  only showing up as an aggregate final-output error.
- **`inference/src/wgsl/*.wgsl`** — four kernels: `conv.wgsl` (direct 3x3
  convolution with LeakyReLU fused into the same dispatch), `upsample.wgsl`
  (nearest x2), `concat.wgsl` (channel concat for skip connections),
  `pixel_shuffle.wgsl` (the final 2x depth-to-space upscale). All NHWC
  throughout, one thread per `(x, y, outChannel)` output element,
  `@workgroup_size(8, 8, 1)`.
- **`inference/src/model_wgsl.ts`** (`WgslUNet`) — wires the 16 conv layers
  + 3 upsample/concat pairs + pixel-shuffle into the same graph topology as
  `training/src/model.py`'s `SpatialUNet.forward()`, with optional GPU
  timestamp-query profiling per dispatch (`{ profile: true }`).
- **`inference/src/main.ts`** — the harness page, three steps in order per
  Spec 4's own step ordering: (1) ORT Web vs PyTorch oracle check, (2)
  hand-written WGSL vs PyTorch per-layer diff, (3) GPU timestamp-query
  profiling at increasing resolutions (128x128 → 240x136 → full deployment
  960x544).

## The layout bug (worth documenting in detail — this was the real Phase 4 bug)

The first per-layer diff run showed large error (max 1.22) starting at
`stem.conv` that *shrank* through the encoder and then *grew again* through
the decoder via already-corrupted skip connections. That specific pattern —
wrong from the very first layer, not localised to one later layer — pointed
at the input itself rather than any kernel, before any kernel code was
re-read. Cause: `test_input_temporal.bin` was saved NCHW (correct for ORT
Web, which expects PyTorch's native layout) but the WGSL harness assumed
NHWC throughout, matching every other tensor in the pipeline. Fixed by
generating a second, NHWC-permuted copy of the same input
(`test_input_temporal_nhwc.bin`) instead of changing what ORT Web receives.
Re-run after the fix: all 10 layers passed, max error 0.000018–0.000687.

## Gate results (per CLAUDE.md's Phase 4 row)

- **WGSL output matches ORT Web within FP16 tolerance (0.01), max/mean
  reported**:
  - Step 1 (ORT Web vs PyTorch): max=0.000001, mean=0.000000.
  - Step 2 (WGSL vs PyTorch, all 10 checkpointed layers + final output): max
    error across all layers 0.000018–0.000687, final output max=0.000468
    mean=0.000105.
  - **PASSED**, with large margin against the 0.01 tolerance.
- **Per-layer profile captured and reported**: full GPU timestamp-query
  profile at 128x128, 240x136, and the real deployment resolution (960x540,
  padded to 960x544 for the encoder's three stride-2 downsamples), 23
  dispatches each, averaged over 20 timed runs at full resolution after 5
  warmup runs. **PASSED** — see below.
- **Running live in a browser tab against the fully trained model**: the
  harness at `inference/` (Vite build+preview, not dev mode — see Known
  gaps in Phase 2's summary for why) ran all three steps against real
  Chrome/D3D12/NVIDIA hardware, loading `temporal_w1.0_final_best.pt`'s
  actual extracted weights, not a placeholder or randomly-initialised
  model. **PASSED.**

## Profiling results (deployment resolution, 960x544, 20-run average)

```
conv:stem.conv            6.4ms
conv:down1.down.conv     78.6ms
conv:down1.refine.conv   72.1ms
conv:down2.down.conv    123.8ms
conv:down2.refine.conv   88.5ms
conv:down3.down.conv     20.2ms
conv:down3.refine.conv   17.3ms
conv:bottleneck.0.conv   17.1ms
conv:bottleneck.1.conv   17.1ms
upsample:up1              1.3ms
concat:up1                3.5ms
conv:up1.conv1.conv     391.9ms
conv:up1.conv2.conv      87.3ms
upsample:up2               4.5ms
concat:up2                11.3ms
conv:up2.conv1.conv    1170.8ms
conv:up2.conv2.conv      72.9ms
upsample:up3               9.3ms
concat:up3                22.1ms
conv:up3.conv1.conv    1095.2ms
conv:up3.conv2.conv      21.4ms
conv:head                  9.6ms
pixel_shuffle               0.4ms

total: 3342.3ms/frame, 23 dispatches
```

**PSSR's own reference is ~2ms/frame — this is 1671x slower.** Not expected
to match a shipped, heavily-optimised production kernel; reported honestly
rather than hidden, per CLAUDE.md.

## Resolution-tier fusion evaluation (Spec 4 step 5 / CLAUDE.md's dispatch-overhead note)

`conv.wgsl`'s header comment states the *a priori* assumption behind
choosing direct convolution over im2col+GEMM: that per-dispatch overhead
would dominate over intra-kernel efficiency at this model's scale, so
avoiding an extra dispatch outweighs any GEMM tiling gains. The full
profiling data above **partially contradicts that assumption**, and it's
worth being explicit about the correction rather than leaving the original
comment as the last word:

- `up1.conv1.conv`, `up2.conv1.conv`, `up3.conv1.conv` — the three conv
  layers that immediately follow an upsample+concat and therefore have the
  largest `Cin` of any layer in the network (196, 140, 84 respectively) —
  together account for **2658ms of the 3342ms total (79.5%)**.
- The genuinely cheap, dispatch-overhead-shaped operations — `upsample`,
  `concat`, `pixel_shuffle` — sum to only **~52ms total** across all three
  resolutions combined, even at full deployment resolution. If fixed
  per-dispatch overhead were the dominant cost, these tiny operations would
  show a large *fixed* floor regardless of size; instead they scale with
  their (tiny) workload and stay cheap.
- Both facts point the same way: **the actual bottleneck is compute/memory
  cost inside three specific large-channel-count conv kernels, not the
  23-dispatch overhead itself.** Fusing dispatches together (e.g.
  upsample+concat+conv1 into one pass) would only recover time on the ~52ms
  side of the ledger — dispatch fusion cannot reduce the number of MACs a
  conv layer performs, only the intermediate buffer write/read round trips
  between passes, which are small relative to the compute at these channel
  counts.
- **A second, independent finding from the same data**: kernel efficiency
  (time per unit of work) gets *worse*, not better, as a layer's per-thread
  work increases and its thread count shrinks. `up3.conv1.conv` has more
  total multiply-adds than `up2.conv1.conv` (1.10e10 vs 9.21e9, from
  `width × height × Cout × Cin × 9`) but *fewer* input channels per thread
  (84 vs 140) and *more* threads (14.6M vs 7.3M, since `Cout` scales the
  z-dimension of the dispatch) — and it runs faster (1095ms vs 1171ms)
  despite doing more work. Same pattern between `down1.down.conv` (more
  MACs, more threads, 78.6ms) and `down2.down.conv` (fewer MACs, fewer
  threads, 123.8ms). Consistent with `conv.wgsl`'s kernel doing zero data
  reuse: every thread independently re-fetches its entire `3×3×Cin` input
  window from global memory (no shared-memory tiling across neighbouring
  output pixels, and no reuse across the `Cout` threads that all read the
  *same* window at a given `(x, y)`) — fewer, "fatter" threads leave less
  work in flight to hide that redundant memory latency.

**Conclusion**: dispatch-count fusion, the specific technique this task
asked to evaluate, is not the right next lever — the data rules it out
concretely rather than by assumption. What *would* address the actual
79.5%-of-total-time bottleneck is workgroup-level shared-memory tiling (so
neighbouring output pixels and output channels reuse the same fetched input
window instead of each re-reading it from global memory) or restructuring
those three specific layers as im2col+GEMM (where the extra dispatch's cost
would be worth paying, unlike the small layers `conv.wgsl`'s comment was
reasoning about). Not implemented here — it's a real kernel rewrite, not a
config change, and is the natural next optimisation task if inference speed
work continues past this phase's gate (which only requires the profile be
captured and reported, not a specific performance target).

## Adapter verification

`gpu.ts` refuses SwiftShader/WARP/llvmpipe/software adapters by name before
proceeding (same discipline as `renderer/src/gpu/device.ts` in Phase 0/1).
Every run this phase — correctness checks and profiling alike — logged
`adapter: nvidia / ... ` and `shader-f16: true`, i.e. real hardware, not a
software fallback. `chrome://gpu` itself is a browser-internal URL that the
available browser-automation tooling cannot navigate to (same limitation
hit in Phase 0/1); the adapter-name check inside `gpu.ts` is the available
substitute and it passed consistently across every run.

## Where the fragile logic lives

- **`inference/src/wgsl/conv.wgsl`** — the NHWC layout assumption, and the
  fact that every thread re-derives its full input window from scratch (see
  fusion evaluation above). Any future change to add shared-memory tiling
  needs to preserve the existing per-thread zero-padding boundary handling
  (`iy`/`ix` bounds checks) exactly, or the "same" padding semantics silently
  break at feature-map edges.
- **`export/extract_weights.py`**'s weight transpose
  (`(Cout,Cin,3,3)` → `(Cout,3,3,Cin)`) — this is the one place PyTorch's
  and the WGSL kernel's layouts are reconciled; get the transpose axes wrong
  here and every layer still "runs" without erroring, it just produces
  silently wrong output (the exact failure mode the Phase 4 layout bug
  above demonstrated for the *input* tensor specifically).
- **`export/gen_diff_fixtures_temporal.py`** and
  **`export/capture_intermediates.py`** — both must stay in sync with
  `model_wgsl.ts`'s checkpoint names (`HOOK_NAMES` / `LAYER_CHECKS`) for the
  per-layer diff to mean anything; a silent mismatch here would make the
  harness compare the wrong tensors without erroring.

## Live demo viewer (post-gate, on request)

`inference/viewer.html` (+ `src/viewer.ts`) — a small standalone page, built
after this phase's gate had already passed, at the owner's explicit request
to *see* the upscaler working rather than only reading numbers. Shows one
of four held-out demo frames (1500/1650/1800/1949, from the Spec 3 gate's
held-out block — genuinely frames the model never trained on) in three
toggled states: the raw 540p input (nearest-upscaled 2x for display, so the
aliasing is visible rather than hidden), the WGSL network's actual 1080p
reconstruction, and the captured 1080p ground truth.

Two scope decisions made explicitly rather than assumed, both raised as
questions before building:

- **Not a live/interactive camera demo.** At 3.3s/frame, a literal
  per-frame toggle during live camera movement would be a slideshow, not a
  real-time comparison — this shows fixed frames instead, computed once and
  cached, so toggling between views after the first computation is instant.
- **Cold-start temporal input**, not real recurrent history. Every frame
  shown is fed zeroed warped-previous-output and a fully-invalid
  disocclusion mask — exactly frame 0's input in every training sequence
  (`train_temporal.py`), a real, already-validated code path, not an
  approximation. Building a live WGSL warp step so history actually
  accumulates across a sequence of demo frames (matching how the model
  performs in the Phase 3 stability gate) is real additional work, out of
  scope here.

`export/copy_demo_frames.py` copies the four frames' raw buffers from
`E:\neural-upscaler\data\seed-20260812` into
`inference/public/demo_frames/` (gitignored, like every other generated
asset under `inference/public/` — rerun the script to regenerate). The
input-construction code in `viewer.ts` reflect-pads 540→544 rows to exactly
match `training/src/dataset.py`'s `pad_to_multiple(mode="reflect")` — the
same preprocessing-parity fragile-logic point as the rest of this phase.

## Known gaps / notes for later work

- **Full-resolution inference is 1671x slower than PSSR's reference**,
  entirely expected for a first hand-written, untiled implementation but
  worth being explicit about: this is nowhere near real-time (3.3s/frame,
  not ms/frame). The fusion evaluation above identifies the concrete next
  step (shared-memory tiling in `conv.wgsl`, or im2col+GEMM specifically for
  the three `up*.conv1.conv` layers) if performance work continues.
- **`shader-f16` was available and used** (feature-detected, not assumed,
  per CLAUDE.md) on the adapter this was tested against, but the profiling
  numbers above were not separately broken out by precision — worth
  checking whether f16 storage is actually reducing memory traffic
  proportionally to its bit-width savings, or whether the `f32` accumulator
  in `conv.wgsl` (line 46, deliberate for accuracy) is masking that benefit.
- **Single adapter tested** (NVIDIA, Ampere-generation, D3D12 backend
  implied by Windows + Chrome). No cross-vendor (AMD/Intel) or cross-OS
  validation attempted — out of scope for this machine but worth flagging
  for anyone extending this beyond the owner's own hardware.

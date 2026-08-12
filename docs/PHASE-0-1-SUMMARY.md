# Phase 0/1 Summary — Rasteriser & Data Pipeline

Covers Spec 1 in full: the WebGPU rasteriser (Part A) and the dataset
generation pipeline (Part B), including the reprojection validation gate.

## What was built

**Renderer** (`/renderer`, TypeScript + Vite + WebGPU):
- `src/gpu/device.ts` — adapter/device acquisition with feature detection.
  Refuses to proceed on a software adapter (SwiftShader/WARP/llvmpipe, name
  heuristic) and logs the real adapter name + `shader-f16` support rather
  than assuming either.
- `src/scene/` — a procedural "city block" test scene (ground plane,
  variable-height boxes, thin cylindrical rods, small accent cubes),
  deterministic from a seed (mulberry32 PRNG). No external assets — chosen
  over sourcing a Sponza glTF to avoid an asset-licensing/sourcing
  dependency before rendering work could start.
- `src/camera/` — a scripted camera path (orbit + radial forward/back motion
  + panning look-target, all seeded, all combined continuously) and
  Halton(2,3) sub-pixel jitter.
- `src/render/` — the MRT render pipeline (colour RGBA16F, linear depth
  R32F, motion vectors RG16F, plus a real depth-stencil attachment for
  z-testing), a GPU compute box-filter downsampler for ground-truth
  antialiasing, and a fullscreen-triangle blit for the interactive preview.
- `src/main.ts` — interactive preview (`npm run dev` in `/renderer`), toggles
  960×540 / 1920×1080 with `R`, shows adapter info and live jitter/frame
  state on screen.

**Headless capture** (`/renderer/scripts/capture.ts`, Node + Dawn via the
`webgpu` npm package): runs the identical `SceneRenderer` used by the
interactive preview — deliberately the same code path, per the project's
validation-chain philosophy of not letting parallel implementations drift.
For each frame, renders a jittered 540p input and an unjittered
3840×2160-supersampled-then-box-filtered 1920×1080 ground-truth colour
frame. Resumable (detects contiguous completed frames and continues);
verified byte-identical output between a fresh run and an interrupted/resumed
run of the same frames. Reports a disk budget (per-buffer and total,
compared against actual free space) before writing anything, and refuses to
proceed above 90% of free space.

**Reprojection validation** (`/renderer/scripts/validate-reprojection.ts` +
`src/validate/`): warps frame N−1's colour into frame N's viewpoint using
frame N's stored motion vectors (bilinear resample), compares against frame
N's actual colour, and excludes disoccluded pixels (detected via reprojected
depth mismatch, or reprojected UV landing off-screen) from the error stats.
Outputs a warped-image PNG and an error-heatmap PNG per frame pair, plus a
JSON stats summary. A small dependency-free PNG encoder was written for this
(`src/validate/png.ts`) rather than adding an image library, since it was a
one-shot need.

## Motion vector convention

Computed in **UV space** (origin top-left, V down — matching framebuffer/
texture-sample space, not NDC which is y-up):

```
mv = current_uv − previous_uv
previous_uv = current_uv − mv
```

So `mv` points from where a surface point was sampled last frame to where it
is now; subtracting it from a pixel's current UV walks back to its
previous-frame location. This is the convention both the shader
(`render/shaders.wgsl`) and the CPU-side reprojection tool
(`validate/reproject.ts`) assume — documented in both places so they can't
silently drift apart.

Depth stored as **linear view-space distance** (`currClip.w`), *not* the
default WGSL perspective-correct interpolation applied naively — using
`clip.w` itself as the interpolated varying is a standard identity that
happens to recover the correct hyperbolically-interpolated view-space depth
per pixel for free (verified by derivation, not assumed).

Jitter is injected into the projection matrix's NDC x/y offset terms
(`camera/projection.ts`, `jitterProjectionMatrix`) — never by translating the
camera eye, which would change parallax and corrupt the motion-vector
geometry (see CLAUDE.md fragile-logic list). Verified against gl-matrix's
actual `perspectiveZO` element layout rather than assumed.

## A real bug found and fixed during validation

The first end-to-end reprojection validation run showed ~14% mean colour
error per frame pair — nowhere near "near-zero except at disocclusions." The
suspect list was: sign/space convention, jitter formula, or the capture
script's frame bookkeeping.

Diagnosis (see CLAUDE.md: diff against a known-good reference before
guessing): swapping the scene's textures for a smooth, low-frequency
gradient and re-running validation dropped mean error to ~0.5%, with the
residual concentrated exactly at the gradient tile's own seam edges. That
isolated the problem to **texture content, not the motion vector math**: the
ground/building textures used literal per-texel-independent white noise
(and near-per-texel checker patterns), which has zero spatial correlation
between neighbouring texels. Point-sampled, that means even mathematically
perfect motion vectors land on an unrelated random value after any sub-texel
reprojection offset — the validation was measuring texture aliasing, not
motion-vector correctness.

Fix: replaced per-texel white noise with **value noise** (a coarse random
lattice, bilinearly upsampled — `scene/texture.ts`), and coarsened checker/
stripe cell sizes and UV tiling scene-wide. This keeps the scene
high-frequency enough to alias visibly at 540p (the point of Spec 1 Part A
step 3) while giving reprojection something spatially coherent to measure.
Nearest/point sampling was kept in the render pipeline throughout — the fix
was to the texture *content*, not the sampling mode, since sampling mode is
what makes 540p rendering alias in the first place. The gate was not
loosened to accommodate the original textures; the scene was fixed instead
(CLAUDE.md hard rule 3).

## Camera-path revisions (post-gate polish)

After the initial gate pass, the camera path was revised three times based
on direct review of the interactive preview:

1. **Slowed down** — the original path moved too much frame-to-frame.
   Added `SPEED_SCALE` in `camera/path.ts` scaling rotation speed and all
   oscillation frequencies (radius/height/pan); amplitudes untouched.
2. **Collision avoidance** — the scripted orbit/oscillation formula has no
   awareness of scene geometry, and at the original radius (14–18) it
   regularly clipped into or through buildings. Added `scene/colliders.ts`:
   every placed object (building, rod, accent cube) gets a conservative
   bounding-cylinder collider built from the *same placement loop* that
   generates its render geometry (not a separately-derived RNG stream, so
   they can't drift out of sync). `camera/sequence.ts`'s `frameState()`
   pushes the raw analytic eye position outside every collider (0.4 unit
   margin) and above the ground plane (0.25 unit margin) before it's used —
   verified programmatically across 2000 frames (zero violations) and
   against the actual generated dataset's manifest (zero violations across
   all 500 frames).
3. **Widened the orbit** — pure collision push-out as the primary
   shot-composer produced awkward pressed-against-the-wall and
   squeezed-between-buildings framing. Scene geometry extends to radius
   ~39.5 at its farthest outlier corners but the ground plane itself only
   extends to 30; the orbit radius was moved to ~39–43 (just past the
   typical geometry field, deliberately not fully past every outlier, since
   going further would put the ground plane's edge in frame as empty void).
   Push-out now triggers on ~4% of frames, each under 1 unit — a rare
   safety net rather than the routine determinant of framing. Then
   `SPEED_SCALE` was reduced again (0.4 → 0.15): widening the radius raised
   the camera's actual linear/tangential speed proportionally
   (linear speed = radius × angular speed) even though angular speed
   in rad/s hadn't changed, so the radius change alone made the path feel
   faster again without any intentional speed-up. Verified: mean per-frame
   eye displacement in the final dataset is ~0.066 world units.

## Gate results

Dataset generated: **500 frames**, seed `20260812`, at
`E:\neural-upscaler\data\seed-20260812` (≈11.6 GB — colour, depth, motion at
960×540, ground-truth colour at 1920×1080, all uncompressed; see
`dataset.json` for the exact per-buffer layout and `manifest.jsonl` for
per-frame camera/jitter records). This is the *final* dataset, generated
after all three camera-path revisions above — the gate was re-run against
it, not left pointing at stale numbers from the pre-revision path.

Reprojection validated across **40 consecutive frame pairs** (frames 1–40).
The scripted camera path combines orbital rotation, radial forward/back
motion, and look-target panning *simultaneously* on every frame (see
`camera/path.ts`), so any contiguous window — not just this one — exercises
all three motion types the gate asks for, rather than needing separate
isolated runs per motion type. This was a judgement call; flagging it as
such rather than presenting it as the only valid interpretation.

| | mean error (included px) | p99 | max | disocclusion frac |
|---|---|---|---|---|
| mean across 40 pairs | 0.039 | 0.385 | — | 0.72% |
| worst single pair | — | — | 0.614 | ~1% (no more large sweeping-occlusion events, since the camera moves far more gently now) |

Colour channels are ~[0,1]; error is mean absolute channel difference.
Compared to the pre-revision gate run (mean 0.030, worst-case max 0.835,
mean disocclusion 9.2%): disocclusion fraction and worst-case max error both
dropped substantially (slower, more distant camera rarely sweeps a building
across the whole frame anymore), while mean error rose slightly. The likely
cause: the camera now orbits much farther out (~40 vs ~15 world units), so
the same fixed-size texture cells subtend fewer screen pixels — a bit closer
to their aliasing limit again, purely a consequence of viewing distance, not
a regression in the motion-vector math. Heatmaps still show black
(near-zero error) interiors with error concentrated at checker/stripe cell
edges, just a somewhat denser edge grid than the closer-orbit version.

**This is the gate the owner should look at personally** (per Spec 1) —
sampled heatmaps were shown inline during the session; the full set for all
40 pairs is on disk at the path above for direct review.

## Where the fragile logic lives

- `render/shaders.wgsl` — motion vector sign/space convention, linear depth
  derivation via the `clip.w` interpolation identity.
- `camera/projection.ts` — jitter-on-projection-matrix, verified against
  gl-matrix's actual matrix layout rather than assumed.
- `scene/texture.ts` — the value-noise fix and *why* it matters for
  reprojection (see comment block at the top of the file).
- `validate/reproject.ts` — disocclusion detection threshold
  (`DEPTH_REL_THRESHOLD = 0.05`, relative depth mismatch) — empirical,
  not derived; worth revisiting if future scenes have very different depth
  ranges.
- `scripts/capture.ts` — resumability depends on `frameState`/`stateViewProj`
  being pure functions of `(seed, frameIndex)`; don't introduce hidden
  per-run state into those without breaking resume-determinism (verified
  byte-identical between fresh and resumed runs during this session).
- `scene/colliders.ts` — camera collision avoidance uses a *conservative*
  bounding cylinder (circumscribed circle) for box colliders rather than a
  true oriented-box test, so clearance near box corners is a bit more
  generous than strictly necessary. If the scene grows much denser, this
  margin of error could start meaningfully constraining valid camera
  positions — worth revisiting with real OBB tests at that point rather
  than assuming the conservative approximation stays cheap.

## Known gaps / notes for later phases

- Dawn's Node bindings (`webgpu` package) don't report `shader-f16` as an
  available feature on this machine even though Chrome does. Not used by any
  current shader, so not blocking, but worth re-checking before Phase 4
  WGSL inference work if f16 becomes relevant to capture-side tooling.
- Capture throughput dropped from ~60 fps to ~23 fps over the 500-frame run
  (visible in the console log) — worth profiling before generating a much
  larger dataset for Phase 2/3, but not investigated further here since it
  wasn't a Phase 0/1 blocker.

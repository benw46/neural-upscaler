# Spec 1 of 4: Rendering & Data Pipeline

Read `CLAUDE.md` first. This spec covers building the renderer and turning it
into a training dataset. Stop at the gate below — do not begin Spec 2.

---

## Part A — WebGPU rasteriser

**Goal:** a minimal forward renderer that draws a test scene at arbitrary
resolution and writes colour, depth, and motion vectors to disk.

1. TypeScript + Vite scaffold. WebGPU adapter/device init with feature
   detection — log adapter name and `shader-f16` availability; fail loudly if
   the adapter is SwiftShader.
2. Render pipeline: buffers, bind groups, depth buffer + depth testing.
3. Test scene: Sponza via glTF if straightforward, otherwise procedural
   geometry with enough high-frequency detail to make upscaling non-trivial.
   Static scene only for now.
4. Camera: scripted path, deterministic from a seed, byte-identical output
   across runs.
5. MRT pass writing colour (RGBA16F), linear depth (R32F), and motion vectors
   (RG16F).
6. Resolution as a parameter — same scene/path renderable at 960×540 and
   1920×1080.
7. Headless capture mode: step the camera path, write targets to disk.

## Part B — Dataset generation

**Goal:** reproducible generation of (jittered 540p input + depth + motion
vectors) → (antialiased 1080p ground truth) frame pairs, plus the tool that
proves the motion vectors are correct.

1. Ground truth: render at 3840×2160, box-filter down to 1920×1080, no
   jitter on this path.
2. Jitter: Halton(2,3) sub-pixel offset applied to the projection matrix per
   frame (see `CLAUDE.md` fragile-logic list — jitter goes on the projection
   matrix, not the camera position). Record the offset used per frame.
3. Motion vectors: previous-frame MVP stored, per-vertex previous clip
   position, fragment-shader screen-space delta. State the sign/space
   convention chosen, explicitly, in the phase summary.
4. Capture orchestration: given a seed and frame count, generate the full
   dataset. Resumable.
5. Storage: uncompressed, on the `E:` drive (e.g. `E:\neural-upscaler\data`),
   laid out for efficient random access. Store full frames, not
   pre-extracted patches — patch extraction happens later, in the training
   dataloader. Make the base path configurable rather than hardcoded, but
   default it to the `E:` location.
6. Manifest: per-frame jitter offset, camera parameters, frame index.
7. **Reprojection validation tool** — given frames N−1 and N plus motion
   vectors, warp N−1 into N and output the warped image, an absolute error
   heatmap, and error stats excluding a disocclusion mask.
8. Disk budget report before generating, to catch a bad config early.

---

## Gate

- Renders the scene at both 960×540 and 1920×1080, dumps colour+depth
  reproducibly.
- Reprojection error is near-zero except at disocclusions/edges, shown across
  ≥20 frame pairs spanning pan, forward motion, and rotation. **Show the
  heatmaps** — this is the one gate the owner should look at personally, not
  just read a summary of, since a sign/space bug here is invisible in every
  later phase until it silently caps quality.

Write the `/docs` summary before stopping. State the motion vector convention
chosen and why, and the projected dataset size actually generated.

# Spec 4 of 4: WebGPU Inference

Prerequisite: Spec 3 gate passed. Read `CLAUDE.md` first.

**Goal:** the trained, temporally-stable model running in a browser tab via
hand-written WGSL, profiled and reasonably optimised. This is the final
phase — after its gate, stop and write the whole-project summary.

---

1. Confirm the ONNX Runtime Web path still works end to end with the final
   trained+temporal model from Spec 3. This is the correctness oracle for
   everything below — don't touch WGSL until this is confirmed.
2. Hand-write the conv layers in WGSL. Choose im2col+GEMM or direct
   convolution per layer based on channel counts — state the reasoning in
   the summary.
3. Feature map layout: use NHWC unless there's a specific reason not to (see
   `CLAUDE.md`).
4. Use FP16 where `shader-f16` is available; feature-detect, don't assume.
5. **Kernel fusion**: fuse conv+activation at minimum; evaluate fusing whole
   resolution tiers. Measure dispatch count and per-dispatch overhead before
   and after fusion — this is expected to matter more than intra-kernel
   tuning (see `CLAUDE.md` on WebGPU dispatch overhead).
6. Numerically diff WGSL output against ONNX Runtime Web output **per
   layer**, not just at the final output — report max/mean error per layer
   so a future bug can be localised quickly.
7. Profile with timestamp queries per layer/dispatch. Report ms/frame,
   compare against PSSR's ~2ms reference point, and report the achieved
   multiple honestly — you will not match it; report what you got and why.
8. Verify the whole pipeline on the actual adapter in `chrome://gpu`
   (Windows, D3D12) — not a SwiftShader fallback.

---

## Gate

- WGSL output matches ONNX Runtime Web within FP16 tolerance, reported per
  layer.
- Full per-layer profile captured and reported.
- Running live in a browser tab against the fully trained model.

## After this gate

Stop. Write a final `/docs` summary covering the whole pipeline end to end,
the key decisions made at each phase, and a list of what's most fragile or
would most reward review. Do not start additional features (higher
resolution, dynamic scenes, larger models) without being asked.

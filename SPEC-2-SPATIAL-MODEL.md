# Spec 2 of 4: Spatial Model

Prerequisite: Spec 1 gate passed. Read `CLAUDE.md` first.

**Goal:** a working 2x upscaler with no temporal component, proven against
baselines, and proven exportable to ONNX/WebGPU before further investment.
This phase deliberately excludes history/temporal logic — that's Spec 3.

---

1. U-Net in PyTorch, ~0.5–1M params: input res → down → down → down →
   bottleneck → up → up → up → output res, with skip connections. Inputs:
   jittered low-res colour + depth (from Spec 1's dataset).
2. Loss: L1 + a perceptual term (LPIPS or VGG features). State in the summary
   why L1 alone is insufficient (see `CLAUDE.md`).
3. Overfit a single patch to near-zero loss before any real training run —
   isolates data/model bugs from training-dynamics issues. Do this first and
   report the result before proceeding.
4. Train at 128×128 input patches, 2x output, using the Spec 1 dataset.
   Baselines to compare against: bilinear, bicubic, Lanczos. Metrics: PSNR,
   SSIM, LPIPS on held-out frames.
5. **Export the untrained, randomly-initialised model to ONNX now**, before
   spending training time. Load it in ONNX Runtime Web with the WebGPU
   backend and confirm every operator has a working kernel. If something
   doesn't export or doesn't run, fix the architecture now, not after
   training.
6. Train to convergence. Export the trained model to ONNX; diff ONNX Runtime
   Web's output against PyTorch's on the same input; report max/mean error.

---

## Gate

- Beats bicubic on held-out frames — report PSNR/SSIM/LPIPS numbers directly,
  not just "better."
- Trained model round-trips through ONNX Runtime Web within FP16 tolerance
  (report the max/mean error).

Write the `/docs` summary before stopping. Include the baseline comparison
table and the ONNX diff numbers.

import blitSource from "./blit.wgsl?raw";

/** Draws a source texture to the canvas via a fullscreen triangle. Preview-only. */
export class BlitPipeline {
  private readonly device: GPUDevice;
  private readonly pipeline: GPURenderPipeline;
  private readonly sampler: GPUSampler;
  private readonly layout: GPUBindGroupLayout;

  constructor(device: GPUDevice, canvasFormat: GPUTextureFormat) {
    this.device = device;
    const module = device.createShaderModule({ label: "blit-shader", code: blitSource });
    this.layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
      ],
    });
    this.pipeline = device.createRenderPipeline({
      label: "blit-pipeline",
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
      vertex: { module, entryPoint: "vs_main" },
      fragment: { module, entryPoint: "fs_main", targets: [{ format: canvasFormat }] },
      primitive: { topology: "triangle-list" },
    });
    this.sampler = device.createSampler({ magFilter: "nearest", minFilter: "nearest" });
  }

  draw(encoder: GPUCommandEncoder, source: GPUTexture, target: GPUTextureView) {
    const bindGroup = this.device.createBindGroup({
      layout: this.layout,
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: source.createView() },
      ],
    });
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: target, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
  }
}

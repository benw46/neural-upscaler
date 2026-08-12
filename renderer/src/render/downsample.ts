/** Box-filters a supersampled colour texture down to exactly half its width
 * and height on the GPU, so capture never has to read the (much larger)
 * supersampled buffer back to CPU. `shaderSource` is injected — see the note
 * on MainPipeline's constructor in pipeline.ts for why. */
export class DownsamplePipeline {
  private readonly pipeline: GPUComputePipeline;
  private readonly layout: GPUBindGroupLayout;
  private readonly device: GPUDevice;

  constructor(device: GPUDevice, downsampleSource: string) {
    this.device = device;
    const module = device.createShaderModule({ label: "downsample-shader", code: downsampleSource });
    this.layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba16float" } },
      ],
    });
    this.pipeline = device.createComputePipeline({
      label: "downsample-pipeline",
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
      compute: { module, entryPoint: "cs_main" },
    });
  }

  /** `dst` must be exactly half `src`'s width/height in each dimension. */
  run(encoder: GPUCommandEncoder, src: GPUTexture, dst: GPUTexture) {
    const bindGroup = this.device.createBindGroup({
      layout: this.layout,
      entries: [
        { binding: 0, resource: src.createView() },
        { binding: 1, resource: dst.createView() },
      ],
    });
    const pass = encoder.beginComputePass({ label: "downsample-pass" });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(dst.width / 8), Math.ceil(dst.height / 8));
    pass.end();
  }
}

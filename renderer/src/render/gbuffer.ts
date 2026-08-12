/** Offscreen MRT targets: colour, linear depth, motion vectors, plus the
 * hardware depth-stencil attachment used for z-testing (not itself saved —
 * `depth` is the saved linear-depth target per Spec 1 Part A step 5). */
export class GBuffer {
  width: number;
  height: number;
  colour!: GPUTexture;
  depth!: GPUTexture;
  motion!: GPUTexture;
  depthStencil!: GPUTexture;
  private readonly device: GPUDevice;

  constructor(device: GPUDevice, width: number, height: number) {
    this.device = device;
    this.width = width;
    this.height = height;
    this.create();
  }

  private create() {
    const usage =
      GPUTextureUsage.RENDER_ATTACHMENT |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_SRC;

    this.colour = this.device.createTexture({
      label: "gbuffer-colour",
      size: [this.width, this.height],
      format: "rgba16float",
      usage,
    });
    this.depth = this.device.createTexture({
      label: "gbuffer-linear-depth",
      size: [this.width, this.height],
      format: "r32float",
      usage,
    });
    this.motion = this.device.createTexture({
      label: "gbuffer-motion",
      size: [this.width, this.height],
      format: "rg16float",
      usage,
    });
    this.depthStencil = this.device.createTexture({
      label: "gbuffer-depth-stencil",
      size: [this.width, this.height],
      format: "depth32float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  resize(width: number, height: number) {
    if (width === this.width && height === this.height) return;
    this.colour.destroy();
    this.depth.destroy();
    this.motion.destroy();
    this.depthStencil.destroy();
    this.width = width;
    this.height = height;
    this.create();
  }

  colorAttachments(clear: boolean): GPURenderPassColorAttachment[] {
    const loadOp: GPULoadOp = clear ? "clear" : "load";
    return [
      { view: this.colour.createView(), loadOp, storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } },
      { view: this.depth.createView(), loadOp, storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } },
      { view: this.motion.createView(), loadOp, storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } },
    ];
  }

  depthStencilAttachment(): GPURenderPassDepthStencilAttachment {
    return {
      view: this.depthStencil.createView(),
      depthLoadOp: "clear",
      depthStoreOp: "store",
      depthClearValue: 1.0,
    };
  }
}

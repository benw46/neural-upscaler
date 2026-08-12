import { GBuffer } from "./gbuffer.ts";
import { MainPipeline, type GpuGroup } from "./pipeline.ts";
import { buildScene } from "../scene/scene.ts";

/** Owns the uploaded scene and pipeline; renders into any GBuffer given
 * explicit view-projection matrices. Deliberately has no notion of "current
 * frame" or jitter — that bookkeeping lives in camera/sequence.ts and is
 * driven externally, so the interactive preview (main.ts) and headless
 * capture (capture/*, Node + Dawn) can share this exact code path for both
 * the jittered input render and the unjittered ground-truth render, per
 * CLAUDE.md's validation-chain philosophy of not letting parallel
 * implementations drift apart. */
export class SceneRenderer {
  private readonly pipeline: MainPipeline;
  private readonly groups: GpuGroup[];
  private readonly device: GPUDevice;

  constructor(device: GPUDevice, mainShaderSource: string) {
    this.device = device;
    this.pipeline = new MainPipeline(device, mainShaderSource);
    this.groups = this.pipeline.uploadGroups(buildScene());
  }

  /** Renders one frame into `gbuffer` using the given (already
   * jittered-or-not) view-projection matrices. `prevViewProj` is whatever
   * was used to render the *previous* output frame in the same sequence —
   * pass a copy of `viewProj` itself for a frame with no real predecessor
   * (frame 0), which yields near-zero motion vectors rather than reading
   * uninitialised state. */
  renderInto(gbuffer: GBuffer, viewProj: Float32Array, prevViewProj: Float32Array, label = "frame") {
    this.pipeline.writeFrameUniforms(viewProj, prevViewProj);

    const encoder = this.device.createCommandEncoder({ label });
    const pass = encoder.beginRenderPass({
      colorAttachments: gbuffer.colorAttachments(true),
      depthStencilAttachment: gbuffer.depthStencilAttachment(),
    });
    pass.setPipeline(this.pipeline.pipeline);
    for (const group of this.groups) {
      pass.setBindGroup(0, group.bindGroup);
      pass.setVertexBuffer(0, group.mesh.vertexBuffer);
      pass.setIndexBuffer(group.mesh.indexBuffer, "uint32");
      pass.drawIndexed(group.mesh.indexCount);
    }
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }
}

import { VERTEX_FLOATS, type Mesh } from "../scene/geometry.ts";
import type { SceneGroup } from "../scene/scene.ts";

const FRAME_UNIFORM_SIZE = 128; // 2x mat4x4<f32>

export interface GpuMesh {
  vertexBuffer: GPUBuffer;
  indexBuffer: GPUBuffer;
  indexCount: number;
}

export interface GpuGroup {
  name: string;
  mesh: GpuMesh;
  bindGroup: GPUBindGroup;
}

export class MainPipeline {
  readonly pipeline: GPURenderPipeline;
  readonly frameUniformBuffer: GPUBuffer;
  private readonly bindGroupLayout: GPUBindGroupLayout;
  private readonly sampler: GPUSampler;
  private readonly device: GPUDevice;

  /**
   * `shaderSource` is injected rather than imported here because this class
   * is shared between the Vite-bundled browser build (which loads it via a
   * `?raw` import) and the Node capture script (which reads the .wgsl file
   * straight off disk) — see shaders.ts in each entry point.
   */
  constructor(device: GPUDevice, shaderSource: string) {
    this.device = device;
    const shaderModule = device.createShaderModule({ label: "main-mrt-shader", code: shaderSource });

    this.bindGroupLayout = device.createBindGroupLayout({
      label: "main-bind-group-layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "non-filtering" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
      ],
    });

    const pipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [this.bindGroupLayout],
    });

    this.pipeline = device.createRenderPipeline({
      label: "main-mrt-pipeline",
      layout: pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: "vs_main",
        buffers: [
          {
            arrayStride: VERTEX_FLOATS * 4,
            attributes: [
              { shaderLocation: 0, offset: 0, format: "float32x3" }, // position
              { shaderLocation: 1, offset: 12, format: "float32x3" }, // normal
              { shaderLocation: 2, offset: 24, format: "float32x2" }, // uv
            ],
          },
        ],
      },
      fragment: {
        module: shaderModule,
        entryPoint: "fs_main",
        targets: [
          { format: "rgba16float" },
          { format: "r32float" },
          { format: "rg16float" },
        ],
      },
      primitive: { topology: "triangle-list", cullMode: "back" },
      depthStencil: { format: "depth32float", depthWriteEnabled: true, depthCompare: "less" },
    });

    this.frameUniformBuffer = device.createBuffer({
      label: "frame-uniforms",
      size: FRAME_UNIFORM_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Point sampling deliberately — filtering would pre-blur the
    // high-frequency textures that are the whole point of the test scene.
    this.sampler = device.createSampler({ magFilter: "nearest", minFilter: "nearest" });
  }

  writeFrameUniforms(viewProj: Float32Array, prevViewProj: Float32Array) {
    this.device.queue.writeBuffer(this.frameUniformBuffer, 0, viewProj);
    this.device.queue.writeBuffer(this.frameUniformBuffer, 64, prevViewProj);
  }

  uploadMesh(mesh: Mesh, label: string): GpuMesh {
    const vertexBuffer = this.device.createBuffer({
      label: `${label}-vertices`,
      size: mesh.vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(vertexBuffer.getMappedRange()).set(mesh.vertices);
    vertexBuffer.unmap();

    const indexBuffer = this.device.createBuffer({
      label: `${label}-indices`,
      size: mesh.indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint32Array(indexBuffer.getMappedRange()).set(mesh.indices);
    indexBuffer.unmap();

    return { vertexBuffer, indexBuffer, indexCount: mesh.indices.length };
  }

  uploadGroups(groups: SceneGroup[]): GpuGroup[] {
    return groups.map((group) => {
      const mesh = this.uploadMesh(group.mesh, group.name);

      const texture = this.device.createTexture({
        label: `${group.name}-texture`,
        size: [group.textureSize, group.textureSize],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      this.device.queue.writeTexture(
        { texture },
        group.textureData,
        { bytesPerRow: group.textureSize * 4 },
        { width: group.textureSize, height: group.textureSize }
      );

      const bindGroup = this.device.createBindGroup({
        label: `${group.name}-bind-group`,
        layout: this.bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.frameUniformBuffer } },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: texture.createView() },
        ],
      });

      return { name: group.name, mesh, bindGroup };
    });
  }
}

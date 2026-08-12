// Fullscreen-triangle blit: samples a source texture and writes it to the
// canvas. Used only for interactive preview — capture mode reads MRT
// textures back to CPU directly and never touches this pipeline.

@group(0) @binding(0) var srcSampler: sampler;
@group(0) @binding(1) var srcTexture: texture_2d<f32>;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0),
  );
  var out: VertexOutput;
  let p = positions[vertexIndex];
  out.position = vec4<f32>(p, 0.0, 1.0);
  out.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
  return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  return textureSample(srcTexture, srcSampler, in.uv);
}

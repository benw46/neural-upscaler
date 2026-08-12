// Neural Upscaler — main forward MRT shader.
//
// Motion vector convention (see CLAUDE.md fragile-logic list — state this
// explicitly, it's invisible-until-it-isn't):
//   - Computed in UV space (origin top-left, V down — i.e. framebuffer /
//     texture-sample space, NOT NDC which is y-up).
//   - mv = current_uv - previous_uv
//   - To reproject: previous_uv = current_uv - mv
//   - So mv points FROM the previous-frame sample location TO the
//     current-frame location; subtracting it from a current pixel's UV
//     walks you back to where that surface point was last frame. This is
//     the convention the reprojection validation tool (validate/reproject.ts)
//     assumes.
//   - The scene is static in Phase 0/1 (camera-only motion), so "previous
//     clip position" only differs from "current clip position" because
//     prevViewProj != viewProj — there is no per-object model motion yet.

struct FrameUniforms {
  viewProj: mat4x4<f32>,
  prevViewProj: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var texSampler: sampler;
@group(0) @binding(2) var tex: texture_2d<f32>;

struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
}

struct VertexOutput {
  @builtin(position) clipPosition: vec4<f32>,
  @location(0) worldNormal: vec3<f32>,
  @location(1) uv: vec2<f32>,
  @location(2) currClip: vec4<f32>,
  @location(3) prevClip: vec4<f32>,
  @location(4) viewSpaceZ: f32,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
  var out: VertexOutput;
  let worldPos = vec4<f32>(in.position, 1.0);

  let currClip = frame.viewProj * worldPos;
  let prevClip = frame.prevViewProj * worldPos;

  out.clipPosition = currClip;
  out.currClip = currClip;
  out.prevClip = prevClip;
  out.worldNormal = in.normal;
  out.uv = in.uv;
  // currClip.w is the view-space -Z for a standard perspective projection;
  // that's exactly the linear depth (distance along view axis) we want.
  out.viewSpaceZ = currClip.w;
  return out;
}

struct FragmentOutput {
  @location(0) colour: vec4<f32>,
  @location(1) linearDepth: vec4<f32>,
  @location(2) motionVector: vec4<f32>,
}

const LIGHT_DIR = vec3<f32>(0.4, 0.8, 0.35); // pre-normalised-ish, normalised below
const AMBIENT = 0.28;

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
  var out: FragmentOutput;

  let texColour = textureSample(tex, texSampler, fract(in.uv));
  let n = normalize(in.worldNormal);
  let l = normalize(LIGHT_DIR);
  let diffuse = max(dot(n, l), 0.0);
  let shade = AMBIENT + (1.0 - AMBIENT) * diffuse;
  out.colour = vec4<f32>(texColour.rgb * shade, 1.0);

  out.linearDepth = vec4<f32>(in.viewSpaceZ, 0.0, 0.0, 1.0);

  let currNdc = in.currClip.xy / in.currClip.w;
  let prevNdc = in.prevClip.xy / in.prevClip.w;
  // NDC (y-up) -> UV (y-down)
  let currUv = vec2<f32>(currNdc.x * 0.5 + 0.5, 0.5 - currNdc.y * 0.5);
  let prevUv = vec2<f32>(prevNdc.x * 0.5 + 0.5, 0.5 - prevNdc.y * 0.5);
  out.motionVector = vec4<f32>(currUv - prevUv, 0.0, 1.0);

  return out;
}

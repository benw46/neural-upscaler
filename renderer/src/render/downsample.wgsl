// 2x2 box filter: reads a supersampled colour texture, writes the averaged
// half-resolution result. Ground-truth antialiasing per Spec 1 Part B step 1
// — a merely-high-resolution render is not sufficient on its own; this is
// the "downfilter" half of "supersample and downfilter".

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dstSize = textureDimensions(dst);
  if (gid.x >= dstSize.x || gid.y >= dstSize.y) {
    return;
  }
  let sx = gid.x * 2u;
  let sy = gid.y * 2u;
  let a = textureLoad(src, vec2<u32>(sx, sy), 0);
  let b = textureLoad(src, vec2<u32>(sx + 1u, sy), 0);
  let c = textureLoad(src, vec2<u32>(sx, sy + 1u), 0);
  let d = textureLoad(src, vec2<u32>(sx + 1u, sy + 1u), 0);
  let avg = (a + b + c + d) * 0.25;
  textureStore(dst, vec2<u32>(gid.x, gid.y), avg);
}

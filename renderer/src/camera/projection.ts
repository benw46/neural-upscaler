import { mat4, vec3 } from "gl-matrix";
import type { CameraPose } from "./path.ts";

export function viewMatrix(pose: CameraPose): mat4 {
  const m = mat4.create();
  mat4.lookAt(m, pose.eye, pose.target, pose.up);
  return m;
}

export function baseProjectionMatrix(fovYRadians: number, aspect: number, near: number, far: number): mat4 {
  const m = mat4.create();
  // WebGPU clip-space Z range is [0, 1] (unlike OpenGL's [-1, 1]) — use the
  // ZO (zero-to-one) variant, and note NDC Y is up while framebuffer Y is
  // down (WebGPU flips it during rasterisation); see CLAUDE.md fragile list.
  mat4.perspectiveZO(m, fovYRadians, aspect, near, far);
  return m;
}

/**
 * Applies a sub-pixel jitter offset to a projection matrix.
 *
 * IMPORTANT: jitter is injected here, as a post-multiply shift of the
 * projection matrix's x/y NDC offset terms — NOT by translating the camera
 * eye position. Translating the camera changes parallax (every point in the
 * scene shifts by a different amount depending on depth), which corrupts
 * the very motion-vector geometry this dataset exists to teach. A projection
 * shift moves every point by the same NDC offset regardless of depth, which
 * is what sub-pixel jitter is supposed to do. See CLAUDE.md fragile-logic
 * list ("Jitter must be applied to the projection matrix").
 *
 * `jitterTexels` is a sub-pixel offset in [-0.5, 0.5) texel units, in the
 * *target render resolution* (width/height passed in). NDC spans 2 units
 * (-1..1) across `width`/`height` pixels, so 1 texel = 2/width (resp.
 * 2/height) NDC units. NDC y is up (see baseProjectionMatrix note), matching
 * this function's `jitterTexels[1]` sign convention directly (no flip here).
 */
export function jitterProjectionMatrix(
  proj: mat4,
  jitterTexels: [number, number],
  width: number,
  height: number
): mat4 {
  const jittered = mat4.clone(proj);
  const ndcX = (2 * jitterTexels[0]) / width;
  const ndcY = (2 * jitterTexels[1]) / height;
  // Column-major mat4: column 2 holds the x/y NDC offset terms (indices 8, 9).
  jittered[8] += ndcX;
  jittered[9] += ndcY;
  return jittered;
}

export function viewProjectionMatrix(view: mat4, proj: mat4): mat4 {
  const vp = mat4.create();
  mat4.multiply(vp, proj, view);
  return vp;
}

export const worldUp = vec3.fromValues(0, 1, 0);

import { vec3, type mat4 } from "gl-matrix";
import { ScriptedCameraPath, type CameraPose } from "./path.ts";
import { baseProjectionMatrix, jitterProjectionMatrix, viewMatrix, viewProjectionMatrix } from "./projection.ts";
import { haltonJitter } from "./halton.ts";
import { resolveAgainstColliders, type Collider } from "../scene/colliders.ts";

export const NEAR = 0.1;
export const FAR = 200;
export const FOV_Y = (60 * Math.PI) / 180;
export const DT = 1 / 30; // fixed simulation step — frameIndex * DT is the only time input, keeps output deterministic

// Clearance kept between the camera eye and any collider surface, and above
// the ground plane. Bigger than NEAR so the near clip plane itself never
// pokes into geometry even when the eye is right at the boundary.
const COLLISION_MARGIN = 0.4;
const GROUND_MARGIN = 0.25;

export interface FrameCameraState {
  frameIndex: number;
  t: number;
  jitter: [number, number];
  pose: CameraPose;
}

/** Pure function of (path, frameIndex, colliders) — no hidden state, so
 * callers can request any frame (including "previous frame") independently
 * and get an identical result every time. This is what keeps capture
 * byte-identical across runs and lets capture mode resume from an arbitrary
 * frame index.
 *
 * When `colliders` is given, the raw analytic eye position is pushed
 * outside every collider (plus margin) and above the ground plane — the
 * scripted path is a simple orbit/oscillation formula with no awareness of
 * scene geometry, so without this the camera can end up inside a building
 * or below ground. The look-at target is left untouched: looking toward a
 * point behind/inside geometry is normal, it's only the eye that must never
 * be inside something solid. */
export function frameState(path: ScriptedCameraPath, frameIndex: number, colliders?: Collider[]): FrameCameraState {
  const t = frameIndex * DT;
  const rawPose = path.poseAt(t);

  let pose = rawPose;
  if (colliders && colliders.length > 0) {
    const [x, y, z] = resolveAgainstColliders(
      [rawPose.eye[0], rawPose.eye[1], rawPose.eye[2]],
      colliders,
      COLLISION_MARGIN,
      GROUND_MARGIN
    );
    pose = { eye: vec3.fromValues(x, y, z), target: rawPose.target, up: rawPose.up };
  }

  return {
    frameIndex,
    t,
    jitter: haltonJitter(frameIndex),
    pose,
  };
}

/** Builds the (optionally jittered) view-projection matrix for a frame state
 * at a given target resolution. Ground-truth rendering passes `jitter: false`
 * per Spec 1 Part B step 1 ("no jitter on this path"). */
export function stateViewProj(
  state: FrameCameraState,
  width: number,
  height: number,
  jittered: boolean
): mat4 {
  const view = viewMatrix(state.pose);
  const baseProj = baseProjectionMatrix(FOV_Y, width / height, NEAR, FAR);
  const proj = jittered ? jitterProjectionMatrix(baseProj, state.jitter, width, height) : baseProj;
  return viewProjectionMatrix(view, proj);
}

export { ScriptedCameraPath };
export type { CameraPose };

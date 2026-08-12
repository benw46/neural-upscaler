import type { mat4 } from "gl-matrix";
import { ScriptedCameraPath, type CameraPose } from "./path.ts";
import { baseProjectionMatrix, jitterProjectionMatrix, viewMatrix, viewProjectionMatrix } from "./projection.ts";
import { haltonJitter } from "./halton.ts";

export const NEAR = 0.1;
export const FAR = 200;
export const FOV_Y = (60 * Math.PI) / 180;
export const DT = 1 / 30; // fixed simulation step — frameIndex * DT is the only time input, keeps output deterministic

export interface FrameCameraState {
  frameIndex: number;
  t: number;
  jitter: [number, number];
  pose: CameraPose;
}

/** Pure function of (path, frameIndex) — no hidden state, so callers can
 * request any frame (including "previous frame") independently and get an
 * identical result every time. This is what keeps capture byte-identical
 * across runs and lets capture mode resume from an arbitrary frame index. */
export function frameState(path: ScriptedCameraPath, frameIndex: number): FrameCameraState {
  const t = frameIndex * DT;
  return {
    frameIndex,
    t,
    jitter: haltonJitter(frameIndex),
    pose: path.poseAt(t),
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

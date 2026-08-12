import { vec3 } from "gl-matrix";

export interface CameraPose {
  eye: vec3;
  target: vec3;
  up: vec3;
}

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Deterministic scripted camera path, reproducible byte-for-byte from `seed`.
 * Combines orbital rotation, radial (forward/back) motion, and a panning
 * look-target sweep, so any ~20+ frame window exercises all three motion
 * types the gate asks for. Pure function of (seed, frameIndex, dt) — no
 * hidden state, so re-running produces identical output. */
export class ScriptedCameraPath {
  private readonly baseRadius: number;
  private readonly radiusAmp: number;
  private readonly radiusFreq: number;
  private readonly radiusPhase: number;
  private readonly angularSpeed: number;
  private readonly anglePhase: number;
  private readonly baseHeight: number;
  private readonly heightAmp: number;
  private readonly heightFreq: number;
  private readonly heightPhase: number;
  private readonly panAmp: number;
  private readonly panFreq: number;
  private readonly panPhase: number;

  constructor(seed: number) {
    const rng = mulberry32(seed);
    this.baseRadius = 14 + rng() * 4;
    this.radiusAmp = 5 + rng() * 3;
    this.radiusFreq = 0.15 + rng() * 0.05;
    this.radiusPhase = rng() * Math.PI * 2;
    this.angularSpeed = 0.25 + rng() * 0.1;
    this.anglePhase = rng() * Math.PI * 2;
    this.baseHeight = 3 + rng() * 2;
    this.heightAmp = 1.5 + rng() * 1;
    this.heightFreq = 0.1 + rng() * 0.05;
    this.heightPhase = rng() * Math.PI * 2;
    this.panAmp = 4 + rng() * 2;
    this.panFreq = 0.12 + rng() * 0.04;
    this.panPhase = rng() * Math.PI * 2;
  }

  /** `t` is seconds of scripted time — pass `frameIndex * dt` for a fixed
   * simulation step, so output depends only on (seed, frameIndex, dt). */
  poseAt(t: number): CameraPose {
    const radius = this.baseRadius + this.radiusAmp * Math.sin(t * this.radiusFreq + this.radiusPhase);
    const angle = this.angularSpeed * t + this.anglePhase;
    const height = this.baseHeight + this.heightAmp * Math.sin(t * this.heightFreq + this.heightPhase);

    const eye = vec3.fromValues(Math.cos(angle) * radius, height, Math.sin(angle) * radius);

    const panX = this.panAmp * Math.sin(t * this.panFreq + this.panPhase);
    const panZ = this.panAmp * Math.cos(t * this.panFreq * 0.7 + this.panPhase);
    const target = vec3.fromValues(panX, this.baseHeight * 0.5, panZ);

    return { eye, target, up: vec3.fromValues(0, 1, 0) };
  }
}

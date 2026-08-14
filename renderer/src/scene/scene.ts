import { makeBox, makeCylinder, makePlane, mergeMeshes, transformMesh, type Mesh } from "./geometry.ts";
import { checkerTexture, colorCheckerTexture, noiseTexture, stripeTexture } from "./texture.ts";
import type { Collider } from "./colliders.ts";

/** One draw-call group: merged geometry sharing a single procedural texture. */
export interface SceneGroup {
  name: string;
  mesh: Mesh;
  textureData: Uint8Array;
  textureSize: number;
}

export interface Scene {
  groups: SceneGroup[];
  /** Camera-avoidance colliders — built from the exact same placement loops
   * as the render geometry below (not a separately-derived RNG stream), so
   * they can never drift out of sync with what's actually rendered. */
  colliders: Collider[];
}

/** Deterministic seeded PRNG (mulberry32) — scene layout must be reproducible. */
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const GROUND_SIZE = 60;
const SEED = 20260812;

/** Builds the static test scene: a ground plane with a grid of "buildings"
 * of varying height, plus thin cylindrical rods scattered between them.
 * All content uses high-frequency procedural textures/geometry deliberately
 * (see texture.ts) so 540p rendering aliases and there's real upscaling
 * work to evaluate. Static scene only, per Spec 1 Part A step 3. */
export function buildScene(): Scene {
  const rng = mulberry32(SEED);
  const colliders: Collider[] = [];

  // uvTile values kept moderate (not maximally fine) — see texture.ts header:
  // tiling density this high combined with per-texel-uncorrelated content
  // made reprojection numerically meaningless even with correct motion
  // vectors, discovered while validating the Phase 0/1 gate.
  const ground = makePlane(GROUND_SIZE, GROUND_SIZE, 4);

  const buildingMeshes: Mesh[] = [];
  const gridN = 6;
  const cellSize = GROUND_SIZE / gridN;
  for (let gx = 0; gx < gridN; gx++) {
    for (let gz = 0; gz < gridN; gz++) {
      // Leave a gap in the centre for camera manoeuvring room.
      const cx = (gx - (gridN - 1) / 2) * cellSize;
      const cz = (gz - (gridN - 1) / 2) * cellSize;
      if (Math.hypot(cx, cz) < cellSize * 1.2) continue;

      const w = cellSize * (0.4 + rng() * 0.3);
      const d = cellSize * (0.4 + rng() * 0.3);
      const h = 1 + rng() * 8;
      const box = makeBox(w, h, d, 1);
      buildingMeshes.push(transformMesh(box, [cx, h / 2, cz], rng() * Math.PI * 2));
      // Circumscribed circle (half-diagonal) — rotation-independent, so no
      // need to track the box's rotation angle for collision purposes.
      colliders.push({ center: [cx, cz], radius: Math.hypot(w / 2, d / 2), minY: 0, maxY: h });
    }
  }

  const rodMeshes: Mesh[] = [];
  const rodCount = 24;
  for (let i = 0; i < rodCount; i++) {
    const angle = rng() * Math.PI * 2;
    const radius = 2 + rng() * (GROUND_SIZE / 2 - 3);
    const rx = Math.cos(angle) * radius;
    const rz = Math.sin(angle) * radius;
    const height = 2 + rng() * 4;
    const rodRadius = 0.06 + rng() * 0.05;
    const cyl = makeCylinder(rodRadius, height, 8);
    rodMeshes.push(transformMesh(cyl, [rx, height / 2, rz]));
    colliders.push({ center: [rx, rz], radius: rodRadius, minY: 0, maxY: height });
  }

  const accentMeshes: Mesh[] = [];
  const accentCount = 10;
  for (let i = 0; i < accentCount; i++) {
    const angle = rng() * Math.PI * 2;
    const radius = 3 + rng() * (GROUND_SIZE / 2 - 4);
    const ax = Math.cos(angle) * radius;
    const az = Math.sin(angle) * radius;
    const s = 0.4 + rng() * 0.6;
    const box = makeBox(s, s, s, 1);
    accentMeshes.push(transformMesh(box, [ax, s / 2, az], rng() * Math.PI * 2));
    colliders.push({ center: [ax, az], radius: Math.hypot(s / 2, s / 2), minY: 0, maxY: s });
  }

  // One fixed, hand-placed accent cube (not drawn from `rng`, so it doesn't
  // perturb the 10 seeded ones above or anything drawn after them) sitting
  // directly in the scripted camera path's early view. With only 10 small
  // (0.4-1.0 unit) accent cubes scattered across the full 60-unit ground
  // plane, the live viewer's realtime mode could run for a long time before
  // the orbit happened to swing past one -- confirmed directly by computing
  // ScriptedCameraPath(SEED)'s actual eye/target at t=0 (renderer/src/camera/path.ts):
  // eye=(40.41, 3.00, 0.20), looking in direction (-0.993, -0.016, -0.117),
  // an 8.5-degree offset from this cube's placement below (comfortably
  // inside the ~60x91-degree vertical/horizontal FOV, not dead-centre so it
  // doesn't sit exactly behind anything on the camera's own forward axis).
  //
  // Placed at (20, 0) rather than directly on that ray on purpose: the
  // building grid (gridN=6, cellSize=10) has cell centres at x,z in
  // {-25,-15,-5,5,15,25}, each building only 4-7 units wide -- (20, 0) sits
  // exactly at the gap corner between the x=15/x=25 columns and z=-5/z=5
  // rows, >=7 units from every neighbouring cell centre, clear of every
  // building regardless of that cell's randomly-drawn width/rotation. A
  // first attempt placed directly along the t=0 view ray (~20 units out)
  // turned out to sit almost exactly on the x=25 column and was occluded --
  // corrected empirically after checking the live render, not just the ray
  // math alone. Sized larger (1.4) than the random ones (max 1.0) so it
  // reads clearly at this distance, not as a speck.
  {
    const s = 1.4;
    const ax = 20, az = 0;
    const box = makeBox(s, s, s, 1);
    accentMeshes.push(transformMesh(box, [ax, s / 2, az], 0));
    colliders.push({ center: [ax, az], radius: Math.hypot(s / 2, s / 2), minY: 0, maxY: s });
  }

  const groups: SceneGroup[] = [
    { name: "ground", mesh: ground, textureData: noiseTexture(128, SEED), textureSize: 128 },
    { name: "buildings", mesh: mergeMeshes(buildingMeshes), textureData: checkerTexture(64), textureSize: 64 },
    { name: "rods", mesh: mergeMeshes(rodMeshes), textureData: stripeTexture(64), textureSize: 64 },
    { name: "accents", mesh: mergeMeshes(accentMeshes), textureData: colorCheckerTexture(64, [220, 90, 60], [40, 60, 200]), textureSize: 64 },
  ];

  return { groups, colliders };
}

import { makeBox, makeCylinder, makePlane, mergeMeshes, transformMesh, type Mesh } from "./geometry.ts";
import { checkerTexture, colorCheckerTexture, noiseTexture, stripeTexture } from "./texture.ts";

/** One draw-call group: merged geometry sharing a single procedural texture. */
export interface SceneGroup {
  name: string;
  mesh: Mesh;
  textureData: Uint8Array;
  textureSize: number;
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
export function buildScene(): SceneGroup[] {
  const rng = mulberry32(SEED);

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
    const cyl = makeCylinder(0.06 + rng() * 0.05, height, 8);
    rodMeshes.push(transformMesh(cyl, [rx, height / 2, rz]));
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
  }

  return [
    { name: "ground", mesh: ground, textureData: noiseTexture(128, SEED), textureSize: 128 },
    { name: "buildings", mesh: mergeMeshes(buildingMeshes), textureData: checkerTexture(64), textureSize: 64 },
    { name: "rods", mesh: mergeMeshes(rodMeshes), textureData: stripeTexture(64), textureSize: 64 },
    { name: "accents", mesh: mergeMeshes(accentMeshes), textureData: colorCheckerTexture(64, [220, 90, 60], [40, 60, 200]), textureSize: 64 },
  ];
}

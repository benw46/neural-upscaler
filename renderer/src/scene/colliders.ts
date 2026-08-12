/** Conservative bounding-cylinder collider for camera-path avoidance.
 * Boxes (buildings/accents) use their circumscribed circle (radius =
 * half-diagonal) rather than a true oriented-box test — slightly more
 * generous clearance at box corners, but simple and impossible to get the
 * rotation math wrong on, which matters more than tightness here. */
export interface Collider {
  center: [number, number]; // x, z
  radius: number;
  minY: number;
  maxY: number;
}

/** Pushes `pos` outside every collider (plus `margin`) and above the ground
 * plane (plus `groundMargin`). Iterates a few passes so a push out of one
 * collider that lands inside a neighbour still gets resolved — scene
 * objects aren't guaranteed non-overlapping with each other. */
export function resolveAgainstColliders(
  pos: [number, number, number],
  colliders: Collider[],
  margin: number,
  groundMargin: number,
  iterations = 4
): [number, number, number] {
  let [x, y, z] = pos;
  y = Math.max(y, groundMargin);

  for (let iter = 0; iter < iterations; iter++) {
    let moved = false;
    for (const c of colliders) {
      if (y < c.minY - margin || y > c.maxY + margin) continue;

      const dx = x - c.center[0];
      const dz = z - c.center[1];
      const dist = Math.hypot(dx, dz);
      const minDist = c.radius + margin;
      if (dist >= minDist) continue;

      moved = true;
      if (dist < 1e-6) {
        // Degenerate: camera sits exactly on the collider's centre axis —
        // any escape direction is equally valid.
        x = c.center[0] + minDist;
        z = c.center[1];
      } else {
        const scale = minDist / dist;
        x = c.center[0] + dx * scale;
        z = c.center[1] + dz * scale;
      }
    }
    if (!moved) break;
  }

  return [x, y, z];
}

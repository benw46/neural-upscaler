/** Halton low-discrepancy sequence, base `b`, 1-indexed (index 0 is skipped
 * since it's always 0 — a degenerate jitter offset). */
export function halton(index: number, b: number): number {
  let f = 1;
  let r = 0;
  let i = index;
  while (i > 0) {
    f = f / b;
    r = r + f * (i % b);
    i = Math.floor(i / b);
  }
  return r;
}

/** Halton(2,3) sub-pixel jitter sequence, in [-0.5, 0.5) texel offsets. */
export function haltonJitter(frameIndex: number): [number, number] {
  const i = (frameIndex % 16) + 1; // 16-frame cycle keeps sequence low-discrepancy without unbounded growth
  const jx = halton(i, 2) - 0.5;
  const jy = halton(i, 3) - 0.5;
  return [jx, jy];
}

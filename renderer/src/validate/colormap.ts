/** Black -> red -> yellow -> white heat ramp for error visualisation.
 * `t` in [0, 1]. Chosen over plain grayscale so near-zero error reads as
 * unambiguously black and any real error pops visually. */
export function heatColor(t: number): [number, number, number] {
  const c = Math.max(0, Math.min(1, t));
  const stops: [number, [number, number, number]][] = [
    [0.0, [0, 0, 0]],
    [0.33, [200, 0, 0]],
    [0.66, [255, 210, 0]],
    [1.0, [255, 255, 255]],
  ];
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, c0] = stops[i];
    const [t1, c1] = stops[i + 1];
    if (c >= t0 && c <= t1) {
      const f = (c - t0) / (t1 - t0);
      return [
        Math.round(c0[0] + (c1[0] - c0[0]) * f),
        Math.round(c0[1] + (c1[1] - c0[1]) * f),
        Math.round(c0[2] + (c1[2] - c0[2]) * f),
      ];
    }
  }
  return [255, 255, 255];
}

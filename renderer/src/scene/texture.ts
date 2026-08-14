/** Procedural textures with deliberately high spatial frequency — the point
 * is to give the 540p render something that *will* alias, so the upscaler
 * has real work to do. Point/nearest sampling is used elsewhere so this
 * detail isn't pre-filtered away by the GPU's own mip/aniso filtering.
 *
 * Important constraint learned while validating reprojection (Spec 1 Part B
 * step 7): "high-frequency" must still mean *locally coherent* detail (fine
 * but continuous variation), not per-texel-independent noise. Point-sampled
 * white noise has zero spatial correlation between adjacent texels, so even
 * mathematically perfect motion vectors can't reproject it — any sub-texel
 * sampling difference lands on an unrelated random value. That showed up
 * during Phase 0/1 validation as ~14% mean reprojection error covering
 * entire surfaces; a diagnostic with a smooth gradient in place of noise
 * dropped mean error to ~0.5%, confirming the motion-vector math itself was
 * correct and the textures were the problem. Value noise below keeps the
 * fine detail (and the aliasing) while staying locally smooth. */

function makeRgba(size: number, fill: (x: number, y: number) => [number, number, number]): Uint8Array {
  const data = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const [r, g, b] = fill(x, y);
      const i = (y * size + x) * 4;
      data[i] = r;
      data[i + 1] = g;
      data[i + 2] = b;
      data[i + 3] = 255;
    }
  }
  return data;
}

/** Multiplies a scalar grayscale value by a [0,1] tint, clamped to a valid
 * byte. `[1,1,1]` (the default everywhere below) reproduces the original
 * grayscale output exactly. Applied as a flat multiply on the existing
 * validated luminance field rather than deriving colour independently, so
 * whatever spatial-correlation property that field already has (see this
 * file's header) carries over unchanged -- tinting can't reintroduce the
 * aliasing failure mode a per-channel-independent scheme could. */
function applyTint(c: number, tint: [number, number, number]): [number, number, number] {
  return [Math.min(255, Math.round(c * tint[0])), Math.min(255, Math.round(c * tint[1])), Math.min(255, Math.round(c * tint[2]))];
}

/** Checkerboard with `cellSize`-texel cells (default 4, not 1) — cell
 * interiors are constant colour, so reprojection error concentrates at cell
 * boundaries (a small fraction of pixels) rather than everywhere. */
export function checkerTexture(size = 64, cellSize = 6, tint: [number, number, number] = [1, 1, 1]): Uint8Array {
  return makeRgba(size, (x, y) => {
    const cellX = Math.floor(x / cellSize);
    const cellY = Math.floor(y / cellSize);
    const c = (cellX + cellY) % 2 === 0 ? 235 : 25;
    return applyTint(c, tint);
  });
}

/** Coloured checkerboard, coarser, used to distinguish surfaces at a glance. */
export function colorCheckerTexture(size = 64, colorA: [number, number, number], colorB: [number, number, number], cellSize = 4): Uint8Array {
  return makeRgba(size, (x, y) => {
    const cellX = Math.floor(x / cellSize), cellY = Math.floor(y / cellSize);
    return (cellX + cellY) % 2 === 0 ? colorA : colorB;
  });
}

function seededRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5; state >>>= 0;
    return state / 0xffffffff;
  };
}

/** Smooth value noise: a coarse random lattice, bilinearly upsampled. Still
 * fine-grained enough to alias under 540p point-sampling, but each texel is
 * correlated with its neighbours (unlike per-texel white noise), which is
 * what makes reprojection against it meaningful — see file header. Lattice
 * wraps at the edges so the tiled texture has no extra hard seam beyond the
 * ones already inherent to tiling. */
export function noiseTexture(size = 128, seed = 1234, latticeSize = 6, tint: [number, number, number] = [1, 1, 1]): Uint8Array {
  const rand = seededRng(seed);
  const lattice: number[][] = [];
  for (let ly = 0; ly <= latticeSize; ly++) {
    const row: number[] = [];
    for (let lx = 0; lx <= latticeSize; lx++) row.push(rand());
    lattice.push(row);
  }
  for (let lx = 0; lx <= latticeSize; lx++) lattice[latticeSize][lx] = lattice[0][lx];
  for (let ly = 0; ly <= latticeSize; ly++) lattice[ly][latticeSize] = lattice[ly][0];

  function sampleLattice(u: number, v: number): number {
    const fx = u * latticeSize;
    const fy = v * latticeSize;
    const x0 = Math.floor(fx);
    const y0 = Math.floor(fy);
    const tx = fx - x0;
    const ty = fy - y0;
    const v00 = lattice[y0][x0];
    const v10 = lattice[y0][x0 + 1];
    const v01 = lattice[y0 + 1][x0];
    const v11 = lattice[y0 + 1][x0 + 1];
    const top = v00 * (1 - tx) + v10 * tx;
    const bottom = v01 * (1 - tx) + v11 * tx;
    return top * (1 - ty) + bottom * ty;
  }

  return makeRgba(size, (x, y) => {
    const n = sampleLattice(x / size, y / size);
    const c = Math.round(40 + n * 200);
    return applyTint(c, tint);
  });
}

/** Alternating stripe pattern, `stripeWidth` texels per band (default 4, not
 * 1) — band interiors are constant colour, same reasoning as checkerTexture. */
export function stripeTexture(size = 64, stripeWidth = 6, tint: [number, number, number] = [1, 1, 1]): Uint8Array {
  return makeRgba(size, (x) => {
    const c = Math.floor(x / stripeWidth) % 2 === 0 ? 250 : 10;
    return applyTint(c, tint);
  });
}

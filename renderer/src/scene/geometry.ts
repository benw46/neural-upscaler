/** Procedural mesh generators. All meshes share the same interleaved vertex
 * layout: position(3) + normal(3) + uv(2) = 8 floats, f32. */

export interface Mesh {
  vertices: Float32Array;
  indices: Uint32Array;
}

export const VERTEX_FLOATS = 8;

function pushVertex(
  out: number[],
  px: number, py: number, pz: number,
  nx: number, ny: number, nz: number,
  u: number, v: number
) {
  out.push(px, py, pz, nx, ny, nz, u, v);
}

/** Axis-aligned box centred at origin, size given as full extents.
 * `uvTile` scales UVs per face so textures repeat instead of stretching —
 * this is what keeps high-frequency checker detail dense on large faces. */
export function makeBox(sx: number, sy: number, sz: number, uvTile = 1): Mesh {
  const hx = sx / 2, hy = sy / 2, hz = sz / 2;
  const verts: number[] = [];
  const idx: number[] = [];

  type Face = {
    normal: [number, number, number];
    corners: [number, number, number][]; // CCW when viewed from outside
    uScale: number;
    vScale: number;
  };

  const faces: Face[] = [
    // +X
    { normal: [1, 0, 0], corners: [[hx, -hy, hz], [hx, -hy, -hz], [hx, hy, -hz], [hx, hy, hz]], uScale: sz, vScale: sy },
    // -X
    { normal: [-1, 0, 0], corners: [[-hx, -hy, -hz], [-hx, -hy, hz], [-hx, hy, hz], [-hx, hy, -hz]], uScale: sz, vScale: sy },
    // +Y
    { normal: [0, 1, 0], corners: [[-hx, hy, hz], [hx, hy, hz], [hx, hy, -hz], [-hx, hy, -hz]], uScale: sx, vScale: sz },
    // -Y
    { normal: [0, -1, 0], corners: [[-hx, -hy, -hz], [hx, -hy, -hz], [hx, -hy, hz], [-hx, -hy, hz]], uScale: sx, vScale: sz },
    // +Z
    { normal: [0, 0, 1], corners: [[-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]], uScale: sx, vScale: sy },
    // -Z
    { normal: [0, 0, -1], corners: [[hx, -hy, -hz], [-hx, -hy, -hz], [-hx, hy, -hz], [hx, hy, -hz]], uScale: sx, vScale: sy },
  ];

  const faceUVs: [number, number][] = [[0, 1], [1, 1], [1, 0], [0, 0]];

  for (const face of faces) {
    const base = verts.length / VERTEX_FLOATS;
    for (let i = 0; i < 4; i++) {
      const [px, py, pz] = face.corners[i];
      const [u, v] = faceUVs[i];
      pushVertex(
        verts, px, py, pz,
        face.normal[0], face.normal[1], face.normal[2],
        u * face.uScale * uvTile, v * face.vScale * uvTile
      );
    }
    idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }

  return { vertices: new Float32Array(verts), indices: new Uint32Array(idx) };
}

/** Flat XZ-plane centred at origin, y=0, facing +Y. `uvTile` = texture repeats across the plane. */
export function makePlane(sx: number, sz: number, uvTile: number): Mesh {
  const hx = sx / 2, hz = sz / 2;
  const verts: number[] = [];
  pushVertex(verts, -hx, 0, hz, 0, 1, 0, 0, uvTile);
  pushVertex(verts, hx, 0, hz, 0, 1, 0, uvTile, uvTile);
  pushVertex(verts, hx, 0, -hz, 0, 1, 0, uvTile, 0);
  pushVertex(verts, -hx, 0, -hz, 0, 1, 0, 0, 0);
  const indices = new Uint32Array([0, 1, 2, 0, 2, 3]);
  return { vertices: new Float32Array(verts), indices };
}

/** Upright cylinder centred at origin, radius `r`, height `h`, `segments` sides.
 * Thin, tall cylinders are good aliasing torture tests (sub-pixel width at distance). */
export function makeCylinder(r: number, h: number, segments: number): Mesh {
  const hh = h / 2;
  const verts: number[] = [];
  const idx: number[] = [];

  // Side wall
  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    const angle = t * Math.PI * 2;
    const nx = Math.cos(angle), nz = Math.sin(angle);
    const px = nx * r, pz = nz * r;
    pushVertex(verts, px, hh, pz, nx, 0, nz, t * 1.5, 1);
    pushVertex(verts, px, -hh, pz, nx, 0, nz, t * 4, 0);
  }
  for (let i = 0; i < segments; i++) {
    const a = i * 2, b = i * 2 + 1, c = i * 2 + 2, d = i * 2 + 3;
    idx.push(a, b, c, b, d, c);
  }

  // Top and bottom caps
  const capBase = verts.length / VERTEX_FLOATS;
  pushVertex(verts, 0, hh, 0, 0, 1, 0, 0.5, 0.5);
  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    const angle = t * Math.PI * 2;
    const px = Math.cos(angle) * r, pz = Math.sin(angle) * r;
    pushVertex(verts, px, hh, pz, 0, 1, 0, 0.5 + Math.cos(angle) * 0.5, 0.5 + Math.sin(angle) * 0.5);
  }
  for (let i = 0; i < segments; i++) {
    idx.push(capBase, capBase + 1 + i, capBase + 2 + i);
  }

  const capBase2 = verts.length / VERTEX_FLOATS;
  pushVertex(verts, 0, -hh, 0, 0, -1, 0, 0.5, 0.5);
  for (let i = 0; i <= segments; i++) {
    const t = i / segments;
    const angle = t * Math.PI * 2;
    const px = Math.cos(angle) * r, pz = Math.sin(angle) * r;
    pushVertex(verts, px, -hh, pz, 0, -1, 0, 0.5 + Math.cos(angle) * 0.5, 0.5 + Math.sin(angle) * 0.5);
  }
  for (let i = 0; i < segments; i++) {
    idx.push(capBase2, capBase2 + 2 + i, capBase2 + 1 + i);
  }

  return { vertices: new Float32Array(verts), indices: new Uint32Array(idx) };
}

/** Concatenate multiple meshes into one vertex/index buffer pair (single draw call, no transform). */
export function mergeMeshes(meshes: Mesh[]): Mesh {
  let vertCount = 0;
  let idxCount = 0;
  for (const m of meshes) {
    vertCount += m.vertices.length;
    idxCount += m.indices.length;
  }
  const vertices = new Float32Array(vertCount);
  const indices = new Uint32Array(idxCount);
  let vOff = 0, iOff = 0, vertBase = 0;
  for (const m of meshes) {
    vertices.set(m.vertices, vOff);
    for (let i = 0; i < m.indices.length; i++) {
      indices[iOff + i] = m.indices[i] + vertBase;
    }
    vOff += m.vertices.length;
    iOff += m.indices.length;
    vertBase += m.vertices.length / VERTEX_FLOATS;
  }
  return { vertices, indices };
}

/** Apply a rigid transform (uniform scale + rotation about Y + translation) to a mesh's positions/normals in place. */
export function transformMesh(
  mesh: Mesh,
  translate: [number, number, number],
  rotateY = 0,
  scale = 1
): Mesh {
  const out = new Float32Array(mesh.vertices);
  const cos = Math.cos(rotateY), sin = Math.sin(rotateY);
  for (let i = 0; i < out.length; i += VERTEX_FLOATS) {
    const x = out[i] * scale, y = out[i + 1] * scale, z = out[i + 2] * scale;
    out[i] = x * cos + z * sin + translate[0];
    out[i + 1] = y + translate[1];
    out[i + 2] = -x * sin + z * cos + translate[2];

    const nx = out[i + 3], nz = out[i + 5];
    out[i + 3] = nx * cos + nz * sin;
    out[i + 5] = -nx * sin + nz * cos;
  }
  return { vertices: out, indices: mesh.indices };
}

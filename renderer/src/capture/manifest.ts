export interface DatasetHeader {
  seed: number;
  frameCount: number;
  inputWidth: number;
  inputHeight: number;
  gtWidth: number;
  gtHeight: number;
  gtSupersampleWidth: number;
  gtSupersampleHeight: number;
  fovYRadians: number;
  near: number;
  far: number;
  dt: number;
  /** Buffer dtype/layout documentation — the actual contract downstream
   * PyTorch dataloaders read against. Every .bin file is tightly packed
   * row-major, no padding, matching these formats exactly. */
  buffers: {
    color: { format: "rgba16float"; width: number; height: number };
    depth: { format: "r32float"; width: number; height: number };
    motion: { format: "rg16float"; width: number; height: number };
    gt_color: { format: "rgba16float"; width: number; height: number };
  };
  createdAt: string;
}

export interface ManifestRecord {
  frameIndex: number;
  t: number;
  jitter: [number, number];
  eye: [number, number, number];
  target: [number, number, number];
  up: [number, number, number];
  viewProj: number[]; // 16 floats, jittered, input resolution
  prevViewProj: number[]; // 16 floats — the previous frame's viewProj, for motion vector reconstruction
}

export function manifestLine(record: ManifestRecord): string {
  return JSON.stringify(record) + "\n";
}

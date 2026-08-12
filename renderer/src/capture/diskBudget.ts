/** Per-frame byte accounting for the dataset layout, and a report string
 * printed before any writes — hard rule 2: report projected size before
 * generating, confirm free space first. */
export interface DiskBudget {
  colorBytes: number;
  depthBytes: number;
  motionBytes: number;
  gtColorBytes: number;
  bytesPerFrame: number;
  totalBytes: number;
}

export function computeDiskBudget(
  inputWidth: number,
  inputHeight: number,
  gtWidth: number,
  gtHeight: number,
  frameCount: number
): DiskBudget {
  const colorBytes = inputWidth * inputHeight * 4 * 2; // rgba16float
  const depthBytes = inputWidth * inputHeight * 1 * 4; // r32float
  const motionBytes = inputWidth * inputHeight * 2 * 2; // rg16float
  const gtColorBytes = gtWidth * gtHeight * 4 * 2; // rgba16float
  const bytesPerFrame = colorBytes + depthBytes + motionBytes + gtColorBytes;
  return {
    colorBytes,
    depthBytes,
    motionBytes,
    gtColorBytes,
    bytesPerFrame,
    totalBytes: bytesPerFrame * frameCount,
  };
}

function formatBytes(bytes: number): string {
  const gb = bytes / 1024 ** 3;
  if (gb >= 1) return `${gb.toFixed(2)} GB`;
  const mb = bytes / 1024 ** 2;
  return `${mb.toFixed(2)} MB`;
}

export function formatDiskBudgetReport(budget: DiskBudget, frameCount: number, freeBytes: number): string {
  const lines = [
    `Disk budget for ${frameCount} frames:`,
    `  colour    : ${formatBytes(budget.colorBytes)}/frame`,
    `  depth     : ${formatBytes(budget.depthBytes)}/frame`,
    `  motion    : ${formatBytes(budget.motionBytes)}/frame`,
    `  gt_colour : ${formatBytes(budget.gtColorBytes)}/frame`,
    `  total     : ${formatBytes(budget.bytesPerFrame)}/frame -> ${formatBytes(budget.totalBytes)} projected`,
    `  free space: ${formatBytes(freeBytes)}`,
  ];
  return lines.join("\n");
}

/** Refuse to proceed if the projected dataset would use more than 90% of
 * currently free space — leaves headroom rather than running the drive dry
 * mid-generation. */
export function fitsOnDisk(budget: DiskBudget, freeBytes: number): boolean {
  return budget.totalBytes < freeBytes * 0.9;
}

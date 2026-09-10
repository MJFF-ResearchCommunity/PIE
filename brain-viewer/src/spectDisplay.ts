import type { Modality, ViewMode } from "./types";

// A reversible visualization preset, NOT a segmentation or diagnostic threshold.
export const INITIAL_SPECT_CUTOFF = 0.5;

export function clipEmissionToAnatomy(
  hasContext: boolean,
  mode: ViewMode,
  anatomyOpacity: number,
) {
  // A fully transparent MRI cannot supply NiiVue's foreground-alpha mask.
  return hasContext && mode === "3d" && anatomyOpacity > 0;
}

export function spectCutoff(
  modality: Modality,
  mode: ViewMode,
  cutoff: number,
) {
  return modality === "SPECT" && mode === "3d"
    ? Math.max(0, Math.min(1, Number.isFinite(cutoff) ? cutoff : 0))
    : 0;
}

/** Preserve the color window and RGB values; change display alpha only.
 * The source voxels, geometry, and probe values never enter this function.
 * NiiVue interpolates this 256-entry transfer function on the GPU.
 */
export function thresholdColormap(lut: Uint8ClampedArray, cutoff: number) {
  const cm = {
    I: [] as number[],
    R: [] as number[],
    G: [] as number[],
    B: [] as number[],
    A: [] as number[],
  };
  for (let i = 0; i < 256; i++) {
    cm.I.push(i);
    cm.R.push(lut[i * 4]);
    cm.G.push(lut[i * 4 + 1]);
    cm.B.push(lut[i * 4 + 2]);
    // Short opacity ramp avoids an artificially hard-edged isosurface.
    const alpha =
      cutoff <= 0 ? 1 : Math.max(0, Math.min(1, (i / 255 - cutoff) / 0.05));
    cm.A.push(Math.round(lut[i * 4 + 3] * alpha));
  }
  return cm;
}

/** Camera target only: bounded sampling of signal above the initial display
 * cutoff. Cap weights at the color-window ceiling so one bright outlier cannot
 * dominate. Returns RAS-buffer voxel coordinates, not a tissue classification.
 */
export function signalCameraTarget(
  dims: number[],
  valueAt: (x: number, y: number, z: number) => number,
  threshold: number,
  ceiling: number,
): number[] | null {
  if (!(ceiling > threshold)) return null;
  const step = Math.max(
    1,
    Math.ceil(Math.cbrt((dims[0] * dims[1] * dims[2]) / 250_000)),
  );
  const sum = [0, 0, 0];
  let weight = 0;
  for (let z = 0; z < dims[2]; z += step)
    for (let y = 0; y < dims[1]; y += step)
      for (let x = 0; x < dims[0]; x += step) {
        const value = valueAt(x, y, z);
        if (!Number.isFinite(value) || value <= threshold) continue;
        const w = Math.min(value, ceiling) - threshold;
        sum[0] += x * w;
        sum[1] += y * w;
        sum[2] += z * w;
        weight += w;
      }
  return weight > 0 ? sum.map((v) => v / weight) : null;
}

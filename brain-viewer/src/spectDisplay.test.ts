import { describe, expect, it } from "vitest";
import {
  INITIAL_SPECT_CUTOFF,
  spectCutoff,
  thresholdColormap,
  signalCameraTarget,
  clipEmissionToAnatomy,
} from "./spectDisplay";

const lut = Uint8ClampedArray.from({ length: 1024 }, (_, i) =>
  i % 4 === 3 ? 255 : Math.floor(i / 4),
);

describe("SPECT display-only transfer function", () => {
  it("does not erase SPECT when the MRI layer is hidden", () => {
    expect(clipEmissionToAnatomy(true, "3d", 0)).toBe(false);
    expect(clipEmissionToAnatomy(true, "3d", 0.8)).toBe(true);
    expect(clipEmissionToAnatomy(false, "3d", 0.8)).toBe(false);
    expect(clipEmissionToAnatomy(true, "multi", 0.8)).toBe(false);
  });
  it("suppresses low signal without changing RGB or the input LUT", () => {
    const original = lut.slice();
    const cm = thresholdColormap(lut, INITIAL_SPECT_CUTOFF);
    expect(cm.A.slice(0, 128).every((a) => a === 0)).toBe(true);
    expect(cm.A[134]).toBeGreaterThan(0);
    expect(cm.A[134]).toBeLessThan(255);
    expect(cm.A[150]).toBe(255);
    expect(cm.R).toEqual(Array.from({ length: 256 }, (_, i) => i));
    expect(lut).toEqual(original);
  });
  it("restores the original alpha at zero cutoff", () => {
    expect(thresholdColormap(lut, 0).A).toEqual(Array(256).fill(255));
  });
  it("leaves slices, four-view and every other modality unthresholded", () => {
    for (const mode of ["multi", "axial", "coronal", "sagittal"] as const)
      expect(spectCutoff("SPECT", mode, 0.5)).toBe(0);
    for (const modality of ["MRI", "DTI", "PET", "CT", "fMRI"] as const)
      expect(spectCutoff(modality, "3d", 0.5)).toBe(0);
    expect(spectCutoff("SPECT", "3d", 0.5)).toBe(0.5);
  });
  it("bounds invalid display-state values", () => {
    expect(spectCutoff("SPECT", "3d", NaN)).toBe(0);
    expect(spectCutoff("SPECT", "3d", -1)).toBe(0);
    expect(spectCutoff("SPECT", "3d", 2)).toBe(1);
  });
  it("centers the camera on signal, ignoring background and nonfinite voxels", () => {
    expect(
      signalCameraTarget(
        [10, 10, 10],
        (x, y, z) =>
          x === 2 && y === 4 && z === 8 ? 0.8 : x === 0 ? NaN : 0.01,
        0.15,
        0.3,
      ),
    ).toEqual([2, 4, 8]);
    expect(signalCameraTarget([2, 2, 2], () => 0, 0.15, 0.3)).toBeNull();
  });
  it("caps camera weights rather than letting a bright outlier dominate", () => {
    expect(
      signalCameraTarget(
        [3, 1, 1],
        (x) => (x === 0 ? 1000 : x === 2 ? 1 : 0),
        0.1,
        1,
      ),
    ).toEqual([1, 0, 0]);
  });
});

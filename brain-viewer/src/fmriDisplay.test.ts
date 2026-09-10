import { describe, it, expect } from "vitest";
import {
  frameSeconds,
  percentFromMean,
  tracePath,
  voxelTimeSeries,
} from "./fmriDisplay";
import { captureLines } from "./captureView";
import type { Geometry, Prepared, Display } from "./types";

describe("BOLD inspection", () => {
  const geometry = {
    frames: 240,
    frame_step: 2.5,
    time_unit: "sec",
  } as Geometry;
  it("separates within-run time from dates and does not guess unknown units", () => {
    expect(frameSeconds(geometry, 239)).toBe(597.5);
    expect(
      frameSeconds({ ...geometry, frame_step: 2500, time_unit: "msec" }, 3),
    ).toBe(7.5);
    expect(frameSeconds({ ...geometry, time_unit: "unknown" }, 3)).toBeNull();
    expect(frameSeconds(geometry, 240)).toBeNull();
    expect(frameSeconds(geometry, -1)).toBeNull();
  });
  it("reads each RAS-buffer frame at one voxel, rejecting out-of-grid points", () => {
    const series = voxelTimeSeries(
      [1, 2, 3],
      [5, 6, 7],
      3,
      (x, y, z, f) => x + y + z + f,
    );
    expect(series).toEqual({ voxel: [1, 2, 3], values: [6, 7, 8] });
    expect(voxelTimeSeries([-1, 2, 3], [5, 6, 7], 3, () => 0)).toBeUndefined();
    expect(voxelTimeSeries([1, 2, 3], [5, 6, 7], 2, () => NaN)?.values).toEqual(
      [null, null],
    );
  });
  it("uses percent from temporal mean without inventing a zero-signal baseline", () => {
    expect(percentFromMean([90, 100, 110])).toEqual([-10, 0, 10]);
    expect(percentFromMean([0, 0])).toBeNull();
    expect(percentFromMean([null, NaN])).toBeNull();
    expect(tracePath([null, 1, 2, null, 4]).path.match(/M/g)?.length).toBe(2);
    expect(tracePath([5, 5]).path).not.toContain("NaN");
  });
  it("exports the actual frame or summary representation with correct units", () => {
    const p = {
      scan: {
        id: "b",
        subject: "001",
        date: "2021-01-01",
        modality: "fMRI",
        units: "BOLD a.u.",
      },
      geometry,
      extra: [
        {
          key: "tsnr",
          name: "Temporal signal-to-noise ratio",
          units: "dimensionless",
        },
      ],
    } as Prepared;
    const display = {
      mode: "axial",
      metric: "bold",
      frame: 239,
      window: [0, 200],
      opacity: 1,
    } as Display;
    expect(captureLines(p, display).join(" ")).toContain(
      "BOLD frame 240/240 · 597.50 s",
    );
    const summary = captureLines(p, { ...display, metric: "tsnr" }).join(" ");
    expect(summary).toContain("Temporal summary of all 240 frames");
    expect(summary).toContain("dimensionless");
    expect(summary).toContain("not activation or connectivity");
  });
});

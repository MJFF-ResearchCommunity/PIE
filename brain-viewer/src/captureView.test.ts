import { describe, expect, it } from "vitest";
import { captureLines } from "./captureView";
import type { Display, Prepared } from "./types";

const display = { structures: false, atlas: false, atlasOutline: false, mode: "multi", window: [0, 100], opacity: 1,
  selectedStructures: [], spectCutoff: 0.5 } as unknown as Display;
const t1 = { scan: { id: "mni152-t1", subject: "MNI152-template", modality: "MRI", units: "", qc: "Template" } } as unknown as Prepared;
const statistic = { scan: { id: "stat-x", kind: "statistic", modality: "Statistic", units: "SD",
  metadata: { statistic: "Cohen's d (A - B)", vmax: 0.7317, threshold: { p_column: "p_fdr", alpha: 0.05 } } } } as unknown as Prepared;

describe("export header", () => {
  it("names a statistic overlay with its threshold rule and colour limits", () => {
    const text = captureLines(t1, display, null, statistic).join(" ");
    for (const part of ["Cohen's d (A - B)", "p_fdr < 0.05", "±0.73 SD", "regional, not voxelwise"])
      expect(text).toContain(part);
  });
  it("adds no overlay line when there is no statistic overlay", () => {
    expect(captureLines(t1, display, null, null)).toEqual(captureLines(t1, display));
  });
});

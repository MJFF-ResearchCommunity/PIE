import { describe, expect, it } from "vitest";
import { statisticDisplay, statisticLabel, statisticNote } from "./statDisplay";

describe("statisticDisplay", () => {
  it("is symmetric so negative effects stay visible", () => {
    const d = statisticDisplay({ vmax: 0.5 });
    expect(d).toMatchObject({ colormap: "warm", colormapNegative: "cool", cal_max: 0.5, cal_minNeg: -0.5, empty: false });
    expect(d.cal_min).toBeGreaterThan(0);
    expect(d.cal_maxNeg).toBeLessThan(0);
  });
  it("marks an all-subthreshold map as empty instead of dividing by zero", () => {
    expect(statisticDisplay({ vmax: 0 }).empty).toBe(true);
  });
});


const scan = (id: string, threshold: unknown) => ({ id, description: "d", units: "SD",
  metadata: { statistic: "Cohen's d (PD - HC)", vmax: 0.5, threshold } });

describe("statistic overlay text", () => {
  it("tells two maps of the same statistic apart", () => {
    const a = statisticLabel(scan("stat-a", { p_column: "p", alpha: 0.05 }));
    const b = statisticLabel(scan("stat-b", null));
    expect(a).not.toEqual(b);
    expect(a).toContain("p < 0.05");
    expect(b).toContain("all matched regions");
  });
  it("describes a statistic overlay instead of an acquisition date", () => {
    const note = statisticNote(scan("stat-a", { p_column: "p_fdr", alpha: 0.05 }));
    expect(note).toContain("p_fdr < 0.05");
    expect(note).toContain("±0.50 SD");
    expect(note).not.toContain("acquired");
  });
});

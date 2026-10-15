import { describe, expect, it } from "vitest";
import { statisticDisplay } from "./statDisplay";

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

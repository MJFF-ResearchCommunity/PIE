import { describe, expect, it } from "vitest";
import { initialWindow, roundSignificant } from "./model";

describe("data-derived window display", () => {
  it("rounds a percentile window to four significant figures", () => {
    // The PET example's window: raw floats filled the Window min/max inputs.
    expect(initialWindow(0.4733345210552, 4984.794921875)).toEqual([0.4733, 4985]);
  });

  it("leaves exact presets and zero unchanged", () => {
    expect(initialWindow(0, 1)).toEqual([0, 1]);
    expect(initialWindow(-400, 1800)).toEqual([-400, 1800]);
    expect(roundSignificant(0)).toBe(0);
  });

  it("returns plain numbers at tiny and large magnitudes", () => {
    expect(initialWindow(0.000123456, 0.00234567)).toEqual([0.0001235, 0.002346]);
    expect(String(initialWindow(1, 1234567.8)[1])).toBe("1235000");
  });

  it("never collapses or inverts a narrow window", () => {
    expect(initialWindow(100.001, 100.002)).toEqual([100.001, 100.002]);
  });
});

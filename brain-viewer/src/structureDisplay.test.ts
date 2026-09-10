import { describe, it, expect } from "vitest";
import { availableStriatum, structureOpacity } from "./structureDisplay";
import { captureLines } from "./captureView";
import type { Display, Prepared, Structures } from "./types";

const display = {
  structures: true,
  mode: "3d",
  leftOpacity: 0,
  rightOpacity: 0.3,
  contextOpacity: 0.2,
  selectedStructures: ["11"],
  window: [0, 1],
  spectCutoff: 0.7,
  opacity: 0.7,
} as Display;
describe("anatomical representation", () => {
  it("hides hemisphere shells independently without hiding selected nuclei", () => {
    expect(structureOpacity("left", display)).toBe(0);
    expect(structureOpacity("right", display)).toBe(0.3);
    expect(structureOpacity("11", display)).toBe(0.7);
    expect(structureOpacity("50", display)).toBe(0);
  });
  it("never adds meshes to measured slice review", () => {
    for (const mode of ["multi", "axial", "sagittal", "coronal"] as const)
      expect(structureOpacity("11", { ...display, mode })).toBe(0);
    expect(structureOpacity("11", { ...display, structures: false })).toBe(0);
  });
  it("does not invent missing striatal labels", () => {
    expect(
      availableStriatum({
        meshes: [{ key: "11" }, { key: "13" }],
      } as Structures),
    ).toEqual(["11"]);
  });
  it("exports hidden-signal and unreviewed status", () => {
    const p = {
      scan: {
        id: "s",
        subject: "001",
        date: "2020-01-01",
        modality: "SPECT",
        units: "counts",
      },
      context: { reference_id: "m", reference_date: "2020-02-01" },
    } as Prepared;
    const lines = captureLines(p, { ...display, hideSignal: true });
    expect(lines.join(" ")).toContain("ALIGNMENT NOT REVIEWED");
    expect(lines.join(" ")).toContain("SPECT HIDDEN");
    expect(lines.join(" ")).toContain("2020-02-01");
    expect(captureLines(p, { ...display, mode: "multi" }).join(" ")).toContain(
      "extra cutoff paused",
    );
  });
});

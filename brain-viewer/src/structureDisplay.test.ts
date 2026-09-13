import { describe, it, expect } from "vitest";
import {
  anatomicalLabelLut,
  anatomyDisplayOpacity,
  availableStriatum,
  selectedStructureIds,
  selectedLabelBuffer,
  structureAtlasState,
  structureOpacity,
  structureViewLabel,
} from "./structureDisplay";
import { captureLines } from "./captureView";
import type { Display, Prepared, Structures } from "./types";

const display = {
  structures: true,
  structureMri: true,
  structureMriOpacity: 0.35,
  structureOpacity: 0.7,
  structureOutlines: true,
  atlas: false,
  atlasOpacity: 0.38,
  atlasOutline: false,
  mode: "3d",
  leftOpacity: 0,
  rightOpacity: 0.3,
  contextOpacity: 0.2,
  selectedStructures: ["11"],
  window: [0, 1],
  spectCutoff: 0.7,
  opacity: 0.7,
} as Display;
const structures = {
  meshes: [
    { key: "left", region_ids: [2, 1002] },
    { key: "11", region_ids: [11] },
    { key: "50", region_ids: [50] },
  ],
} as Structures;
describe("anatomical representation", () => {
  it("filters a separate display buffer while preserving the full atlas for picking", () => {
    const original = new Uint16Array([0, 11, 10, 50, 1002, 11]);
    const visible = selectedLabelBuffer(original, [11]);
    expect(Array.from(visible)).toEqual([0, 11, 0, 0, 0, 11]);
    expect(Array.from(original)).toEqual([0, 11, 10, 50, 1002, 11]);
    expect(visible.buffer).not.toBe(original.buffer);
    expect(Array.from(selectedLabelBuffer(original, []))).toEqual([
      0, 0, 0, 0, 0, 0,
    ]);
  });
  it("uses transparent label interiors for the renderer's boundary-only pass", () => {
    const regions = [
      { id: 11, name: "Caudate", color: [122, 186, 220] },
    ] as Prepared["regions"];
    const outline = anatomicalLabelLut(regions, [11], true);
    expect(outline.A).toEqual([0, 0]);
    expect(outline.R).toEqual([0, 122]);
  });
  it("keeps MRI visible with independent opacity in combined 3D mode", () => {
    expect(anatomyDisplayOpacity(display, 1, structures)).toBe(0.35);
    expect(
      anatomyDisplayOpacity(
        { ...display, structureMriOpacity: 0.8 },
        0.5,
        structures,
      ),
    ).toBe(0.8);
    expect(structureOpacity("11", { ...display, structureOpacity: 0.2 })).toBe(
      0.2,
    );
  });
  it("supports structures-only 3D without hiding MRI in slices", () => {
    const hidden = { ...display, structureMri: false };
    expect(anatomyDisplayOpacity(hidden, 1, structures)).toBe(0);
    for (const mode of ["multi", "axial", "sagittal", "coronal"] as const)
      expect(anatomyDisplayOpacity({ ...hidden, mode }, 1, structures)).toBe(1);
  });
  it("turning structures off restores the user's original image and atlas settings", () => {
    const off = {
      ...display,
      structures: false,
      atlas: true,
      atlasOutline: false,
    };
    expect(anatomyDisplayOpacity(off, 0.82, structures)).toBe(0.82);
    expect(structureOpacity("11", off)).toBe(0);
    expect(structureAtlasState(off, structures)).toEqual({
      outlines: false,
      opacity: 0.38,
      ids: null,
    });
  });
  it("does not change rendering when a structure dataset is unavailable", () => {
    expect(anatomyDisplayOpacity(display, 0.6, null)).toBe(0.6);
    expect(structureAtlasState(display, null)).toEqual({
      outlines: false,
      opacity: 0,
      ids: null,
    });
    expect(structureViewLabel(display, null)).toBeNull();
  });
  it("links selected deep structures to outlines in every measured slice mode", () => {
    for (const mode of ["multi", "axial", "sagittal", "coronal"] as const)
      expect(structureAtlasState({ ...display, mode }, structures)).toEqual({
        outlines: true,
        opacity: 0.7,
        ids: [11],
      });
    expect(structureAtlasState(display, structures).opacity).toBe(0);
    expect(
      structureAtlasState(
        { ...display, mode: "axial", structureOutlines: false },
        structures,
      ).opacity,
    ).toBe(0);
  });
  it("uses only present selected deep labels, never a generic hemisphere or absent label", () => {
    expect(
      selectedStructureIds(
        { ...display, selectedStructures: ["left", "50", "999"] },
        structures,
      ),
    ).toEqual([50]);
    expect(
      structureAtlasState({ ...display, selectedStructures: [] }, structures)
        .ids,
    ).toEqual([]);
  });
  it("filters the display LUT without changing label IDs, colors or source region data", () => {
    const regions = [
      { id: 11, name: "Left Caudate", color: [122, 186, 220] },
      { id: 50, name: "Right Caudate", color: [122, 186, 220] },
    ] as Prepared["regions"];
    const before = JSON.stringify(regions);
    const lut = anatomicalLabelLut(regions, [50]);
    expect(lut.I).toEqual([0, 11, 50]);
    expect(lut.A).toEqual([0, 0, 255]);
    expect(lut.R).toEqual([0, 122, 122]);
    expect(anatomicalLabelLut(regions, []).A).toEqual([0, 0, 0]);
    expect(anatomicalLabelLut(regions, null).A).toEqual([0, 255, 255]);
    expect(JSON.stringify(regions)).toBe(before);
  });
  it("clearly distinguishes combined, structures-only and slice representations in exports", () => {
    const p = {
      scan: {
        id: "m",
        subject: "001",
        modality: "MRI",
        units: "arbitrary intensity",
      },
    } as Prepared;
    const combined = captureLines(p, display, structures).join(" ");
    expect(combined).toContain(
      "Measured MRI + participant segmentation boundaries",
    );
    expect(combined).toContain("see-through 3D · MRI 35%");
    expect(combined).toContain("not validated pial surfaces");
    expect(
      captureLines(p, { ...display, structureMri: false }, structures).join(
        " ",
      ),
    ).toContain("MRI hidden in 3D");
    expect(
      captureLines(p, { ...display, structureMriOpacity: 0 }, structures).join(
        " ",
      ),
    ).toContain("MRI hidden in 3D");
    expect(
      captureLines(p, { ...display, mode: "multi" }, structures).join(" "),
    ).toContain("selected slice outlines on");
    expect(
      structureViewLabel({ ...display, structures: false }, structures),
    ).toBeNull();
  });
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

import { describe, it, expect } from "vitest";
import { canOverlay, nearestScan, dateLabel, nearbyAtlasLabel } from "./model";
import type { Scan, Subject } from "./types";

const scan = (patch: Partial<Scan> = {}): Scan => ({
  id: "t1",
  subject: "1",
  modality: "MRI",
  date: "2020-01-01",
  visit: "BL",
  description: "Test",
  space: "t1:1",
  kind: "scalar",
  units: "a.u.",
  reference_id: null,
  registration: "native",
  provenance: "",
  qc: "",
  tracer: null,
  has_atlas: false,
  has_anatomy: false,
  metadata: {},
  ...patch,
});

it("bounds surface label lookup in physical mm even with anisotropic voxels", () => {
  const affine = [
    [-2, 0, 0, 10],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ];
  const dims = [10, 10, 10];
  const inside = nearbyAtlasLabel(
    [6, 2, 2],
    [2, 2, 2],
    affine,
    dims,
    (x, y, z) => (x === 3 && y === 2 && z === 2 ? 12 : 0),
  );
  expect(inside).toEqual({ id: 12, distance: 2 });
  expect(
    nearbyAtlasLabel([6, 2, 2], [2, 2, 2], affine, dims, (x) =>
      x === 4 ? 12 : 0,
    ),
  ).toBeNull();
});
describe("fusion guardrails", () => {
  it("requires explicit same-participant, reference-specific registration", () => {
    const base = scan();
    const overlay = scan({
      id: "pet",
      modality: "PET",
      reference_id: "t1",
      registration: "verified",
    });
    expect(canOverlay(base, overlay)).toBe(true);
    expect(canOverlay(base, { ...overlay, subject: "2" })).toBe(false);
    expect(canOverlay(base, { ...overlay, space: "native-pet" })).toBe(false);
    expect(canOverlay(base, { ...overlay, registration: "native" })).toBe(
      false,
    );
    expect(canOverlay(base, { ...overlay, reference_id: "other-t1" })).toBe(
      false,
    );
    expect(canOverlay(base, base)).toBe(false);
    expect(canOverlay(base, { ...overlay, registration: "unreviewed" })).toBe(
      false,
    );
  });
  it("permits different acquisition dates only with explicit registration", () => {
    expect(
      canOverlay(
        scan(),
        scan({
          id: "fMRI",
          date: "2020-01-04",
          registration: "verified",
          reference_id: "t1",
        }),
      ),
    ).toBe(true);
  });
});
describe("acquisition chronology", () => {
  const subject: Subject = {
    id: "1",
    group: "Control",
    cohort: "Control",
    dates: [],
    modalities: ["MRI"],
    scans: [
      scan(),
      scan({ id: "year-1", date: "2021-01-01" }),
      scan({ id: "unknown", date: null }),
    ],
  };
  it("selects the nearest actual scan and never constructs an interpolated one", () => {
    expect(nearestScan(subject, "MRI", "2020-12-15")?.id).toBe("year-1");
    expect(nearestScan(subject, "PET", "2020-12-15")).toBeUndefined();
  });
  it("keeps exact dates stable regardless of browser timezone", () => {
    expect(dateLabel("2020-01-01")).toBe("January 1, 2020");
    expect(dateLabel(null)).toBe("Date not established");
  });
  it("prefers a full BOLD run over a short reference on the same date", () => {
    const candidates = {
      ...subject,
      scans: [
        scan({
          id: "a-reference",
          modality: "fMRI",
          metadata: { short_reference: true },
        }),
        scan({ id: "b-full", modality: "fMRI" }),
      ],
    };
    expect(nearestScan(candidates, "fMRI", "2020-01-01")?.id).toBe("b-full");
    candidates.scans[1].date = "2021-01-01";
    expect(nearestScan(candidates, "fMRI", "2020-01-01")?.id).toBe(
      "a-reference",
    );
  });
});

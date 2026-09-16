import { describe, expect, it } from "vitest";
import { bundleLine, comparisonExample } from "./model";
import type { Scan, Subject } from "./types";

const mri = (id: string, date: string | null): Scan => ({
  id,
  subject: "s",
  modality: "MRI",
  date,
  visit: "visit",
  description: "T1",
  space: "native",
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
});
const subject = (id: string, scans: Scan[]): Subject => ({
  id,
  group: "G",
  cohort: "C",
  scans,
  dates: [],
  modalities: ["MRI"],
});

describe("Compare visits example", () => {
  it("names the first catalogue participant with two dated MRIs", () => {
    const subjects = [
      subject("one-date", [mri("a", "2000-01-01"), mri("b", "2000-01-01")]),
      subject("undated", [mri("c", null), mri("d", null)]),
      subject("two-dates", [mri("e", "2000-01-01"), mri("f", "2001-01-01")]),
      subject("also-two", [mri("g", "2000-01-01"), mri("h", "2002-01-01")]),
    ];
    expect(comparisonExample(subjects)?.id).toBe("two-dates");
  });

  it("returns nothing when no participant has two dated MRIs", () => {
    const bold = { ...mri("bold", "2000-01-01"), modality: "fMRI" as const };
    expect(comparisonExample([])).toBeUndefined();
    expect(
      comparisonExample([subject("open-data", [mri("t1", null), bold])]),
    ).toBeUndefined();
  });
});

describe("next bundle line", () => {
  it("lists the plan's dates and never invents one", () => {
    expect(
      bundleLine({ modality: "SPECT", dates: ["2000-01-01", "2001-01-01"], evidence: [] }),
    ).toBe("SPECT · 2000-01-01, 2001-01-01");
    expect(bundleLine({ modality: "PET", dates: [], evidence: [] })).toBe(
      "PET · dates not established in the local tables",
    );
  });
});

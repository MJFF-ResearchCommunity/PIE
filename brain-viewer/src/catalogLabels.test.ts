import { describe, expect, it } from "vitest";
import { exampleShortcuts, subjectLabel } from "./model";
import type { Scan, Subject } from "./types";

const scan = (patch: Partial<Scan>): Scan => ({
  id: "s",
  subject: "1",
  modality: "fMRI",
  date: "2000-01-01",
  visit: "BL",
  description: "BOLD",
  space: "native",
  kind: "timeseries",
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
const subject = (patch: Partial<Subject>): Subject => ({
  id: "1",
  group: "G",
  cohort: "C",
  scans: [],
  dates: [],
  modalities: ["fMRI"],
  ...patch,
});

describe("catalog labels", () => {
  it("names the data collection only when the catalog declares one", () => {
    expect(subjectLabel({ id: "7", collection: "PPMI" })).toBe("PPMI 7");
    expect(subjectLabel({ id: "sub-01" })).toBe("sub-01");
  });

  it("labels each fMRI example with that participant's own cohort", () => {
    const subjects = [
      subject({
        id: "1",
        cohort: "Cohort A",
        collection: "PPMI",
        scans: [
          scan({ id: "short", metadata: { example: true, short_reference: true } }),
          scan({ id: "full", metadata: { example: true } }),
        ],
      }),
      subject({
        id: "2",
        cohort: "Cohort B",
        scans: [
          scan({ id: "only-short", metadata: { example: true, short_reference: true } }),
        ],
      }),
      subject({ id: "3", cohort: "Cohort C", scans: [scan({ id: "plain" })] }),
    ];
    expect(exampleShortcuts(subjects)).toEqual([
      { subjectId: "1", scanId: "full", label: "PPMI 1 · Cohort A" },
      { subjectId: "2", scanId: "only-short", label: "2 · Cohort B" },
    ]);
  });
});

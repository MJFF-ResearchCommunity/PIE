import type {
  BundleRequest,
  Scan,
  Subject,
  Modality,
  Region,
} from "./types";

export function canOverlay(base: Scan, overlay: Scan): boolean {
  return (
    base.id !== overlay.id &&
    base.subject === overlay.subject &&
    base.space === overlay.space &&
    overlay.registration === "verified" &&
    overlay.reference_id === base.id
  );
}

export function nearestScan(
  subject: Subject,
  modality: Modality | null,
  date: string | null,
): Scan | undefined {
  const choices = subject.scans.filter(
    (s) => !modality || s.modality === modality,
  );
  const distance = (s: Scan) =>
    !date
      ? s.date
        ? 0
        : 1
      : !s.date
        ? Infinity
        : Math.abs(Date.parse(s.date) - Date.parse(date));
  return [...choices].sort(
    (a, b) =>
      distance(a) - distance(b) ||
      Number(a.modality === "fMRI" && !!a.metadata?.short_reference) -
        Number(b.modality === "fMRI" && !!b.metadata?.short_reference) ||
      a.id.localeCompare(b.id),
  )[0];
}

/** "<collection> <id>", or the bare ID when the catalog names no collection. */
export function subjectLabel(subject: Pick<Subject, "id" | "collection">) {
  return subject.collection ? `${subject.collection} ${subject.id}` : subject.id;
}

/** One shortcut per participant with a prepared fMRI example, labelled with that
 * participant's own cohort. The full run is preferred over a short reference. */
export function exampleShortcuts(subjects: Subject[]) {
  return subjects.flatMap((p) => {
    const examples = p.scans.filter(
      (s) => s.modality === "fMRI" && !!s.metadata.example,
    );
    const run = examples.find((s) => !s.metadata.short_reference) ?? examples[0];
    return run
      ? [
          {
            subjectId: p.id,
            scanId: run.id,
            label: `${subjectLabel(p)} · ${p.cohort}`,
          },
        ]
      : [];
  });
}

/** One request line from the local plan. Dates come from the tables, never invented. */
export function bundleLine(request: BundleRequest) {
  return `${request.modality} · ${request.dates.length ? request.dates.join(", ") : "dates not established in the local tables"}`;
}

/** The first catalogue participant with two or more dated MRIs, in catalogue order,
 * so the Compare visits hint names a participant this index actually has. */
export function comparisonExample(subjects: Subject[]): Subject | undefined {
  return subjects.find(
    (s) =>
      new Set(
        s.scans
          .filter((scan) => scan.modality === "MRI" && scan.date)
          .map((scan) => scan.date),
      ).size >= 2,
  );
}

/** A data-derived value at display precision (4 significant figures), as a plain number. */
export function roundSignificant(value: number, digits = 4) {
  return Number.isFinite(value) && value !== 0
    ? Number(value.toPrecision(digits))
    : value;
}

/** Initial window from data percentiles, rounded for display. Rendering uses the
 * same values; rounding never collapses or inverts a narrow window. */
export function initialWindow(min: number, max: number): [number, number] {
  const lo = roundSignificant(min),
    hi = roundSignificant(max);
  return hi > lo ? [lo, hi] : [min, max];
}

export function dateLabel(date: string | null, short = false) {
  if (!date) return "Date not established";
  return new Intl.DateTimeFormat("en-US", {
    month: short ? "short" : "long",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  }).format(new Date(`${date}T00:00:00Z`));
}

export function regionName(name: string) {
  return name
    .replace(/^ctx-lh-/, "Left · ")
    .replace(/^ctx-rh-/, "Right · ")
    .replace(/-/g, " ")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace("superiorfrontal", "superior frontal")
    .replace("rostralmiddlefrontal", "rostral middle frontal")
    .replace("caudalmiddlefrontal", "caudal middle frontal")
    .replace("lateraloccipital", "lateral occipital")
    .replace("inferiorparietal", "inferior parietal")
    .replace("superiorparietal", "superior parietal")
    .replace("superiortemporal", "superior temporal")
    .replace("middletemporal", "middle temporal")
    .replace("inferiortemporal", "inferior temporal")
    .replace("posteriorcingulate", "posterior cingulate")
    .replace("caudalanteriorcingulate", "caudal anterior cingulate")
    .replace("rostralanteriorcingulate", "rostral anterior cingulate")
    .replace("medialorbitofrontal", "medial orbitofrontal")
    .replace("lateralorbitofrontal", "lateral orbitofrontal")
    .replace("transversetemporal", "transverse temporal")
    .replace("isthmuscingulate", "isthmus cingulate")
    .replace("parahippocampal", "parahippocampal")
    .replace("parsopercularis", "pars opercularis")
    .replace("parsorbitalis", "pars orbitalis")
    .replace("parstriangularis", "pars triangularis");
}

export function regionContext(region: Region): string {
  const name = region.name.toLowerCase();
  if (name.includes("putamen"))
    return "A component of the dorsal striatum involved in motor circuits. This participant-specific segmentation marks anatomy; image intensity alone does not establish dopaminergic loss.";
  if (name.includes("caudate"))
    return "Part of the dorsal striatum, participating in associative and motor circuits. Inspect corresponding acquisitions and their processing before comparing signals.";
  if (name.includes("pallidum"))
    return "A basal ganglia structure in cortico-basal-ganglia-thalamic circuits. The segmentation does not distinguish internal and external pallidal subdivisions.";
  if (name.includes("hippocampus"))
    return "A medial temporal structure involved in learning and memory. Its boundary is an automated segmentation and should be checked on the anatomical slices.";
  if (name.includes("thalamus"))
    return "A deep gray-matter structure linking cortical and subcortical circuits. This whole-structure label does not resolve individual thalamic nuclei.";
  if (name.includes("precentral"))
    return "The precentral gyrus contains primary motor cortex. Atlas boundaries describe anatomy; they are not a patient-specific functional localization.";
  if (name.includes("vent"))
    return "A cerebrospinal-fluid compartment. Displayed volume is a voxel-count estimate from this segmentation, without partial-volume correction.";
  if (name.includes("white"))
    return "Segmented white matter. Fractional anisotropy describes diffusion anisotropy; tractography streamlines estimate pathways and are not individual axons.";
  return "An anatomical label from the aligned segmentation. Review its boundary in the linked slices. Region labels do not by themselves indicate function, disease, or abnormality.";
}

export function downloadJson(value: unknown, filename: string) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Bounded surface-pick fallback. Distances are in native world mm, not voxels. */
export function nearbyAtlasLabel(
  mm: number[],
  voxel: number[],
  affine: number[][],
  dims: number[],
  valueAt: (x: number, y: number, z: number) => number,
) {
  const spacing = [0, 1, 2].map((c) =>
    Math.hypot(affine[0][c], affine[1][c], affine[2][c]),
  );
  const radius = Math.min(12, Math.ceil(3 / Math.min(...spacing)) + 1);
  let best: { id: number; distance: number } | null = null;
  for (let dx = -radius; dx <= radius; dx++)
    for (let dy = -radius; dy <= radius; dy++)
      for (let dz = -radius; dz <= radius; dz++) {
        const v = [
          Math.round(voxel[0]) + dx,
          Math.round(voxel[1]) + dy,
          Math.round(voxel[2]) + dz,
        ];
        if (v.some((x, i) => x < 0 || x >= dims[i])) continue;
        const world = affine
          .slice(0, 3)
          .map((row) => row[0] * v[0] + row[1] * v[1] + row[2] * v[2] + row[3]);
        const distance = Math.hypot(...world.map((x, i) => x - mm[i]));
        if (distance > 3 || (best && distance >= best.distance)) continue;
        const id = Math.round(valueAt(v[0], v[1], v[2]));
        if (id > 0) best = { id, distance };
      }
  return best;
}

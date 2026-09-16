export type Modality = "CT" | "DTI" | "MRI" | "PET" | "SPECT" | "fMRI";
export const MODALITIES: Modality[] = [
  "MRI",
  "DTI",
  "SPECT",
  "PET",
  "CT",
  "fMRI",
];
export interface Scan {
  id: string;
  subject: string;
  modality: Modality;
  date: string | null;
  visit: string;
  description: string;
  space: string;
  kind: "scalar" | "timeseries" | "tracts";
  units: string;
  reference_id: string | null;
  registration: string;
  provenance: string;
  qc: string;
  tracer: string | null;
  has_atlas: boolean;
  has_anatomy: boolean;
  metadata: Record<string, unknown>;
  atlas_name?: string;
}
export interface Subject {
  id: string;
  group: string;
  cohort: string;
  /** Data source named in the label ("PPMI" for discovered PIE outputs); absent for unnamed imports. */
  collection?: string;
  sex?: string;
  age_at_scan?: string;
  scans: Scan[];
  dates: string[];
  modalities: Modality[];
}
export interface Catalog {
  subjects: Subject[];
  modalities: Modality[];
  warnings: string[];
  scan_count: number;
  source: string;
}
/** One modality to request for a participant, from the local sample plan. */
export interface BundleRequest {
  modality: string;
  dates: string[];
  evidence: string[];
}
export interface BundleEntry {
  subject: string;
  cohort: string;
  local_modalities: string[];
  requests: BundleRequest[];
}
export interface Region {
  id: number;
  name: string;
  color: number[];
  voxels: number;
  volume_mm3: number;
  center_mm: number[];
  focus_mm?: number[];
}
export interface Volume {
  url: string;
  name: string;
  role: "primary" | "atlas" | "anatomy" | "overlay";
  colormap: string;
  cal_min: number;
  cal_max: number;
  opacity: number;
}
export interface Geometry {
  spatial_unit?: string;
  shape: number[];
  spacing: number[];
  orientation: string;
  affine: number[][];
  frames: number;
  frame_step: number | null;
  time_unit: string;
  cal_min: number;
  cal_max: number;
}
export interface Prepared {
  fmri_unavailable?: string;
  fmri?: {
    frames: number;
    tr_seconds: number | null;
    mean_signal: (number | null)[];
    raw_dvars: (number | null)[];
    foreground_voxels: number;
    excluded_nonfinite_voxels: number;
    foreground_threshold: number;
    foreground_definition: string;
    method: string;
    warning: string;
    source_fingerprint: string;
  };
  histogram?: {
    counts: number[];
    edges: number[];
    sample_count: number;
    scope: string;
  } | null;
  fingerprint?: string;
  context?: {
    reference_id: string;
    reference_date: string | null;
    status: "unreviewed";
    processing: string;
    flip_lr: boolean;
    reference_geometry: Geometry;
    source_geometry: Geometry;
    fixed_to_moving_ras: number[][];
    note: string;
  };
  scan: Scan;
  volumes: Volume[];
  meshes: { url: string; name: string }[];
  regions: Region[];
  geometry: Geometry | null;
  extra?: {
    name: string;
    units: string;
    url: string;
    key: string;
    cal_min?: number;
    cal_max?: number;
    colormap?: string;
  }[];
}
export interface Probe {
  timeSeries?: { values: (number | null)[]; voxel: number[]; mm: number[] };
  region_distance_mm?: number;
  mm: number[];
  value: number | null;
  region: Region | null;
}
export type ViewMode = "3d" | "multi" | "axial" | "coronal" | "sagittal";
export interface Structures {
  reference_id: string;
  space: string;
  fingerprint: string;
  note: string;
  meshes: {
    key: string;
    name: string;
    url: string;
    color: number[];
    region_ids: number[];
  }[];
}
export interface Display {
  atlasOutline: boolean;
  hideSignal: boolean;
  structures: boolean;
  structureMri: boolean;
  structureMriOpacity: number;
  structureOpacity: number;
  structureOutlines: boolean;
  leftOpacity: number;
  rightOpacity: number;
  contextOpacity: number;
  selectedStructures: string[];
  mode: ViewMode;
  atlas: boolean;
  atlasOpacity: number;
  opacity: number;
  overlayOpacity: number;
  anatomyOpacity: number;
  window: [number, number];
  colormap: string;
  clip: number;
  spectCutoff: number;
  clipAxis: number;
  frame: number;
  illumination: boolean;
  crosshair: boolean;
  metric: string;
}

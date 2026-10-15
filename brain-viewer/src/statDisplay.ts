/** Diverging display for a regional statistic map (pie.imaging.regionmap): symmetric limits from the sidecar's vmax,
 * so negative effects (e.g. PD < HC) stay visible in the cool scale. */
export interface StatisticDisplay {
  colormap: "warm";
  colormapNegative: "cool";
  cal_min: number;
  cal_max: number;
  cal_minNeg: number;
  cal_maxNeg: number;
  empty: boolean;
}

export function statisticDisplay(meta: { vmax?: unknown; threshold_value?: unknown }): StatisticDisplay {
  const vmax = Math.abs(Number(meta.vmax)) || 0;
  const t = Number(meta.threshold_value);
  const lo = Number.isFinite(t) && t > 0 && t < vmax ? t : vmax / 1000;
  return { colormap: "warm", colormapNegative: "cool", cal_min: lo, cal_max: vmax, cal_minNeg: -vmax, cal_maxNeg: -lo, empty: vmax === 0 };
}

/** The overlay descriptor fields for a statistic map (NiiVue negative colormap + colour bar). */
export function statisticVolume(meta: { vmax?: unknown; threshold_value?: unknown }) {
  const { empty: _empty, ...d } = statisticDisplay(meta);
  return { ...d, colorbarVisible: true };
}

type Sidecar = { statistic?: string; vmax?: number; threshold?: { p_column: string; alpha: number } | null };
interface StatisticScan {
  id: string;
  description: string;
  units: string;
  metadata: Record<string, unknown>;
}

export function statisticRule(meta: Record<string, unknown>) {
  const t = (meta as Sidecar).threshold;
  return t ? `${t.p_column} < ${t.alpha}` : "all matched regions";
}

/** Menu label: statistic, threshold rule and map id, so two maps of one statistic stay distinguishable. */
export function statisticLabel(scan: StatisticScan) {
  return `${(scan.metadata as Sidecar).statistic ?? scan.description} · ${statisticRule(scan.metadata)} · ${scan.id}`;
}

/** What a statistic overlay shows, for the inspector note and the export header (it has no acquisition date). */
export function statisticNote(scan: StatisticScan) {
  const m = scan.metadata as Sidecar;
  return `${m.statistic ?? scan.description} · ${statisticRule(scan.metadata)} · ±${Number(m.vmax ?? 0).toPrecision(2)} ${scan.units} · painted on this scan's own label grid; regional, not voxelwise`;
}

/** The command that makes a statistic map for the open scan: participant mode (its row of a per-scan table, e.g.
 * normative z-scores) when PIE found the T1, otherwise a per-region table painted on the scan's own segmentation. */
export function statisticRecipe(scan: { id: string; metadata: Record<string, unknown> }) {
  const run = "venv_imaging/bin/python -m pie.imaging.regionmap";
  const imageId = scan.metadata.image_id;
  return imageId
    ? `${run} --table zscores.csv --image-id ${imageId} --measure vol --units z --out maps/${scan.id}_z --manifest viewer.json`
    : `${run} --table results.csv --value d --fastsurfer-dir <this scan's FastSurfer folder> --reference-id ${scan.id} --out maps/${scan.id}_d --manifest <this viewer's manifest>`;
}

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

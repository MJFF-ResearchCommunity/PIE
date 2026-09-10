import type { Geometry } from "./types";

export function frameSeconds(
  geometry: Geometry | null,
  frame: number,
): number | null {
  if (
    !geometry ||
    frame < 0 ||
    !Number.isInteger(frame) ||
    frame >= geometry.frames
  )
    return null;
  const factor = (
    { sec: 1, msec: 0.001, usec: 0.000001 } as Record<string, number>
  )[geometry.time_unit];
  const step = geometry.frame_step;
  return factor && step && Number.isFinite(step) && step > 0
    ? frame * step * factor
    : null;
}

export function voxelTimeSeries(
  voxel: number[],
  dimensions: number[],
  frames: number,
  read: (x: number, y: number, z: number, frame: number) => number,
) {
  const xyz = voxel.slice(0, 3).map(Math.round);
  if (
    xyz.length !== 3 ||
    xyz.some((v, i) => !Number.isFinite(v) || v < 0 || v >= dimensions[i])
  )
    return undefined;
  if (!Number.isInteger(frames) || frames < 2 || frames > 10000)
    return undefined;
  return {
    voxel: xyz,
    values: Array.from({ length: frames }, (_, f) => {
      const v = read(xyz[0], xyz[1], xyz[2], f);
      return Number.isFinite(v) ? v : null;
    }),
  };
}

export function percentFromMean(values: (number | null)[]) {
  const finite = values.filter(
    (v): v is number => v !== null && Number.isFinite(v),
  );
  const mean = finite.reduce((a, b) => a + b, 0) / finite.length;
  if (!Number.isFinite(mean) || mean <= 0) return null;
  return values.map((v) =>
    v === null || !Number.isFinite(v) ? null : (100 * (v - mean)) / mean,
  );
}

export function tracePath(values: (number | null)[], width = 680, height = 94) {
  const finite = values.filter(
    (v): v is number => v !== null && Number.isFinite(v),
  );
  if (!finite.length || values.length < 2)
    return { path: "", min: null, max: null };
  const min = Math.min(...finite),
    max = Math.max(...finite);
  const span = max - min || 1;
  let start = true;
  const path = values
    .map((v, i) => {
      if (v === null || !Number.isFinite(v)) {
        start = true;
        return "";
      }
      const point = `${start ? "M" : "L"}${((i / (values.length - 1)) * width).toFixed(2)},${(max === min ? height / 2 : height - ((v - min) / span) * height).toFixed(2)}`;
      start = false;
      return point;
    })
    .join(" ");
  return { path, min, max };
}

import type { Display, Region, Structures } from "./types";
import type { NVImage } from "@niivue/niivue";

export const STRIATUM = ["11", "12", "50", "51"];
export function structureOpacity(key: string, display: Display) {
  if (!display.structures || display.mode !== "3d") return 0;
  if (key === "left") return display.leftOpacity;
  if (key === "right") return display.rightOpacity;
  if (key === "context") return display.contextOpacity;
  return display.selectedStructures.includes(key)
    ? display.structureOpacity
    : 0;
}
export function availableStriatum(structures: Structures) {
  return STRIATUM.filter((key) => structures.meshes.some((m) => m.key === key));
}

export function anatomyDisplayOpacity(
  display: Display,
  normalOpacity: number,
  structures?: Structures | null,
) {
  // 3D visibility never removes measured MRI from anatomical slice review.
  if (!display.structures || !structures || display.mode !== "3d")
    return normalOpacity;
  return display.structureMri ? display.structureMriOpacity : 0;
}

export function selectedStructureIds(display: Display, structures: Structures) {
  return structures.meshes
    .filter(
      (m) => /^\d+$/.test(m.key) && display.selectedStructures.includes(m.key),
    )
    .flatMap((m) => m.region_ids);
}

export function structureAtlasState(
  display: Display,
  structures?: Structures | null,
) {
  const active = display.structures && !!structures;
  const outlines = active && display.structureOutlines && display.mode !== "3d";
  return {
    outlines: active ? outlines : display.atlasOutline,
    opacity: active
      ? outlines
        ? display.structureOpacity
        : 0
      : display.atlas
        ? display.atlasOpacity
        : 0,
    // null restores the ordinary full atlas; [] intentionally hides all labels.
    ids: active ? selectedStructureIds(display, structures) : null,
  };
}

export function anatomicalLabelLut(
  regions: Region[],
  ids: number[] | null,
  outlinesOnly = false,
) {
  const selected = ids === null ? null : new Set(ids);
  return {
    I: [0, ...regions.map((r) => r.id)],
    R: [0, ...regions.map((r) => r.color[0])],
    G: [0, ...regions.map((r) => r.color[1])],
    B: [0, ...regions.map((r) => r.color[2])],
    A: [
      0,
      ...regions.map((r) =>
        !outlinesOnly && (selected === null || selected.has(r.id)) ? 255 : 0,
      ),
    ],
    labels: ["Background", ...regions.map((r) => r.name)],
  };
}

/** A disposable display buffer. Never edit the source atlas or probe from this copy.
 * NiiVue's outline shader overrides LUT alpha at every nonzero label boundary,
 * so unselected labels must be absent from this display-only buffer.
 */
export function selectedLabelBuffer(
  source: NonNullable<NVImage["img"]>,
  ids: number[],
) {
  const selected = new Set(ids);
  const filtered = source.slice();
  for (let i = 0; i < filtered.length; i++) {
    if (!selected.has(filtered[i])) filtered[i] = 0;
  }
  return filtered;
}

export function structureViewLabel(
  display: Display,
  structures?: Structures | null,
) {
  if (!display.structures || !structures) return null;
  if (display.mode !== "3d")
    return display.structureOutlines
      ? "Measured MRI + selected structure outlines"
      : "Measured slices · structure outlines off";
  return display.structureMri && display.structureMriOpacity > 0
    ? "Measured MRI + participant segmentation boundaries"
    : "Segmentation boundaries only · MRI hidden in 3D";
}

import type { Display, Structures } from "./types";

export const STRIATUM = ["11", "12", "50", "51"];
export function structureOpacity(key: string, display: Display) {
  if (!display.structures || display.mode !== "3d") return 0;
  if (key === "left") return display.leftOpacity;
  if (key === "right") return display.rightOpacity;
  if (key === "context") return display.contextOpacity;
  return display.selectedStructures.includes(key) ? 0.7 : 0;
}
export function availableStriatum(structures: Structures) {
  return STRIATUM.filter((key) => structures.meshes.some((m) => m.key === key));
}

import type { Display, Prepared, Structures } from "./types";
import { frameSeconds } from "./fmriDisplay";
import { structureViewLabel } from "./structureDisplay";

export function captureLines(
  p: Prepared,
  d: Display,
  structures?: Structures | null,
) {
  const boundaries = d.structures && !!structures && d.mode === "3d";
  const structureLabel = structureViewLabel(d, structures);
  const metric = p.extra?.find((e) => e.key === d.metric);
  const bold = p.scan.modality === "fMRI" && (p.geometry?.frames ?? 0) > 1;
  const seconds = frameSeconds(p.geometry, d.frame);
  const warning =
    p.context || p.scan.registration === "unreviewed"
      ? " · ALIGNMENT NOT REVIEWED"
      : "";
  return [
    `PIE ${p.scan.subject} · ${p.scan.id} · ${p.scan.date ?? "undated"}${warning}`,
    p.context
      ? `MRI ${p.context.reference_id} · ${p.context.reference_date ?? "undated"} / SPECT ${p.scan.date ?? "undated"}`
      : p.scan.qc || p.scan.description,
    `${d.mode} · ${structureLabel ?? metric?.name ?? "measured image"} · window ${d.window.map((x) => x.toPrecision(4)).join("–")} ${metric?.units ?? p.scan.units}`,
    p.scan.modality === "SPECT"
      ? `${d.hideSignal ? "SPECT HIDDEN" : d.mode === "3d" ? `${Math.round(d.spectCutoff * 100)}% display cutoff` : "Full-signal slices; extra cutoff paused"} · tracer signal, not cortical activity or a disease boundary`
      : bold
        ? `${metric ? `Temporal summary of all ${p.geometry?.frames} frames` : `BOLD frame ${d.frame + 1}/${p.geometry?.frames} · ${seconds === null ? "timing unknown" : `${seconds.toFixed(2)} s from first frame`}`} · unprocessed EPI, not activation or connectivity`
        : "Research visualization · no quantitative change or diagnosis established",
    structureLabel
      ? `Estimated labels, not validated pial surfaces · ${boundaries ? `see-through 3D · MRI ${d.structureMri ? Math.round(d.structureMriOpacity * 100) : 0}%` : `selected slice outlines ${d.structureOutlines ? "on" : "off"}`} · structure opacity ${Math.round(d.structureOpacity * 100)}% · selected labels ${d.selectedStructures.join(", ")}`
      : `Atlas ${d.atlas ? (d.atlasOutline ? "outlines" : "labels") : "off"} · signal opacity ${Math.round(d.opacity * 100)}% · fingerprint ${p.fingerprint ?? "browser import"}`,
  ];
}
export function captureView(
  source: HTMLCanvasElement,
  p: Prepared,
  d: Display,
  structures?: Structures | null,
) {
  const lines = captureLines(p, d, structures);
  const exported = document.createElement("canvas");
  const header = lines.length * 24 + 24;
  exported.width = source.width;
  exported.height = source.height + header;
  const ctx = exported.getContext("2d")!;
  ctx.fillStyle = "#111416";
  ctx.fillRect(0, 0, exported.width, exported.height);
  ctx.drawImage(source, 0, header);
  ctx.fillStyle = "#e2e6dc";
  ctx.font = `${Math.max(12, Math.min(17, source.width / 65))}px sans-serif`;
  lines.forEach((line, i) =>
    ctx.fillText(line, 16, 26 + i * 24, exported.width - 32),
  );
  const link = document.createElement("a");
  link.download = `PIE-${p.scan.subject}-${p.scan.modality}-${p.scan.date ?? "undated"}${p.context || p.scan.registration === "unreviewed" ? "-UNREVIEWED" : ""}.png`;
  link.href = exported.toDataURL("image/png");
  link.click();
}

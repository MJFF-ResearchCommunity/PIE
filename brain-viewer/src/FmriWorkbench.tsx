import { useState } from "react";
import { Pause, Play, ArrowLeft, ArrowRight, Download } from "lucide-react";
import { downloadJson } from "./model";
import { frameSeconds, percentFromMean, tracePath } from "./fmriDisplay";
import type { Display, Prepared, Probe } from "./types";

interface TransportProps {
  prepared: Prepared;
  frame: number;
  playing: boolean;
  speed: number;
  busy: boolean;
  seek: (frame: number) => void;
  setPlaying: (play: boolean) => void;
  setSpeed: (speed: number) => void;
}
export function FmriTransport({
  prepared,
  frame,
  playing,
  speed,
  busy,
  seek,
  setPlaying,
  setSpeed,
}: TransportProps) {
  const count = prepared.geometry?.frames ?? 1;
  const seconds = frameSeconds(prepared.geometry, frame);
  return (
    <div className="fmri-transport" aria-label="Within-scan BOLD playback">
      <button
        className="button"
        disabled={busy}
        aria-label={playing ? "Pause BOLD frames" : "Play BOLD frames"}
        onClick={() => {
          if (!playing && frame === count - 1) seek(0);
          setPlaying(!playing);
        }}
      >
        {playing ? <Pause size={15} /> : <Play size={15} />}{" "}
        {playing ? "Pause" : "Play"}
      </button>
      <button
        className="icon-button"
        aria-label="Previous BOLD frame"
        disabled={busy || frame === 0}
        onClick={() => seek(frame - 1)}
      >
        <ArrowLeft size={15} />
      </button>
      <label className="fmri-frame-range">
        Within this scan{" "}
        <output>
          Frame {frame + 1} / {count}
          {seconds !== null
            ? ` · ${seconds.toFixed(1)} s`
            : " · timing unknown"}
        </output>
        <input
          type="range"
          aria-label="BOLD image frame"
          min={0}
          max={count - 1}
          step={1}
          value={frame}
          disabled={busy}
          onChange={(e) => seek(Number(e.target.value))}
        />
      </label>
      <button
        className="icon-button"
        aria-label="Next BOLD frame"
        disabled={busy || frame === count - 1}
        onClick={() => seek(frame + 1)}
      >
        <ArrowRight size={15} />
      </button>
      <label className="fmri-speed">
        Playback
        <select
          aria-label="BOLD display speed"
          value={speed}
          onChange={(e) => setSpeed(Number(e.target.value))}
        >
          <option value={1}>1 frame/s</option>
          <option value={2}>2 frames/s</option>
          <option value={4}>4 frames/s</option>
        </select>
      </label>
    </div>
  );
}

function Trace({
  label,
  values,
  frame,
  seek,
  units,
  duration,
  interactive,
}: {
  label: string;
  values: (number | null)[];
  frame: number;
  seek: (f: number) => void;
  units: string;
  duration: number | null;
  interactive: boolean;
}) {
  const { path, min, max } = tracePath(values);
  const current = values[frame];
  return (
    <figure className="fmri-trace">
      <figcaption>
        <strong>{label}</strong>
        <span>
          {current == null ? "—" : current.toPrecision(4)} {units}
        </span>
      </figcaption>
      <div className="trace-plot">
        <div className="trace-y">
          <span>{max?.toPrecision(3) ?? "—"}</span>
          <span>{min?.toPrecision(3) ?? "—"}</span>
        </div>
        <svg
          viewBox="0 0 680 94"
          preserveAspectRatio="none"
          role="img"
          aria-label={`${label}. Range ${min ?? "unavailable"} to ${max ?? "unavailable"} ${units}. ${interactive ? "Click to select a frame; the BOLD image frame slider provides keyboard access." : "All acquired frames."}`}
          onClick={(e) => {
            if (!interactive) return;
            const rect = e.currentTarget.getBoundingClientRect();
            seek(
              Math.max(
                0,
                Math.min(
                  values.length - 1,
                  Math.round(
                    ((e.clientX - rect.left) / rect.width) *
                      (values.length - 1),
                  ),
                ),
              ),
            );
          }}
        >
          <path
            d={path}
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            vectorEffect="non-scaling-stroke"
          />
          {interactive && (
            <line
              x1={(frame / Math.max(1, values.length - 1)) * 680}
              x2={(frame / Math.max(1, values.length - 1)) * 680}
              y1={0}
              y2={94}
              stroke="currentColor"
              strokeDasharray="4 3"
              opacity={0.6}
            />
          )}
        </svg>
      </div>
      <div className="trace-x">
        <span>{duration !== null ? "0 s" : "Frame 1"}</span>
        <span>
          {duration !== null
            ? `${duration.toFixed(1)} s from first frame`
            : `Frame ${values.length}`}
        </span>
      </div>
    </figure>
  );
}

export default function FmriWorkbench(
  props: TransportProps & {
    display: Display;
    probe: Probe | null;
    updateMetric: (metric: string) => void;
  },
) {
  const { prepared: p, display, probe, updateMetric, ...transport } = props;
  const [percent, setPercent] = useState(false);
  const raw = display.metric === "bold";
  const summary = p.fmri;
  const series = probe?.timeSeries;
  const relative = series ? percentFromMean(series.values) : null;
  const duration = frameSeconds(p.geometry, (p.geometry?.frames ?? 1) - 1);
  const tr = frameSeconds(p.geometry, 1);
  const sidecar = (p.scan.metadata.sidecar ?? {}) as Record<string, unknown>;
  const candidates = [
    { key: "bold", name: "Raw BOLD frames" },
    ...(p.extra ?? []),
  ];
  return (
    <section
      className="fmri-workbench"
      aria-label="BOLD time-series inspection"
    >
      <header>
        <div>
          <span className="eyebrow">WITHIN-SCAN fMRI</span>
          <h2>
            {p.scan.metadata.short_reference
              ? "Short EPI reference candidate"
              : "Resting-state BOLD inspection"}
          </h2>
        </div>
        <span>
          {p.geometry?.frames} frames · TR{" "}
          {tr !== null ? `${tr.toFixed(2)} s` : "unknown"} · PE{" "}
          {String(sidecar.PhaseEncodingDirection ?? "unknown")}
        </span>
      </header>
      <div className="preset-row" role="group" aria-label="fMRI representation">
        {candidates.map((c) => (
          <button
            key={c.key}
            aria-pressed={display.metric === c.key}
            disabled={transport.busy}
            onClick={() => updateMetric(c.key)}
          >
            {c.name}
          </button>
        ))}
      </div>
      {raw ? (
        <FmriTransport {...transport} prepared={p} frame={display.frame} />
      ) : (
        <p className="empty-note">
          Summary across all {p.geometry?.frames} frames. Return to Raw BOLD
          frames to play or scrub this acquisition.
        </p>
      )}
      <p className="fmri-warning">
        {p.scan.metadata.short_reference
          ? "Only a short opposite-encoding series: potential distortion-correction reference, not a full resting-state run. "
          : ""}
        Native EPI; no added motion correction, distortion correction or
        denoising. These are signal and variability views—not activation or
        connectivity maps. Within-scan seconds are separate from visit dates.
      </p>
      {raw && (
        <section
          className="voxel-trace-section"
          aria-label="Selected voxel BOLD signal"
        >
          <div className="trace-heading">
            <h3>Selected voxel</h3>
            <label>
              <input
                type="checkbox"
                checked={percent && !!relative}
                disabled={!relative}
                onChange={(e) => setPercent(e.target.checked)}
              />{" "}
              % from this voxel’s temporal mean
            </label>
          </div>
          {series ? (
            <>
              <p className="empty-note">
                Sampled voxel center · RAS{" "}
                {series.mm.map((n) => n.toFixed(1)).join(" / ")} mm · no
                anatomical label established. A fixed scanner voxel may sample
                different tissue if the participant moves.
              </p>
              <Trace
                label="Voxel signal"
                values={percent && relative ? relative : series.values}
                frame={display.frame}
                seek={transport.seek}
                units={percent && relative ? "%" : "a.u."}
                duration={duration}
                interactive={!transport.busy}
              />
              <button
                className="text-link"
                onClick={() =>
                  downloadJson(
                    {
                      scan_id: p.scan.id,
                      date: p.scan.date,
                      fingerprint: p.fingerprint,
                      ras_mm: series.mm,
                      selected_location_mm: probe!.mm,
                      ras_voxel: series.voxel,
                      tr_seconds: tr,
                      frames: series.values.map((value, i) => ({
                        frame: i + 1,
                        seconds: frameSeconds(p.geometry, i),
                        signal: value,
                        percent_from_temporal_mean: relative?.[i] ?? null,
                      })),
                      note: "ras_mm is the center of the exact sampled voxel; selected_location_mm is the potentially fractional crosshair. Unprocessed fixed-voxel signal, not activation or motion-corrected tissue tracking.",
                    },
                    `PIE-${p.scan.subject}-${p.scan.id}-voxel-signal.json`,
                  )
                }
              >
                <Download size={13} /> Export voxel signal
              </button>
            </>
          ) : (
            <p className="empty-note">
              Click a brain voxel in an axial, coronal or sagittal slice to plot
              its signal through the run. Use Four-view to inspect all planes
              together.
            </p>
          )}
        </section>
      )}
      {summary ? (
        <details className="fmri-diagnostics">
          <summary>Run diagnostics · foreground signal &amp; raw DVARS</summary>
          <Trace
            label="Foreground mean signal"
            values={summary.mean_signal}
            frame={display.frame}
            seek={transport.seek}
            units="a.u."
            duration={duration}
            interactive={raw && !transport.busy}
          />
          <Trace
            label="Raw DVARS · successive-frame differences"
            values={summary.raw_dvars}
            frame={display.frame}
            seek={transport.seek}
            units="a.u."
            duration={duration}
            interactive={raw && !transport.busy}
          />
          <p>
            {summary.foreground_voxels.toLocaleString()} foreground voxels.{" "}
            {summary.foreground_definition}
          </p>
          <p>{summary.method}</p>
          <p>
            {summary.warning}{" "}
            {summary.excluded_nonfinite_voxels > 0
              ? `${summary.excluded_nonfinite_voxels} voxels with nonfinite samples excluded.`
              : ""}
          </p>
          <button
            className="text-link"
            onClick={() =>
              downloadJson(
                { scan: p.scan, geometry: p.geometry, summary },
                `PIE-${p.scan.id}-descriptive-fmri.json`,
              )
            }
          >
            Export descriptive diagnostics
          </button>
        </details>
      ) : (
        <p className="empty-note">
          {p.fmri_unavailable ??
            "Descriptive run summaries are available for indexed local BOLD examples; this import has frame inspection only."}
        </p>
      )}
    </section>
  );
}

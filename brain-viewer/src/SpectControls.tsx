import { cmapper } from "@niivue/niivue";
import type { Display, Prepared } from "./types";

export function SpectPresets({
  prepared,
  update,
  display,
}: {
  prepared: Prepared;
  display: Display;
  update: (patch: Partial<Display>) => void;
}) {
  const preset = (mode: "context" | "signal" | "review") =>
    update({
      mode: mode === "review" ? "multi" : "3d",
      structures: false,
      hideSignal: false,
      anatomyOpacity: mode === "signal" ? 0 : mode === "review" ? 1 : 0.8,
      opacity: mode === "review" ? 0.45 : 0.7,
      spectCutoff: prepared.context ? 0.7 : 0.5,
      atlas: false,
      atlasOutline: true,
      crosshair: mode === "review",
      clip: 100,
    });
  return (
    <section className="control-section spect-presets">
      <div className="section-label">
        <h3>SPECT display</h3>
      </div>
      <div
        className="preset-row"
        role="group"
        aria-label="SPECT display presets"
      >
        <button disabled={!prepared.context} onClick={() => preset("context")}>
          Anatomy context
        </button>
        <button onClick={() => preset("signal")}>Signal only</button>
        <button onClick={() => preset("review")}>Alignment review</button>
      </div>
      {prepared.context && (
        <button
          className="button"
          aria-pressed={display.hideSignal}
          onClick={() => update({ hideSignal: !display.hideSignal })}
        >
          {display.hideSignal
            ? "Show SPECT overlay"
            : "Hide SPECT · compare MRI"}
        </button>
      )}
      <p className="empty-note">
        Presets change appearance only.{" "}
        {display.hideSignal
          ? "SPECT is temporarily hidden."
          : "Color shows tracer signal, not a disease boundary."}
      </p>
    </section>
  );
}

export function SpectHistogram({
  prepared,
  display,
}: {
  prepared: Prepared;
  display: Display;
}) {
  const h = prepared.histogram;
  if (!h || h.counts.length < 2) return null;
  const lo = h.edges[0],
    hi = h.edges[h.edges.length - 1];
  const cutoff =
    display.window[0] +
    display.spectCutoff * (display.window[1] - display.window[0]);
  const active = display.mode === "3d" && !display.hideSignal;
  const lut = cmapper.colormap(display.colormap);
  const max = Math.max(...h.counts, 1);
  return (
    <figure className="signal-histogram">
      <figcaption>
        Signal distribution{" "}
        <span>{h.sample_count.toLocaleString()} samples</span>
      </figcaption>
      <svg
        viewBox="0 0 256 72"
        role="img"
        aria-label={`Sampled signal histogram. ${display.hideSignal ? "SPECT is hidden; distribution retained for reference." : active ? `3D display cutoff ${cutoff.toPrecision(3)}; shaded bars are below the cutoff.` : "Additional 3D cutoff is not applied to slices."}`}
      >
        {h.counts.map((n, i) => {
          const v = (h.edges[i] + h.edges[i + 1]) / 2;
          const t = Math.max(
            0,
            Math.min(
              255,
              Math.round(
                ((v - display.window[0]) /
                  (display.window[1] - display.window[0])) *
                  255,
              ),
            ),
          );
          const height = (n / max) * 66;
          return (
            <rect
              key={i}
              x={i * 4}
              y={70 - height}
              width={3.5}
              height={height}
              fill={`rgb(${lut[t * 4]},${lut[t * 4 + 1]},${lut[t * 4 + 2]})`}
              opacity={active && v < cutoff ? 0.22 : 1}
            />
          );
        })}
        {active && cutoff >= lo && cutoff <= hi && (
          <line
            x1={((cutoff - lo) / (hi - lo)) * 256}
            x2={((cutoff - lo) / (hi - lo)) * 256}
            y1={0}
            y2={72}
            stroke="#e3e8dd"
            strokeDasharray="3 2"
          />
        )}
      </svg>
      <div className="scale-values">
        <span>{lo.toPrecision(3)}</span>
        <span>{hi.toPrecision(3)}</span>
      </div>
      <p>
        {h.scope}.{" "}
        {display.hideSignal
          ? "SPECT is hidden; distribution retained for reference."
          : active
            ? "Dimmed bars fall below the display cutoff."
            : "Full-signal slices: extra cutoff paused."}{" "}
        Not a regional binding-ratio measurement.
      </p>
    </figure>
  );
}

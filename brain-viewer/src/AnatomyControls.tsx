import { useEffect, useState } from "react";
import { availableStriatum } from "./structureDisplay";
import type { Display, Prepared, Region, Structures } from "./types";

export default function AnatomyControls({
  prepared,
  structures,
  onLoad,
  display,
  update,
  focus,
}: {
  prepared: Prepared;
  structures: Structures | null;
  onLoad: (s: Structures) => void;
  display: Display;
  update: (d: Partial<Display>) => void;
  focus: (r: Region) => void;
}) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [request, setRequest] = useState(false);
  const reference =
    prepared.context?.reference_id ??
    (prepared.scan.modality === "MRI" ? prepared.scan.id : null);
  useEffect(() => {
    if (!request || !reference) return;
    const controller = new AbortController();
    setLoading(true);
    setError("");
    fetch(`/api/scans/${encodeURIComponent(reference)}/structures`, {
      signal: controller.signal,
    })
      .then(async (r) => {
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail);
        return data as Structures;
      })
      .then((s) => {
        if (!controller.signal.aborted) {
          onLoad(s);
          update({
            structures: true,
            selectedStructures: availableStriatum(s),
          });
        }
      })
      .catch((e) => {
        if (e.name !== "AbortError") setError(String(e.message));
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setLoading(false);
          setRequest(false);
        }
      });
    return () => controller.abort();
    // Load only by explicit request; parent callbacks update on render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [request, reference]);
  if (
    !reference ||
    !prepared.regions.length ||
    prepared.scan.atlas_name !== "DKT + aseg"
  )
    return null;
  const striatum = () => {
    if (!structures) return;
    update({
      structures: true,
      mode: "3d",
      leftOpacity: 0.08,
      rightOpacity: 0.08,
      contextOpacity: 0.08,
      selectedStructures: availableStriatum(structures),
      crosshair: true,
      clip: 100,
    });
    const region =
      prepared.regions.find((r) => r.id === 11) ??
      prepared.regions.find((r) => r.id === 50);
    if (region) focus(region);
  };
  return (
    <section className="control-section anatomy-explorer">
      <div className="section-label">
        <h3>Inside the brain</h3>
      </div>
      {!structures ? (
        <>
          <p className="empty-note">
            Explore boundaries from this participant’s segmentation.
          </p>
          <button
            className="button"
            disabled={loading}
            onClick={() => setRequest(true)}
          >
            {loading ? "Preparing structures…" : "Load anatomical structures"}
          </button>
        </>
      ) : (
        <>
          <div
            className="preset-row"
            role="group"
            aria-label="Anatomical representation"
          >
            <button
              aria-pressed={!display.structures}
              onClick={() => update({ structures: false })}
            >
              Measured MRI
            </button>
            <button
              aria-pressed={display.structures}
              onClick={() => update({ structures: true, mode: "3d" })}
            >
              Segmentation boundaries
            </button>
          </div>
          <button className="button" onClick={striatum}>
            Focus on striatum
          </button>
          <p className="empty-note">
            Boundary mode · 3D only. Structure colors are anatomical labels, not
            tracer intensity. Slices retain the measured images.
          </p>
          {(
            [
              ["leftOpacity", "Left hemisphere shell"],
              ["rightOpacity", "Right hemisphere shell"],
              ["contextOpacity", "Cerebellum / brainstem"],
            ] as const
          ).map(([key, label]) => (
            <label className="range-field" key={key}>
              <span>
                {label}
                <output>{Math.round(display[key] * 100)}%</output>
              </span>
              <input
                type="range"
                aria-label={label}
                min={0}
                max={1}
                step={0.01}
                value={display[key]}
                disabled={!display.structures}
                onChange={(e) => update({ [key]: Number(e.target.value) })}
              />
            </label>
          ))}
          <details>
            <summary>
              Deep structures · {display.selectedStructures.length} selected
            </summary>
            <div className="structure-list">
              {structures.meshes
                .filter((m) => /^\d+$/.test(m.key))
                .map((m) => (
                  <div key={m.key}>
                    <label>
                      <input
                        type="checkbox"
                        checked={display.selectedStructures.includes(m.key)}
                        onChange={(e) =>
                          update({
                            selectedStructures: e.target.checked
                              ? [...display.selectedStructures, m.key]
                              : display.selectedStructures.filter(
                                  (k) => k !== m.key,
                                ),
                          })
                        }
                      />
                      <i style={{ background: `rgb(${m.color.join(",")})` }} />
                      {m.name}
                    </label>
                    <button
                      aria-label={`Locate ${m.name}`}
                      onClick={() => {
                        const r = prepared.regions.find((r) =>
                          m.region_ids.includes(r.id),
                        );
                        if (r) focus(r);
                      }}
                    >
                      Locate
                    </button>
                  </div>
                ))}
            </div>
          </details>
          <small>{structures.note}</small>
        </>
      )}
      {error && <p role="alert">{error}</p>}
    </section>
  );
}

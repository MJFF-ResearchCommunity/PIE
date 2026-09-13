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
            structureMri: true,
            structureMriOpacity: 0.35,
            structureOutlines: true,
            leftOpacity: 0,
            rightOpacity: 0,
            contextOpacity: 0,
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
      leftOpacity: 0,
      rightOpacity: 0,
      contextOpacity: 0,
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
            Keep the measured MRI visible and add this participant’s estimated
            structure boundaries. No generic brain model is used.
          </p>
          <button
            className="button"
            disabled={loading}
            onClick={() => setRequest(true)}
          >
            {loading ? "Preparing structures…" : "Show MRI + structures"}
          </button>
        </>
      ) : (
        <>
          <button
            className={`button structure-master ${display.structures ? "enabled" : ""}`}
            aria-pressed={display.structures}
            onClick={() => update({ structures: !display.structures })}
          >
            {display.structures ? "Turn off structures" : "Turn on structures"}
          </button>
          <p className="structure-status" role="status">
            {display.structures
              ? "On · participant-specific estimated anatomy"
              : "Off · original image display restored"}
          </p>
          <label className="toggle-row">
            <span>Show MRI in 3D</span>
            <input
              type="checkbox"
              role="switch"
              checked={display.structureMri}
              disabled={!display.structures}
              onChange={(e) =>
                update({
                  structureMri: e.target.checked,
                  structureMriOpacity: display.structureMriOpacity || 0.35,
                })
              }
            />
          </label>
          <label className="range-field">
            <span>
              MRI visibility in 3D
              <output>
                {display.structureMri
                  ? Math.round(display.structureMriOpacity * 100)
                  : 0}
                %
              </output>
            </span>
            <input
              type="range"
              aria-label="MRI visibility in 3D"
              min={0}
              max={1}
              step={0.01}
              value={display.structureMriOpacity}
              disabled={
                !display.structures ||
                !display.structureMri ||
                display.mode !== "3d"
              }
              onChange={(e) =>
                update({ structureMriOpacity: Number(e.target.value) })
              }
            />
          </label>
          <label className="range-field">
            <span>
              Selected structure opacity
              <output>{Math.round(display.structureOpacity * 100)}%</output>
            </span>
            <input
              type="range"
              aria-label="Selected structure opacity"
              min={0}
              max={1}
              step={0.01}
              value={display.structureOpacity}
              disabled={!display.structures}
              onChange={(e) =>
                update({ structureOpacity: Number(e.target.value) })
              }
            />
          </label>
          <label className="toggle-row">
            <span>Selected outlines on MRI slices</span>
            <input
              type="checkbox"
              role="switch"
              checked={display.structureOutlines}
              disabled={!display.structures}
              onChange={(e) => update({ structureOutlines: e.target.checked })}
            />
          </label>
          <button
            className="button"
            disabled={!display.structures}
            onClick={() =>
              update({
                mode: "multi",
                structureOutlines: true,
                crosshair: true,
              })
            }
          >
            Review boundaries on MRI slices
          </button>
          <button
            className="button"
            disabled={!display.structures}
            onClick={striatum}
          >
            Focus on striatum
          </button>
          <p className="empty-note">
            Gray: measured MRI. Color: estimated anatomical boundaries, not
            activity. In 3D, boundaries are visible through the MRI; use slices
            to check their actual location. Hiding MRI affects 3D only.
          </p>
          <details className="structure-shells">
            <summary>Outer anatomy shells · optional</summary>
            <p className="empty-note">
              Simplified segmentation shells, separate from the measured MRI.
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
                  disabled={!display.structures || display.mode !== "3d"}
                  onChange={(e) => update({ [key]: Number(e.target.value) })}
                />
              </label>
            ))}
          </details>
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
                        disabled={!display.structures}
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
                      disabled={!display.structures}
                      onClick={() => {
                        const r = prepared.regions.find((r) =>
                          m.region_ids.includes(r.id),
                        );
                        if (r) {
                          update({
                            crosshair: true,
                            selectedStructures:
                              display.selectedStructures.includes(m.key)
                                ? display.selectedStructures
                                : [...display.selectedStructures, m.key],
                          });
                          focus(r);
                        }
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

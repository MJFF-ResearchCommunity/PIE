import { useEffect, useRef, useState } from "react";
import BrainCanvas from "./BrainCanvas";
import type { ViewerApi } from "./BrainCanvas";
import { dateLabel, downloadJson, initialWindow, subjectLabel } from "./model";
import type { Display, Prepared, Subject, ViewMode } from "./types";

interface Pair {
  baseline: Prepared;
  followup: Prepared;
  registration?: Record<string, unknown>;
  fingerprint?: string;
}
async function fetchJson(url: string, signal: AbortSignal) {
  const r = await fetch(url, { signal });
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail ?? `Request failed (${r.status})`);
  return data;
}
export default function ComparisonWorkspace({
  subject,
  example,
  defaults,
}: {
  subject?: Subject;
  /** A participant this local index has two dated MRIs for, if there is one. */
  example?: Subject;
  defaults: Display;
}) {
  const scans = (subject?.scans ?? [])
    .filter((s) => s.modality === "MRI" && s.date)
    .sort((a, b) => a.date!.localeCompare(b.date!));
  const [baseline, setBaseline] = useState(scans[0]?.id ?? "");
  const [followup, setFollowup] = useState(scans.at(-1)?.id ?? "");
  const [aligned, setAligned] = useState(false);
  const [pair, setPair] = useState<Pair | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [paneErrors, setPaneErrors] = useState(["", ""]);
  const [paneBusy, setPaneBusy] = useState([false, false]);
  const [retry, setRetry] = useState([0, 0]);
  const [mode, setMode] = useState<ViewMode>("axial");
  const [linked, setLinked] = useState(true);
  const [windows, setWindows] = useState<[number, number][]>([
    [0, 200],
    [0, 200],
  ]);
  const apis = useRef<(ViewerApi | null)[]>([null, null]);
  const active = useRef(0);
  const valid =
    baseline !== followup &&
    !!scans.find((s) => s.id === baseline)?.date &&
    scans.find((s) => s.id === baseline)!.date! <
      (scans.find((s) => s.id === followup)?.date ?? "");
  useEffect(() => {
    if (!valid) {
      setPair(null);
      return;
    }
    const controller = new AbortController();
    setBusy(true);
    setError("");
    setPair(null);
    setPaneErrors(["", ""]);
    const task = aligned
      ? fetchJson(
          `/api/comparison/${encodeURIComponent(baseline)}/${encodeURIComponent(followup)}`,
          controller.signal,
        )
      : Promise.all(
          [baseline, followup].map((id) =>
            fetchJson(
              `/api/scans/${encodeURIComponent(id)}`,
              controller.signal,
            ),
          ),
        ).then(([a, b]) => ({ baseline: a, followup: b }));
    task
      .then((p: Pair) => {
        if (controller.signal.aborted) return;
        setPair(p);
        setWindows(
          [p.baseline, p.followup].map((p) => {
            const v = p.volumes.find((v) => v.role === "primary")!;
            return initialWindow(v.cal_min, v.cal_max);
          }),
        );
      })
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setBusy(false);
      });
    return () => controller.abort();
  }, [baseline, followup, aligned, valid]);
  useEffect(() => {
    if (!linked || !pair) return;
    let previous = "";
    const timer = window.setInterval(() => {
      const source = apis.current[active.current],
        target = apis.current[1 - active.current];
      if (!source || !target) return;
      const pose = source.readView(),
        key = JSON.stringify([active.current, pose]);
      if (key === previous) return;
      previous = key;
      target.applyView(pose, !!pair.registration);
    }, 120);
    return () => clearInterval(timer);
  }, [linked, pair]);
  const caption = (id: string) => {
    const s = scans.find((s) => s.id === id);
    return s
      ? `${dateLabel(s.date, true)} · ${s.visit} · ${s.id}`
      : "Choose scan";
  };
  return (
    <main className="comparison-workspace">
      <div className="comparison-heading">
        <div>
          <span className="eyebrow">LONGITUDINAL INSPECTION</span>
          <h2>{subject ? subjectLabel(subject) : "—"} · MRI comparison</h2>
        </div>
        {pair && (
          <button
            className="button"
            onClick={() =>
              downloadJson(
                {
                  version: 1,
                  baseline: pair.baseline.scan,
                  followup: pair.followup.scan,
                  registration: pair.registration ?? null,
                  fingerprint: pair.fingerprint,
                  mode,
                  windows,
                  views: apis.current.map((a) => a?.readView() ?? null),
                  quantitative_change: "not established",
                },
                `PIE-${subject?.id}-comparison.json`,
              )
            }
          >
            Export comparison provenance
          </button>
        )}
      </div>
      {new Set(scans.map((s) => s.date)).size < 2 ? (
        <div className="comparison-empty">
          <h3>A follow-up MRI is needed for this participant.</h3>
          <p>
            The same modality must have two distinct acquisition dates.
            Different modalities on different dates are not longitudinal
            follow-ups.
          </p>
          {example ? (
            <p>
              The local participant <strong>{subjectLabel(example)}</strong> has
              MRIs on two dates. Find them in the participant browser.
            </p>
          ) : (
            <p>
              No participant in this local index has two dated MRIs yet. Add a
              second dated MRI through a viewer manifest, for example with
              scripts/prepare_viewer_followup.py.
            </p>
          )}
        </div>
      ) : (
        <>
          <div className="comparison-selects">
            {(
              [
                ["Baseline MRI", baseline, setBaseline],
                ["Follow-up MRI", followup, setFollowup],
              ] as const
            ).map(([label, id, set]) => (
              <label key={label}>
                {label}
                <select
                  value={id}
                  onChange={(e) => {
                    set(e.target.value);
                    setAligned(false);
                  }}
                >
                  {scans.map((s) => (
                    <option key={s.id} value={s.id}>
                      {caption(s.id)}
                    </option>
                  ))}
                </select>
              </label>
            ))}
          </div>
          {!valid && (
            <p role="alert">
              Choose an earlier baseline and a later follow-up.
            </p>
          )}
          <div className="comparison-options">
            <div
              className="preset-row"
              role="group"
              aria-label="Comparison geometry"
            >
              <button aria-pressed={!aligned} onClick={() => setAligned(false)}>
                Native spaces
              </button>
              <button
                aria-pressed={aligned}
                disabled={!valid || busy}
                onClick={() => setAligned(true)}
              >
                Prepare rigid alignment preview
              </button>
            </div>
            <label>
              <input
                type="checkbox"
                checked={linked}
                onChange={(e) => setLinked(e.target.checked)}
              />{" "}
              Link {aligned ? "camera and slice position" : "3D camera only"}
            </label>
            <select
              aria-label="Comparison view layout"
              value={mode}
              onChange={(e) => setMode(e.target.value as ViewMode)}
            >
              <option value="axial">Axial</option>
              <option value="coronal">Coronal</option>
              <option value="sagittal">Sagittal</option>
              <option value="multi">Four-view</option>
              <option value="3d">3D native head</option>
            </select>
          </div>
          <p className="comparison-warning">
            {aligned
              ? "AUTOMATIC RIGID ALIGNMENT — NOT REVIEWED. Follow-up is resampled into baseline geometry for visual inspection; linked coordinates do not establish anatomical correspondence."
              : "Native spaces: slices remain independent because scanner coordinates do not establish anatomical correspondence."}{" "}
            Intensity windows are independent. No disease progression,
            difference map, or quantitative change is inferred.
          </p>
          {busy && (
            <p role="status">
              {aligned
                ? "Estimating rigid alignment locally…"
                : "Loading comparison scans…"}
            </p>
          )}
          {error && <p role="alert">{error}</p>}
          {pair && (
            <div className="comparison-panes">
              {[pair.baseline, pair.followup].map((p, i) => (
                <section
                  key={p.fingerprint}
                  onPointerDownCapture={() => {
                    active.current = i;
                  }}
                  onWheelCapture={() => {
                    active.current = i;
                  }}
                  onFocusCapture={() => {
                    active.current = i;
                  }}
                >
                  <header>
                    <strong>{i ? "Follow-up" : "Baseline"}</strong>
                    <span>
                      {dateLabel(p.scan.date)} · {p.scan.visit}
                    </span>
                    <button
                      aria-label={`Save ${i ? "follow-up" : "baseline"} image`}
                      onClick={() => apis.current[i]?.capture()}
                    >
                      Save PNG
                    </button>
                  </header>
                  <div className="comparison-canvas">
                    {!paneErrors[i] && (
                      <BrainCanvas
                        key={`${p.fingerprint}-${retry[i]}`}
                        prepared={p}
                        overlay={null}
                        display={{
                          ...defaults,
                          mode,
                          crosshair: true,
                          opacity: 1,
                          window: windows[i],
                        }}
                        onModeChange={setMode}
                        onReady={(api) => {
                          apis.current[i] = api;
                        }}
                        onProbe={() => {}}
                        onBusy={(b) =>
                          setPaneBusy((a) => a.map((v, n) => (n === i ? b : v)))
                        }
                        onError={(e) =>
                          setPaneErrors((a) =>
                            a.map((v, n) => (n === i ? e : v)),
                          )
                        }
                        onContextLost={() =>
                          setPaneErrors((a) =>
                            a.map((v, n) =>
                              n === i
                                ? "Graphics context lost. Retry this pane."
                                : v,
                            ),
                          )
                        }
                        onLightingUnavailable={() => {}}
                      />
                    )}
                    {paneBusy[i] && !paneErrors[i] && (
                      <span className="comparison-loading" role="status">
                        Loading image…
                      </span>
                    )}
                    {paneErrors[i] && (
                      <div className="comparison-loading" role="alert">
                        {paneErrors[i]}
                        <button
                          className="button"
                          onClick={() => {
                            setPaneErrors((a) =>
                              a.map((v, n) => (n === i ? "" : v)),
                            );
                            setRetry((a) =>
                              a.map((v, n) => (n === i ? v + 1 : v)),
                            );
                          }}
                        >
                          Retry pane
                        </button>
                      </div>
                    )}
                  </div>
                  <div className="comparison-window">
                    {[0, 1].map((j) => (
                      <label key={j}>
                        {j ? "Window max" : "Window min"}
                        <input
                          type="number"
                          step="any"
                          aria-label={`${i ? "Follow-up" : "Baseline"} window ${j ? "maximum" : "minimum"}`}
                          value={windows[i][j]}
                          onChange={(e) => {
                            const value = Number(e.target.value),
                              next = [...windows[i]] as [number, number];
                            next[j] = value;
                            if (Number.isFinite(value) && next[0] < next[1])
                              setWindows((w) =>
                                w.map((v, n) => (n === i ? next : v)),
                              );
                          }}
                        />
                      </label>
                    ))}
                  </div>
                  <details>
                    <summary>Acquisition and processing</summary>
                    <p>{p.scan.qc || "Source quality not reviewed"}</p>
                    <p>
                      {p.geometry?.spacing.join(" × ")} mm ·{" "}
                      {p.geometry?.orientation}
                    </p>
                    <p>
                      {String(
                        p.scan.metadata.manufacturer ?? "Unknown manufacturer",
                      )}{" "}
                      · {String(p.scan.metadata.model ?? "Unknown scanner")}
                    </p>
                    <p>
                      {String(p.scan.metadata.protocol ?? p.scan.description)}
                    </p>
                    <p>{p.scan.provenance}</p>
                  </details>
                </section>
              ))}
            </div>
          )}
          <div className="comparison-measures">
            <h3>Regional change is not established</h3>
            <p>
              The viewer has no reviewed longitudinal segmentations for these
              scans. Regional trend charts and quantitative
              differences remain unavailable; no values are inferred from
              display brightness.
            </p>
          </div>
        </>
      )}
    </main>
  );
}

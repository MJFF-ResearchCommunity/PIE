import { useEffect, useRef, useState } from "react";
import type { FormEvent, ReactNode } from "react";
import { NVImage } from "@niivue/niivue";
import pieLogo from "../../assets/icon.png";
import {
  Activity,
  ArrowDownToLine,
  ArrowLeft,
  ArrowRight,
  Box,
  Camera,
  Check,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  Crosshair,
  FilePlus2,
  Focus,
  Layers2,
  LoaderCircle,
  Maximize2,
  Minus,
  Minimize2,
  PanelLeftClose,
  PanelLeftOpen,
  Pause,
  Play,
  Plus,
  RotateCcw,
  Search,
  SlidersHorizontal,
  X,
} from "lucide-react";
import BrainCanvas from "./BrainCanvas";
import AnatomyControls from "./AnatomyControls";
import ReviewPanel from "./ReviewPanel";
import { SpectPresets, SpectHistogram } from "./SpectControls";
import ComparisonWorkspace from "./ComparisonWorkspace";
import FmriWorkbench, { FmriTransport } from "./FmriWorkbench";
import { INITIAL_SURFACE_LIGHTING } from "./renderSafety";
import { INITIAL_SPECT_CUTOFF } from "./spectDisplay";
import { structureViewLabel } from "./structureDisplay";
import type { ViewerApi } from "./BrainCanvas";
import { MODALITIES } from "./types";
import type {
  Catalog,
  Display,
  Modality,
  Prepared,
  Probe,
  Scan,
  Subject,
  ViewMode,
  Structures,
} from "./types";
import {
  canOverlay,
  dateLabel,
  downloadJson,
  nearestScan,
  regionContext,
  regionName,
} from "./model";

const names: Record<Modality, string> = {
  MRI: "Structural MRI",
  DTI: "Diffusion imaging",
  SPECT: "SPECT emission",
  PET: "PET emission",
  CT: "Computed tomography",
  fMRI: "Functional MRI",
};
const modalityNotes: Record<Modality, string> = {
  MRI: "Native anatomy from the measured MRI. The brain mask removes extracranial tissue; anatomical labels come from the participant’s segmentation.",
  DTI: "FA is shown over the mean b0 in diffusion space. FA measures diffusion anisotropy, not tract count or neuronal density. T1 fusion requires registration.",
  SPECT:
    "This is a reconstructed emission volume. Projection frames must be reconstructed before viewing. Native counts are not a voxelwise binding-ratio map.",
  PET: "Interpret signal in the context of the tracer, acquisition timing, reconstruction, and intensity units. Uncalibrated PET counts are not SUV or SUVR.",
  CT: "Window and level control Hounsfield-unit contrast when the source is calibrated. Bone and brain windows show the same measured data differently.",
  fMRI: "Raw BOLD frames describe a time series, not an activation map. Statistical maps require an explicit analysis and registration to the anatomical reference.",
};
const defaults: Display = {
  atlasOutline: false,
  hideSignal: false,
  structures: false,
  structureMri: true,
  structureMriOpacity: 0.35,
  structureOpacity: 0.7,
  structureOutlines: true,
  leftOpacity: 0.18,
  rightOpacity: 0.18,
  contextOpacity: 0.18,
  selectedStructures: [],
  mode: "3d",
  atlas: false,
  atlasOpacity: 0.38,
  opacity: 1,
  overlayOpacity: 0.6,
  anatomyOpacity: 1,
  window: [0, 200],
  colormap: "gray",
  spectCutoff: INITIAL_SPECT_CUTOFF,
  clip: 100,
  clipAxis: 0,
  frame: 0,
  illumination: INITIAL_SURFACE_LIGHTING,
  crosshair: false,
  metric: "fa",
};

async function get<T>(url: string, signal?: AbortSignal): Promise<T> {
  const r = await fetch(url, { signal });
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    throw new Error(
      typeof body.detail === "string"
        ? body.detail
        : `Request failed (${r.status})`,
    );
  }
  return r.json();
}

function IconButton({
  title,
  children,
  onClick,
  disabled,
  active,
}: {
  title: string;
  children: ReactNode;
  onClick: () => void;
  disabled?: boolean;
  active?: boolean;
}) {
  return (
    <button
      className={`icon-button ${active ? "active" : ""}`}
      title={title}
      aria-label={title}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  );
}
function Range({
  label,
  value,
  min = 0,
  max = 100,
  step = 1,
  text,
  onChange,
  disabled,
}: {
  label: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  text?: string;
  onChange: (n: number) => void;
  disabled?: boolean;
}) {
  return (
    <label className="range-field">
      <span>
        {label}
        <output>{text ?? value}</output>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        aria-label={label}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}

export default function App() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [subjectId, setSubjectId] = useState("");
  const [scanId, setScanId] = useState("");
  const [prepared, setPrepared] = useState<Prepared | null>(null);
  const [structures, setStructures] = useState<Structures | null>(null);
  const [spectAnatomy, setSpectAnatomy] = useState(true);
  const [overlay, setOverlay] = useState<Prepared | null>(null);
  const [display, setDisplay] = useState<Display>(defaults);
  const [query, setQuery] = useState("");
  const [cohort, setCohort] = useState("all");
  const [regionQuery, setRegionQuery] = useState("");
  const [probe, setProbe] = useState<Probe | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [section, setSection] = useState<
    "explore" | "acquisitions" | "samples" | "compare"
  >("explore");
  const [importing, setImporting] = useState(false);
  const [sidebar, setSidebar] = useState(true);
  const [help, setHelp] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(2);
  const [retry, setRetry] = useState(0);
  const api = useRef<ViewerApi | null>(null);
  const localScans = useRef(new Map<string, Prepared>());
  const urls = useRef<string[]>([]);
  const requestCounter = useRef(0);
  const [overlayBusy, setOverlayBusy] = useState(false);
  const [graphicsWarning, setGraphicsWarning] = useState("");
  const [lightingBlocked, setLightingBlocked] = useState(false);
  const recoveryCount = useRef(0);
  const viewerFrame = useRef<HTMLDivElement>(null);
  const fullscreenButton = useRef<HTMLButtonElement>(null);
  const [fullscreen, setFullscreen] = useState(false);
  const [fullscreenError, setFullscreenError] = useState("");
  const isBold =
    prepared?.scan.modality === "fMRI" && (prepared.geometry?.frames ?? 0) > 1;

  useEffect(() => {
    let wasFullscreen = false;
    const changed = () => {
      const active =
        document.fullscreenElement === viewerFrame.current &&
        !!viewerFrame.current;
      setFullscreen(active);
      if (wasFullscreen && !active)
        fullscreenButton.current?.focus({ preventScroll: true });
      wasFullscreen = active;
    };
    document.addEventListener("fullscreenchange", changed);
    return () => document.removeEventListener("fullscreenchange", changed);
  }, []);

  async function toggleFullscreen() {
    const frame = viewerFrame.current;
    if (!frame) return;
    setFullscreenError("");
    try {
      if (document.fullscreenElement === frame) await document.exitFullscreen();
      else if (frame.requestFullscreen) await frame.requestFullscreen();
      else throw new Error("Fullscreen is not supported");
    } catch {
      setFullscreenError(
        "The browser could not enter fullscreen. Allow fullscreen for this local page and try again.",
      );
    }
  }

  useEffect(() => {
    recoveryCount.current = 0;
  }, [scanId]);

  function recoverRenderer() {
    setLightingBlocked(true);
    setGraphicsWarning(
      "The graphics driver reset. The brain was reloaded with surface lighting off; 3D rotation, slices and measurements are still available.",
    );
    setDisplay((d) => ({ ...d, illumination: false }));
    if (recoveryCount.current++ < 1) setRetry((r) => r + 1);
    else {
      setBusy(false);
      setError(
        "The browser keeps losing its WebGL graphics context. Close other graphics-heavy tabs, then choose Try again. Your image data are unchanged.",
      );
    }
  }

  useEffect(() => {
    const controller = new AbortController();
    get<Catalog>("/api/catalog", controller.signal)
      .then((c) => {
        setCatalog(c);
        const initial = c.subjects[0];
        if (initial) {
          setSubjectId(initial.id);
          setScanId(
            initial.scans.find((s) => s.modality === "MRI")?.id ??
              initial.scans[0].id,
          );
        }
      })
      .catch((e) => {
        if (e.name !== "AbortError")
          setError(
            "The imaging service is unavailable. Start the PIE viewer server on port 8765, then reload.",
          );
      });
    return () => {
      controller.abort();
      urls.current.forEach(URL.revokeObjectURL);
    };
  }, []);

  useEffect(() => {
    if (!scanId) return;
    const controller = new AbortController();
    requestCounter.current++;
    setBusy(true);
    setError("");
    setProbe(null);
    setPrepared(null);
    setStructures(null);
    setOverlay(null);
    setPlaying(false);
    setOverlayBusy(false);
    const selected = catalog?.subjects
      .find((s) => s.id === subjectId)
      ?.scans.find((s) => s.id === scanId);
    const withAnatomy = spectAnatomy && !!selected?.metadata.anatomy_preview;
    const load = localScans.current.has(scanId)
      ? Promise.resolve(localScans.current.get(scanId)!)
      : get<Prepared>(
          `/api/scans/${encodeURIComponent(scanId)}${withAnatomy ? "/anatomy-preview" : ""}`,
          controller.signal,
        );
    load
      .then((p) => {
        if (controller.signal.aborted) return;
        const v = p.volumes.find((v) => v.role === "primary");
        setDisplay((d) => ({
          ...defaults,
          mode: p.scan.modality === "fMRI" ? "axial" : d.mode,
          metric: p.scan.modality === "fMRI" ? "bold" : defaults.metric,
          crosshair: p.scan.modality === "fMRI",
          window: [v?.cal_min ?? 0, v?.cal_max ?? 1],
          colormap: v?.colormap ?? "gray",
          opacity: v?.opacity ?? 1,
          anatomyOpacity: p.context ? 0.8 : 1,
          spectCutoff: p.context ? 0.7 : INITIAL_SPECT_CUTOFF,
        }));
        setPrepared(p);
      })
      .catch((e) => {
        if (e.name !== "AbortError") {
          setError(e.message);
          setBusy(false);
        }
      });
    return () => controller.abort();
    // Catalog changes from imports do not reload an already selected acquisition.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scanId, retry, spectAnatomy]);

  useEffect(() => {
    if (
      !playing ||
      busy ||
      section !== "explore" ||
      !prepared?.geometry ||
      prepared.geometry.frames < 2 ||
      (isBold && display.metric !== "bold")
    )
      return;
    const frames = prepared.geometry.frames;
    if (display.frame >= frames - 1) {
      setPlaying(false);
      return;
    }
    const timer = window.setInterval(
      () =>
        setDisplay((d) => ({ ...d, frame: Math.min(d.frame + 1, frames - 1) })),
      1000 / (isBold ? playbackSpeed : 4),
    );
    return () => clearInterval(timer);
  }, [
    playing,
    prepared,
    busy,
    section,
    isBold,
    display.metric,
    display.frame,
    playbackSpeed,
  ]);
  useEffect(() => {
    setPlaying(false);
  }, [section]);
  useEffect(() => {
    const pauseWhenHidden = () => {
      if (document.hidden) setPlaying(false);
    };
    document.addEventListener("visibilitychange", pauseWhenHidden);
    return () =>
      document.removeEventListener("visibilitychange", pauseWhenHidden);
  }, []);

  const subject = catalog?.subjects.find((p) => p.id === subjectId);
  const selectedScan = subject?.scans.find((s) => s.id === scanId);
  const scan = prepared?.scan.id === scanId ? prepared.scan : selectedScan;
  const dates = subject?.dates ?? [];
  const dateIndex = scan?.date ? dates.indexOf(scan.date) : -1;
  const currentDate = scan?.date ?? null;
  const groups = [
    ...new Set(catalog?.subjects.map((s) => s.cohort) ?? []),
  ].sort();
  const filtered =
    catalog?.subjects.filter(
      (s) =>
        (cohort === "all" || s.cohort === cohort) &&
        `${s.id} ${s.group} ${s.cohort}`
          .toLowerCase()
          .includes(query.toLowerCase()),
    ) ?? [];
  const compatible = prepared?.context
    ? []
    : (subject?.scans.filter((s) => scan && canOverlay(scan, s)) ?? []);
  const region = probe?.region;
  const visibleRegions =
    prepared?.regions.filter((r) =>
      regionName(r.name).toLowerCase().includes(regionQuery.toLowerCase()),
    ) ?? [];
  const update = (patch: Partial<Display>) =>
    setDisplay((d) => ({ ...d, ...patch }));
  const seekBold = (frame: number) => {
    setPlaying(false);
    update({ frame });
  };
  const updateBoldMetric = (metric: string) => {
    if (metric === display.metric) return;
    const extra = prepared?.extra?.find((e) => e.key === metric);
    const primary = prepared?.volumes.find((v) => v.role === "primary");
    setPlaying(false);
    setProbe(null);
    setBusy(true);
    update({
      metric,
      window: [
        extra?.cal_min ?? primary?.cal_min ?? 0,
        extra?.cal_max ?? primary?.cal_max ?? 1,
      ],
      colormap: extra?.colormap ?? "gray",
    });
  };
  const selectSubject = (p: Subject) => {
    setSubjectId(p.id);
    setScanId(p.scans.find((s) => s.modality === "MRI")?.id ?? p.scans[0].id);
  };
  const selectDate = (i: number) => {
    if (!subject || !dates[i]) return;
    const match =
      nearestScan(
        { ...subject, scans: subject.scans.filter((s) => s.date === dates[i]) },
        scan?.modality ?? null,
        dates[i],
      ) ?? subject.scans.find((s) => s.date === dates[i]);
    if (match) setScanId(match.id);
  };
  const selectModality = (m: Modality) => {
    if (!subject) return;
    const next = nearestScan(subject, m, currentDate);
    if (next) setScanId(next.id);
  };
  async function selectOverlay(id: string) {
    const epoch = ++requestCounter.current;
    setOverlay(null);
    if (!id) return;
    setOverlayBusy(true);
    try {
      const p =
        localScans.current.get(id) ??
        (await get<Prepared>(`/api/scans/${encodeURIComponent(id)}`));
      if (epoch === requestCounter.current && scan && canOverlay(scan, p.scan))
        setOverlay(p);
    } catch (e) {
      if (epoch === requestCounter.current) setError(String(e));
    } finally {
      if (epoch === requestCounter.current) setOverlayBusy(false);
    }
  }
  function importScan(p: Prepared, group: string) {
    localScans.current.set(p.scan.id, p);
    urls.current.push(...p.volumes.map((v) => v.url));
    setCatalog((c) => {
      const base = c ?? {
        subjects: [],
        modalities: MODALITIES,
        scan_count: 0,
        source: "Local browser imports",
        warnings: [],
      };
      const old = base.subjects.find((s) => s.id === p.scan.subject);
      const scans = [...(old?.scans ?? []), p.scan];
      const next: Subject = {
        ...old,
        id: p.scan.subject,
        group: old?.group ?? group,
        cohort: old?.cohort ?? group,
        scans,
        dates: [
          ...new Set(
            scans.map((s) => s.date).filter((d): d is string => Boolean(d)),
          ),
        ].sort(),
        modalities: [...new Set(scans.map((s) => s.modality))],
      };
      return {
        ...base,
        scan_count: base.scan_count + 1,
        subjects: [...base.subjects.filter((s) => s.id !== next.id), next],
      };
    });
    setSubjectId(p.scan.subject);
    setScanId(p.scan.id);
    setImporting(false);
    setSection("explore");
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            setSection("explore");
          }}
          aria-label="PIE Brain Explorer home"
        >
          <img
            className="brand-logo"
            src={pieLogo}
            alt=""
            width={40}
            height={40}
          />
          <span>
            PIE
            <span className="brand-divider" />{" "}
            <span className="brand-product">Brain Explorer</span>
          </span>
          <span className="preview-tag">RESEARCH</span>
        </a>
        <nav aria-label="Workspace">
          <button
            className={section === "explore" ? "nav-active" : ""}
            onClick={() => setSection("explore")}
          >
            Explorer
          </button>
          <button
            className={section === "acquisitions" ? "nav-active" : ""}
            onClick={() => setSection("acquisitions")}
          >
            Acquisitions
          </button>
          <button
            className={section === "compare" ? "nav-active" : ""}
            onClick={() => setSection("compare")}
          >
            Compare visits
          </button>
          <button
            className={section === "samples" ? "nav-active" : ""}
            onClick={() => setSection("samples")}
          >
            Sample plan
          </button>
        </nav>
        <div className="header-actions">
          <span className="local-status">
            <i />
            Local workspace
          </span>
          <button className="button primary" onClick={() => setImporting(true)}>
            <Plus size={16} /> Import scan
          </button>
        </div>
      </header>

      {importing && (
        <ImportPanel
          onClose={() => setImporting(false)}
          onImport={importScan}
          defaultSubject={subjectId}
        />
      )}

      <div className="workspace-heading">
        <div className="workspace-title">
          <span className="eyebrow">PARKINSON’S INSIGHT ENGINE</span>
          <h1 className="sr-only">
            {section === "samples"
              ? "Sample plan"
              : section === "compare"
                ? "Compare visits"
                : section === "acquisitions"
                  ? "Acquisitions"
                  : "Brain Explorer"}
          </h1>
        </div>
        <div className="workspace-tools">
          {section !== "samples" && (
            <>
              <IconButton
                title={
                  sidebar
                    ? "Hide participant browser"
                    : "Show participant browser"
                }
                onClick={() => setSidebar(!sidebar)}
              >
                {sidebar ? (
                  <PanelLeftClose size={18} />
                ) : (
                  <PanelLeftOpen size={18} />
                )}
              </IconButton>
              <button
                className="button"
                disabled={!prepared || busy}
                onClick={() =>
                  downloadJson(
                    {
                      version: 1,
                      scan: prepared?.scan,
                      display,
                      overlay: overlay?.scan.id,
                      selected_region: region?.id,
                      note: "View state only; patient images are not embedded.",
                      anatomy_context: prepared?.context ?? null,
                      fingerprint: prepared?.fingerprint,
                      structures: structures
                        ? {
                            reference_id: structures.reference_id,
                            fingerprint: structures.fingerprint,
                            note: structures.note,
                          }
                        : null,
                    },
                    `PIE-view-${subjectId}.json`,
                  )
                }
              >
                <ArrowDownToLine size={15} /> View state
              </button>
            </>
          )}
          <IconButton
            title="Viewer controls and geometry guide"
            onClick={() => setHelp(!help)}
            active={help}
          >
            <CircleHelp size={19} />
          </IconButton>
        </div>
      </div>
      {help && (
        <section className="help-strip" aria-label="Viewer guide">
          <div>
            <strong>Explore</strong>
            <p>
              Drag to rotate. Scroll to zoom. Click the 3D brain to inspect a
              region. Arrow keys rotate the focused canvas; the region list
              provides keyboard access. Use the expand button (or F with the
              viewer focused) for fullscreen; Escape exits.
            </p>
          </div>
          <div>
            <strong>Read the anatomy</strong>
            <p>
              Use Four-view or the cutaway to inspect deep structures.
              Coordinates are scanner RAS millimetres; L/R labels describe the
              patient.
            </p>
          </div>
          <div>
            <strong>Compare responsibly</strong>
            <p>
              The timeline selects acquired scans without interpolation. Fusion
              requires a documented same-subject registration. BOLD frame time
              is separate from visit date.
            </p>
          </div>
          <IconButton title="Close guide" onClick={() => setHelp(false)}>
            <X size={16} />
          </IconButton>
        </section>
      )}

      {section === "samples" ? (
        <SamplePlan />
      ) : (
        <div
          className={`workspace ${!sidebar ? "sidebar-hidden" : ""} ${section === "acquisitions" ? "acquisition-workspace" : ""}`}
        >
          {sidebar && (
            <aside
              className="participant-panel"
              aria-label="Participant browser"
            >
              <div className="panel-title">
                <h2>Participants</h2>
                <span className="count">
                  {catalog?.subjects.length.toLocaleString() ?? "—"}
                </span>
              </div>
              <label className="search-field">
                <Search size={15} />
                <input
                  aria-label="Search participant ID or cohort"
                  placeholder="Find participant…"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
                <kbd>⌕</kbd>
              </label>
              <label className="cohort-select">
                <select
                  aria-label="Filter participants by cohort"
                  value={cohort}
                  onChange={(e) => setCohort(e.target.value)}
                >
                  <option value="all">All cohorts</option>
                  {groups.map((g) => (
                    <option key={g}>{g}</option>
                  ))}
                </select>
                <ChevronDown size={13} />
              </label>
              {!!catalog?.subjects.some((p) =>
                p.scans.some(
                  (s) => s.metadata.example && s.modality === "fMRI",
                ),
              ) && (
                <div
                  className="example-shortcuts"
                  aria-label="Local fMRI examples"
                >
                  <span className="eyebrow">fMRI EXAMPLES · PRODROMAL</span>
                  {catalog.subjects
                    .filter((p) =>
                      p.scans.some(
                        (s) => s.metadata.example && s.modality === "fMRI",
                      ),
                    )
                    .map((p) => (
                      <button
                        className="text-link"
                        key={p.id}
                        onClick={() => {
                          setCohort("all");
                          setQuery(p.id);
                          setSubjectId(p.id);
                          setScanId(
                            p.scans.find(
                              (s) =>
                                s.modality === "fMRI" &&
                                !s.metadata.short_reference,
                            )!.id,
                          );
                          setSection("explore");
                        }}
                      >
                        PPMI {p.id} <ArrowRight size={12} />
                      </button>
                    ))}
                </div>
              )}
              <div
                className="participant-list"
                role="list"
                aria-label="Participants"
              >
                {!catalog && !error && (
                  <p className="empty-note">Reading local imaging index…</p>
                )}
                {catalog && filtered.length === 0 && (
                  <p className="empty-note">
                    No participants match this search.
                  </p>
                )}
                {filtered.slice(0, 70).map((p) => (
                  <button
                    role="listitem"
                    key={p.id}
                    className={`participant ${p.id === subjectId ? "selected" : ""}`}
                    onClick={() => selectSubject(p)}
                    aria-label={`Open participant ${p.id}, ${p.group}`}
                    aria-current={p.id === subjectId}
                  >
                    <span className="participant-number">
                      <span className="subject-dot" />
                      PPMI {p.id}
                      <ChevronRight size={13} />
                    </span>
                    <span className="participant-cohort">{p.group}</span>
                    <span className="mini-modalities">
                      {p.modalities.map((m) => (
                        <span key={m}>{m}</span>
                      ))}
                    </span>
                  </button>
                ))}
                {filtered.length > 70 && (
                  <p className="empty-note">
                    Showing 70 of {filtered.length.toLocaleString()}. Search an
                    ID to find any participant.
                  </p>
                )}
              </div>
              <div className="collection-note">
                <span className="eyebrow">CONNECTED COLLECTION</span>
                <strong>PPMI · local imaging</strong>
                <p>
                  {catalog?.scan_count.toLocaleString() ?? "—"} indexed scans
                  <br />
                  Images stay on this machine.
                </p>
                <button onClick={() => setSection("samples")}>
                  Plan more downloads <ArrowRight size={13} />
                </button>
              </div>
            </aside>
          )}

          {section === "compare" ? (
            <ComparisonWorkspace
              key={subjectId}
              subject={subject}
              defaults={defaults}
            />
          ) : section === "acquisitions" ? (
            <Acquisitions
              subject={subject}
              onSelect={(id) => {
                setScanId(id);
                setSection("explore");
              }}
            />
          ) : (
            <>
              <main className="viewer-column">
                <section
                  className="subject-bar"
                  aria-label="Selected participant"
                >
                  <div>
                    <span className="eyebrow">PARTICIPANT</span>
                    <h2>
                      {subject ? `PPMI ${subject.id}` : "Choose a participant"}{" "}
                      <span className="group-badge">{subject?.group}</span>
                    </h2>
                  </div>
                  <div className="subject-facts">
                    <span>
                      {subject?.sex === "M"
                        ? "Male"
                        : subject?.sex === "F"
                          ? "Female"
                          : "Sex not recorded"}
                    </span>
                    <span>
                      {subject?.age_at_scan
                        ? `${Math.round(Number(subject.age_at_scan))} at index scan`
                        : "Age not recorded"}
                    </span>
                    <span>{subject?.scans.length ?? 0} acquisitions</span>
                  </div>
                </section>
                <div
                  className="modality-bar"
                  role="group"
                  aria-label="Imaging modality"
                >
                  {MODALITIES.map((m) => {
                    const available = subject?.modalities.includes(m);
                    return (
                      <button
                        key={m}
                        className={scan?.modality === m ? "active" : ""}
                        aria-pressed={scan?.modality === m}
                        disabled={!available}
                        title={
                          available
                            ? `View ${names[m]}; selects nearest actual acquisition`
                            : `No local ${m} for this participant. Import a scan or consult Sample plan.`
                        }
                        onClick={() => selectModality(m)}
                      >
                        {m}
                        <span
                          className={`modality-dot ${available ? "available" : ""}`}
                        />
                      </button>
                    );
                  })}
                  <span className="modality-space">
                    <Box size={13} />{" "}
                    {prepared?.context
                      ? "MRI reference space"
                      : "Patient space"}
                  </span>
                </div>

                {scan?.modality === "SPECT" && (
                  <div className="anatomy-context-bar">
                    <div
                      className="context-switch"
                      role="group"
                      aria-label="SPECT representation"
                    >
                      <button
                        aria-pressed={
                          spectAnatomy &&
                          !!selectedScan?.metadata.anatomy_preview
                        }
                        disabled={!selectedScan?.metadata.anatomy_preview}
                        onClick={() => setSpectAnatomy(true)}
                      >
                        MRI + SPECT preview
                      </button>
                      <button
                        aria-pressed={
                          !spectAnatomy ||
                          !selectedScan?.metadata.anatomy_preview
                        }
                        onClick={() => setSpectAnatomy(false)}
                      >
                        Native SPECT
                      </button>
                    </div>
                    <p>
                      {prepared?.context
                        ? `Alignment not reviewed · SPECT ${dateLabel(scan.date, true)} · MRI ${dateLabel(prepared.context.reference_date, true)}`
                        : selectedScan?.metadata.anatomy_preview
                          ? "Patient MRI is available as an explicitly unreviewed alignment preview."
                          : "No exact MRI reference and transform found. Native SPECT cannot supply cortical anatomy."}
                    </p>
                  </div>
                )}
                <div
                  className={`canvas-frame ${isBold ? "has-fmri" : ""} ${isBold && display.metric === "bold" ? "has-bold-transport" : ""}`}
                  ref={viewerFrame}
                  onKeyDown={(e) => {
                    if (
                      e.key.toLowerCase() === "f" &&
                      !e.repeat &&
                      !e.ctrlKey &&
                      !e.metaKey &&
                      !e.altKey &&
                      !(e.target instanceof HTMLInputElement) &&
                      !(e.target instanceof HTMLSelectElement)
                    ) {
                      e.preventDefault();
                      void toggleFullscreen();
                    }
                  }}
                >
                  <div className="canvas-topline">
                    <div>
                      <span className="eyebrow">
                        {scan
                          ? names[scan.modality].toUpperCase()
                          : "IMAGING WORKSPACE"}
                      </span>
                      <span className="scan-subtitle">
                        {structureViewLabel(display, structures) ??
                          (prepared?.context
                            ? "MRI anatomy + internal SPECT · see-through composite"
                            : isBold
                              ? `${prepared?.extra?.find((e) => e.key === display.metric)?.name ?? `Raw BOLD · frame ${display.frame + 1}/${prepared?.geometry?.frames}`} · native EPI · not an activation map`
                              : scan?.modality === "MRI"
                                ? "Measured anatomy · 3D volume"
                                : scan?.description)}
                      </span>
                    </div>
                    <div className="canvas-actions">
                      {display.structures && structures && (
                        <button
                          className="canvas-button structures-off"
                          onClick={() => update({ structures: false })}
                        >
                          <X size={14} /> Turn off structures
                        </button>
                      )}
                      <button
                        className="canvas-button"
                        disabled={!prepared || busy}
                        onClick={() => api.current?.capture()}
                        title="Save brain view as PNG"
                        aria-label="Save brain view as PNG"
                      >
                        <Camera size={17} />
                      </button>
                      <button
                        ref={fullscreenButton}
                        className="canvas-button"
                        onClick={() => void toggleFullscreen()}
                        title={
                          fullscreen
                            ? "Exit fullscreen (Esc)"
                            : "Enter fullscreen (F)"
                        }
                        aria-label={
                          fullscreen ? "Exit fullscreen" : "Enter fullscreen"
                        }
                        aria-keyshortcuts="F"
                        aria-pressed={fullscreen}
                      >
                        {fullscreen ? (
                          <Minimize2 size={17} />
                        ) : (
                          <Maximize2 size={17} />
                        )}
                      </button>
                    </div>
                  </div>
                  {(fullscreenError ||
                    scan?.modality === "SPECT" ||
                    (display.structures && structures)) && (
                    <div className="canvas-notice" role="status">
                      {fullscreenError ||
                        (scan?.modality !== "SPECT"
                          ? display.mode === "3d"
                            ? "ESTIMATED ANATOMY · colors identify structures, not activity · boundaries are visible through MRI"
                            : "ESTIMATED BOUNDARIES · selected labels on measured MRI · inspect alignment in all three planes"
                          : display.hideSignal
                            ? "SPECT HIDDEN · showing anatomical reference · alignment not reviewed"
                            : prepared?.context
                              ? `ALIGNMENT NOT REVIEWED · MRI ${dateLabel(prepared.context.reference_date, true)} + SPECT ${dateLabel(scan?.date ?? null, true)} · ${display.mode === "3d" ? `${Math.round(display.spectCutoff * 100)}% display cutoff` : "full-signal slices"}`
                              : display.mode === "3d"
                                ? `SPECT signal · ${display.spectCutoff > 0 ? `${Math.round(display.spectCutoff * 100)}% display cutoff` : "background cutoff off"} · not an anatomical surface`
                                : "Full SPECT signal · 3D background cutoff not applied")}
                    </div>
                  )}
                  <div
                    className={`canvas-mount ${display.mode === "multi" ? "review-grid" : ""}`}
                  >
                    {prepared && !error && (
                      <BrainCanvas
                        key={`${scanId}-${retry}-${prepared.context ? "anatomy" : "native"}-${overlay?.scan.id ?? "base"}-${display.metric}-${structures?.fingerprint ?? "volume"}`}
                        structures={structures}
                        onModeChange={(mode) => update({ mode })}
                        onFrameChange={(frame) => update({ frame })}
                        prepared={prepared}
                        overlay={overlay}
                        display={display}
                        onProbe={setProbe}
                        onBusy={setBusy}
                        onError={setError}
                        onContextLost={recoverRenderer}
                        onLightingUnavailable={() => {
                          setLightingBlocked(true);
                          setDisplay((d) => ({ ...d, illumination: false }));
                          setGraphicsWarning(
                            "Surface lighting is unavailable on this graphics configuration. The measured brain remains available in standard 3D and slice views.",
                          );
                        }}
                        onReady={(v) => {
                          api.current = v;
                        }}
                      />
                    )}
                  </div>
                  {busy && !error && (
                    <div className="canvas-loading" role="status">
                      <LoaderCircle size={25} className="spin" />
                      <strong>Preparing scan</strong>
                      <span>Loading image geometry and local voxel data…</span>
                    </div>
                  )}
                  {error && (
                    <div className="canvas-empty" role="alert">
                      <Activity size={28} />
                      <h3>Unable to open this scan</h3>
                      <p>{error}</p>
                      <button
                        className="button"
                        onClick={() => {
                          recoveryCount.current = 0;
                          catalog
                            ? setRetry((r) => r + 1)
                            : window.location.reload();
                        }}
                      >
                        Try again
                      </button>
                    </div>
                  )}
                  {!scan && !busy && !error && (
                    <div className="canvas-empty">
                      <Box size={32} />
                      <h3>Your imaging workspace</h3>
                      <p>
                        Select a local participant or import a NIfTI scan to
                        begin.
                      </p>
                      <button
                        className="button primary"
                        onClick={() => setImporting(true)}
                      >
                        Import a scan
                      </button>
                    </div>
                  )}
                  <div className="canvas-toolbar">
                    <div
                      className="view-switch"
                      role="group"
                      aria-label="View layout"
                    >
                      {(
                        [
                          [
                            "3d",
                            scan?.modality === "SPECT" && !prepared?.context
                              ? "3D signal"
                              : "3D brain",
                          ],
                          ["multi", "Four-view"],
                          ["axial", "Axial"],
                          ["coronal", "Coronal"],
                          ["sagittal", "Sagittal"],
                        ] as [ViewMode, string][]
                      ).map(([mode, text]) => (
                        <button
                          key={mode}
                          disabled={!prepared}
                          aria-pressed={display.mode === mode}
                          className={display.mode === mode ? "active" : ""}
                          onClick={() => update({ mode })}
                        >
                          {mode === "3d" && <Box size={13} />}
                          {text}
                        </button>
                      ))}
                    </div>
                    <div className="zoom-tools">
                      <IconButton
                        title="Zoom out"
                        disabled={!prepared || busy}
                        onClick={() => api.current?.zoom(0.85)}
                      >
                        <Minus size={16} />
                      </IconButton>
                      <IconButton
                        title="Reset camera"
                        disabled={!prepared || busy}
                        onClick={() => {
                          update({ clip: 100 });
                          api.current?.home();
                        }}
                      >
                        <RotateCcw size={15} />
                      </IconButton>
                      <IconButton
                        title="Zoom in"
                        disabled={!prepared || busy}
                        onClick={() => api.current?.zoom(1.15)}
                      >
                        <Plus size={16} />
                      </IconButton>
                    </div>
                  </div>
                  <div className="canvas-bottomline">
                    <span>
                      <i />
                      {prepared?.geometry
                        ? `${prepared.geometry.spacing.join(" × ")} mm`
                        : "NATIVE GEOMETRY"}
                      <span className="canvas-separator">/</span>RAS coordinates
                    </span>
                    <span>
                      {display.mode === "multi"
                        ? "Linked slices · Double-click a panel to enlarge"
                        : "Drag to rotate · Scroll to zoom · Double-click for Four-view"}
                    </span>
                  </div>
                  {isBold && prepared && display.metric === "bold" && (
                    <div className="fullscreen-transport">
                      <FmriTransport
                        prepared={prepared}
                        frame={display.frame}
                        playing={playing}
                        speed={playbackSpeed}
                        busy={busy}
                        seek={seekBold}
                        setPlaying={setPlaying}
                        setSpeed={setPlaybackSpeed}
                      />
                    </div>
                  )}
                </div>

                {isBold && prepared && (
                  <FmriWorkbench
                    key={prepared.fingerprint ?? prepared.scan.id}
                    prepared={prepared}
                    display={display}
                    probe={probe}
                    frame={display.frame}
                    playing={playing}
                    speed={playbackSpeed}
                    busy={busy}
                    seek={seekBold}
                    setPlaying={setPlaying}
                    setSpeed={setPlaybackSpeed}
                    updateMetric={updateBoldMetric}
                  />
                )}
                <section className="timeline" aria-label="Acquisition timeline">
                  <div className="timeline-header">
                    <div>
                      <span className="eyebrow">ACQUISITION TIMELINE</span>
                      <strong>
                        {dateLabel(currentDate)} <span>{scan?.visit}</span>
                      </strong>
                    </div>
                    <div className="timeline-arrows">
                      <button
                        className="text-link"
                        onClick={() => setSection("compare")}
                      >
                        Compare visits side by side <ArrowRight size={13} />
                      </button>
                      <span>
                        {dates.length} distinct{" "}
                        {dates.length === 1 ? "date" : "dates"}
                      </span>
                      <IconButton
                        title="Previous acquisition date"
                        disabled={dateIndex <= 0 || busy}
                        onClick={() => selectDate(dateIndex - 1)}
                      >
                        <ArrowLeft size={16} />
                      </IconButton>
                      <IconButton
                        title="Next acquisition date"
                        disabled={
                          dateIndex < 0 || dateIndex >= dates.length - 1 || busy
                        }
                        onClick={() => selectDate(dateIndex + 1)}
                      >
                        <ArrowRight size={16} />
                      </IconButton>
                    </div>
                  </div>
                  <input
                    type="range"
                    className="timeline-range"
                    aria-label="Patient acquisition date"
                    min={0}
                    max={Math.max(0, dates.length - 1)}
                    step={1}
                    value={Math.max(0, dateIndex)}
                    disabled={dates.length < 2 || busy}
                    onChange={(e) => selectDate(Number(e.target.value))}
                  />
                  <div className="timeline-labels">
                    <span>
                      {dates[0]
                        ? dateLabel(dates[0], true)
                        : "No established acquisition dates"}
                    </span>
                    <span>Acquired scans only · no interpolation</span>
                    <span>
                      {dates.length > 1
                        ? dateLabel(dates[dates.length - 1], true)
                        : "Single-date collection"}
                    </span>
                  </div>
                  {subject &&
                    subject.scans.filter((s) => s.date === currentDate).length >
                      1 && (
                      <label className="series-select">
                        Series on this date
                        <select
                          aria-label="Select acquisition series on this date"
                          value={scanId}
                          onChange={(e) => setScanId(e.target.value)}
                        >
                          {subject.scans
                            .filter((s) => s.date === currentDate)
                            .map((s) => (
                              <option key={s.id} value={s.id}>
                                {s.modality} · {s.description} (
                                {(s.metadata.image_id as string) ?? s.id})
                              </option>
                            ))}
                        </select>
                      </label>
                    )}
                </section>
                <footer className="viewer-footer">
                  <span>
                    <Check size={13} />{" "}
                    {prepared?.context
                      ? "MRI anatomy · SPECT placement uses an unreviewed transform"
                      : "Local rendering · native voxel geometry"}
                  </span>
                  <button onClick={() => setSection("acquisitions")}>
                    View acquisition details <ArrowRight size={12} />
                  </button>
                </footer>
              </main>

              <aside
                className="inspector"
                aria-label="Image controls and region information"
              >
                <div className="panel-title">
                  <h2>
                    <SlidersHorizontal size={16} /> View controls
                  </h2>
                  <span className="small-label">LIVE</span>
                </div>
                {prepared && (
                  <AnatomyControls
                    key={`${scanId}-${prepared.context ? "preview" : "native"}`}
                    prepared={prepared}
                    structures={structures}
                    onLoad={setStructures}
                    display={display}
                    update={update}
                    focus={(r) => api.current?.focus(r)}
                  />
                )}
                {prepared?.scan.modality === "SPECT" && (
                  <SpectPresets
                    prepared={prepared}
                    display={display}
                    update={update}
                  />
                )}
                {prepared?.fingerprint && (
                  <ReviewPanel key={prepared.fingerprint} prepared={prepared} />
                )}
                <section className="control-section">
                  <div className="section-label">
                    <Layers2 size={14} />
                    <h3>Image layers</h3>
                  </div>
                  <div className="layer-row">
                    <span
                      className={`layer-dot ${scan?.modality?.toLowerCase() ?? ""}`}
                    />
                    <div>
                      <strong>
                        {scan ? names[scan.modality] : "No image loaded"}
                      </strong>
                      <small>
                        {prepared?.context
                          ? "Alignment preview · datscan_full"
                          : scan?.modality === "DTI"
                            ? "FA + mean b0 · same native grid"
                            : scan?.kind === "timeseries"
                              ? "4D image sequence"
                              : "Native volume"}
                      </small>
                    </div>
                    <Check size={15} />
                  </div>
                  {scan?.modality === "DTI" && (
                    <label className="compact-field">
                      Diffusion measure
                      <select
                        aria-label="Diffusion measure"
                        value={display.metric}
                        onChange={(e) => {
                          const metric = e.target.value;
                          update({
                            metric,
                            window: [0, metric === "md" ? 0.002 : 1],
                          });
                        }}
                      >
                        <option value="fa">Fractional anisotropy</option>
                        {prepared?.extra?.map((e) => (
                          <option key={e.key} value={e.key}>
                            {e.name}
                          </option>
                        ))}
                      </select>
                    </label>
                  )}
                  {prepared?.context && (
                    <div className="anatomy-layer-controls">
                      <div className="layer-row">
                        <span className="layer-dot mri" />
                        <div>
                          <strong>Patient MRI anatomy</strong>
                          <small>
                            Measured cortex ·{" "}
                            {dateLabel(prepared.context.reference_date, true)}
                          </small>
                        </div>
                        <Check size={15} />
                      </div>
                      <Range
                        label="MRI anatomy opacity"
                        value={Math.round(display.anatomyOpacity * 100)}
                        text={`${Math.round(display.anatomyOpacity * 100)}%`}
                        onChange={(n) => update({ anatomyOpacity: n / 100 })}
                        disabled={
                          display.structures &&
                          !!structures &&
                          display.mode === "3d"
                        }
                      />
                      <p className="empty-note">
                        Gray folds and labels come from MRI. Colored uptake
                        remains inside the volume; it is not painted onto the
                        cortex.
                      </p>
                      {display.anatomyOpacity === 0 && (
                        <p className="empty-note">
                          MRI hidden · foreground clipping is off. Extracranial
                          SPECT signal may be visible.
                        </p>
                      )}
                      <details className="fusion-details">
                        <summary>
                          Alignment provenance <ChevronDown size={13} />
                        </summary>
                        <p>
                          Automatic PIE datscan_full transform to{" "}
                          {prepared.context.reference_id}. Alignment has not
                          been reviewed.
                        </p>
                        <p>
                          SPECT source grid:{" "}
                          {prepared.context.source_geometry.spacing.join(" × ")}{" "}
                          mm. MRI does not add SPECT resolution. The pipeline’s
                          recorded left/right flip is{" "}
                          {prepared.context.flip_lr ? "applied" : "not applied"}
                          ; no new orientation is guessed.
                        </p>
                        <p>
                          3D emission is clipped to MRI foreground for display.
                          Four-view and slices retain the full signal for
                          alignment review. Native SPECT opens the original
                          standalone reconstruction.
                        </p>
                      </details>
                    </div>
                  )}
                  <Range
                    label="Signal opacity"
                    value={Math.round(display.opacity * 100)}
                    text={`${Math.round(display.opacity * 100)}%`}
                    onChange={(n) => update({ opacity: n / 100 })}
                    disabled={
                      !prepared ||
                      (scan?.modality === "MRI" &&
                        display.structures &&
                        !!structures &&
                        display.mode === "3d")
                    }
                  />
                  {display.structures && structures ? (
                    <p className="empty-note">
                      Selected structure outlines are controlled in Inside the
                      brain. Turn off structures to return to the full region
                      atlas controls. MRI visibility in 3D is also controlled
                      above.
                    </p>
                  ) : (
                    <>
                      <label className="toggle-row">
                        <span>
                          <strong>Region atlas</strong>
                          <small>
                            {prepared?.regions.length
                              ? `${prepared.regions.length} anatomical labels`
                              : "No registered segmentation"}
                          </small>
                        </span>
                        <input
                          type="checkbox"
                          role="switch"
                          aria-label="Show anatomical region atlas"
                          checked={display.atlas}
                          disabled={!prepared?.regions.length}
                          onChange={(e) => update({ atlas: e.target.checked })}
                        />
                      </label>
                      {display.atlas && (
                        <Range
                          label="Atlas opacity"
                          value={Math.round(display.atlasOpacity * 100)}
                          text={`${Math.round(display.atlasOpacity * 100)}%`}
                          onChange={(n) => update({ atlasOpacity: n / 100 })}
                        />
                      )}
                    </>
                  )}
                  <details className="fusion-details">
                    <summary>
                      Registered overlay <Plus size={13} />
                    </summary>
                    {compatible.length > 0 ? (
                      <label className="compact-field">
                        Aligned to this acquisition
                        <select
                          aria-label="Registered image overlay"
                          disabled={overlayBusy}
                          value={overlay?.scan.id ?? ""}
                          onChange={(e) => void selectOverlay(e.target.value)}
                        >
                          <option value="">No additional overlay</option>
                          {compatible.map((s) => (
                            <option key={s.id} value={s.id}>
                              {s.modality} · {dateLabel(s.date, true)} ·{" "}
                              {s.description}
                            </option>
                          ))}
                        </select>
                      </label>
                    ) : (
                      <p>
                        No verified cross-modality registration targets this
                        scan. Native scans remain independently viewable.
                        Register and review an image, then add it with its
                        reference ID in a viewer manifest.
                      </p>
                    )}
                    {overlay && (
                      <div>
                        <Range
                          label="Overlay opacity"
                          value={Math.round(display.overlayOpacity * 100)}
                          text={`${Math.round(display.overlayOpacity * 100)}%`}
                          onChange={(n) => update({ overlayOpacity: n / 100 })}
                        />
                        <p className="fusion-note">
                          {overlay.scan.modality} acquired{" "}
                          {dateLabel(overlay.scan.date, true)}. Reference
                          acquired {dateLabel(scan?.date ?? null, true)}.{" "}
                          {overlay.scan.date !== scan?.date &&
                            "These are different acquisition dates."}
                        </p>
                      </div>
                    )}
                  </details>
                </section>

                <section className="control-section">
                  <div className="section-label">
                    <Focus size={14} />
                    <h3>Appearance</h3>
                  </div>
                  <label className="compact-field">
                    Color map
                    <select
                      aria-label="Image color map"
                      value={display.colormap}
                      disabled={!prepared}
                      onChange={(e) => update({ colormap: e.target.value })}
                    >
                      {[
                        "gray",
                        "viridis",
                        "inferno",
                        "magma",
                        "hot",
                        "warm",
                        "cool",
                        "ct_bone",
                      ].map((c) => (
                        <option key={c} value={c}>
                          {
                            (
                              {
                                gray: "Grayscale",
                                viridis: "Viridis",
                                inferno: "Inferno",
                                magma: "Magma",
                                hot: "Hot",
                                warm: "Warm",
                                cool: "Cool",
                                ct_bone: "CT bone",
                              } as Record<string, string>
                            )[c]
                          }
                        </option>
                      ))}
                    </select>
                  </label>
                  <div className={`color-scale scale-${display.colormap}`} />
                  <div className="scale-values">
                    <span>{display.window[0].toPrecision(3)}</span>
                    <span>
                      {(
                        (display.window[0] + display.window[1]) /
                        2
                      ).toPrecision(3)}
                    </span>
                    <span>{display.window[1].toPrecision(3)}</span>
                  </div>
                  <div className="window-inputs">
                    <label>
                      Window min
                      <input
                        type="number"
                        aria-label="Window minimum"
                        step="any"
                        value={display.window[0]}
                        disabled={!prepared}
                        onChange={(e) => {
                          const n = Number(e.target.value);
                          if (Number.isFinite(n) && n < display.window[1])
                            update({ window: [n, display.window[1]] });
                        }}
                      />
                    </label>
                    <label>
                      Window max
                      <input
                        type="number"
                        aria-label="Window maximum"
                        step="any"
                        value={display.window[1]}
                        disabled={!prepared}
                        onChange={(e) => {
                          const n = Number(e.target.value);
                          if (Number.isFinite(n) && n > display.window[0])
                            update({ window: [display.window[0], n] });
                        }}
                      />
                    </label>
                  </div>
                  <p className="units">
                    {prepared?.extra?.find((e) => e.key === display.metric)
                      ?.units ??
                      scan?.units ??
                      "Image intensity"}
                  </p>
                  {scan?.modality === "SPECT" && (
                    <div className="spect-display-controls">
                      {prepared && (
                        <SpectHistogram prepared={prepared} display={display} />
                      )}
                      <Range
                        label="3D background cutoff"
                        value={Math.round(display.spectCutoff * 100)}
                        max={95}
                        text={`${Math.round(display.spectCutoff * 100)}%`}
                        disabled={!prepared || display.mode !== "3d"}
                        onChange={(n) => update({ spectCutoff: n / 100 })}
                      />
                      <p>
                        {display.mode === "3d"
                          ? `Display only: fade signal below ${(display.window[0] + display.spectCutoff * (display.window[1] - display.window[0])).toPrecision(3)}. This is not a brain mask or a diagnostic threshold.`
                          : "Cutoff paused. Slice views and Four-view retain the full signal."}
                      </p>
                      <div className="preset-row">
                        <button
                          disabled={!prepared}
                          onClick={() => update({ mode: "axial" })}
                        >
                          Inspect full-signal slices
                        </button>
                        <button
                          disabled={!prepared || display.mode !== "3d"}
                          onClick={() =>
                            update({
                              spectCutoff: prepared?.context
                                ? 0.7
                                : INITIAL_SPECT_CUTOFF,
                            })
                          }
                        >
                          Reset cutoff
                        </button>
                      </div>
                      <p>
                        {prepared?.context
                          ? "MRI provides anatomy, not additional SPECT resolution. Inspect alignment in Four-view before interpreting spatial correspondence."
                          : "SPECT alone does not define the cortical surface. Use the MRI preview when an explicit reference transform is available."}
                      </p>
                    </div>
                  )}
                  {scan?.modality === "CT" && (
                    <div className="preset-row">
                      <button
                        onClick={() =>
                          update({ window: [0, 80], colormap: "gray" })
                        }
                      >
                        Brain window
                      </button>
                      <button
                        onClick={() =>
                          update({ window: [-400, 1800], colormap: "gray" })
                        }
                      >
                        Bone window
                      </button>
                    </div>
                  )}
                  <label className="toggle-row">
                    <span>
                      Surface lighting{" "}
                      <small>
                        Optional · standard rendering is the reliable default
                      </small>
                    </span>
                    <input
                      type="checkbox"
                      role="switch"
                      aria-label="Surface lighting"
                      checked={display.illumination}
                      disabled={!prepared || lightingBlocked}
                      onChange={(e) =>
                        update({ illumination: e.target.checked })
                      }
                    />
                  </label>
                  {graphicsWarning && (
                    <p className="graphics-warning" role="status">
                      {graphicsWarning}
                    </p>
                  )}
                  <label className="toggle-row">
                    <span>Crosshair</span>
                    <input
                      type="checkbox"
                      role="switch"
                      aria-label="Show crosshair"
                      checked={display.crosshair}
                      disabled={!prepared}
                      onChange={(e) => update({ crosshair: e.target.checked })}
                    />
                  </label>
                  {!(display.structures && structures) && (
                    <label className="toggle-row">
                      <span>Segmentation outlines</span>
                      <input
                        type="checkbox"
                        role="switch"
                        aria-label="Segmentation outlines"
                        checked={display.atlasOutline}
                        disabled={!prepared?.regions.length}
                        onChange={(e) =>
                          update({
                            atlasOutline: e.target.checked,
                            atlas: true,
                          })
                        }
                      />
                    </label>
                  )}
                  <Range
                    label="Cutaway depth"
                    value={display.clip}
                    text={
                      display.clip === 100 ? "Full volume" : `${display.clip}%`
                    }
                    onChange={(n) => update({ clip: n })}
                    disabled={!prepared || display.mode !== "3d"}
                  />
                  {display.structures && (
                    <p className="empty-note">
                      Cutaway applies to image volumes. Use hemisphere shell
                      sliders to reveal segmentation boundaries.
                    </p>
                  )}
                  {display.clip < 100 && (
                    <div
                      className="preset-row"
                      role="group"
                      aria-label="Cutaway plane"
                    >
                      {["Coronal", "Sagittal", "Axial"].map((name, i) => (
                        <button
                          key={name}
                          aria-pressed={display.clipAxis === i}
                          className={display.clipAxis === i ? "active" : ""}
                          onClick={() => update({ clipAxis: i })}
                        >
                          {name}
                        </button>
                      ))}
                    </div>
                  )}
                </section>

                {!isBold &&
                  prepared?.geometry &&
                  prepared.geometry.frames > 1 && (
                    <section className="control-section">
                      <div className="section-label">
                        <Activity size={14} />
                        <h3>Within-scan time</h3>
                        <IconButton
                          title={
                            playing ? "Pause image frames" : "Play image frames"
                          }
                          onClick={() => setPlaying(!playing)}
                        >
                          {playing ? <Pause size={13} /> : <Play size={13} />}
                        </IconButton>
                      </div>
                      <Range
                        label="Image frame"
                        min={0}
                        max={prepared.geometry.frames - 1}
                        value={display.frame}
                        text={`${display.frame + 1} / ${prepared.geometry.frames}`}
                        onChange={(n) => update({ frame: n })}
                      />
                      <p className="empty-note">
                        {prepared.geometry.time_unit === "sec"
                          ? `${((prepared.geometry.frame_step ?? 0) * display.frame).toFixed(2)} s from first frame`
                          : "Frame time units not established"}
                        . This is within one scan, separate from the acquisition
                        timeline.
                      </p>
                    </section>
                  )}

                <section className="control-section region-section">
                  <div className="section-label">
                    <Crosshair size={14} />
                    <h3>Region inspector</h3>
                  </div>
                  {prepared?.context && (
                    <p className="empty-note">
                      MRI anatomical labels · SPECT alignment unreviewed. Voxel
                      signals are exploratory; no regional SPECT quantification
                      is provided here.
                    </p>
                  )}
                  {region ? (
                    <div className="region-detail">
                      <span
                        className="region-swatch"
                        style={{ background: `rgb(${region.color.join(",")})` }}
                      />
                      <span className="eyebrow">
                        {scan?.atlas_name ?? "Aligned atlas"} · LABEL{" "}
                        {region.id}
                      </span>
                      <h3>{regionName(region.name)}</h3>
                      {probe?.region_distance_mm !== undefined && (
                        <p>
                          Nearby label · {probe.region_distance_mm.toFixed(1)}{" "}
                          mm from the picked surface. Voxel signal is from the
                          original pick; use slices for exact label boundaries.
                        </p>
                      )}
                      <p>{regionContext(region)}</p>
                      <div className="region-measure">
                        <span>Segmented volume</span>
                        <strong>
                          {(region.volume_mm3 / 1000).toFixed(2)}{" "}
                          <small>mL</small>
                        </strong>
                      </div>
                      <small className="measure-note">
                        Voxel-count estimate · no partial-volume correction
                      </small>
                    </div>
                  ) : (
                    <div className="region-empty">
                      <Crosshair size={23} />
                      <strong>
                        {prepared?.regions.length
                          ? "Explore a region"
                          : "Voxel inspection"}
                      </strong>
                      <p>
                        {prepared?.regions.length
                          ? "Click the brain, inspect a slice, or choose a region below."
                          : "Click a voxel to read its coordinates and signal. Region names require an aligned segmentation."}
                      </p>
                    </div>
                  )}
                  {probe && (
                    <div className="probe-readout">
                      <span>
                        RAS{" "}
                        <b>
                          {probe.mm.map((x) => x.toFixed(1)).join(" · ")} mm
                        </b>
                      </span>
                      <span>
                        Voxel signal{" "}
                        <b>
                          {probe.value === null
                            ? "Outside volume"
                            : Number(probe.value.toPrecision(5))}
                        </b>
                      </span>
                    </div>
                  )}
                  {!!prepared?.regions.length && (
                    <>
                      <label className="search-field small">
                        <Search size={13} />
                        <input
                          aria-label="Search anatomical regions"
                          placeholder="Find a region…"
                          value={regionQuery}
                          onChange={(e) => setRegionQuery(e.target.value)}
                        />
                      </label>
                      <div className="region-list">
                        {visibleRegions.length === 0 && (
                          <p className="empty-note">No matching regions.</p>
                        )}
                        {visibleRegions.map((r) => (
                          <button
                            key={r.id}
                            className={region?.id === r.id ? "selected" : ""}
                            onClick={() => {
                              update({ crosshair: true, mode: "multi" });
                              setProbe({
                                mm: r.center_mm,
                                value: null,
                                region: r,
                              });
                              api.current?.focus(r);
                            }}
                          >
                            <span
                              style={{
                                background: `rgb(${r.color.join(",")})`,
                              }}
                            />
                            {regionName(r.name)}
                            <ChevronRight size={11} />
                          </button>
                        ))}
                      </div>
                    </>
                  )}
                </section>
              </aside>
            </>
          )}
        </div>
      )}
      {section === "explore" && scan && (
        <div className="scientific-note">
          <span>{names[scan.modality]}</span>
          <p>
            {modalityNotes[scan.modality]} <strong>{scan.qc}.</strong>
            {prepared?.geometry?.spatial_unit === "unknown" &&
              " Source spatial units are unspecified: coordinates and volumes assume millimetres. Verify the source header before interpreting measurements."}
          </p>
        </div>
      )}
      <footer className="app-footer">
        <span>PIE / Brain Explorer</span>
        <span>
          Research visualization ·{" "}
          {scan?.tracer
            ? `Tracer: ${scan.tracer}`
            : "Acquisition-aware, participant-specific anatomy"}
        </span>
        <a href="https://niivue.com/" target="_blank" rel="noreferrer">
          Rendering with NiiVue ↗
        </a>
      </footer>
    </div>
  );
}

function Acquisitions({
  subject,
  onSelect,
}: {
  subject?: Subject;
  onSelect: (id: string) => void;
}) {
  return (
    <main className="acquisitions-page">
      <span className="eyebrow">PARTICIPANT RECORD</span>
      <h2>{subject ? `PPMI ${subject.id}` : "Select a participant"}</h2>
      <p>
        Every row is a real acquisition or a derived image. A shared date does
        not establish image registration.
      </p>
      {subject && (
        <>
          <div className="record-summary">
            <div>
              <small>Current cohort</small>
              <strong>{subject.cohort}</strong>
            </div>
            <div>
              <small>Archive group</small>
              <strong>{subject.group}</strong>
            </div>
            <div>
              <small>Distinct acquisition dates</small>
              <strong>{subject.dates.length}</strong>
            </div>
          </div>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Acquisition</th>
                  <th>Modality</th>
                  <th>Series</th>
                  <th>Geometry / provenance</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {subject.scans.map((s) => (
                  <tr key={s.id}>
                    <td>
                      <strong>{dateLabel(s.date, true)}</strong>
                      <small>{s.visit}</small>
                    </td>
                    <td>
                      <span className="modality-pill">{s.modality}</span>
                    </td>
                    <td>
                      <strong>{s.description}</strong>
                      <small>{s.id}</small>
                    </td>
                    <td>
                      <strong>
                        {s.registration === "verified"
                          ? "Verified reference registration"
                          : "Native image space"}
                      </strong>
                      <small>{s.space}</small>
                      <details>
                        <summary>Processing details</summary>
                        <p>{s.provenance}</p>
                        <p>
                          Units: {s.units}. {s.qc}.
                        </p>
                        {s.tracer && <p>Tracer: {s.tracer}</p>}
                        <pre>{JSON.stringify(s.metadata, null, 2)}</pre>
                      </details>
                    </td>
                    <td>
                      <button className="button" onClick={() => onSelect(s.id)}>
                        Open <ArrowRight size={14} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </main>
  );
}

interface Plan {
  available: boolean;
  message?: string;
  notes?: string[];
  coverage?: {
    cohort: string;
    modality: string;
    local_subjects: number;
    candidate_subjects: number;
  }[];
  shortlist?: {
    subject: string;
    modality: string;
    cohort: string;
    dates: string[];
    evidence: string[];
    tracers: string[];
    sources: string[];
  }[];
  archive_groups?: {
    archive_group: string;
    local_subjects: number;
    suggested_subjects: string[];
    note: string;
  }[];
}
function SamplePlan() {
  const [plan, setPlan] = useState<Plan | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    get<Plan>("/api/sample-plan")
      .then(setPlan)
      .catch((e) => setError(e.message));
  }, []);
  return (
    <main className="sample-page">
      <div className="sample-intro">
        <div>
          <span className="eyebrow">EVIDENCE-BACKED SAMPLING</span>
          <h2>The next scans to bring in.</h2>
          <p>
            Start with matched anatomy, then add modalities and genuinely
            different visits. These candidates come from your local PPMI tables;
            confirm image availability in IDA.
          </p>
        </div>
        <button
          className="button"
          disabled={!plan?.available}
          onClick={() => downloadJson(plan, "PPMI-viewer-download-plan.json")}
        >
          <ArrowDownToLine size={15} /> Export plan
        </button>
      </div>
      {error && <p role="alert">{error}</p>}
      {!plan && !error && <p>Reading the local sample plan…</p>}
      <section className="next-downloads-note">
        <h3>Next bundle: participant 3123</h3>
        <p>
          Follow-up T1 MRI and reconstructed SPECT: June 2013 / May 2014. AV-133
          PET: July 2013 / June 2014. Confirm exact acquisitions in IDA.
        </p>
        <p>
          Both parts of fMRI collection 1 are complete by image-ID inventory;
          examples 101685 and 218968 are now available in Explorer. Participant
          116869 supplies a local two-visit MRI example. The older inventory
          below predates these additions.
        </p>
        <a className="button" href="/api/download-guide" download>
          Download the updated IDA checklist
        </a>
      </section>
      {plan && !plan.available && <p>{plan.message}</p>}
      {plan?.available && (
        <>
          <div className="sample-priorities">
            <section>
              <span>01</span>
              <h3>Anatomy first</h3>
              <p>
                3D T1w MRI, ideally around 1 mm isotropic, for every selected
                participant. Include a second visit where available.
              </p>
            </section>
            <section>
              <span>02</span>
              <h3>Matched signals</h3>
              <p>
                DTI with gradients and reverse-phase b0; full 4D resting BOLD;
                reconstructed PET / SPECT with tracer and units.
              </p>
            </section>
            <section>
              <span>03</span>
              <h3>Keep the evidence</h3>
              <p>
                Download the IDA collection CSV and Advanced Download metadata.
                Preserve exact date, image ID, visit and archive group.
              </p>
            </section>
          </div>
          <h3>Modality × current cohort</h3>
          <p className="table-caption">
            Values show local participants / acquisition candidates. A clinical
            form is not proof that an image can be downloaded.
          </p>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Cohort</th>
                  {MODALITIES.map((m) => (
                    <th key={m}>{m}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[...new Set(plan.coverage?.map((c) => c.cohort))].map((c) => (
                  <tr key={c}>
                    <th>{c}</th>
                    {MODALITIES.map((m) => {
                      const cell = plan.coverage?.find(
                        (r) => r.cohort === c && r.modality === m,
                      );
                      return (
                        <td key={m}>
                          <strong>{cell?.local_subjects ?? 0}</strong>
                          <span className="muted">
                            {" "}
                            / {cell?.candidate_subjects ?? 0}
                          </span>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <h3>Suggested participants</h3>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Modality</th>
                  <th>Cohort</th>
                  <th>PATNO</th>
                  <th>Acquisition hints</th>
                  <th>Evidence / tracer</th>
                </tr>
              </thead>
              <tbody>
                {plan.shortlist?.map((s, i) => (
                  <tr key={i}>
                    <td>
                      <span className="modality-pill">{s.modality}</span>
                    </td>
                    <td>{s.cohort}</td>
                    <td>
                      <strong>{s.subject}</strong>
                    </td>
                    <td>
                      {s.dates.slice(0, 4).join(", ")}
                      {s.dates.length > 4 && ` +${s.dates.length - 4}`}
                    </td>
                    <td>
                      <small>{s.evidence.join("; ")}</small>
                      <small>{s.tracers.join(", ")}</small>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <h3>Archive research groups</h3>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>IDA checkbox</th>
                  <th>Local participants</th>
                  <th>Suggested IDs</th>
                  <th>Interpretation</th>
                </tr>
              </thead>
              <tbody>
                {plan.archive_groups?.map((g) => (
                  <tr key={g.archive_group}>
                    <th>{g.archive_group}</th>
                    <td>{g.local_subjects}</td>
                    <td>{g.suggested_subjects.join(", ") || "Query IDA"}</td>
                    <td>{g.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="plan-notes">
            {plan.notes?.map((n) => (
              <p key={n}>{n}</p>
            ))}
          </div>
        </>
      )}
    </main>
  );
}

function ImportPanel({
  onClose,
  onImport,
  defaultSubject,
}: {
  onClose: () => void;
  onImport: (p: Prepared, group: string) => void;
  defaultSubject: string;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [subject, setSubject] = useState(defaultSubject || "local-001");
  const [date, setDate] = useState("");
  const [modality, setModality] = useState<Modality>("MRI");
  const [group, setGroup] = useState("Imported");
  const [tracer, setTracer] = useState("");
  const [working, setWorking] = useState(false);
  const [error, setError] = useState("");
  const [units, setUnits] = useState("arbitrary intensity");
  const first = useRef<HTMLInputElement>(null);
  useEffect(() => {
    first.current?.focus();
  }, []);
  async function submit(e: FormEvent) {
    e.preventDefault();
    if (!file) return;
    setWorking(true);
    setError("");
    let url: string | undefined;
    try {
      if (!/\.(nii(\.gz)?|mgz)$/i.test(file.name))
        throw new Error(
          "Choose a NIfTI (.nii/.nii.gz) or MGZ volume. Convert DICOM with dcm2niix first; raw SPECT projections require reconstruction.",
        );
      if (file.size > 512 * 1024 * 1024)
        throw new Error(
          "This browser import is limited to 512 MB. Use a local viewer manifest for larger acquisitions.",
        );
      const image = await NVImage.loadFromFile({ file, name: file.name });
      const hdr = image.hdr;
      if (
        !hdr ||
        ![3, 4].includes(hdr.dims[0]) ||
        hdr.dims[1] < 2 ||
        hdr.dims[2] < 2 ||
        hdr.dims[3] < 2
      )
        throw new Error("Expected a reconstructed 3D or 4D volume.");
      const affine = hdr.affine;
      if (![0, 2].includes(hdr.xyzt_units & 7))
        throw new Error(
          "Convert the image's declared spatial units to millimetres before import. The viewer does not silently rescale geometry.",
        );
      const determinant =
        affine[0][0] *
          (affine[1][1] * affine[2][2] - affine[1][2] * affine[2][1]) -
        affine[0][1] *
          (affine[1][0] * affine[2][2] - affine[1][2] * affine[2][0]) +
        affine[0][2] *
          (affine[1][0] * affine[2][1] - affine[1][1] * affine[2][0]);
      if (!affine.flat().every(Number.isFinite) || Math.abs(determinant) < 1e-9)
        throw new Error("Image does not contain a usable spatial affine.");
      const id = `local-${crypto.randomUUID()}`;
      url = URL.createObjectURL(file);
      const frames = Math.max(1, image.nFrame4D ?? 1);
      const range: [number, number] =
        modality === "CT"
          ? [0, 80]
          : modality === "DTI"
            ? [0, 1]
            : [image.cal_min ?? 0, image.cal_max ?? 1];
      const cmap =
        modality === "PET" || modality === "SPECT"
          ? "inferno"
          : modality === "DTI"
            ? "viridis"
            : "gray";
      const s: Scan = {
        id,
        subject: subject.trim(),
        date: date || null,
        modality,
        visit: "Imported acquisition",
        description: file.name,
        space: `sub-${subject.trim()}:native:${id}`,
        kind: frames > 1 ? "timeseries" : "scalar",
        units,
        reference_id: null,
        registration: "native",
        provenance:
          "Browser-local import. No registration, brain extraction, atlas, or intensity calibration was performed by the viewer.",
        qc: "User import · not reviewed",
        tracer: tracer || null,
        has_atlas: false,
        has_anatomy: false,
        metadata: { filename: file.name, persistent: false },
      };
      onImport(
        {
          scan: s,
          volumes: [
            {
              url,
              name: file.name,
              role: "primary",
              colormap: cmap,
              opacity: 1,
              cal_min: range[0],
              cal_max: range[1],
            },
          ],
          meshes: [],
          regions: [],
          geometry: {
            shape: hdr.dims.slice(1, frames > 1 ? 5 : 4),
            spacing: hdr.pixDims.slice(1, 4),
            orientation: "NIfTI affine",
            spatial_unit: (hdr.xyzt_units & 7) === 2 ? "mm" : "unknown",
            affine,
            frames,
            frame_step: frames > 1 ? hdr.pixDims[4] : null,
            time_unit: (hdr.xyzt_units & 56) === 8 ? "sec" : "unknown",
            cal_min: range[0],
            cal_max: range[1],
          },
        },
        group,
      );
    } catch (e) {
      if (url) URL.revokeObjectURL(url);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setWorking(false);
    }
  }
  return (
    <section className="import-panel" aria-label="Import local image">
      <div className="import-heading">
        <div>
          <span className="eyebrow">LOCAL IMAGE IMPORT</span>
          <h2>Bring a scan into view.</h2>
          <p>
            Files are read in this browser and stay on this machine. Imports
            last for this session.
          </p>
        </div>
        <IconButton title="Close image import" onClick={onClose}>
          <X size={20} />
        </IconButton>
      </div>
      <form onSubmit={submit}>
        <label className="file-field">
          <FilePlus2 size={25} />
          <span>
            {file ? file.name : "Choose a NIfTI or MGZ volume"}
            <small>.nii · .nii.gz · .mgz</small>
          </span>
          <input
            ref={first}
            type="file"
            accept=".nii,.nii.gz,.mgz"
            aria-label="Local neuroimaging file"
            required
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </label>
        <div className="import-fields">
          <label>
            Participant ID
            <input
              required
              value={subject}
              onChange={(e) => setSubject(e.target.value)}
              aria-label="Import participant ID"
            />
          </label>
          <label>
            Acquisition date
            <input
              type="date"
              value={date}
              onChange={(e) => setDate(e.target.value)}
              aria-label="Import acquisition date"
            />
          </label>
          <label>
            Modality
            <select
              value={modality}
              onChange={(e) => setModality(e.target.value as Modality)}
              aria-label="Import modality"
            >
              {MODALITIES.map((m) => (
                <option key={m}>{m}</option>
              ))}
            </select>
          </label>
          <label>
            Cohort
            <input
              value={group}
              onChange={(e) => setGroup(e.target.value)}
              aria-label="Import cohort"
            />
          </label>
          <label>
            Signal units
            <input
              value={units}
              onChange={(e) => setUnits(e.target.value)}
              aria-label="Import signal units"
            />
          </label>
          {(modality === "PET" || modality === "SPECT") && (
            <label>
              Tracer
              <input
                required
                value={tracer}
                onChange={(e) => setTracer(e.target.value)}
                aria-label="Import tracer"
                placeholder="e.g. 18F-FDG"
              />
            </label>
          )}
        </div>
        <p className="import-note">
          Use reconstructed volumes. DICOM series can be converted with
          dcm2niix. For aligned overlays, segmentations, tractography and
          persistent collections, use a viewer manifest (see the repository’s
          Brain Explorer guide).
        </p>
        {error && (
          <p className="form-error" role="alert">
            {error}
          </p>
        )}
        <div className="import-actions">
          <button type="button" className="button" onClick={onClose}>
            Cancel
          </button>
          <button
            className="button primary"
            disabled={!file || working || !subject.trim()}
          >
            {working ? (
              <LoaderCircle className="spin" size={16} />
            ) : (
              <Plus size={16} />
            )}{" "}
            {working ? "Reading volume…" : "Open image"}
          </button>
        </div>
      </form>
    </section>
  );
}

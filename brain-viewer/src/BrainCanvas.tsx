import { useEffect, useRef, useState } from "react";
import { Niivue, cmapper } from "@niivue/niivue";
import { nearbyAtlasLabel } from "./model";
import { probeLightingTarget } from "./renderSafety";
import { structureOpacity } from "./structureDisplay";
import { captureView } from "./captureView";
import { voxelTimeSeries } from "./fmriDisplay";
import {
  INITIAL_SPECT_CUTOFF,
  clipEmissionToAnatomy,
  signalCameraTarget,
  spectCutoff,
  thresholdColormap,
} from "./spectDisplay";
import type {
  Prepared,
  Display,
  Probe,
  Region,
  Volume,
  Structures,
  ViewMode,
} from "./types";

export interface ViewerApi {
  readView: () => ViewPose;
  applyView: (pose: ViewPose, linkPosition: boolean) => void;
  home: () => void;
  zoom: (factor: number) => void;
  orient: (azimuth: number, elevation: number) => void;
  focus: (region: Region) => void;
  capture: () => void;
}
export interface ViewPose {
  azimuth: number;
  elevation: number;
  scale: number;
  mm: number[];
  pan: number[];
}
interface Props {
  onFrameChange?: (frame: number) => void;
  onModeChange?: (mode: ViewMode) => void;
  structures?: Structures | null;
  prepared: Prepared;
  overlay: Prepared | null;
  display: Display;
  onProbe: (probe: Probe) => void;
  onBusy: (busy: boolean) => void;
  onError: (error: string) => void;
  onContextLost: () => void;
  onLightingUnavailable: () => void;
  onReady: (api: ViewerApi | null) => void;
}

// NiiVue recomputes the pivot on every 3D draw. Change only the camera target,
// never the volume affine, to center off-axis SPECT signal after hiding its FOV.
class BrainRenderer extends Niivue {
  afterDraw = () => {};
  signalPivot: number[] | null = null;
  retired = false;
  override updateGLVolume() {
    if (!this.retired) super.updateGLVolume();
  }
  override refreshLayers(...args: Parameters<Niivue["refreshLayers"]>) {
    if (!this.retired) super.refreshLayers(...args);
  }
  override drawScene() {
    if (!this.retired) {
      const result = super.drawScene();
      this.afterDraw();
      return result;
    }
  }
  override setPivot3D() {
    super.setPivot3D();
    if (this.signalPivot) this.pivot3D = [...this.signalPivot];
  }
}

export default function BrainCanvas({
  onFrameChange,
  onModeChange,
  structures,
  prepared,
  overlay,
  display,
  onProbe,
  onBusy,
  onError,
  onContextLost,
  onLightingUnavailable,
  onReady,
}: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [tiles, setTiles] = useState<
    { mode: ViewMode; left: number; top: number }[]
  >([]);
  const nvRef = useRef<BrainRenderer | null>(null);
  const callbacks = useRef({
    onFrameChange,
    onProbe,
    onBusy,
    onError,
    onReady,
    onContextLost,
    onLightingUnavailable,
  });
  callbacks.current = {
    onFrameChange,
    onProbe,
    onBusy,
    onError,
    onReady,
    onContextLost,
    onLightingUnavailable,
  };
  const settings = useRef(display);
  settings.current = display;
  const descriptors = useRef<Volume[]>([]);
  const interaction = useRef({ x: 0, y: 0 });
  const renderReady = useRef(false);
  const lossReported = useRef(false);
  const transferKey = useRef("");
  const signalPivot = useRef<number[] | null>(null);
  const transferName = `pie-spect-${prepared.scan.id}`;

  function reportLostContext() {
    if (lossReported.current) return;
    lossReported.current = true;
    renderReady.current = false;
    callbacks.current.onContextLost();
  }

  async function applyLighting() {
    const nv = nvRef.current;
    if (!nv || !renderReady.current || !nv.volumes.length) return;
    try {
      if (nv.gl.isContextLost()) {
        reportLostContext();
        return;
      }
      if (
        settings.current.illumination &&
        !probeLightingTarget(nv.gl, nv.volumes[0].hdr!.dims.slice(1, 4))
      ) {
        callbacks.current.onLightingUnavailable();
        return;
      }
      await nv.setVolumeRenderIllumination(
        settings.current.illumination ? 0.6 : 0,
      );
      if (nvRef.current !== nv) return;
      if (nv.gl.isContextLost()) reportLostContext();
    } catch (e) {
      if (nvRef.current !== nv) return;
      if (nv.gl.isContextLost()) reportLostContext();
      else
        callbacks.current.onError(
          `The graphics renderer failed: ${e instanceof Error ? e.message : String(e)}`,
        );
    }
  }

  function applyDisplay() {
    const nv = nvRef.current;
    if (!nv || !renderReady.current) return;
    if (nv.gl.isContextLost()) {
      reportLostContext();
      return;
    }
    const s = settings.current;
    nv.opts.atlasOutline = s.atlasOutline ? 1 : 0;
    // Keep native/full-signal slice views for alignment review. In the 3D
    // preview only, clip emission to the MRI foreground so extracranial uptake
    // cannot obscure the anatomical reference. This does not alter voxel data.
    nv.backgroundMasksOverlays = clipEmissionToAnatomy(
      !!prepared.context,
      s.mode,
      s.anatomyOpacity,
    )
      ? 1
      : 0;
    const modes = {
      "3d": nv.sliceTypeRender,
      multi: nv.sliceTypeMultiplanar,
      axial: nv.sliceTypeAxial,
      coronal: nv.sliceTypeCoronal,
      sagittal: nv.sliceTypeSagittal,
    };
    nv.setSliceType(modes[s.mode]);
    nv.opts.show3Dcrosshair = s.crosshair;
    nv.setCrosshairWidth(
      s.crosshair ? (prepared.scan.modality === "fMRI" ? 0.2 : 0.8) : 0,
    );
    const primaryIndex = descriptors.current.findIndex(
      (v) => v.role === "primary",
    );
    const primary = nv.volumes[primaryIndex];
    if (primary) {
      const cutoff = spectCutoff(prepared.scan.modality, s.mode, s.spectCutoff);
      nv.signalPivot = cutoff > 0 ? signalPivot.current : null;
      let colormap = s.colormap;
      if (cutoff > 0) {
        const key = `${s.colormap}:${cutoff}`;
        if (transferKey.current !== key) {
          cmapper.addColormap(
            transferName,
            thresholdColormap(cmapper.colormap(s.colormap), cutoff),
          );
          transferKey.current = key;
          nv.refreshColormaps();
        }
        colormap = transferName;
      }
      if (primary.colormap !== colormap) primary.setColormap(colormap);
      // setColormap can recalibrate: apply the user's window afterwards.
      primary.cal_min = s.window[0];
      primary.cal_max = s.window[1];
      primary.opacity = s.hideSignal && prepared.context ? 0 : s.opacity;
      if ((primary.nFrame4D ?? 1) > 1)
        nv.setFrame4D(
          primary.id,
          Math.min(s.frame, (primary.nFrame4D ?? 1) - 1),
        );
    }
    descriptors.current.forEach((v, i) => {
      if (v.role === "anatomy" && nv.volumes[i])
        nv.volumes[i].opacity =
          s.structures && structures && s.mode === "3d" ? 0 : s.anatomyOpacity;
      if (
        v.role === "primary" &&
        prepared.scan.modality === "MRI" &&
        s.structures &&
        structures &&
        s.mode === "3d"
      )
        nv.volumes[i].opacity = 0;
      if (v.role === "atlas" && nv.volumes[i])
        nv.volumes[i].opacity = s.atlas ? s.atlasOpacity : 0;
      if (v.role === "overlay" && nv.volumes[i])
        nv.volumes[i].opacity = s.overlayOpacity;
    });
    if (structures) {
      // Structure geometry lives in MRI RAS. It never replaces voxel data or labels.
      const offset = prepared.meshes.length + (overlay?.meshes.length ?? 0);
      structures.meshes.forEach((m, i) => {
        if (nv.meshes[offset + i])
          nv.meshes[offset + i].opacity = structureOpacity(m.key, s);
      });
      if (s.structures && s.mode === "3d") nv.backgroundMasksOverlays = 0;
    }
    const planes = [
      [0, 0],
      [90, 0],
      [0, 90],
    ];
    nv.setClipPlane(
      s.clip >= 100
        ? [2, 0, 0]
        : [-1 + (2 * s.clip) / 100, ...planes[s.clipAxis]],
    );
    nv.updateGLVolume();
    nv.drawScene();
  }

  useEffect(() => {
    let disposed = false;
    let loadFinished = false;
    let contextReleased = false;
    renderReady.current = false;
    lossReported.current = false;
    const element = canvas.current!;
    const lost = (event: Event) => {
      event.preventDefault();
      if (!disposed) reportLostContext();
    };
    element.addEventListener("webglcontextlost", lost);
    const nv = new BrainRenderer({
      backColor: [0.065, 0.079, 0.085, 1],
      fontColor: [0.77, 0.82, 0.82, 1],
      crosshairColor: [0.87, 0.74, 0.39, 0.85],
      show3Dcrosshair: false,
      isColorbar: false,
      isOrientCube: true,
      isRuler: false,
      isCornerOrientationText: true,
      isRadiologicalConvention: false,
      isSliceMM: true,
      textHeight: 0.025,
      loadingText: "",
      dragAndDropEnabled: false,
      multiplanarForceRender: true,
      multiplanarEqualSize: true,
      multiplanarLayout: 2,
      showLegend: false,
      logLevel: "error",
      renderOverlayBlend: 0.6,
    });
    nvRef.current = nv;
    let layoutKey = "";
    nv.afterDraw = () => {
      if (disposed || !renderReady.current || settings.current.mode !== "multi")
        return;
      const names: Record<number, ViewMode> = {
        0: "axial",
        1: "coronal",
        2: "sagittal",
        4: "3d",
      };
      const next = nv.screenSlices
        .filter((t) => names[t.axCorSag])
        .map((t) => ({
          mode: names[t.axCorSag],
          left: (t.leftTopWidthHeight[0] / element.width) * 100,
          top: (t.leftTopWidthHeight[1] / element.height) * 100,
        }));
      const key = JSON.stringify(next);
      if (key !== layoutKey) {
        layoutKey = key;
        setTiles(next);
      }
    };
    const registry = new Map(prepared.regions.map((r) => [r.id, r]));
    let previousProbe = "";
    let previousVoxel = "";
    let timeSeries: Probe["timeSeries"];
    nv.onFrameChange = (volume, frame) => {
      if (
        disposed ||
        !renderReady.current ||
        volume !==
          nv.volumes[descriptors.current.findIndex((v) => v.role === "primary")]
      )
        return;
      if (frame !== settings.current.frame)
        callbacks.current.onFrameChange?.(frame);
    };
    nv.onLocationChange = (location: unknown) => {
      if (disposed || !renderReady.current) return;
      const loc = location as { mm: number[]; values: { value: number }[] };
      const mm = Array.from(loc.mm).slice(0, 3);
      const atlasIndex = descriptors.current.findIndex(
        (v) => v.role === "atlas",
      );
      const primaryIndex = descriptors.current.findIndex(
        (v) => v.role === "primary",
      );
      let label =
        atlasIndex >= 0 ? Math.round(loc.values[atlasIndex]?.value ?? 0) : 0;
      let regionDistance: number | undefined;
      const atlas = nv.volumes[atlasIndex];
      if (!label && atlas && settings.current.mode === "3d") {
        // NiiVue's mm2vox/getValue address its RAS-reordered buffer, not the
        // source header's voxel axes. Use the matching RAS matrix and dimensions.
        const ras = Array.from(atlas.matRAS!);
        const affine = [
          ras.slice(0, 4),
          ras.slice(4, 8),
          ras.slice(8, 12),
          ras.slice(12, 16),
        ];
        const nearby = nearbyAtlasLabel(
          mm,
          Array.from(atlas.mm2vox(mm)),
          affine,
          Array.from(atlas.dimsRAS!).slice(1, 4),
          (x, y, z) => atlas.getValue(x, y, z),
        );
        if (nearby && registry.has(nearby.id)) {
          label = nearby.id;
          regionDistance = nearby.distance;
        }
      }
      const value = loc.values[primaryIndex]?.value;
      const primary = nv.volumes[primaryIndex];
      if (
        prepared.scan.modality === "fMRI" &&
        primary &&
        (primary.nFrame4D ?? 0) > 1
      ) {
        const voxel = Array.from(primary.mm2vox(mm))
          .slice(0, 3)
          .map(Math.round);
        const voxelKey = voxel.join(",");
        if (voxelKey !== previousVoxel) {
          previousVoxel = voxelKey;
          const sampled = voxelTimeSeries(
            voxel,
            Array.from(primary.dimsRAS!).slice(1, 4),
            primary.nFrame4D!,
            (x, y, z, frame) => primary.getValue(x, y, z, frame),
          );
          // The click may fall between voxel centers. Export the center of the
          // exact RAS-buffer voxel sampled, not a rounded display/click location.
          timeSeries = sampled
            ? {
                ...sampled,
                mm: Array.from(
                  primary.vox2mm(sampled.voxel, primary.matRAS!),
                ).slice(0, 3),
              }
            : undefined;
        }
      } else timeSeries = undefined;
      const key = `${label}:${mm.map((x) => x.toFixed(1)).join()}:${value}`;
      if (key === previousProbe) return;
      previousProbe = key;
      callbacks.current.onProbe({
        mm,
        region: registry.get(label) ?? null,
        region_distance_mm: regionDistance,
        value: Number.isFinite(value) ? value : null,
        timeSeries,
      });
    };
    async function start() {
      callbacks.current.onBusy(true);
      try {
        await nv.attachToCanvas(canvas.current!);
        if (disposed) return;
        const volumes = prepared.volumes.map((v) => ({ ...v }));
        const extra = prepared.extra?.find(
          (e) => e.key === settings.current.metric,
        );
        const primary = volumes.find((v) => v.role === "primary");
        if (extra && primary) {
          primary.url = extra.url;
          primary.name = extra.name;
        }
        if (overlay) {
          const ov = overlay.volumes.find((v) => v.role === "primary");
          if (ov)
            volumes.splice(
              volumes.findIndex((v) => v.role === "atlas") < 0
                ? volumes.length
                : volumes.findIndex((v) => v.role === "atlas"),
              0,
              {
                ...ov,
                role: "overlay",
                colormap:
                  overlay.scan.modality === "fMRI" ? "warm" : ov.colormap,
                opacity: 0.6,
              },
            );
        }
        descriptors.current = volumes;
        if (volumes.length)
          await nv.loadVolumes(
            volumes.map((v) => ({
              ...v,
              name: v.url.startsWith("blob:")
                ? v.name
                : decodeURIComponent(v.url.split("/").pop()!),
              colorbarVisible: false,
              trustCalMinMax: true,
            })),
          );
        if (disposed) return;
        signalPivot.current = null;
        if (
          prepared.scan.modality === "SPECT" &&
          !overlay &&
          nv.volumes.length === 1
        ) {
          const image = nv.volumes[0];
          const window = settings.current.window;
          const target = signalCameraTarget(
            Array.from(image.dimsRAS!).slice(1, 4),
            (x, y, z) => image.getValue(x, y, z),
            window[0] + INITIAL_SPECT_CUTOFF * (window[1] - window[0]),
            window[1],
          );
          if (target)
            signalPivot.current = Array.from(
              image.vox2mm(target, image.matRAS!),
            );
        }
        const meshes = [...prepared.meshes, ...(overlay?.meshes ?? [])];
        if (meshes.length)
          await nv.loadMeshes(
            meshes.map((m) => ({
              ...m,
              name: decodeURIComponent(m.url.split("/").pop()!),
              rgba255: [215, 221, 210, 255],
              opacity: 1,
            })),
          );
        if (structures) {
          for (const mesh of structures.meshes) {
            if (disposed) return;
            await nv.addMeshFromUrl({
              url: mesh.url,
              name: mesh.url.split("/").pop(),
              rgba255: [...mesh.color, 255] as [number, number, number, number],
              opacity: 0,
            });
          }
        }
        if (disposed) return;
        const atlasIndex = volumes.findIndex((v) => v.role === "atlas");
        if (atlasIndex >= 0) {
          const regions = prepared.regions;
          // Explicit label intent selects NiiVue's integer-label/outline shader.
          // Legacy cached atlases lacked it and were treated as scalar maps.
          nv.volumes[atlasIndex].hdr!.intent_code = 1002;
          nv.volumes[atlasIndex].setColormapLabel({
            I: [0, ...regions.map((r) => r.id)],
            R: [0, ...regions.map((r) => r.color[0])],
            G: [0, ...regions.map((r) => r.color[1])],
            B: [0, ...regions.map((r) => r.color[2])],
            A: [0, ...regions.map(() => 255)],
            labels: ["Background", ...regions.map((r) => r.name)],
          });
        }
        nv.setRenderAzimuthElevation(120, 18);
        nv.setScale(1.6);
        renderReady.current = true;
        applyDisplay();
        if (settings.current.illumination) await applyLighting();
        if (disposed) return;
        if (nv.gl.isContextLost()) {
          reportLostContext();
          return;
        }
        callbacks.current.onReady({
          readView: () => ({
            azimuth: nv.scene.renderAzimuth,
            elevation: nv.scene.renderElevation,
            scale: nv.scene.volScaleMultiplier,
            mm: Array.from(nv.frac2mm(nv.scene.crosshairPos)),
            pan: Array.from(nv.scene.pan2Dxyzmm),
          }),
          applyView: (pose, linkPosition) => {
            nv.scene.renderAzimuth = pose.azimuth;
            nv.scene.renderElevation = pose.elevation;
            nv.scene.volScaleMultiplier = pose.scale;
            if (linkPosition) {
              nv.scene.crosshairPos = nv.mm2frac(
                pose.mm as [number, number, number],
                0,
                true,
              );
              nv.scene.pan2Dxyzmm = new Float32Array(pose.pan);
            }
            nv.drawScene();
          },
          home: () => {
            nv.setRenderAzimuthElevation(120, 18);
            nv.setScale(1.6);
            nv.scene.crosshairPos = new Float32Array([0.5, 0.5, 0.5]);
            nv.drawScene();
          },
          zoom: (factor) =>
            nv.setScale(
              Math.max(0.3, Math.min(4, nv.scene.volScaleMultiplier * factor)),
            ),
          orient: (a, e) => nv.setRenderAzimuthElevation(a, e),
          focus: (region) => {
            nv.scene.crosshairPos = nv.mm2frac(
              (region.focus_mm ?? region.center_mm) as [number, number, number],
              0,
              true,
            );
            nv.drawScene();
            nv.createOnLocationChange();
          },
          capture: () => {
            nv.drawScene();
            captureView(element, prepared, settings.current, structures);
          },
        });
        // Browser inspection hook: local-only, holds just this renderer.
        if (import.meta.env.DEV)
          (window as unknown as { __pieViewer: Niivue }).__pieViewer = nv;
      } catch (e) {
        if (import.meta.env.DEV) console.error("PIE viewer load failed", e);
        if (!disposed)
          callbacks.current.onError(e instanceof Error ? e.message : String(e));
      } finally {
        if (!disposed) callbacks.current.onBusy(false);
      }
    }
    const releaseContext = () => {
      if (contextReleased) return;
      contextReleased = true;
      element
        .getContext("webgl2")
        ?.getExtension("WEBGL_lose_context")
        ?.loseContext();
    };
    void start().finally(() => {
      loadFinished = true;
      if (disposed) releaseContext();
    });
    return () => {
      disposed = true;
      nv.retired = true;
      nv.afterDraw = () => {};
      renderReady.current = false;
      element.removeEventListener("webglcontextlost", lost);
      callbacks.current.onReady(null);
      if (nvRef.current === nv) nvRef.current = null;
      nv.onLocationChange = () => {};
      nv.cleanup();
      delete cmapper.cluts[transferName];
      transferKey.current = "";
      // NiiVue's pending image/font loads are not cancellable. Stop their draw
      // callbacks immediately, but release GL only after startup has settled.
      // Otherwise a late load attempts a gradient pass on a destroyed context.
      // This keyed canvas is never reused by the next acquisition.
      if (loadFinished) releaseContext();
    };
    // Remount the WebGL context only when the data, overlay, or diffusion metric change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prepared, overlay, display.metric, structures]);

  useEffect(applyDisplay, [display]);
  useEffect(() => {
    void applyLighting();
  }, [display.illumination]);

  return (
    <div className="brain-canvas-host">
      <canvas
        ref={canvas}
        className="brain-canvas"
        aria-label="Interactive patient brain. Drag to rotate, scroll to zoom, click to inspect a region. Use the region list for keyboard access."
        tabIndex={0}
        onDoubleClickCapture={(e) => {
          e.preventDefault();
          e.stopPropagation();
          const nv = nvRef.current;
          if (!nv || !renderReady.current) return;
          if (display.mode !== "multi") {
            onModeChange?.("multi");
            return;
          }
          const rect = e.currentTarget.getBoundingClientRect();
          const x =
            ((e.clientX - rect.left) * e.currentTarget.width) / rect.width;
          const y =
            ((e.clientY - rect.top) * e.currentTarget.height) / rect.height;
          const tile = nv.screenSlices.find((t) => {
            const [l, top, w, h] = t.leftTopWidthHeight;
            return x >= l && x < l + w && y >= top && y < top + h;
          });
          const mode = (
            { 0: "axial", 1: "coronal", 2: "sagittal", 4: "3d" } as const
          )[tile?.axCorSag as 0 | 1 | 2 | 4];
          if (mode) onModeChange?.(mode);
        }}
        onPointerDown={(e) => {
          interaction.current = { x: e.clientX, y: e.clientY };
        }}
        onPointerUp={(e) => {
          if (
            display.mode !== "3d" ||
            Math.hypot(
              e.clientX - interaction.current.x,
              e.clientY - interaction.current.y,
            ) > 4 ||
            e.button !== 0
          )
            return;
          const nv = nvRef.current;
          if (!nv || !renderReady.current) return;
          const rect = e.currentTarget.getBoundingClientRect();
          nv.mousePos = [
            ((e.clientX - rect.left) * e.currentTarget.width) / rect.width,
            ((e.clientY - rect.top) * e.currentTarget.height) / rect.height,
          ];
          nv.uiData.mouseDepthPicker = true;
          nv.drawScene();
          nv.createOnLocationChange();
        }}
        onKeyDown={(e) => {
          const nv = nvRef.current;
          if (!nv || !renderReady.current) return;
          if (
            ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(e.key)
          ) {
            e.preventDefault();
            nv.setRenderAzimuthElevation(
              nv.scene.renderAzimuth +
                (e.key === "ArrowLeft" ? -10 : e.key === "ArrowRight" ? 10 : 0),
              nv.scene.renderElevation +
                (e.key === "ArrowUp" ? 10 : e.key === "ArrowDown" ? -10 : 0),
            );
          }
        }}
      />
      {display.mode === "multi" &&
        tiles.map((tile, i) => (
          <button
            key={`${tile.mode}-${i}`}
            className="tile-heading"
            style={{ left: `${tile.left}%`, top: `${tile.top}%` }}
            aria-label={`Enlarge ${tile.mode === "3d" ? "3D" : tile.mode} panel`}
            onClick={() => onModeChange?.(tile.mode)}
          >
            {tile.mode === "3d"
              ? "3D"
              : tile.mode[0].toUpperCase() + tile.mode.slice(1)}{" "}
            <span aria-hidden="true">↗</span>
          </button>
        ))}
    </div>
  );
}

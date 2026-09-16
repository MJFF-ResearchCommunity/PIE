"""Local-only scan API and production frontend hosting."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .catalog import Catalog
from .images import ImageStore

log = logging.getLogger(__name__)


def create_app(repo: Path | None = None, ppmi: Path | None = None, manifest: Path | None = None,
               cache: Path | None = None, require_cache_mount: bool = False, sample_plan: Path | None = None):
    if require_cache_mount and cache is None:
        raise ValueError("--require-cache-mount needs --cache-dir naming the mounted volume; "
                         "the default cache location is never assumed")
    if require_cache_mount and not cache.is_mount():
        raise ValueError("Required viewer cache filesystem is not mounted; no fallback")
    repo = (repo or Path(__file__).resolve().parents[3]).resolve()
    local_collection = repo / "Imaging/derived/viewer_collection/manifest.json"
    catalog = Catalog(repo, ppmi, manifest or (local_collection if local_collection.is_file() else None))
    for message in catalog.warnings:
        log.warning("Viewer index: %s", message)
    store = ImageStore(cache or repo / "Imaging/derived/viewer_cache")
    plan_dir = sample_plan or repo / "Imaging/derived/viewer_sample_plan"
    guide = repo / "documentation/viewer_next_downloads.md"
    app = FastAPI(title="PIE Brain Explorer", version="0.1.0", docs_url="/api/docs", redoc_url=None)
    # Block DNS-rebinding access to a service that holds local research images.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"])
    app.state.catalog, app.state.store = catalog, store

    @app.middleware("http")
    async def local_headers(request, call_next):
        if require_cache_mount and request.url.path.startswith("/api/") and not cache.is_mount():
            return JSONResponse(status_code=503, content={
                "detail": "Viewer cache filesystem is not mounted; restore it before retrying. No fallback."})
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/health")
    def health():
        return {"status": "ok", "subjects": len(catalog.subjects), "scans": len(catalog.scans)}

    @app.get("/api/download-guide")
    def download_guide():
        if not guide.is_file():
            raise HTTPException(404, "Download checklist not installed")
        return FileResponse(guide, media_type="text/markdown", filename="PIE-next-downloads.md")

    @app.get("/api/catalog")
    def list_catalog():
        return catalog.public()

    @app.get("/api/scans/{scan_id}")
    def scan(scan_id: str):
        selected = catalog.scans.get(scan_id)
        if not selected:
            raise HTTPException(404, "Scan not found")
        try:
            return store.prepare(selected)
        except (ValueError, OSError) as e:
            raise HTTPException(422, f"Cannot prepare scan: {e}") from e

    @app.get("/api/assets/{asset_id}/{filename}")
    def asset(asset_id: str, filename: str):
        path = store.assets.get(asset_id)
        if not path or path.name != filename or not path.is_file():
            raise HTTPException(404, "Asset not found; load its scan first")
        return FileResponse(path, media_type="application/octet-stream")

    @app.get("/api/scans/{scan_id}/anatomy-preview")
    def anatomy_preview(scan_id: str):
        from .anatomy import prepare_preview
        selected = catalog.scans.get(scan_id)
        preview = catalog.anatomy_previews.get(scan_id)
        if not selected or not preview:
            raise HTTPException(404, "No unambiguous MRI reference and transform for this SPECT")
        try:
            return prepare_preview(store, selected, preview)
        except (ValueError, OSError, RuntimeError) as e:
            raise HTTPException(422, f"Cannot prepare anatomy preview: {e}") from e

    @app.get("/api/sample-plan")
    def read_sample_plan():
        path = plan_dir / "plan.json"
        if not path.is_file():
            return {"available": False, "download_guide": guide.is_file(),
                    "message": f"No sample plan at {path}. Run python -m pie.imaging.viewer sample-plan to build it."}
        return {"available": True, "download_guide": guide.is_file(), **json.loads(path.read_text())}

    @app.get("/api/scans/{scan_id}/structures")
    def structures(scan_id: str):
        from .structures import prepare_structures
        selected = catalog.scans.get(scan_id)
        if not selected or selected.modality != "MRI":
            raise HTTPException(404, "Select an indexed MRI anatomical reference")
        try:
            return prepare_structures(store, selected)
        except (ValueError, OSError) as e:
            raise HTTPException(422, f"Cannot prepare segmentation boundaries: {e}") from e

    @app.get("/api/comparison/{baseline_id}/{followup_id}")
    def comparison(baseline_id: str, followup_id: str):
        from .comparison import prepare_comparison
        baseline, followup = catalog.scans.get(baseline_id), catalog.scans.get(followup_id)
        if not baseline or not followup:
            raise HTTPException(404, "Comparison acquisition not found")
        try:
            return prepare_comparison(store, baseline, followup)
        except (ValueError, OSError, RuntimeError) as e:
            raise HTTPException(422, f"Cannot prepare comparison: {e}") from e

    frontend = repo / "brain-viewer/dist"
    if frontend.is_dir():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app

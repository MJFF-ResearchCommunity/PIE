"""Explicit viewer cache routing and no-fallback mount safeguards."""
import sys
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from pie.imaging.viewer import __main__, server


def test_explicit_cache_requires_mount_before_catalog_read(tmp_path):
    with pytest.raises(ValueError, match="not mounted; no fallback"):
        server.create_app(tmp_path, cache=tmp_path / "cache", require_cache_mount=True)
    with pytest.raises(ValueError, match="not mounted; no fallback"):
        server.create_app(tmp_path, require_cache_mount=True)


def test_lost_cache_mount_blocks_requests_without_fallback(tmp_path, monkeypatch):
    cache = tmp_path / "bounded-cache"
    cache.mkdir()
    monkeypatch.setattr(Path, "is_mount", lambda self: self == cache)
    app = server.create_app(tmp_path, cache=cache, require_cache_mount=True)
    assert app.state.store.cache == cache
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        monkeypatch.setattr(Path, "is_mount", lambda self: False)
        response = client.get("/api/health")
        assert response.status_code == 503
        assert "No fallback" in response.json()["detail"]
        assert list(cache.iterdir()) == []


def test_cli_forwards_cache_parameters_without_machine_paths(tmp_path, monkeypatch):
    import uvicorn
    captured = {}
    def create(*args, **kwargs):
        captured.update(kwargs)
        return "test-app"
    monkeypatch.setattr(server, "create_app", create)
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: captured.update(app=app, **kwargs))
    cache = tmp_path / "caller-selected-cache"
    monkeypatch.setattr(sys, "argv", ["viewer", "serve", "--cache-dir", str(cache),
                                      "--require-cache-mount"])
    __main__.main()
    assert captured == dict(cache=cache, require_cache_mount=True, app="test-app",
                            host="127.0.0.1", port=8765)

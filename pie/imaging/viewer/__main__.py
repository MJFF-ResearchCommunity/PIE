"""python -m pie.imaging.viewer [serve|sample-plan]"""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="PIE Brain Explorer — local neuroimaging")
    parser.add_argument("command", choices=("serve", "sample-plan"), nargs="?", default="serve")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--ppmi-dir", type=Path)
    parser.add_argument("--manifest", type=Path, help="Additional scans in an explicit local JSON manifest")
    parser.add_argument("--cache-dir", type=Path, help="Viewer-only prepared-image cache directory")
    parser.add_argument("--require-cache-mount", action="store_true",
                        help="Refuse cache access unless --cache-dir is a mounted filesystem")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "sample-plan":
        from .sample_plan import build_plan
        build_plan(args.repo, args.ppmi_dir or args.repo / "PPMI", args.output or args.repo / "Imaging/derived/viewer_sample_plan")
    else:
        import uvicorn
        from .server import create_app
        uvicorn.run(create_app(args.repo, args.ppmi_dir, args.manifest, cache=args.cache_dir,
                              require_cache_mount=args.require_cache_mount),
                    host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()

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
    parser.add_argument("--output", type=Path,
                        help="Sample-plan folder, written by sample-plan and read by serve "
                             "(default Imaging/derived/viewer_sample_plan)")
    parser.add_argument("--if-missing", action="store_true",
                        help="sample-plan: keep an existing plan.json instead of rebuilding it")
    args = parser.parse_args()
    if args.command == "sample-plan":
        output = args.output or args.repo / "Imaging/derived/viewer_sample_plan"
        if args.if_missing and (output / "plan.json").is_file():
            print(f"Keeping existing sample plan: {output / 'plan.json'}")
            return
        from .sample_plan import build_plan
        build_plan(args.repo, args.ppmi_dir or args.repo / "PPMI", output)
        return
    import uvicorn
    from . import server
    try:
        app = server.create_app(args.repo, args.ppmi_dir, args.manifest, cache=args.cache_dir,
                                require_cache_mount=args.require_cache_mount, sample_plan=args.output)
    except ValueError as error:  # configuration or manifest problem: a message, not a traceback
        parser.exit(2, f"PIE Brain Explorer: {error}\n")
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()

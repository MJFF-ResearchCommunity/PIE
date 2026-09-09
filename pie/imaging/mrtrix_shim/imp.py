"""Minimal stand-in for the ``imp`` module removed in Python 3.12, for MRtrix3 3.0.x Python scripts (``dwi2response``,
``dwifslpreproc``, ...) whose launcher does ``imp.find_module`` / ``imp.load_module`` / ``imp.load_source``. Put this
directory on PYTHONPATH only for those subprocesses (pie.imaging.fba does)."""

import importlib.util
import os
import sys

PY_SOURCE, PKG_DIRECTORY = 1, 5


def find_module(name, path=None):
    for p in (path if path is not None else sys.path):
        cand = os.path.join(p, name)
        if os.path.isfile(os.path.join(cand, "__init__.py")):
            return None, cand, ("", "", PKG_DIRECTORY)
        if os.path.isfile(cand + ".py"):
            return open(cand + ".py"), cand + ".py", (".py", "r", PY_SOURCE)
    raise ImportError(f"No module named {name}")


def load_module(name, fp, pathname, description):
    if fp is not None:
        fp.close()
    if description[2] == PKG_DIRECTORY:
        spec = importlib.util.spec_from_file_location(name, os.path.join(pathname, "__init__.py"), submodule_search_locations=[pathname])
    else:
        spec = importlib.util.spec_from_file_location(name, pathname)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_source(name, path):
    return load_module(name, None, path, (".py", "r", PY_SOURCE))

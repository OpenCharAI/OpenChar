"""Locate the Inline Studio frontend Core serves on its own port (mirrors ComfyUI's frontend pkg).

Resolution order, most specific first:
  1. ``INLINE_FRONTEND_ROOT`` - a local SPA build dir (set directly or via ``main.py
     --front-end-root``); the dev loop - rebuild the UI locally without republishing the package.
  2. the installed ``omnichar_frontend`` package's ``static/`` dir - the default for end users
     (``pip install`` pulls the built frontend; no Node needed).
  3. ``None`` - Core runs API-only (no UI mounted).

A dir only counts when it actually holds an ``index.html``.
"""

from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
from types import ModuleType

#: New name first, old name still accepted: the wheel was renamed with the product, and an install
#: that predates the rename would otherwise serve no UI at all after an engine upgrade.
FRONTEND_MODULES = ("omnichar_frontend", "openchar_frontend")


def frontend_package() -> ModuleType | None:
    """The installed prebuilt-UI package under whichever name it carries, or None if absent."""
    for name in FRONTEND_MODULES:
        try:
            return import_module(name)
        except ModuleNotFoundError:
            continue
    return None


def package_static() -> Path | None:
    """The ``static/`` dir inside that package, wherever pip put it."""
    package = frontend_package()
    pkg_file = getattr(package, "__file__", None) if package else None
    return Path(pkg_file).parent / "static" if pkg_file else None


def _has_index(path: Path) -> bool:
    return (path / "index.html").is_file()


def resolve_frontend_root() -> str | None:
    env = os.environ.get("INLINE_FRONTEND_ROOT", "").strip()
    if env:
        root = Path(env)
        return str(root) if _has_index(root) else None

    static = package_static()
    if static is None:
        return None
    return str(static) if _has_index(static) else None

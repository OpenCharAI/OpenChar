"""Inline Core: the generation engine behind OpenChar Studio."""

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _source_version() -> str | None:
    """The checkout's own version, for a server run in place rather than installed."""
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        parsed = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    found = parsed.get("project", {}).get("version")
    return found if isinstance(found, str) else None


def _running_version() -> str:
    # The checkout wins outright: an editable install records its version once and then reports a
    # stale number after every bump, while a wheel install has no pyproject beside it to find.
    from_source = _source_version()
    if from_source:
        return from_source
    for distribution in ("openchar-core", "inline-core"):
        try:
            return version(distribution)
        except PackageNotFoundError:
            continue
    return "0.0.0"


__version__ = _running_version()

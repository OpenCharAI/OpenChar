"""GET /api: the Scalar reference, themed like the app and served entirely from this package."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from scalar_fastapi import AgentScalarConfig, Layout, Theme, get_scalar_api_reference

#: Bump with the vendored bundle; it is in the URL so the cache header can say immutable.
SCALAR_VERSION = "1.68.0"

_VENDOR = Path(__file__).parent / "vendor"
_BUNDLE = _VENDOR / "scalar.standalone.js"
_LOGO = _VENDOR / "logo.svg"
_SCRIPT_PATH = f"/api/scalar-{SCALAR_VERSION}.js"
_LOGO_PATH = "/api/logo.svg"

#: Pinned, never `@latest`. Reached only when the vendored copy was pruned from an install.
_CDN_FALLBACK = f"https://cdn.jsdelivr.net/npm/@scalar/api-reference@{SCALAR_VERSION}"

#: The version is in the filename, so a stale copy can never be the wrong one.
_IMMUTABLE = {"Cache-Control": "public, max-age=31536000, immutable"}

#: The app's tailwind tokens on Scalar's variables; dark only, and `#2e3026` is accent over panel.
SCALAR_CSS = """
:root, .light-mode, .dark-mode {
  --scalar-background-1: #16171b;
  --scalar-background-2: #1d1f24;
  --scalar-background-3: #2a2d34;
  --scalar-background-accent: #2e3026;
  --scalar-border-color: #2a2d34;
  --scalar-border-width: 1px;
  --scalar-color-1: #f4f4f5;
  --scalar-color-2: #a1a1aa;
  --scalar-color-3: #71717a;
  --scalar-color-accent: #dce775;
  --scalar-button-1: #dce775;
  --scalar-button-1-color: #16171b;
  --scalar-button-1-hover: #e9f09a;
  --scalar-link-color: #dce775;
  --scalar-link-color-hover: #e9f09a;
  --scalar-text-decoration: none;
  --scalar-text-decoration-hover: underline;
  --scalar-color-green: #34d399;
  --scalar-color-red: #ef4444;
  --scalar-color-yellow: #f59e0b;
  --scalar-color-blue: #60a5fa;
  --scalar-color-orange: #fb923c;
  --scalar-color-purple: #c084fc;
  --scalar-color-danger: #ef4444;
  --scalar-background-danger: #2a1b1b;
  --scalar-color-alert: #f59e0b;
  --scalar-background-alert: #2a2418;
  --scalar-scrollbar-color: rgba(244, 244, 245, 0.16);
  --scalar-scrollbar-color-active: rgba(244, 244, 245, 0.32);
  --scalar-radius: 6px;
  --scalar-radius-md: 6px;
  --scalar-radius-lg: 8px;
  --scalar-radius-xl: 12px;
  --scalar-radius-full: 9999px;
  --scalar-shadow-1: 0 1px 2px rgba(0, 0, 0, 0.4);
  --scalar-shadow-2: 0 8px 24px rgba(0, 0, 0, 0.5);
  --scalar-font: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --scalar-font-code: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
    'Liberation Mono', 'Courier New', monospace;
  --scalar-header-background-1: #16171b;
}
.t-doc__sidebar, .sidebar {
  --scalar-sidebar-background-1: #1d1f24;
  --scalar-sidebar-border-color: #2a2d34;
  --scalar-sidebar-color-1: #e4e4e7;
  --scalar-sidebar-color-2: #a1a1aa;
  --scalar-sidebar-color-active: #dce775;
  --scalar-sidebar-item-hover-color: currentColor;
  --scalar-sidebar-item-hover-background: #2a2d34;
  --scalar-sidebar-item-active-background: #2e3026;
  --scalar-sidebar-indent-border-active: #dce775;
  --scalar-sidebar-search-background: #16171b;
  --scalar-sidebar-search-border-color: #2a2d34;
  --scalar-sidebar-search-color: #71717a;
  --scalar-sidebar-search--color: #71717a;
}
body { background: #16171b; }
"""


def mount_reference(app: FastAPI) -> None:
    """Register /api and the two assets it loads. Must run before the SPA's catch-all mount."""

    if not _BUNDLE.is_file():
        # CI asserts the bundle is in the wheel, so reaching the CDN means an install lost it.
        logging.getLogger("inline_core").warning(
            "The vendored Scalar bundle is missing, so /api will load it from a CDN and needs "
            "a network. Re-vendor with scripts/vendor_scalar.py."
        )

    @app.get("/api", include_in_schema=False)
    async def api_reference() -> HTMLResponse:
        return get_scalar_api_reference(
            openapi_url=app.openapi_url or "/openapi.json",
            title=app.title,
            # DEFAULT injects its own palette, which ours would then fight for the same variables.
            theme=Theme.NONE,
            custom_css=SCALAR_CSS,
            layout=Layout.MODERN,
            dark_mode=True,
            force_dark_mode_state="dark",
            # The app has no light mode, so a toggle would only reveal an unthemed one.
            hide_dark_mode_toggle=True,
            # Otherwise the bundle fetches its fonts from the network and /api is not offline.
            with_default_fonts=False,
            scalar_js_url=_SCRIPT_PATH if _BUNDLE.is_file() else _CDN_FALLBACK,
            scalar_favicon_url=_LOGO_PATH,
            default_open_all_tags=True,
            telemetry=False,
            agent=AgentScalarConfig(disabled=True),
            show_developer_tools="never",
        )

    @app.get(_SCRIPT_PATH, include_in_schema=False)
    async def scalar_bundle() -> FileResponse:
        return FileResponse(_BUNDLE, media_type="text/javascript", headers=_IMMUTABLE)

    @app.get(_LOGO_PATH, include_in_schema=False)
    async def api_logo() -> FileResponse:
        return FileResponse(_LOGO, media_type="image/svg+xml", headers=_IMMUTABLE)

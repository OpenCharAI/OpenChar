"""Fetch the Scalar browser bundle into server/vendor/ so GET /api works with no network.

    PYTHONPATH=src .venv/bin/python scripts/vendor_scalar.py --version 1.68.0

Prints the sha256 and byte count for the provenance header. The bytes belong in git: a wheel built
without them serves a page whose own script tag 404s.
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

_CDN = "https://cdn.jsdelivr.net/npm/@scalar/api-reference@{version}/dist/browser/standalone.js"
_TARGET = Path(__file__).resolve().parents[1] / "src/inline_core/server/vendor/scalar.standalone.js"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()

    url = _CDN.format(version=args.version)
    print(f"fetching {url}")
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - a pinned CDN URL
        payload: bytes = response.read()
    # The entry point the page calls. Scanned over the whole payload: it is minified, and the
    # readable name does not appear until a quarter of the way in.
    if b"createApiReference" not in payload:
        raise SystemExit("that does not look like the Scalar bundle; refusing to write it")
    _TARGET.parent.mkdir(parents=True, exist_ok=True)
    _TARGET.write_bytes(payload)
    print(f"wrote {_TARGET}")
    print(f"  version: {args.version}")
    print(f"  bytes:   {len(payload)}")
    print(f"  sha256:  {hashlib.sha256(payload).hexdigest()}")
    print("Update the header in vendor/__init__.py and SCALAR_VERSION in server/reference.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

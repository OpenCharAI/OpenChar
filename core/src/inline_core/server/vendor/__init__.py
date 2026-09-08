"""Third-party assets served by GET /api, vendored so the reference works with no network.

    upstream: https://github.com/scalar/scalar
    package:  @scalar/api-reference
    version:  1.68.0
    source:   https://cdn.jsdelivr.net/npm/@scalar/api-reference@1.68.0/dist/browser/standalone.js
    bytes:    3738202
    sha256:   6a1407db14f57f7be9c98464b6d7e8899ba417c63b0d81363db2efb3e1022e1f
    taken:    2026-09-07

A published npm release rather than a branch, so the version pins it and there is no commit sha to
record; the sha256 is what a re-sync verifies against. Verbatim, byte for byte, including its
trailing sourceMappingURL comment. Re-vendor with ``python scripts/vendor_scalar.py --version X``
and bump ``SCALAR_VERSION`` in ``server/reference.py`` in the same commit; never hand-edit it.

``logo.svg`` is a copy of ``src/renderer/assets/logo.svg`` from the Studio repo, which is outside
this subtree and so cannot be read at build time. It is the /api favicon.
"""

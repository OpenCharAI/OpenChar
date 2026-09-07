"""The Z-Image (Alibaba Tongyi) runtime: a diffusers-backed text-to-image / img2img runner.

Optional subpackage. Nothing is re-exported here on purpose: `requirements` is torch-free and
`models/controlspace.py` imports it on the runtime-less path, so a re-export of the runner would
pull torch in through this file and take the whole server down before it could skip the model.
`server.bootstrap` imports the runner from `zimage.runner` directly, best-effort.
"""

from __future__ import annotations

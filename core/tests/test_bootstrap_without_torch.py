"""A runtime-less install must still boot: the models are skipped, the server is not.

`requirements.txt` offers `openchar-core[server]` for a hosted-only setup with no local GPU, and
this is the only test holding that promise. It was broken in a released build: Control Space is
registered unguarded on the belief it is torch-free, and importing it reached `zimage.requirements`
through `zimage/__init__.py`, which eagerly imported the runner and torch.

Run in a subprocess because blocking an import means surgery on `sys.modules`, and doing that in
the test process leaves every later test importing a half-torn-down `inline_core`.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

_BLOCK_TORCH = """
import sys

class _NoTorch:
    def find_spec(self, name, path=None, target=None):
        # ModuleNotFoundError, not ImportError: that is what an absent module raises, and
        # `device.detect` catches only the narrower one.
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError("No module named 'torch'")
        return None

sys.meta_path.insert(0, _NoTorch())
"""


def _run(body: str) -> subprocess.CompletedProcess[str]:
    src = str(Path(__file__).resolve().parents[1] / "src")
    script = _BLOCK_TORCH + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": src, "PATH": "/usr/bin:/bin"},
        timeout=120,
    )


def test_control_space_imports_without_torch() -> None:
    """It is registered unguarded, so its import chain has to stay genuinely torch-free."""
    done = _run(
        """
        from inline_core.models.controlspace import ControlSpaceProvider
        assert ControlSpaceProvider is not None
        print("OK")
        """
    )

    assert done.returncode == 0, done.stderr
    assert "OK" in done.stdout


def test_the_model_registry_still_builds_with_no_runtime_installed() -> None:
    done = _run(
        """
        from inline_core.device.memory import MemoryPolicy
        from inline_core.graph.registry import Registry
        from inline_core.server.bootstrap import register_models

        registered, extensions = register_models(Registry(), None, MemoryPolicy())
        # The torch-backed runners are skipped; Control Space survives by being torch-free.
        assert "controlSpace" in registered, registered
        assert "zimage" not in registered, registered
        assert extensions == []
        print("OK")
        """
    )

    assert done.returncode == 0, done.stderr
    assert "OK" in done.stdout

from __future__ import annotations

import pytest

from inline_core.config import server_host, server_port


def test_server_host_defaults_to_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INLINE_HOST", raising=False)
    assert server_host() == "127.0.0.1"


def test_server_host_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INLINE_HOST", "0.0.0.0")
    assert server_host() == "0.0.0.0"


def test_server_port_defaults_and_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INLINE_PORT", raising=False)
    assert server_port() == 8848
    monkeypatch.setenv("INLINE_PORT", "9000")
    assert server_port() == 9000


def test_server_port_falls_back_on_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INLINE_PORT", "not-a-port")
    assert server_port() == 8848


def test_assets_dir_defaults_to_the_old_store_and_reads_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path

    from inline_core.config import assets_dir

    monkeypatch.delenv("INLINE_ASSET_DIR", raising=False)
    # The default must stay ./.inline-assets, or an existing install loses sight of its uploads.
    assert assets_dir() == Path(".inline-assets")
    monkeypatch.setenv("INLINE_ASSET_DIR", "/opt/run/assets")
    assert assets_dir() == Path("/opt/run/assets")


def test_characters_dir_defaults_under_the_models_root_and_reads_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path

    from inline_core.config import characters_dir

    monkeypatch.setenv("INLINE_MODELS_DIR", "/m")
    monkeypatch.delenv("INLINE_CHARACTERS_DIR", raising=False)
    assert characters_dir() == Path("/m/characters")
    monkeypatch.setenv("INLINE_CHARACTERS_DIR", "/opt/run/characters")
    assert characters_dir() == Path("/opt/run/characters")


def test_run_data_dir_defaults_to_the_data_dir_and_reads_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cloud volume shares the data dir for its fetched model configs, so takes, the run database
    and per-run caches need a place of their own; without the setting nothing moves."""
    from pathlib import Path

    from inline_core.config import run_data_dir

    monkeypatch.setenv("INLINE_DATA_DIR", "/d")
    monkeypatch.delenv("INLINE_RUN_DATA_DIR", raising=False)
    assert run_data_dir() == Path("/d")
    monkeypatch.setenv("INLINE_RUN_DATA_DIR", "/opt/run/data")
    assert run_data_dir() == Path("/opt/run/data")


def test_per_run_caches_follow_the_run_data_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """The character payload cache and prompt embeddings are built from what a person gave a run."""
    from pathlib import Path

    from inline_core.characters import apply
    from inline_core.models.flux2 import embeds

    monkeypatch.setenv("INLINE_DATA_DIR", "/shared")
    monkeypatch.setenv("INLINE_RUN_DATA_DIR", "/private")
    assert apply._cache_root() == Path("/private/characters")
    assert embeds._root() == Path("/private/embeds/flux2")

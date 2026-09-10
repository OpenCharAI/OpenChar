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

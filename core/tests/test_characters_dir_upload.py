"""A character imported over /upload/character lands in INLINE_CHARACTERS_DIR, not a models root."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from inline_core.characters import charfile as cf
from inline_core.characters import library
from inline_core.server.app import create_app
from inline_core.studio.store import StudioStore


# Deliberately without the encoder fixture test_characters_rpc uses: importing a .char needs no
# scoring models, and a fixture that skips without them would skip this too.
@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("INLINE_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("INLINE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("INLINE_EXTRA_MODELS_DIRS", raising=False)
    monkeypatch.delenv("INLINE_CHARACTERS_DIR", raising=False)
    # models_dirs() always appends the relative ./models, so the checkout's real one leaks in.
    monkeypatch.chdir(tmp_path)
    store = StudioStore(tmp_path / "appdata", tmp_path / "workspace")
    app = create_app(
        studio_store=store,
        asset_dir=str(tmp_path / "assets"),
        takes_dir=str(tmp_path / "takes"),
    )
    with TestClient(app) as c:
        yield c


def _char_bytes(tmp_path: Path) -> bytes:
    manifest = cf.Manifest(
        char_id="7f1c0d2e-0000-4000-8000-000000000002",
        name="Ada",
        created_at=1755000000,
        modified_at=1755000000,
        text={"path": "text/description.md", "sha256": cf.sha256_bytes(b"green jacket")},
    )
    manifest.refs.append({"path": "refs/000.png", "sha256": cf.sha256_bytes(b"x")})
    members = {"refs/000.png": b"x", "text/description.md": b"green jacket"}
    source = tmp_path / "upload-source.char"
    cf.write(source, cf.CharDoc(manifest=manifest, members=members))
    return source.read_bytes()


def _import(client: TestClient, tmp_path: Path) -> str:
    imported = client.post("/upload/character?name=Ada.char", content=_char_bytes(tmp_path))
    body = imported.json()
    assert body["ok"] is True
    return str(body["value"]["file"])


def test_with_no_characters_dir_an_import_lands_where_it_always_did(
    client: TestClient, tmp_path: Path
) -> None:
    landed = _import(client, tmp_path)
    assert (tmp_path / "models" / "characters" / landed).is_file()


def test_an_import_lands_in_the_characters_dir_and_not_the_models_root(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private = tmp_path / "private"
    monkeypatch.setenv("INLINE_CHARACTERS_DIR", str(private))
    landed = _import(client, tmp_path)
    assert (private / landed).is_file()
    # The point of the knob on a cloud worker: a shared models root never receives the upload.
    assert not (tmp_path / "models" / "characters" / landed).exists()


def test_an_imported_character_resolves_by_name_for_character_load(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private = tmp_path / "private"
    monkeypatch.setenv("INLINE_CHARACTERS_DIR", str(private))
    landed = _import(client, tmp_path)
    # character/load and the node cache both go through library.resolve.
    assert library.resolve(landed) == private / landed

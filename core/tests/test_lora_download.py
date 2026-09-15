"""The Trainer's LoRA download route: the browser has no filesystem, so a finished run's
.safetensors is fetched over GET /download/lora/{run_id} rather than by copying a path."""

from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from inline_core.graph.registry import build_default_registry
from inline_core.server.app import create_app
from inline_core.studio import training_store as ts
from inline_core.studio.store import StudioStore


@pytest.fixture
def client(tmp_path, monkeypatch):
    # The route resolves the file under config.models_dir(); point that at the test models root so
    # it agrees with where a run's LoRA would actually be written.
    models = tmp_path / "models"
    (models / "loras").mkdir(parents=True)
    monkeypatch.setenv("INLINE_MODELS_DIR", str(models))

    store = StudioStore(tmp_path / "appdata", tmp_path / "workspace")
    app = create_app(
        registry=build_default_registry(),
        studio_store=store,
        asset_dir=str(tmp_path / "assets"),
        models_root=str(models),
        takes_dir=str(tmp_path / "takes"),
    )
    with TestClient(app) as c:
        yield c, store, models


def _finished_run(store: StudioStore, rel: str) -> str:
    # The project connection lives on the server thread (opened by project:create), so write the
    # run row over our own connection to the same project.db to avoid SQLite's thread affinity.
    conn = sqlite3.connect(str(store.folder() / "project.db"), isolation_level=None)
    conn.row_factory = sqlite3.Row
    dataset = ts.create_dataset(conn, "chars", "sks")
    run = ts.create_run(conn, dataset["id"], "my-run", {"baseMode": "raw"})
    ts.update_run(conn, run["id"], {"status": "done", "outputLoraPath": rel})
    conn.close()
    return run["id"]


def test_download_streams_the_lora_as_an_attachment(client) -> None:
    c, store, models = client
    args = [{"name": "F", "parentDir": None}]
    assert c.post("/rpc", json={"channel": "project:create", "args": args}).json()["ok"] is True

    (models / "loras" / "my-run.safetensors").write_bytes(b"LORA-BYTES")
    run_id = _finished_run(store, "loras/my-run.safetensors")

    res = c.get(f"/download/lora/{run_id}")
    assert res.status_code == 200
    assert res.content == b"LORA-BYTES"
    assert "attachment" in res.headers.get("content-disposition", "")
    assert "my-run.safetensors" in res.headers.get("content-disposition", "")


def test_unknown_run_is_404(client) -> None:
    c, _store, _models = client
    c.post("/rpc", json={"channel": "project:create", "args": [{"name": "F", "parentDir": None}]})
    assert c.get("/download/lora/does-not-exist").status_code == 404


def test_path_traversal_is_refused(client) -> None:
    c, store, _models = client
    c.post("/rpc", json={"channel": "project:create", "args": [{"name": "F", "parentDir": None}]})
    # A run whose stored path tries to escape loras/ must not serve an arbitrary file.
    run_id = _finished_run(store, "loras/../../secret.txt")
    assert c.get(f"/download/lora/{run_id}").status_code in (403, 404)


def test_a_lora_in_the_private_folder_downloads(client, tmp_path, monkeypatch) -> None:
    """A worker that keeps trained adapters off its shared models root records them by absolute
    path; the route serves from that folder instead of models/loras."""
    c, store, _models = client
    private = tmp_path / "private-loras"
    private.mkdir()
    monkeypatch.setenv("INLINE_TRAINED_LORAS_DIR", str(private))
    c.post("/rpc", json={"channel": "project:create", "args": [{"name": "F", "parentDir": None}]})

    (private / "my-run.safetensors").write_bytes(b"PRIVATE")
    run_id = _finished_run(store, str(private / "my-run.safetensors"))

    res = c.get(f"/download/lora/{run_id}")
    assert res.status_code == 200
    assert res.content == b"PRIVATE"


def test_the_private_folder_does_not_widen_what_can_be_read(client, tmp_path, monkeypatch) -> None:
    c, store, models = client
    private = tmp_path / "private-loras"
    private.mkdir()
    monkeypatch.setenv("INLINE_TRAINED_LORAS_DIR", str(private))
    c.post("/rpc", json={"channel": "project:create", "args": [{"name": "F", "parentDir": None}]})

    # A recorded path outside the configured folder, even one under models/loras, is not served.
    (models / "loras" / "someone-else.safetensors").write_bytes(b"NOT YOURS")
    run_id = _finished_run(store, "loras/someone-else.safetensors")
    assert c.get(f"/download/lora/{run_id}").status_code in (403, 404)

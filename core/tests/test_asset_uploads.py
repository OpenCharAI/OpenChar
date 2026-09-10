"""Uploads end to end: stored where INLINE_ASSET_DIR says, and usable by id through /v1/runs."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from inline_core.server.app import (
    _resolve_uploads,  # pyright: ignore[reportPrivateUsage]
    create_app,
)
from inline_core.server.assets import AssetStore, is_asset_id

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("INLINE_ASSET_DIR", str(tmp_path / "uploads"))
    monkeypatch.chdir(tmp_path)
    with TestClient(create_app(takes_dir=str(tmp_path / "takes"))) as c:
        yield c


def _graph(asset: dict[str, Any]) -> dict[str, Any]:
    node = {"id": "img", "type": "input/image", "params": {"asset": asset}, "inputs": {}}
    return {"schemaVersion": 1, "nodes": [node]}


def _upload(client: TestClient, data: bytes = PNG) -> str:
    response = client.post("/v1/assets", content=data, headers={"content-type": "image/png"})
    assert response.status_code == 200
    return str(response.json()["id"])


def _status(client: TestClient, run_id: str) -> str:
    deadline = time.monotonic() + 10
    status = ""
    while time.monotonic() < deadline:
        status = str(client.get(f"/v1/runs/{run_id}").json()["status"])
        if status in ("done", "error", "cancelled"):
            return status
        time.sleep(0.05)
    return status


def test_an_upload_lands_in_the_configured_store(client: TestClient, tmp_path: Path) -> None:
    asset_id = _upload(client)
    assert (tmp_path / "uploads" / asset_id).read_bytes() == PNG


# The reason this exists: an asset ref used to be accepted at submit and then read by nothing,
# because every reader opens a path. The run has to get as far as done.
def test_a_run_can_use_an_upload_by_id(client: TestClient) -> None:
    asset_id = _upload(client)
    response = client.post(
        "/v1/runs", json={"graph": _graph({"ref": "asset", "id": asset_id}), "target": "img"}
    )
    assert response.status_code == 201
    assert _status(client, response.json()["runId"]) == "done"


def test_an_unknown_upload_is_refused_at_submit_not_mid_run(client: TestClient) -> None:
    missing = "sha256-" + "0" * 64
    response = client.post(
        "/v1/runs", json={"graph": _graph({"ref": "asset", "id": missing}), "target": "img"}
    )
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_graph"
    assert error["nodeId"] == "img"


@pytest.mark.parametrize(
    "bad", ["../../etc/passwd", "/etc/passwd", "sha256-XYZ", "", "sha256-" + "a" * 63]
)
def test_an_id_the_store_could_not_have_minted_is_refused(client: TestClient, bad: str) -> None:
    response = client.post(
        "/v1/runs", json={"graph": _graph({"ref": "asset", "id": bad}), "target": "img"}
    )
    assert response.status_code == 422


def test_a_path_ref_is_passed_through_untouched(client: TestClient, tmp_path: Path) -> None:
    image = tmp_path / "local.png"
    image.write_bytes(PNG)
    response = client.post(
        "/v1/runs", json={"graph": _graph({"ref": "path", "path": str(image)}), "target": "img"}
    )
    assert response.status_code == 201
    assert _status(client, response.json()["runId"]) == "done"


def test_resolve_rewrites_only_upload_refs_on_source_nodes(tmp_path: Path) -> None:
    store = AssetStore(tmp_path / "store")
    asset_id = store.put(PNG, "image/png").id
    graph: dict[str, Any] = {
        "schemaVersion": 1,
        "nodes": [
            {
                "id": "a",
                "type": "input/image",
                "params": {"asset": {"ref": "asset", "id": asset_id}},
                "inputs": {},
            },
            {
                "id": "p",
                "type": "input/image",
                "params": {"asset": {"ref": "path", "path": "/x.png"}},
                "inputs": {},
            },
            {
                "id": "o",
                "type": "some/node",
                "params": {"asset": {"ref": "asset", "id": "untouched"}},
                "inputs": {},
            },
        ],
    }
    out = _resolve_uploads(graph, store)
    by_id = {node["id"]: node for node in out["nodes"]}
    assert by_id["a"]["params"]["asset"] == {
        "ref": "path",
        "path": str((tmp_path / "store" / asset_id).resolve()),
    }
    assert by_id["p"]["params"]["asset"] == {"ref": "path", "path": "/x.png"}
    assert by_id["o"]["params"]["asset"] == {"ref": "asset", "id": "untouched"}
    # Pure: the caller's graph is not modified.
    assert graph["nodes"][0]["params"]["asset"]["ref"] == "asset"


def test_resolve_leaves_anything_that_is_not_a_graph_for_the_parser(tmp_path: Path) -> None:
    store = AssetStore(tmp_path / "store")
    assert _resolve_uploads(None, store) is None
    assert _resolve_uploads({"nodes": "nope"}, store) == {"nodes": "nope"}


def test_the_store_only_answers_for_ids_it_could_have_minted(tmp_path: Path) -> None:
    store = AssetStore(tmp_path / "store")
    (tmp_path / "secret").write_text("x")
    assert store.path("../secret") is None
    assert store.path("secret") is None
    assert is_asset_id("sha256-" + "a" * 64)
    assert not is_asset_id("sha256-" + "A" * 64)


def test_with_no_env_the_store_is_where_it_always_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("INLINE_ASSET_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    with TestClient(create_app(takes_dir=str(tmp_path / "takes"))) as client:
        asset_id = _upload(client)
    assert (tmp_path / ".inline-assets" / asset_id).is_file()


# The design choice this pins: the store object resolves the id, not the environment, so a caller
# that passes asset_dir explicitly (every existing test does) still gets its uploads found.
def test_an_explicit_asset_dir_wins_and_runs_still_find_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INLINE_ASSET_DIR", str(tmp_path / "from-env"))
    monkeypatch.chdir(tmp_path)
    app = create_app(asset_dir=str(tmp_path / "explicit"), takes_dir=str(tmp_path / "takes"))
    with TestClient(app) as client:
        asset_id = _upload(client)
        assert (tmp_path / "explicit" / asset_id).is_file()
        assert not (tmp_path / "from-env").exists()
        response = client.post(
            "/v1/runs", json={"graph": _graph({"ref": "asset", "id": asset_id}), "target": "img"}
        )
        assert response.status_code == 201
        assert _status(client, response.json()["runId"]) == "done"

"""The published document describes what the server actually serves."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple

import pytest
from fastapi.testclient import TestClient

from inline_core import __version__
from inline_core.device.memory import MemoryPolicy
from inline_core.graph.registry import build_default_registry
from inline_core.runtime.file_store import FileTakeStore
from inline_core.server.app import create_app
from inline_core.server.bootstrap import register_models
from inline_core.server.docs import documented_namespaces, namespace_of
from inline_core.server.reference import SCALAR_VERSION
from inline_core.server.rpc import RpcRouter
from inline_core.studio.store import StudioStore

_SRC = Path(__file__).resolve().parents[1] / "src" / "inline_core"

#: Recorded in vendor/__init__.py when the bundle was fetched.
_PINNED_SHA256 = "6a1407db14f57f7be9c98464b6d7e8899ba417c63b0d81363db2efb3e1022e1f"


class Wired(NamedTuple):
    client: TestClient
    router: RpcRouter


@pytest.fixture
def wired(tmp_path: Path) -> Iterator[Wired]:
    """A fully-registered app plus the router it uses, so the index can be checked against truth."""
    registry = build_default_registry()
    register_models(registry, FileTakeStore(tmp_path / "takes"), MemoryPolicy())
    router = RpcRouter()
    app = create_app(
        registry=registry,
        rpc=router,
        studio_store=StudioStore(tmp_path / "appdata", tmp_path / "workspace"),
        asset_dir=str(tmp_path / "assets"),
        models_root=str(tmp_path / "models"),
        takes_dir=str(tmp_path / "takes"),
    )
    with TestClient(app) as client:
        yield Wired(client, router)


def _studio_tag(wired: Wired) -> str:
    tags = _schema(wired.client)["tags"]
    return str(next(t for t in tags if t["name"] == "Studio RPC")["description"])


def _schema(client: TestClient) -> dict[str, Any]:
    return dict(client.get("/openapi.json").json())


def _operations(schema: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    return [
        (method, path, operation)
        for path, item in schema["paths"].items()
        for method, operation in item.items()
    ]


def test_the_document_is_named_and_versioned(wired: Wired) -> None:
    info = _schema(wired.client)["info"]

    assert info["title"] == "OpenChar Studio APIs"
    assert info["version"] == __version__


def test_every_operation_is_tagged_with_a_declared_tag(wired: Wired) -> None:
    """An untagged operation lands in Scalar's `default` bucket and reads as a mistake."""
    schema = _schema(wired.client)
    declared = {tag["name"] for tag in schema["tags"]}

    for method, path, operation in _operations(schema):
        assert operation.get("tags"), f"{method.upper()} {path} has no tag"
        assert set(operation["tags"]) <= declared, f"{method.upper()} {path} uses an unknown tag"


def test_every_write_declares_the_body_it_reads(wired: Wired) -> None:
    """Asserted by iteration, so a sixth POST added later fails here instead of shipping bare."""
    for method, path, operation in _operations(_schema(wired.client)):
        if method != "post":
            continue
        body = operation.get("requestBody")
        assert body and body.get("content"), f"POST {path} declares no request body"


def test_every_operation_declares_what_it_returns(wired: Wired) -> None:
    for method, path, operation in _operations(_schema(wired.client)):
        ok = [r for code, r in operation["responses"].items() if code.startswith("2")]
        assert ok, f"{method.upper()} {path} declares no success response"
        # Content, never description: FastAPI emits one for every operation, so it cannot fail.
        assert any(r.get("content") for r in ok), f"{method.upper()} {path} returns an empty schema"


def test_the_domain_types_reached_the_components(wired: Wired) -> None:
    components = _schema(wired.client)["components"]["schemas"]

    for name in ("SubmitRunIn", "RpcCallIn", "RunStateOut", "TakeOut", "ErrorOut", "HealthOut"):
        assert name in components


def test_every_registered_channel_is_in_the_index(wired: Wired) -> None:
    listed = set(re.findall(r"`([a-zA-Z]+:[a-zA-Z0-9:]+)`", _studio_tag(wired)))
    registered = set(wired.router.channels())

    assert registered, "the fixture registered no channels; the assertion below would be vacuous"
    assert registered <= listed


def test_every_namespace_in_a_wired_app_has_a_description(wired: Wired) -> None:
    """Taken from the router, not the rendered prose, which also names browser-native channels."""
    described = documented_namespaces()

    for namespace in {namespace_of(c) for c in wired.router.channels()}:
        assert namespace in described, f"namespace {namespace!r} has no line of its own"


def test_the_rpc_section_warns_that_a_failure_is_still_a_200(wired: Wired) -> None:
    """An unknown channel and a handler that raises both answer 200, so clients must read ok."""
    assert "still HTTP 200" in _studio_tag(wired)


def test_both_websockets_are_documented_since_openapi_cannot_hold_them(wired: Wired) -> None:
    schema = _schema(wired.client)
    by_tag = {t["name"]: t["description"] for t in schema["tags"]}

    assert "/v1/runs/{run_id}/events" in by_tag["Runs"]
    assert "4404" in by_tag["Runs"]
    assert "/events" in by_tag["Studio RPC"]


def test_every_event_channel_the_engine_broadcasts_reaches_the_page(wired: Wired) -> None:
    """Scanned from the repo root, so a wrong scan root inside docs.py fails rather than agrees."""
    broadcast = {
        name
        for path in _SRC.rglob("*.py")
        for name in re.findall(r'"(events:[A-Za-z]+)"', path.read_text(encoding="utf-8"))
    }
    documented = _studio_tag(wired)

    assert broadcast, "the scan found nothing; fix the regex, not the test"
    assert {c for c in broadcast if c not in documented} == set()


def test_a_channel_registered_after_boot_reaches_the_document(wired: Wired) -> None:
    """Extensions register channels while the server runs, so the document cannot be cached."""
    assert "ext:demo:ping" not in _studio_tag(wired)

    async def _handler(_args: list[Any]) -> str:
        return "pong"

    wired.router.register("ext:demo:ping", _handler)

    assert "ext:demo:ping" in _studio_tag(wired)


# --- the page ------------------------------------------------------------------------------


def _page(tmp_path: Path) -> Any:
    """The reference needs no project store, so a headless engine install gets docs too."""
    return TestClient(create_app(takes_dir=str(tmp_path / "takes")))


def test_the_reference_page_is_served(tmp_path: Path) -> None:
    with _page(tmp_path) as client:
        page = client.get("/api")

    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert "OpenChar Studio APIs" in page.text


@pytest.mark.parametrize(
    "host", ["cdn.jsdelivr.net", "fonts.scalar.com", "fastapi.tiangolo.com", "proxy.scalar.com"]
)
def test_the_reference_fetches_nothing_from_the_network(tmp_path: Path, host: str) -> None:
    """The app runs offline on people's own machines, so /api has to as well."""
    with _page(tmp_path) as client:
        page = client.get("/api")

    assert host not in page.text


def test_the_bundle_and_favicon_are_served_from_this_package(tmp_path: Path) -> None:
    with _page(tmp_path) as client:
        bundle = client.get(f"/api/scalar-{SCALAR_VERSION}.js")
        logo = client.get("/api/logo.svg")

    assert bundle.status_code == 200
    assert bundle.headers["content-type"].startswith("text/javascript")
    assert len(bundle.content) > 3_000_000
    assert "immutable" in bundle.headers["cache-control"]
    assert logo.status_code == 200 and logo.headers["content-type"] == "image/svg+xml"


def test_the_vendored_bundle_is_the_pinned_one() -> None:
    """The header records a hash; a bundle swapped without re-vendoring fails here."""
    payload = (_SRC / "server" / "vendor" / "scalar.standalone.js").read_bytes()

    assert hashlib.sha256(payload).hexdigest() == _PINNED_SHA256
    assert b"createApiReference" in payload


def test_the_reference_wins_over_the_spa_catch_all(tmp_path: Path) -> None:
    """Registered after the mount, /api would be served index.html and vanish in silence."""
    spa = tmp_path / "dist-web"
    (spa / "assets").mkdir(parents=True)
    (spa / "index.html").write_text("<!doctype html><title>SPA</title>")
    app = create_app(frontend_root=str(spa), takes_dir=str(tmp_path / "takes"))

    with TestClient(app) as client:
        assert client.get("/").text.count("SPA") == 1
        assert "OpenChar Studio APIs" in client.get("/api").text
        assert client.get(f"/api/scalar-{SCALAR_VERSION}.js").status_code == 200

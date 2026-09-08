"""The declared /v1 types and serialize.py's builders are the same shape, checked both ways."""

from __future__ import annotations

from dataclasses import is_dataclass

import pytest

from inline_core.graph.descriptor import NodeDescriptor, ParamField, Port, Widget
from inline_core.graph.schema import PortKind
from inline_core.media import MediaKind
from inline_core.runtime import progress
from inline_core.runtime.progress import (
    CancelledEvent,
    ErrorEvent,
    NodeDoneEvent,
    Phase,
    ProgressEvent,
    RunDoneEvent,
    RunEvent,
    RunStartedEvent,
)
from inline_core.runtime.run import NodeRuntimeState, NodeState, RunError, RunState, RunStatus
from inline_core.server import schemas
from inline_core.server.serialize import (
    descriptor_json,
    event_json,
    node_json,
    param_json,
    port_json,
    run_json,
    run_summary_json,
    take_json,
)
from inline_core.takes import Take


def _take() -> Take:
    return Take(
        id="t1",
        run_id="r1",
        node_id="n1",
        kind=MediaKind.IMAGE,
        uri="/takes/t1.png",
        hash="abc",
        params={"seed": 1},
        created_at=1700000000,
    )


def _descriptor(*, hidden: bool = False) -> NodeDescriptor:
    return NodeDescriptor(
        type="test/node",
        title="Test",
        category="test",
        icon="wand",
        inputs=(Port("prompt", "Prompt", PortKind.TEXT),),
        outputs=(Port("image", "Image", PortKind.IMAGE),),
        params=(
            ParamField("seed", "Seed", Widget.SEED, 0),
            ParamField("steps", "Steps", Widget.NUMBER, 4, min=1, max=50, step=1, advanced=True),
        ),
        hidden=hidden,
    )


def test_a_port_matches_its_declared_shape() -> None:
    assert schemas.PortOut.model_validate(port_json(Port("a", "A", PortKind.IMAGE)))


@pytest.mark.parametrize("hidden", [False, True])
def test_a_descriptor_matches_its_declared_shape(hidden: bool) -> None:
    assert schemas.DescriptorOut.model_validate(descriptor_json(_descriptor(hidden=hidden)))


def test_every_param_widget_matches_its_declared_shape() -> None:
    """Widget decides `kind`, and the optional keys are omitted, never nulled."""
    for widget in Widget:
        field = ParamField("k", "K", widget, None)
        assert schemas.ParamOut.model_validate(param_json(field))


def test_a_take_matches_its_declared_shape() -> None:
    assert schemas.TakeOut.model_validate(take_json(_take()))


def test_a_node_state_matches_its_declared_shape() -> None:
    bare = NodeRuntimeState(state=NodeState.QUEUED)
    full = NodeRuntimeState(
        state=NodeState.RUNNING, phase="sample", fraction=0.5, step=2, step_count=4, status="…"
    )

    assert schemas.NodeStateOut.model_validate(node_json(bare))
    assert schemas.NodeStateOut.model_validate(node_json(full))


def test_a_run_state_matches_its_declared_shape() -> None:
    state = RunState(
        run_id="r1",
        target="n1",
        status=RunStatus.RUNNING,
        fraction=0.5,
        nodes={"n1": NodeRuntimeState(state=NodeState.RUNNING)},
        takes=[_take()],
        error=RunError(message="boom", node_id="n1"),
    )

    assert schemas.RunStateOut.model_validate(run_json(state))
    assert schemas.RunStateOut.model_validate(run_json(RunState(run_id="r2", target="n1")))


def test_a_run_summary_matches_its_declared_shape() -> None:
    state = RunState(run_id="r1", target="n1")

    assert schemas.RunSummaryOut.model_validate(run_summary_json(state, 3))
    assert schemas.RunSummaryOut.model_validate(run_summary_json(state, None))


_EVENTS: list[RunEvent] = [
    ProgressEvent(run_id="r1", node_id="n1", phase=Phase.SAMPLE, fraction=0.5),
    ProgressEvent(
        run_id="r1", node_id="n1", phase=Phase.SAMPLE, fraction=0.5, step=1, step_count=4, eta_ms=10
    ),
    NodeDoneEvent(run_id="r1", node_id="n1", cached=False, takes=[_take()]),
    RunStartedEvent(run_id="r1"),
    RunDoneEvent(run_id="r1"),
    CancelledEvent(run_id="r1"),
    ErrorEvent(run_id="r1", message="boom", node_id="n1"),
]


@pytest.mark.parametrize("event", _EVENTS, ids=lambda e: type(e).__name__)
def test_every_run_event_matches_its_declared_shape(event: RunEvent) -> None:
    payload = event_json(event)
    by_type = {
        "progress": schemas.ProgressEventOut,
        "node_done": schemas.NodeDoneEventOut,
        "run_started": schemas.RunStartedEventOut,
        "run_done": schemas.RunDoneEventOut,
        "cancelled": schemas.CancelledEventOut,
        "error": schemas.ErrorEventOut,
    }

    assert by_type[str(payload["type"])].model_validate(payload)


def test_every_run_event_subclass_has_a_declared_model() -> None:
    """Read off `progress` itself, so a seventh subclass fails here instead of going unseen."""
    subclasses = {
        cls
        for cls in vars(progress).values()
        if isinstance(cls, type) and is_dataclass(cls) and cls.__name__.endswith("Event")
    }
    covered = {type(e) for e in _EVENTS}

    assert subclasses, "no event dataclasses found; fix the discovery, not the test"
    assert subclasses - covered == set(), f"undocumented event types: {subclasses - covered}"

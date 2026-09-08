"""The /v1 contract as declared types, mirroring serialize.py and pinned by a contract test."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Out(BaseModel):
    """Documentation only, and extras are forbidden so drift fails in both directions."""

    model_config = ConfigDict(extra="forbid")


class PortOut(_Out):
    id: str
    label: str
    kind: str
    required: bool


class ParamOptionOut(_Out):
    value: str
    label: str


class ParamOut(_Out):
    key: str
    label: str
    widget: str
    default: Any = None
    kind: str
    # Omitted by param_json unless set on the field, so absent and null are different facts.
    min: float | None = None
    max: float | None = None
    step: float | None = None
    optionsFrom: str | None = None
    options: list[ParamOptionOut] | None = None
    advanced: bool | None = None
    onFace: bool | None = None


class DescriptorOut(_Out):
    type: str
    title: str
    category: str
    icon: str
    source: str
    outputKind: str | None = None
    inputs: list[PortOut]
    outputs: list[PortOut]
    params: list[ParamOut]
    hidden: bool | None = None


class ModelsOut(_Out):
    registryVersion: str
    models: list[DescriptorOut]


class TakeOut(_Out):
    id: str
    runId: str
    nodeId: str
    kind: str
    uri: str
    hash: str
    params: dict[str, Any]
    createdAt: float


class NodeStateOut(_Out):
    state: str
    fraction: float
    phase: str | None = None
    step: int | None = None
    stepCount: int | None = None
    status: str | None = None


class RunErrorOut(_Out):
    nodeId: str | None = None
    message: str


class RunStateOut(_Out):
    runId: str
    status: str
    target: str
    fraction: float
    nodes: dict[str, NodeStateOut]
    takes: list[TakeOut]
    error: RunErrorOut | None = None


class RunSummaryOut(_Out):
    runId: str
    status: str
    target: str
    fraction: float
    queuePosition: int | None = None


class RunListOut(_Out):
    runs: list[RunSummaryOut]


class RunAcceptedOut(_Out):
    runId: str
    status: str


class AssetStoredOut(_Out):
    id: str
    kind: str
    bytes: int


class DeviceOut(_Out):
    kind: str
    profile: str
    vramBudgetMb: int | None = None
    vramFreeMb: int | None = None
    ramFreeMb: int | None = None


class SchemaWindowOut(_Out):
    min: int
    max: int


class HealthOut(BaseModel):
    """The one model FastAPI applies live, since /v1/health returns a bare dict: keep it exact."""

    ok: bool
    apiVersion: str
    schemaVersions: SchemaWindowOut
    registryVersion: str
    device: DeviceOut


class ErrorDetail(_Out):
    code: str
    message: str
    nodeId: str | None = None


class ErrorOut(_Out):
    error: ErrorDetail


class ProgressEventOut(_Out):
    type: Literal["progress"]
    runId: str
    nodeId: str
    phase: str
    fraction: float
    status: str
    step: int | None = None
    stepCount: int | None = None
    etaMs: int | None = None


class NodeDoneEventOut(_Out):
    type: Literal["node_done"]
    runId: str
    nodeId: str
    cached: bool
    takes: list[TakeOut]


class RunStartedEventOut(_Out):
    type: Literal["run_started"]
    runId: str


class RunDoneEventOut(_Out):
    type: Literal["run_done"]
    runId: str


class CancelledEventOut(_Out):
    type: Literal["cancelled"]
    runId: str


class ErrorEventOut(_Out):
    type: Literal["error"]
    runId: str
    message: str
    nodeId: str | None = None


RunEventOut = (
    ProgressEventOut
    | NodeDoneEventOut
    | RunStartedEventOut
    | RunDoneEventOut
    | CancelledEventOut
    | ErrorEventOut
)


class EdgeIn(_Out):
    """One wire. ``from`` is a Python keyword, so the field carries the alias."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_: str = Field(alias="from")
    output: str


class GraphNodeIn(_Out):
    id: str
    type: str
    params: dict[str, Any] = Field(default_factory=dict)
    # A list port keeps its wiring order, because that order is what a prompt addresses.
    inputs: dict[str, EdgeIn | list[EdgeIn]] = Field(default_factory=dict)


class GraphIn(_Out):
    schemaVersion: Literal[1]
    nodes: list[GraphNodeIn]


class SubmitRunIn(_Out):
    graph: GraphIn
    target: str
    # The same clientRunId replays the first run; a different graph under it is a 409.
    clientRunId: str | None = None
    meta: dict[str, Any] | None = None


class RpcCallIn(_Out):
    channel: str
    args: list[Any] = Field(default_factory=list)


class RpcResultOut(BaseModel):
    """The Result envelope. A failed call is still HTTP 200, so read ``ok``, never the status."""

    ok: bool
    value: Any = None
    error: str | None = None

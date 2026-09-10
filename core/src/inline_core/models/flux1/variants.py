"""Which FLUX.1 checkpoint a file is, read from its own tensor shapes.

``diffusion_models/`` is shared across every architecture, so a checkpoint is identified by content
and never by filename. The geometry is reconstructed from the header rather than bundled per build,
which is what lets one node load dev, schnell, Kontext and the Fill/Control builds - and a future
one - with no code change.

Torch-free and header-only: nothing here reads tensor data or imports torch, so the model popup
works on an install with no ML stack.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "VARIANTS",
    "Flux1Variant",
    "config_for",
    "derive_transformer_config",
    "detect",
    "folder_config",
    "get",
    "is_prequantized",
    "quantization_of",
    "single_file_blocker",
    "trainable",
]

#: The one number no tensor shape reveals. FLUX.1 splits its rotary embedding 16/56/56 across the
#: three position axes, which sums to the 128 head dim the shapes do give.
_FIXED_GEOMETRY: dict[str, object] = {
    "patch_size": 1,
    "out_channels": None,
    "axes_dims_rope": [16, 56, 56],
}

#: Every FLUX.1 build diffusers can load through ``FluxTransformer2DModel``. The three that share a
#: geometry are told apart by name; everything else falls out of the shapes (see ``detect``).
_INNER_DIM = 3072


@dataclass(frozen=True)
class Flux1Variant:
    """One FLUX.1 checkpoint: how to build it, and what it wants at sampling time."""

    key: str
    label: str
    #: Which diffusers pipeline family to build: "t2i" | "kontext" | "fill" | "control".
    pipeline: str
    #: **Step**-distilled. schnell alone is; dev is *guidance*-distilled, which is a different thing
    #: and is trained through by pinning guidance at 1 rather than around.
    distilled: bool
    #: 64 for a plain latent, 128 for a Control build's stacked hint, 384 for Fill's mask channels.
    in_channels: int
    guidance_embeds: bool
    #: The loader arch key. One asset bundle serves the family: same VAE, same CLIP-L, same T5-XXL.
    arch: str
    steps: int
    guidance: float


VARIANTS: tuple[Flux1Variant, ...] = (
    Flux1Variant(
        key="dev", label="dev", pipeline="t2i", distilled=False, in_channels=64,
        guidance_embeds=True, arch="flux1", steps=28, guidance=3.5,
    ),
    Flux1Variant(
        key="schnell", label="schnell", pipeline="t2i", distilled=True, in_channels=64,
        guidance_embeds=False, arch="flux1", steps=4, guidance=0.0,
    ),
    Flux1Variant(
        key="krea-dev", label="Krea dev", pipeline="t2i", distilled=False, in_channels=64,
        guidance_embeds=True, arch="flux1", steps=28, guidance=4.5,
    ),
    Flux1Variant(
        key="kontext-dev", label="Kontext dev", pipeline="kontext", distilled=False, in_channels=64,
        guidance_embeds=True, arch="flux1", steps=28, guidance=2.5,
    ),
    Flux1Variant(
        key="fill-dev", label="Fill dev", pipeline="fill", distilled=False, in_channels=384,
        guidance_embeds=True, arch="flux1", steps=50, guidance=30.0,
    ),
    Flux1Variant(
        key="canny-dev", label="Canny dev", pipeline="control", distilled=False, in_channels=128,
        guidance_embeds=True, arch="flux1", steps=50, guidance=30.0,
    ),
    Flux1Variant(
        key="depth-dev", label="Depth dev", pipeline="control", distilled=False, in_channels=128,
        guidance_embeds=True, arch="flux1", steps=30, guidance=10.0,
    ),
)

_BY_KEY = {v.key: v for v in VARIANTS}


def get(key: str | None) -> Flux1Variant | None:
    return _BY_KEY.get((key or "").strip())


def trainable(variant: Flux1Variant) -> bool:
    """Whether a LoRA run can train against this build.

    Three exclusions, all so a run fails now rather than twenty minutes into a precache. schnell is
    step-distilled and collapses the same way a distilled FLUX.2 does. Fill and Control take a mask
    or a stacked hint in their extra input channels, which the dataset exporter does not produce -
    their shapes only disagree once the first batch reaches the transformer. Kontext would train:
    it is 64-channel and undistilled, and conditions through the token sequence rather than the
    channels. But it learns an *edit* between a pair, and a dataset of single images teaches it
    nothing it is used for, so it is left out until paired datasets exist.
    """
    return not variant.distilled and variant.in_channels == 64 and variant.pipeline == "t2i"


# --- identifying a checkpoint --------------------------------------------------------------------

#: Prefixes ComfyUI-style repacks put in front of the keys. Stripped before matching so a Comfy
#: single file and a diffusers export identify the same way.
_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "model.")

_BLOCK_RE = re.compile(r"^transformer_blocks\.(\d+)\.")
_SINGLE_BLOCK_RE = re.compile(r"^single_transformer_blocks\.(\d+)\.")
_WEIGHT_SUFFIXES = (".safetensors", ".sft")

#: The shipped checkpoints use BFL's key layout, not diffusers'. diffusers converts at load time
#: (``convert_flux_transformer_checkpoint_to_diffusers``), but identification runs on the raw
#: header, so the handful of keys we read are renamed here first. Only the identifying keys are
#: mapped - this is not a checkpoint converter, and it must not become one.
_BFL_RENAMES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^double_blocks\."), "transformer_blocks."),
    (re.compile(r"^single_blocks\."), "single_transformer_blocks."),
    (re.compile(r"^txt_in\.weight$"), "context_embedder.weight"),
    (re.compile(r"^img_in\.weight$"), "x_embedder.weight"),
    (
        re.compile(r"^vector_in\.in_layer\.weight$"),
        "time_text_embed.text_embedder.linear_1.weight",
    ),
    (
        re.compile(r"^guidance_in\.in_layer\.weight$"),
        "time_text_embed.guidance_embedder.linear_1.weight",
    ),
    (re.compile(r"(img_)?attn\.norm\.query_norm\.scale$"), "attn.norm_q.weight"),
    (re.compile(r"(?<=\.)norm\.query_norm\.scale$"), "attn.norm_q.weight"),
)


def _strip(key: str) -> str:
    for prefix in _PREFIXES:
        if key.startswith(prefix):
            key = key[len(prefix) :]
            break
    for pattern, replacement in _BFL_RENAMES:
        key = pattern.sub(replacement, key)
    return key


def derive_transformer_config(shapes: dict[str, list[int]]) -> dict[str, object] | None:
    """Reconstruct a ``FluxTransformer2DModel`` config from a checkpoint's tensor shapes.

    Returns None when the file is not a FLUX.1 transformer. ``text_embedder`` is what separates
    FLUX.1 from FLUX.2: only FLUX.1 conditions on a pooled CLIP vector, so FLUX.2's checkpoints
    have no such key and cannot be claimed here.
    """
    keys = {_strip(key): shape for key, shape in shapes.items()}
    context = keys.get("context_embedder.weight")
    x_embed = keys.get("x_embedder.weight")
    pooled = keys.get("time_text_embed.text_embedder.linear_1.weight")
    if not context or not x_embed or not pooled:
        return None
    if len(context) != 2 or len(x_embed) != 2 or len(pooled) != 2:
        return None

    inner_dim, joint_attention_dim = context[0], context[1]
    # diffusers' single-file converter hardcodes inner_dim 3072 and mlp_ratio 4.0 to split each
    # single block's fused linear1, so a wider checkpoint would load mis-split rather than raise.
    if inner_dim != _INNER_DIM:
        return None
    head_dim = next(
        (shape[0] for key, shape in keys.items() if key.endswith("attn.norm_q.weight") and shape),
        0,
    )
    if not head_dim or inner_dim % head_dim:
        return None

    layers = {int(m.group(1)) for key in keys if (m := _BLOCK_RE.match(key))}
    single_layers = {int(m.group(1)) for key in keys if (m := _SINGLE_BLOCK_RE.match(key))}
    if not layers or not single_layers:
        return None

    return {
        **_FIXED_GEOMETRY,
        "attention_head_dim": head_dim,
        "guidance_embeds": any(
            "guidance_embedder" in key or key.startswith("guidance_in.") for key in keys
        ),
        "in_channels": x_embed[1],
        "joint_attention_dim": joint_attention_dim,
        "num_attention_heads": inner_dim // head_dim,
        "num_layers": len(layers),
        "num_single_layers": len(single_layers),
        "pooled_projection_dim": pooled[1],
    }


def folder_config(path: str | Path) -> dict[str, object] | None:
    """The transformer config of a diffusers-format checkpoint **folder**, or None.

    A folder ships its shards beside a ``config.json``, so there is no header to derive geometry
    from - and no need. Recognised by its ``_class_name``, so a folder belonging to another
    architecture is left alone.
    """
    folder = Path(path)
    marker = folder / "config.json"
    if not folder.is_dir() or not marker.is_file():
        return None
    try:
        import json

        config = json.loads(marker.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict):
        return None
    if config.get("_class_name") != "FluxTransformer2DModel":
        return None
    return {k: v for k, v in config.items() if not k.startswith("_")}


#: Safetensors spells fp8 as ``F8_E4M3`` / ``F8_E5M2``. Never match on a ``.scale`` suffix alone: a
#: plain FLUX.1 checkpoint carries 152 RMSNorm ``norm.query_norm.scale`` weights that are not
#: quantization scales at all. Only a scale riding on a *fused qkv* is a repack artifact, and it is
#: the one diffusers' converter chokes on.
_FP8_PREFIX = "F8_"
_INT_DTYPES = frozenset({"I8", "U8"})
_QKV_SCALE_SUFFIXES = ("qkv.scale", "qkv.weight_scale", "qkv.scale_weight")


def quantization_of(path: str | Path) -> str | None:
    """The quantization a checkpoint already carries, or None if it is plain weights.

    A folder declares it in ``config.json``. A single file does not, so it is sniffed from the
    header: an fp8 dtype, or the per-tensor scale tensors every int8/fp8 repack ships alongside.
    """
    target = Path(path)
    if target.is_dir():
        try:
            import json

            config = json.loads((target / "config.json").read_text())
        except (OSError, ValueError):
            return None
        if not isinstance(config, dict):
            return None
        declared = config.get("quantization_config")
        if not isinstance(declared, dict):
            return None
        return str(declared.get("quant_method", "quantized"))

    if not target.is_file() or target.suffix.lower() not in _WEIGHT_SUFFIXES:
        return None
    try:
        from ..checkpoint import CheckpointReader

        dtypes = CheckpointReader(target).dtypes()
    except Exception:  # noqa: BLE001 - unreadable means "not something we can classify"
        return None
    weights = {d for key, d in dtypes.items() if key.endswith(".weight")}
    if any(d.startswith(_FP8_PREFIX) for d in weights):
        return "fp8"
    if weights & _INT_DTYPES:
        return "int8"
    if any(key.endswith(_QKV_SCALE_SUFFIXES) for key in dtypes):
        return "quantized"
    return None


def is_prequantized(path: str | Path) -> bool:
    """Whether a checkpoint carries its own quantization (an NF4 folder, an fp8 single file).

    Such a checkpoint must not be quantized again: its on-disk size already is its resident size,
    and handing diffusers a second, different quantization config is a hard error.
    """
    return quantization_of(path) is not None


def single_file_blocker(path: str | Path) -> str | None:
    """Why a failed single-file load probably failed, or None if this is not the known cause.

    Diagnostic only, never a gate. The converter maps a ``.scale`` key to a weight and chunks it
    into q/k/v, so a **per-tensor** (0-dim) scale on a fused qkv dies deep inside diffusers without
    naming the file.
    """
    target = Path(path)
    if not target.is_file() or target.suffix.lower() not in _WEIGHT_SUFFIXES:
        return None
    try:
        from ..checkpoint import CheckpointReader

        shapes = CheckpointReader(target).shapes()
    except Exception:  # noqa: BLE001 - unreadable is not our call to make here
        return None
    scalar = [k for k, s in shapes.items() if k.endswith(_QKV_SCALE_SUFFIXES) and not s]
    if not scalar:
        return None
    return f"it carries a single scale value on each of {len(scalar)} fused qkv tensors"


def config_for(path: str | Path) -> dict[str, object] | None:
    """The transformer geometry for a checkpoint, whether it is a single file or a folder."""
    folder = folder_config(path)
    if folder is not None:
        return folder
    shapes = _shapes_of(path)
    return derive_transformer_config(shapes) if shapes else None


def _shapes_of(path: str | Path) -> dict[str, list[int]] | None:
    file = Path(path)
    if not file.is_file() or file.suffix.lower() not in _WEIGHT_SUFFIXES:
        return None
    try:
        from ..checkpoint import CheckpointReader

        return CheckpointReader(file).shapes()
    except Exception:  # noqa: BLE001 - an unreadable or foreign file is simply "not FLUX.1"
        return None


def _name_flags(name: str) -> tuple[bool, bool]:
    """(is_kontext, is_krea) read from a filename.

    dev, Kontext dev and Krea dev are byte-identical geometry - same widths, same block counts,
    same guidance embedder - so no shape tells them apart and the name is the only signal left.
    Everything else here is identified by content.
    """
    padded = "-" + re.sub(r"[^a-z0-9]+", "-", name.lower()) + "-"
    return "-kontext-" in padded, "-krea-" in padded


def detect(path: str | Path, shapes: dict[str, list[int]] | None = None) -> Flux1Variant | None:
    """Which FLUX.1 variant a checkpoint is, or None if it is not one.

    Handles both a single ``.safetensors`` and a diffusers folder. Pass ``shapes`` when the header
    has already been read, so it is not read twice. Never reads tensor data, never imports torch.
    """
    file = Path(path)
    config = derive_transformer_config(shapes) if shapes is not None else config_for(file)
    if config is None:
        return None
    family = [v for v in VARIANTS if v.in_channels == config["in_channels"]]
    if not family:
        return None
    # No guidance embedder means schnell, whatever the file is called.
    if not config["guidance_embeds"]:
        return next((v for v in family if v.distilled), None)
    family = [v for v in family if not v.distilled]
    is_kontext, is_krea = _name_flags(file.name)
    if len(family) > 1:
        wanted = "kontext-dev" if is_kontext else "krea-dev" if is_krea else "dev"
        return next((v for v in family if v.key == wanted), family[0])
    return family[0] if family else None

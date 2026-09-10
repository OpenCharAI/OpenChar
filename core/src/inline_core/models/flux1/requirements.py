"""What FLUX.1 needs on disk, and whether it is there - the data behind the node's model popup.

**No hidden downloads.** A component counts as present only because the user placed the file under
``models/`` or fetched it through the popup. Nothing here ever reaches the network.

Torch-free (pure filesystem + safetensors headers) so the popup works on an install with no ML
stack. Files are matched by reading each candidate's header rather than trusting its name, which is
what lets a Z-Image, Krea 2, FLUX.1 and FLUX.2 checkpoint share ``diffusion_models/`` safely - and
what keeps FLUX.1's two encoders out of FLUX.2's encoder slot, which they would otherwise fit.

**Licence.** dev's weights are non-commercial whichever mirror they come from, and a LoRA trained on
them is a derivative that inherits it. The row labels say so; that is the only place a user sees it.
"""

from __future__ import annotations

import os
from pathlib import Path

from ...config import models_dir
from ..catalog import resolve_picked
from ..requirements import ModelComponent
from . import variants as V

__all__ = [
    "CLIP_FILE",
    "DIFFUSION_FILE",
    "TEXT_ENCODER_FILE",
    "VAE_FILE",
    "download_target",
    "flux1_checkpoints",
    "flux1_encoders",
    "footprint_bytes",
    "flux1_requirements",
    "resolve_clip",
    "resolve_diffusion",
    "resolve_text_encoder",
    "resolve_vae",
    "resolved_variant",
]

#: ComfyUI's repackaged single files: ungated, one consolidated ``.safetensors`` per component.
#: BFL's own repos are gated for every FLUX.1 build - schnell included - which breaks the popup.
#: Note the transformer sits at the repo root here, unlike the FLUX.2 mirror's ``split_files/``.
DIFFUSION_REPO = "Comfy-Org/flux1-dev"
DIFFUSION_FILE = "flux1-dev.safetensors"
#: Both encoders come from one ungated Apache-2.0 repo.
ENCODER_REPO = "comfyanonymous/flux_text_encoders"
TEXT_ENCODER_FILE = "t5xxl_fp16.safetensors"
CLIP_FILE = "clip_l.safetensors"
#: FLUX.1's VAE is ``ae.safetensors``, and Z-Image ships the *same weights* under the same name -
#: identical tensor names, shapes and values, and the same 0.3611/0.1159 scale and shift. So a
#: Z-Image install already has this file, and pointing at its repo means no second 335MB download.
VAE_REPO = "Comfy-Org/z_image"
VAE_FILE = "ae.safetensors"

_WEIGHT_SUFFIXES = (".safetensors", ".sft", ".gguf")

#: Offered as suggestions; none of them block a run.
_EXTRAS: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "diffusion_fp8",
        "FLUX.1 dev fp8 (12 GB, non-commercial)",
        "diffusion_models",
        "Kijai/flux-fp8",
        "flux1-dev-fp8.safetensors",
    ),
    (
        "text_encoder_fp8",
        "T5-XXL fp8 scaled (5 GB, for a smaller card)",
        "text_encoders",
        ENCODER_REPO,
        "t5xxl_fp8_e4m3fn_scaled.safetensors",
    ),
)


# --- filesystem resolution -----------------------------------------------------------------------


def _category(name: str) -> Path:
    return models_dir() / name


def _weight_files(category: str) -> list[Path]:
    """Every candidate in a category: consolidated single files, plus diffusers-format folders."""
    root = _category(category)
    if not root.is_dir():
        return []
    return sorted(
        p
        for p in root.iterdir()
        if (p.is_file() and p.suffix.lower() in _WEIGHT_SUFFIXES)
        or (p.is_dir() and (p / "config.json").is_file())
    )


#: Header reads are cheap but the popup opens often, so identification is memoized on
#: (path, size, mtime) - a replaced file re-identifies, an untouched one does not.
_IDENTIFIED: dict[tuple[str, int, int], V.Flux1Variant | None] = {}


def _identify(path: Path) -> V.Flux1Variant | None:
    try:
        target = path / "config.json" if path.is_dir() else path
        stat = target.stat()
    except OSError:
        return None
    key = (str(path), stat.st_size, int(stat.st_mtime))
    if key not in _IDENTIFIED:
        _IDENTIFIED[key] = V.detect(path)
    return _IDENTIFIED[key]


def flux1_checkpoints() -> list[Path]:
    """Every installed file that identifies as a FLUX.1 transformer."""
    return [p for p in _weight_files("diffusion_models") if _identify(p) is not None]


def _encoder_kind(path: Path) -> str | None:
    """``"t5xxl"`` | ``"clip-l"`` | None, from the header alone.

    By content, never by name: both files live in ``text_encoders/`` beside Qwen3 and Mistral-3, and
    T5's 4096-wide embedding is exactly the width FLUX.2 klein 9B looks for.
    """
    if path.is_dir() or path.suffix.lower() not in (".safetensors", ".sft"):
        return None
    try:
        from ..checkpoint import CheckpointReader

        keys = CheckpointReader(path).shapes()
    except Exception:  # noqa: BLE001 - an unreadable file simply does not match
        return None
    if any(k.startswith("encoder.block.") for k in keys) and "shared.weight" in keys:
        return "t5xxl"
    if any(k.startswith("text_model.") for k in keys):
        return "clip-l"
    return None


def _is_flux1_vae(path: Path) -> bool:
    """Whether this is FLUX.1's LDM-style VAE, from the header alone.

    ``vae/`` is shared, and a name check is not enough: FLUX.2's file is called ``flux2-vae`` and
    would match any "flux" fallback, while being a different VAE entirely. It is told apart by the
    running batch-norm buffers it carries **in place of** FLUX.1's scalar scale and shift.
    """
    if path.is_dir() or path.suffix.lower() not in (".safetensors", ".sft"):
        return False
    try:
        from ..checkpoint import CheckpointReader

        keys = CheckpointReader(path).shapes()
    except Exception:  # noqa: BLE001 - an unreadable file simply does not match
        return False
    if any(k.startswith("bn.") or ".bn." in k for k in keys):
        return False  # FLUX.2's VAE
    return "decoder.conv_out.weight" in keys and "encoder.conv_out.weight" in keys


def resolve_diffusion(params: dict[str, object] | None = None) -> Path | None:
    """The FLUX.1 checkpoint to load: an explicit pick, ``INLINE_FLUX1_MODEL``, else the first file
    in ``diffusion_models/`` that identifies as FLUX.1."""
    env = os.environ.get("INLINE_FLUX1_MODEL", "").strip()
    if env:
        path = Path(env)
        return path if path.exists() else None
    chosen = (params or {}).get("model")
    if str(chosen or "").strip():
        return resolve_picked("diffusion_models", chosen)
    return next(iter(flux1_checkpoints()), None)


def resolved_variant(params: dict[str, object] | None = None) -> V.Flux1Variant | None:
    """Which variant the node will run: the explicit ``variant`` param, else what the resolved
    checkpoint identifies as."""
    forced = V.get(str((params or {}).get("variant") or ""))
    if forced is not None:
        return forced
    diffusion = resolve_diffusion(params)
    return _identify(diffusion) if diffusion is not None else None


def resolve_vae(params: dict[str, object] | None = None) -> Path | None:
    """The FLUX.1 VAE, which is also Z-Image's - the same file under the same name.

    ``vae/`` is shared, so the canonical name is matched first and a FLUX-named repack only after.
    Krea 2's Qwen-Image VAE and FLUX.2's carry their own names and are never reached."""
    env = os.environ.get("INLINE_FLUX1_VAE", "").strip()
    if env:
        path = Path(env)
        return path if path.exists() else None
    chosen = (params or {}).get("vae")
    if str(chosen or "").strip():
        return resolve_picked("vae", chosen)
    exact = _category("vae") / VAE_FILE
    if exact.is_file() and _is_flux1_vae(exact):
        return exact
    return next((p for p in _weight_files("vae") if _is_flux1_vae(p)), None)


def resolve_text_encoder(params: dict[str, object] | None = None) -> Path | None:
    """T5-XXL, identified by its own key layout rather than by name or width."""
    env = os.environ.get("INLINE_FLUX1_TEXT_ENCODER", "").strip()
    if env:
        path = Path(env)
        return path if path.exists() else None
    chosen = (params or {}).get("text_encoder")
    if str(chosen or "").strip():
        return resolve_picked("text_encoders", chosen)
    return next((p for p in _weight_files("text_encoders") if _encoder_kind(p) == "t5xxl"), None)


def resolve_clip(params: dict[str, object] | None = None) -> Path | None:
    """CLIP-L, the pooled half of FLUX.1's conditioning."""
    env = os.environ.get("INLINE_FLUX1_CLIP", "").strip()
    if env:
        path = Path(env)
        return path if path.exists() else None
    chosen = (params or {}).get("clip")
    if str(chosen or "").strip():
        return resolve_picked("text_encoders", chosen)
    return next((p for p in _weight_files("text_encoders") if _encoder_kind(p) == "clip-l"), None)


def flux1_encoders() -> list[Path]:
    """Both of FLUX.1's encoders, so the node's picker offers neither Qwen3 nor Mistral-3."""
    return [p for p in _weight_files("text_encoders") if _encoder_kind(p) is not None]


# --- memory footprint ----------------------------------------------------------------------------


def _file_bytes(path: object) -> int:
    text = str(path or "").strip()
    if not text:
        return 0
    try:
        p = Path(text)
        if p.is_dir():
            return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        return p.stat().st_size if p.is_file() else 0
    except OSError:
        return 0


def footprint_bytes(
    diffusion: object = None,
    vae: object = None,
    text_encoder: object = None,
    clip: object = None,
) -> dict[str, int]:
    """On-disk sizes keyed to match ``ModelFootprint``, for the device policy's fit estimate.

    CLIP-L is folded into the text-encoder total rather than dropped: it is only 246 MB, but both
    encoders are resident together while the prompt is encoded.
    """
    return {
        "diffusion_bytes": _file_bytes(diffusion),
        "text_encoder_bytes": _file_bytes(text_encoder) + _file_bytes(clip),
        "vae_bytes": _file_bytes(vae),
        "controlnet_bytes": 0,
    }


# --- the requirements view (the popup's data) -----------------------------------------------------


def _component(
    *, id: str, label: str, category: str, filename: str, present: bool, repo: str, repo_file: str,
    optional: bool = False,
) -> ModelComponent:
    return ModelComponent(
        id=id,
        label=label,
        category=category,
        present=present,
        filename=filename,
        repo=repo,
        repo_file=repo_file,
        optional=optional,
    )


def flux1_requirements(params: dict[str, object] | None = None) -> list[ModelComponent]:
    """The popup's rows: the four required components, then the optional extras."""
    variant = resolved_variant(params)
    label = f" ({variant.label})" if variant else ""
    required = [
        _component(
            id="diffusion",
            # The licence rides on the label because this is the only place a user reads it.
            label=f"Diffusion model{label} - non-commercial licence",
            category="diffusion_models",
            filename=DIFFUSION_FILE,
            present=resolve_diffusion(params) is not None,
            repo=DIFFUSION_REPO,
            repo_file=DIFFUSION_FILE,
        ),
        _component(
            id="text_encoder",
            label="Text encoder (T5-XXL)",
            category="text_encoders",
            filename=TEXT_ENCODER_FILE,
            present=resolve_text_encoder(params) is not None,
            repo=ENCODER_REPO,
            repo_file=TEXT_ENCODER_FILE,
        ),
        _component(
            id="clip",
            label="CLIP-L (pooled conditioning)",
            category="text_encoders",
            filename=CLIP_FILE,
            present=resolve_clip(params) is not None,
            repo=ENCODER_REPO,
            repo_file=CLIP_FILE,
        ),
        _component(
            id="vae",
            label="VAE",
            category="vae",
            filename=VAE_FILE,
            present=resolve_vae(params) is not None,
            repo=VAE_REPO,
            repo_file=VAE_FILE,
        ),
    ]
    extras = [
        _component(
            id=extra_id,
            label=extra_label,
            category=category,
            filename=Path(repo_file).name,
            present=(_category(category) / Path(repo_file).name).is_file(),
            repo=repo,
            repo_file=repo_file,
            optional=True,
        )
        for extra_id, extra_label, category, repo, repo_file in _EXTRAS
    ]
    return required + extras


def download_target(component: ModelComponent) -> Path:
    """Where the component's file lands: its category folder, flat, under the models root."""
    return _category(component.category)

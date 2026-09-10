"""Which files the FLUX.1 node picks off disk, and what the popup reports.

``text_encoders/`` is the crowded one: FLUX.1 puts T5-XXL and CLIP-L in the same folder that already
holds Qwen3 for Z-Image and FLUX.2 klein, and Mistral-3 for FLUX.2 dev. T5's embedding is 4096 wide,
which is exactly what klein 9B looks for, so identification is by key layout and never by width.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from inline_core.models.flux1 import requirements as reqs
from inline_core.models.flux1 import variants as V
from tests.test_flux1_variants import BFL_DEV_HEADER
from tests.test_flux2_resolve import _write_header_only

#: The identifying subset of each real encoder's header.
T5_HEADER = {
    "shared.weight": [32128, 4096],
    "encoder.embed_tokens.weight": [32128, 4096],
    "encoder.block.0.layer.0.SelfAttention.q.weight": [4096, 4096],
}
CLIP_HEADER = {
    "text_model.embeddings.token_embedding.weight": [49408, 768],
    "text_model.encoder.layers.0.self_attn.q_proj.weight": [768, 768],
}
#: FLUX.1's VAE, which Z-Image ships as the same file. FLUX.2's carries running batch-norm buffers
#: in place of the scalar scale and shift, which is what tells the two apart.
VAE_HEADER = {"encoder.conv_out.weight": [32, 512, 3, 3], "decoder.conv_out.weight": [3, 128, 3, 3]}
FLUX2_VAE_HEADER = {**VAE_HEADER, "bn.running_mean": [128], "bn.running_var": [128]}


@pytest.fixture
def models(tmp_path: Path, monkeypatch: Any) -> Path:
    root = tmp_path / "models"
    monkeypatch.setenv("INLINE_MODELS_DIR", str(root))
    for category in ("diffusion_models", "vae", "text_encoders"):
        (root / category).mkdir(parents=True)
    reqs._IDENTIFIED.clear()  # keyed by path, and tmp_path is reused
    return root


def test_resolve_skips_checkpoints_from_other_architectures(models: Path) -> None:
    _write_header_only(models / "diffusion_models" / "z_image_bf16.safetensors", {"foo": [4, 4]})
    assert reqs.resolve_diffusion() is None

    flux = _write_header_only(
        models / "diffusion_models" / "flux1-dev.safetensors", BFL_DEV_HEADER
    )
    assert reqs.resolve_diffusion() == flux
    assert reqs.resolved_variant() is V.get("dev")


def test_the_two_encoders_are_told_apart_by_their_key_layout(models: Path) -> None:
    # Named to sort the wrong way round, so an order-based pick would swap them.
    t5 = _write_header_only(models / "text_encoders" / "zzz_t5.safetensors", T5_HEADER)
    clip = _write_header_only(models / "text_encoders" / "aaa_clip.safetensors", CLIP_HEADER)
    assert reqs.resolve_text_encoder() == t5
    assert reqs.resolve_clip() == clip


def test_a_qwen3_encoder_is_never_offered_to_flux1(models: Path) -> None:
    # The other direction of the shared-folder problem: Z-Image and klein's encoder is not ours.
    _write_header_only(
        models / "text_encoders" / "qwen_3_4b.safetensors",
        {"model.embed_tokens.weight": [151936, 2560]},
    )
    assert reqs.resolve_text_encoder() is None
    assert reqs.resolve_clip() is None


def test_a_flux1_t5_encoder_is_not_offered_to_flux2(models: Path) -> None:
    """The regression this guards, from the other side: T5's ``encoder.embed_tokens.weight`` is
    32128 x 4096, and 4096 is exactly the width FLUX.2 klein 9B matches on."""
    from inline_core.models.flux2 import requirements as flux2_reqs
    from tests.test_flux2_variants import KLEIN_9B
    from tests.test_flux2_variants import _shapes as _flux2_shapes

    flux2_reqs._IDENTIFIED.clear()
    _write_header_only(models / "text_encoders" / "aaa_t5xxl.safetensors", T5_HEADER)
    _write_header_only(models / "text_encoders" / "aab_clip_l.safetensors", CLIP_HEADER)
    _write_header_only(
        models / "diffusion_models" / "flux-2-klein-9b.safetensors", _flux2_shapes(KLEIN_9B)
    )
    assert flux2_reqs.resolve_text_encoder() is None, "FLUX.1's encoders are not klein 9B's"


def test_the_popup_lists_four_required_components_then_the_extras(models: Path) -> None:
    components = reqs.flux1_requirements()
    required = [c for c in components if not c.optional]
    assert [c.id for c in required] == ["diffusion", "text_encoder", "clip", "vae"]
    assert not any(c.present for c in required), "an empty models dir has nothing"
    # The licence is on the row, because the popup is the only place a user reads it.
    assert "non-commercial" in next(c.label for c in required if c.id == "diffusion")
    assert {c.id for c in components if c.optional} == {"diffusion_fp8", "text_encoder_fp8"}

    _write_header_only(models / "diffusion_models" / "flux1-dev.safetensors", BFL_DEV_HEADER)
    _write_header_only(models / "text_encoders" / "t5xxl_fp16.safetensors", T5_HEADER)
    _write_header_only(models / "text_encoders" / "clip_l.safetensors", CLIP_HEADER)
    _write_header_only(models / "vae" / "ae.safetensors", VAE_HEADER)
    reqs._IDENTIFIED.clear()
    assert all(c.present for c in reqs.flux1_requirements() if not c.optional)


def test_an_env_override_wins_over_the_scan(models: Path, monkeypatch: Any) -> None:
    picked = _write_header_only(models / "elsewhere.safetensors", BFL_DEV_HEADER)
    monkeypatch.setenv("INLINE_FLUX1_MODEL", str(picked))
    assert reqs.resolve_diffusion() == picked


def test_the_node_offers_only_files_it_can_load(models: Path) -> None:
    from inline_core.models.flux1.provider import Flux1Provider

    _write_header_only(models / "diffusion_models" / "flux1-dev.safetensors", BFL_DEV_HEADER)
    _write_header_only(models / "diffusion_models" / "z_image.safetensors", {"foo": [4, 4]})
    _write_header_only(models / "text_encoders" / "t5xxl_fp16.safetensors", T5_HEADER)
    _write_header_only(models / "text_encoders" / "clip_l.safetensors", CLIP_HEADER)
    _write_header_only(
        models / "text_encoders" / "qwen_3_4b.safetensors",
        {"model.embed_tokens.weight": [151936, 2560]},
    )
    reqs._IDENTIFIED.clear()
    provider = Flux1Provider()
    assert provider.catalog_options("diffusion_models") == ["flux1-dev.safetensors"]
    assert sorted(provider.catalog_options("text_encoders") or []) == [
        "clip_l.safetensors",
        "t5xxl_fp16.safetensors",
    ]
    assert provider.resolved()["variant"] == "dev"


def test_the_vae_is_shared_with_z_image_but_never_taken_from_flux2(models: Path) -> None:
    """FLUX.1's VAE is ``ae.safetensors``, and Z-Image ships the same weights under the same name -
    identical tensor names, shapes and values. FLUX.2's is a different VAE that would match any
    "flux" name check, so the two are told apart by the batch-norm buffers FLUX.2 carries."""
    _write_header_only(models / "vae" / "flux2-vae.safetensors", FLUX2_VAE_HEADER)
    assert reqs.resolve_vae() is None, "FLUX.2's VAE is not FLUX.1's"

    shared = _write_header_only(models / "vae" / "ae.safetensors", VAE_HEADER)
    assert reqs.resolve_vae() == shared

"""Which FLUX.1 checkpoint a file is, derived from its own tensor shapes.

``diffusion_models/`` is shared with Z-Image, Krea 2 and FLUX.2, so identification is by content.
The values pinned here are read from the real ``Comfy-Org/flux1-dev/flux1-dev.safetensors`` header:
780 BF16 tensors, ``txt_in [3072, 4096]``, ``img_in [3072, 64]``, ``vector_in.in_layer
[3072, 768]``, ``guidance_in.in_layer [3072, 256]``, a ``[128]`` QK-norm vector, and 19+38 blocks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inline_core.models.flux1 import variants as V

DEV: dict[str, object] = {
    "attention_head_dim": 128,
    "axes_dims_rope": [16, 56, 56],
    "guidance_embeds": True,
    "in_channels": 64,
    "joint_attention_dim": 4096,
    "num_attention_heads": 24,
    "num_layers": 19,
    "num_single_layers": 38,
    "out_channels": None,
    "patch_size": 1,
    "pooled_projection_dim": 768,
}
SCHNELL = {**DEV, "guidance_embeds": False}
FILL = {**DEV, "in_channels": 384}
CANNY = {**DEV, "in_channels": 128}


def _shapes(config: dict[str, object]) -> dict[str, list[int]]:
    """Real parameter shapes for a config, straight from diffusers on the meta device."""
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    with torch.device("meta"):
        model = diffusers.FluxTransformer2DModel(**config)
    shapes = {name: list(p.shape) for name, p in model.named_parameters()}
    shapes.update({name: list(b.shape) for name, b in model.named_buffers()})
    return shapes


#: The identifying subset of the real checkpoint, in BFL's key layout rather than diffusers'.
BFL_DEV_HEADER: dict[str, list[int]] = {
    "txt_in.weight": [3072, 4096],
    "img_in.weight": [3072, 64],
    "vector_in.in_layer.weight": [3072, 768],
    "guidance_in.in_layer.weight": [3072, 256],
    "final_layer.linear.weight": [64, 3072],
    "single_blocks.0.linear1.weight": [21504, 3072],
    "single_blocks.0.linear2.weight": [3072, 15360],
    **{f"double_blocks.{i}.img_attn.norm.query_norm.scale": [128] for i in range(19)},
    **{f"single_blocks.{i}.norm.query_norm.scale": [128] for i in range(38)},
}


@pytest.mark.parametrize(
    ("name", "config"), [("dev", DEV), ("schnell", SCHNELL), ("fill", FILL), ("canny", CANNY)]
)
def test_config_round_trips_from_tensor_shapes(name: str, config: dict[str, object]) -> None:
    assert V.derive_transformer_config(_shapes(config)) == config, name


def test_the_shipped_bfl_checkpoint_derives_the_dev_config() -> None:
    # The published file is in BFL's key layout; diffusers renames at load, but identification runs
    # on the raw header, so the renames have to happen here first.
    assert V.derive_transformer_config(BFL_DEV_HEADER) == DEV
    assert V.detect("flux1-dev.safetensors", BFL_DEV_HEADER) is V.get("dev")


def test_comfy_style_key_prefixes_are_stripped() -> None:
    prefixed = {f"model.diffusion_model.{k}": v for k, v in BFL_DEV_HEADER.items()}
    assert V.derive_transformer_config(prefixed) == DEV


def test_schnell_is_identified_by_its_missing_guidance_embedder() -> None:
    """schnell is the one step-distilled build, and no filename is needed to spot it: it is the
    only FLUX.1 checkpoint with no guidance embedder at all."""
    schnell = {k: v for k, v in BFL_DEV_HEADER.items() if not k.startswith("guidance_in.")}
    assert V.detect("some-name.safetensors", schnell) is V.get("schnell")
    assert not V.trainable(V.get("schnell"))


def test_the_three_identical_builds_are_told_apart_by_name_and_nothing_else() -> None:
    # dev, Kontext dev and Krea dev share every shape, so the name is the only signal left.
    assert V.detect("flux1-dev.safetensors", BFL_DEV_HEADER) is V.get("dev")
    assert V.detect("flux1-kontext-dev.safetensors", BFL_DEV_HEADER) is V.get("kontext-dev")
    assert V.detect("flux1-krea-dev.safetensors", BFL_DEV_HEADER) is V.get("krea-dev")


def test_the_control_and_fill_builds_are_told_apart_by_their_input_channels() -> None:
    assert V.detect("flux1-fill-dev.safetensors", _shapes(FILL)) is V.get("fill-dev")
    assert V.detect("flux1-canny-dev.safetensors", _shapes(CANNY)) is V.get("canny-dev")


def test_only_a_plain_undistilled_build_trains() -> None:
    """Fill and Control resolve as undistilled, but their extra input channels carry a mask or a
    stacked hint the dataset exporter does not produce - a shape error twenty minutes into a
    precache rather than a refusal up front. Kontext would train, and is left out for a different
    reason: it learns an edit between a pair, so single images teach it nothing it is used for."""
    assert V.trainable(V.get("dev")) and V.trainable(V.get("krea-dev"))
    assert not V.trainable(V.get("fill-dev"))
    assert not V.trainable(V.get("canny-dev"))
    assert not V.trainable(V.get("schnell"))
    assert not V.trainable(V.get("kontext-dev"))


def test_a_wider_checkpoint_is_refused_rather_than_loaded_mis_split() -> None:
    """diffusers' single-file converter hardcodes inner_dim 3072 and mlp_ratio 4.0 to split each
    single block's fused linear1, so a wider build would load silently mis-split - a wrong image,
    not an error. Refused at identification instead."""
    wide = {k: ([6144, v[1]] if k in ("txt_in.weight", "img_in.weight") else v)
            for k, v in BFL_DEV_HEADER.items()}
    wide["vector_in.in_layer.weight"] = [6144, 768]
    assert V.derive_transformer_config(wide) is None


def test_a_checkpoint_without_pooled_conditioning_is_not_flux1() -> None:
    # Only FLUX.1 conditions on a pooled CLIP vector; this is what keeps FLUX.2 out.
    no_pooled = {k: v for k, v in BFL_DEV_HEADER.items() if k != "vector_in.in_layer.weight"}
    assert V.derive_transformer_config(no_pooled) is None


def test_flux1_and_flux2_never_claim_each_other() -> None:
    """Both directions. The FLUX.2 half passes because 4096 matches no FLUX.2 joint width, which a
    future variant row could change - so it is pinned rather than left to luck."""
    from inline_core.models.flux2 import variants as V2
    from tests.test_flux2_variants import DEV as FLUX2_DEV
    from tests.test_flux2_variants import KLEIN_4B, KLEIN_9B
    from tests.test_flux2_variants import _shapes as _flux2_shapes

    assert V2.detect("flux1-dev.safetensors", BFL_DEV_HEADER) is None
    for config in (KLEIN_4B, KLEIN_9B, FLUX2_DEV):
        assert V.derive_transformer_config(_flux2_shapes(config)) is None


def test_rmsnorm_scales_do_not_make_a_plain_checkpoint_look_quantized(tmp_path: Path) -> None:
    # A plain FLUX.1 file carries 152 `norm.query_norm.scale` weights that are not quant scales.
    from tests.test_flux2_resolve import _write_header_only

    plain = _write_header_only(tmp_path / "flux1-dev.safetensors", BFL_DEV_HEADER)
    assert V.quantization_of(plain) is None
    assert not V.is_prequantized(plain)
    assert V.single_file_blocker(plain) is None


def test_every_variant_names_a_loader_arch_and_a_pipeline() -> None:
    assert {v.arch for v in V.VARIANTS} == {"flux1"}
    assert {v.pipeline for v in V.VARIANTS} <= {"t2i", "kontext", "fill", "control"}
    assert len({v.key for v in V.VARIANTS}) == len(V.VARIANTS)

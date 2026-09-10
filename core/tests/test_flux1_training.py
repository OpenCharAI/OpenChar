"""The FLUX.1 LoRA training arch: what it adapts, what it predicts, and which base it demands.

Two rules decide whether a run is worth anything, and neither raises when broken - they produce a
plausible-but-wrong adapter hours later. Guidance is trained at 1, not at the 3.5 dev generates
with, because dev is guidance-distilled and the embedder is part of the model. And conditioning is
two tensors, not one: T5's sequence and CLIP's pooled vector, both at the compute dtype.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inline_core.training import arch as archs
from tests.test_flux1_variants import BFL_DEV_HEADER, DEV
from tests.test_flux2_resolve import _write_header_only

models = pytest.importorskip("inline_core.training.models")

#: A miniature FLUX.1 with the real topology - both block types, the pre_only single-block
#: attention, the guidance embedder - so the forward and adapter run without a 12B checkpoint.
#: ``axes_dims_rope`` must sum to ``attention_head_dim``; ``in_channels`` stays 64 because that is
#: what a 16-channel latent becomes once the pipeline folds it 2x2.
_TINY = {
    "attention_head_dim": 32,
    "axes_dims_rope": [8, 12, 12],
    "guidance_embeds": True,
    "in_channels": 64,
    "joint_attention_dim": 192,
    "num_attention_heads": 4,
    "num_layers": 2,
    "num_single_layers": 2,
    "out_channels": None,
    "patch_size": 1,
    "pooled_projection_dim": 64,
}
_EMBED, _POOLED = 192, 64


def _tiny_model():
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    return diffusers.FluxTransformer2DModel(**_TINY).to(torch.float32).eval()


def test_flux1_is_a_registered_training_arch() -> None:
    a = archs.get("flux1")
    assert a.key == archs.FLUX1
    # Rectified flow, Krea 2's convention: x_t = (1-s)*clean + s*noise, so d/ds is noise - clean.
    assert a.target(clean=2.0, noise=5.0) == 3.0
    assert a.timestep(0.25) == 0.25
    assert a.clip is None, "FLUX.1 trains on stills"


def _linears() -> set[str]:
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    # On meta, so the real 19 + 38 topology costs nothing and the counts below are the real ones.
    with torch.device("meta"):
        model = diffusers.FluxTransformer2DModel(**DEV)
    return {n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}


def _resolved(target: str, linears: set[str]) -> set[str]:
    """The modules PEFT would attach to for one target: an exact name, or a dotted suffix."""
    return {n for n in linears if n == target or n.endswith("." + target)}


def test_every_target_matches_the_real_model() -> None:
    linears = _linears()
    for target in archs.get("flux1").target_modules:
        assert _resolved(target, linears), f"{target} matches nothing"


def test_to_out_reaches_the_double_blocks_alone() -> None:
    """FLUX.1's single blocks are ``pre_only``, so they carry no ``attn.to_out`` - the ModuleList
    versus Linear suffix clash that forces FLUX.2 to drop its single-block output projection does
    not arise here."""
    linears = _linears()
    hit = _resolved("to_out.0", linears)
    assert len(hit) == 19
    assert not any("single_transformer_blocks" in n for n in hit)


def test_proj_out_reaches_the_single_blocks_and_the_models_own_tail() -> None:
    """The double match is a decision, not an accident: PEFT matches by suffix, so ``proj_out``
    takes the 38 single-block projections *and* the model's own final ``proj_out``. Both are plain
    Linears, so nothing breaks. Pinned here so a later change to it is deliberate."""
    hit = _resolved("proj_out", _linears())
    assert len(hit) == 39
    assert "proj_out" in hit
    assert sum(1 for n in hit if n.startswith("single_transformer_blocks.")) == 38


def test_the_adaln_modulation_linears_are_left_alone() -> None:
    # Adapting the modulation projections fights the checkpoint the way H3's adaln_proj does.
    targets = archs.get("flux1").target_modules
    assert not {"norm1.linear", "norm1_context.linear", "norm.linear", "norm_out.linear"} & set(
        targets
    )


def test_attention_scope_narrows_to_the_projections() -> None:
    narrowed = archs.target_modules(archs.get("flux1"), "attention")
    assert {"to_q", "to_k", "to_v", "to_out.0"}.issubset(set(narrowed))
    assert not {"proj_mlp", "proj_out", "ff.net.2", "x_embedder"} & set(narrowed)
    assert set(narrowed) < set(archs.get("flux1").target_modules)


def test_one_training_step_produces_a_prediction_shaped_like_its_target() -> None:
    torch = pytest.importorskip("torch")
    a = archs.get("flux1")
    torch.manual_seed(0)
    model = _tiny_model()

    clean = torch.randn(16, 32, 32)  # a 16-channel H/8 latent, as the VAE produces
    noise = torch.randn_like(clean)
    sigma = a.sigma("cpu", 3.0)
    noisy = (1 - sigma) * clean + sigma * noise
    item = {"embed": torch.randn(77, _EMBED), "pooled": torch.randn(_POOLED)}
    pred = a.forward(model, noisy, a.timestep(sigma), item)

    assert pred.shape == a.target(clean, noise).shape
    assert torch.isfinite(pred).all()


def test_the_forward_passes_guidance_of_one_and_the_pooled_projections() -> None:
    """Neither is visible in a shape test, and both are wrong in ways that only show up as a bad
    adapter: dev generates at 3.5, and training the embedder there teaches the LoRA to change how
    guidance behaves rather than what the images look like."""
    torch = pytest.importorskip("torch")
    a = archs.get("flux1")
    # No model here, so these are the real widths: T5's 4096 sequence and CLIP's 768 pooled vector.
    seen: dict[str, object] = {}

    def capture(**kwargs: object) -> tuple[object]:
        seen.update(kwargs)
        return (torch.zeros(1, 256, 64),)

    a.forward(
        capture, torch.randn(16, 32, 32), torch.tensor(0.5),
        {"embed": torch.randn(77, 4096), "pooled": torch.randn(768)},
    )
    guidance = seen["guidance"]
    assert tuple(guidance.shape) == (1,)
    assert float(guidance[0]) == 1.0, "dev is guidance-distilled; training pins guidance at 1"
    assert tuple(seen["pooled_projections"].shape) == (1, 768)
    # 2-D on purpose: a 3-D txt_ids is deprecated and silently indexed back down.
    assert seen["txt_ids"].dim() == 2


def test_a_two_tensor_conditioning_round_trips_through_the_precache_store(tmp_path: Path) -> None:
    """FLUX.1 is the first arch whose conditioning is two tensors, so the on-disk cache has to carry
    both. Nothing in the store is arch-aware, which is what makes this worth pinning rather than
    assuming."""
    torch = pytest.importorskip("torch")
    from inline_core.training import precache_store as ps

    items = [{"latent": torch.randn(4, 8, 8), "embed": torch.randn(77, 16),
              "pooled": torch.randn(32)}]
    uncond = {"embed": torch.randn(77, 16), "pooled": torch.randn(32)}
    ps.save(tmp_path, "k", items, uncond, 3.0)
    loaded, loaded_uncond, shift = ps.load(tmp_path, "k")

    assert shift == 3.0
    assert torch.equal(loaded[0]["pooled"], items[0]["pooled"])
    assert torch.equal(loaded[0]["embed"], items[0]["embed"])
    assert loaded_uncond is not None
    assert torch.equal(loaded_uncond["pooled"], uncond["pooled"])


def test_the_pooled_vector_takes_the_compute_dtype() -> None:
    """Left out of the activation set it would still reach the device, but stay fp32 while the
    model runs bf16 - a dtype error at best, a silent upcast under autocast at worst."""
    torch = pytest.importorskip("torch")
    from inline_core.training import trainer

    assert "pooled" in trainer._ACTIVATION_KEYS
    item = {"latent": torch.randn(2, 2), "embed": torch.randn(2, 2), "pooled": torch.randn(4)}
    moved = trainer._to_device(item, torch.device("cpu"), torch.float16)
    assert moved["pooled"].dtype is torch.float16


def test_a_lora_attaches_to_both_block_types_and_receives_gradient() -> None:
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    a = archs.get("flux1")
    torch.manual_seed(0)
    model = _tiny_model()
    model.requires_grad_(False)
    model.add_adapter(
        peft.LoraConfig(r=4, lora_alpha=4, target_modules=a.target_modules, init_lora_weights=False)
    )
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora" in n for n, _ in trainable)

    clean = torch.randn(16, 32, 32)
    noise = torch.randn_like(clean)
    sigma = a.sigma("cpu", 3.0)
    noisy = (1 - sigma) * clean + sigma * noise
    pred = a.forward(
        model, noisy, a.timestep(sigma),
        {"embed": torch.randn(77, _EMBED), "pooled": torch.randn(_POOLED)},
    )
    torch.nn.functional.mse_loss(pred.float(), a.target(clean, noise).float()).backward()

    got = [n for n, p in trainable if p.grad is not None and p.grad.abs().sum() > 0]
    assert got, "no adapter parameter received gradient"
    assert any("transformer_blocks." in n and "single" not in n for n in got)
    assert any("single_transformer_blocks." in n for n in got)


# --- which checkpoint a run trains against -------------------------------------------------------


@pytest.fixture
def models_root(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "models"
    (root / "diffusion_models").mkdir(parents=True)
    monkeypatch.setenv("INLINE_MODELS_DIR", str(root))
    from inline_core.models.flux1 import requirements as reqs

    reqs._IDENTIFIED.clear()  # keyed by path, and tmp_path is reused across tests
    return root


def _put(root: Path, name: str, header: dict | None = None) -> Path:
    return _write_header_only(root / "diffusion_models" / name, header or BFL_DEV_HEADER)


def test_dev_is_the_training_base(models_root: Path) -> None:
    _put(models_root, "flux1-dev.safetensors")
    assert models._base_file(models_root, "flux1", "raw").endswith("flux1-dev.safetensors")


def test_schnell_is_refused_with_a_pointer_to_dev(models_root: Path) -> None:
    # Identified by content, not by name: schnell is the one build with no guidance embedder.
    schnell = {k: v for k, v in BFL_DEV_HEADER.items() if not k.startswith("guidance_in.")}
    _put(models_root, "flux1-schnell.safetensors", schnell)
    with pytest.raises(RuntimeError, match="schnell"):
        models._base_file(models_root, "flux1", "raw")


def test_an_empty_models_dir_says_where_to_get_a_checkpoint(models_root: Path) -> None:
    with pytest.raises(RuntimeError, match="model popup"):
        models._base_file(models_root, "flux1", "raw")


def test_flux1_has_no_de_distillation_adapter(models_root: Path) -> None:
    """dev is *guidance*-distilled, not step-distilled, so it is itself the training base - unlike
    Z-Image and Krea 2 there is nothing to fuse first."""
    with pytest.raises(RuntimeError, match="guidance-distilled"):
        models._adapter_path(models_root, "flux1", "turbo_adapter")
    assert models._adapter_path(models_root, "flux1", "raw") is None


def test_flux1_can_train_in_4bit() -> None:
    # A 24GB bf16 base needs the NF4 rung to reach the cards people have.
    assert archs.FLUX1 in models._QUANTIZABLE


def test_the_four_components_are_all_required_for_a_run() -> None:
    from inline_core.models import trainingreqs

    ids = [c.id for c in trainingreqs.base_components("flux1", "raw")]
    # No row swap, unlike FLUX.2: the popup's default checkpoint already is the training base.
    assert ids == ["diffusion", "text_encoder", "clip", "vae"]

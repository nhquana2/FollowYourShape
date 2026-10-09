"""Small CPU checks; no model downloads, checkpoints, or GPU are used."""

import json

import numpy as np
from PIL import Image
import pytest
import torch

from flux.model import Flux, FluxParams
from flux.sampling import build_inject_list, denoise, denoise_with_TDM
from flux.attn_diff import MidpointAttentionDifference, PerStepAttentionMask, parse_attn_layers


def test_layer_selection_validation():
    assert parse_attn_layers("13, 15,18") == (13, 15, 18)
    for value in ("", "1,", "-1", "1,1", "one"):
        with pytest.raises(ValueError):
            parse_attn_layers(value)
    with pytest.raises(ValueError, match="indices"):
        MidpointAttentionDifference([19], [0], num_blocks=19)


def test_cache_is_detached_and_distance_is_mean_of_layer_norms():
    cache = MidpointAttentionDifference([0, 2], [0], num_blocks=3)
    source = torch.zeros(1, 2, 2, dtype=torch.bfloat16, requires_grad=True)
    capture = cache.source_collector(0, 0.5)
    capture(0, source)
    capture(1, source)  # Unselected block.
    capture(2, source)
    with torch.no_grad():
        source.add_(100)
    assert set(cache.source[0]) == {0, 2}
    assert cache.source[0][0].dtype == torch.bfloat16
    assert not cache.source[0][0].requires_grad
    assert torch.count_nonzero(cache.source[0][0]) == 0

    compare = cache.target_collector(0, 0.5)
    assert not cache.source  # Ownership transferred to the streaming collector.
    compare(0, torch.tensor([[[3, 4], [0, 2]]], dtype=torch.bfloat16))
    compare(2, torch.tensor([[[-3, -4], [0, -4]]], dtype=torch.bfloat16))
    result = compare.result()
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, torch.tensor([[5.0, 3.0]]))
    assert not compare.pending


def test_incomplete_or_misaligned_cache_fails_explicitly():
    cache = MidpointAttentionDifference([0, 1], [0], num_blocks=2)
    assert cache.source_collector(1, 0.1) is None
    assert cache.target_collector(1, 0.1) is None
    with pytest.raises(RuntimeError, match="inversion first"):
        cache.target_collector(0, 0.5)
    capture = cache.source_collector(0, 0.5)
    capture(0, torch.zeros(1, 2, 2))
    with pytest.raises(RuntimeError, match="Missing source"):
        cache.validate_source(0)
    capture(1, torch.zeros(1, 2, 2))
    with pytest.raises(ValueError, match="Midpoint mismatch"):
        cache.target_collector(0, 0.4)
    compare = cache.target_collector(0, 0.5)
    compare(0, torch.zeros(1, 2, 2))
    with pytest.raises(RuntimeError, match="Missing target"):
        compare.result()


def test_collection_reads_preprojection_image_output_without_changing_prediction():
    torch.manual_seed(7)
    model = Flux(FluxParams(
        in_channels=4, vec_in_dim=4, context_in_dim=8, hidden_size=32,
        mlp_ratio=1.0, num_heads=4, depth=3, depth_single_blocks=1,
        axes_dim=[2, 2, 4], theta=10000, qkv_bias=True, guidance_embed=True,
    )).eval()
    inputs = dict(
        img=torch.randn(1, 6, 4), img_ids=torch.zeros(1, 6, 3),
        txt=torch.randn(1, 4, 8), txt_ids=torch.zeros(1, 4, 3),
        timesteps=torch.tensor([0.5]), y=torch.randn(1, 4), guidance=torch.ones(1),
    )
    expected = {}
    handles = [
        model.double_blocks[layer].img_attn.proj.register_forward_pre_hook(
            lambda module, args, layer=layer: expected.__setitem__(layer, args[0].detach().clone())
        )
        for layer in (0, 2)
    ]
    cache = MidpointAttentionDifference([0, 2], [0], num_blocks=3)
    try:
        with torch.no_grad():
            baseline, _ = model(**inputs)
            captured, returned_info = model(**inputs, attn_capture=cache.source_collector(0, 0.5))
    finally:
        for handle in handles:
            handle.remove()
    assert returned_info is None
    torch.testing.assert_close(captured, baseline, rtol=0, atol=0)
    for layer in (0, 2):
        assert cache.source[0][layer].shape == (1, 6, 32)
        torch.testing.assert_close(cache.source[0][layer], expected[layer], rtol=0, atol=0)


class TinyProbeModel:
    """Deterministic stand-in exercising the real solver and visualization path."""

    def __init__(self, constant_attention=False, vary_attention=False):
        self.calls = []
        self.constant_attention = constant_attention
        self.vary_attention = vary_attention
        self.kv_cache = set()

    def __call__(self, *, img, timesteps, txt, info=None, attn_capture=None, **kwargs):
        self.calls.append({
            "condition": float(txt.mean()), "latent": img.clone(),
            "time": float(timesteps[0]), "capture": attn_capture is not None,
            "inverse": info is not None and info["inverse"],
            "controlled": info is not None,
            "has_edit_map": info is not None and info.get("edit_map") is not None,
            "inject": info is not None and info["inject"],
            "second_order": info["second_order"] if info is not None else None,
            "edit_indices": info["edit_map"].tolist() if info is not None and info.get("edit_map") is not None else None,
            "edit_weight": info["edit_weight"].tolist() if info is not None and info.get("edit_weight") is not None else None,
        })
        if info is not None and info.get("dynamic_mask") is not None and info["inject"]:
            key = (info["t"], info["second_order"])
            if info["inverse"]:
                self.kv_cache.add(key)
            else:
                self.kv_cache.remove(key)  # Fails if the required inversion evaluation was not cached.
        positions = torch.arange(1, img.shape[1] + 1, dtype=img.dtype).view(1, -1, 1)
        condition = txt.mean()
        velocity = 0.1 * img + 0.02 * timesteps[:, None, None] + 0.01 * condition * positions
        if attn_capture is not None:
            if self.constant_attention:
                features = torch.zeros(img.shape[0], img.shape[1], 2)
            else:
                features = torch.cat((condition * positions, condition * positions.square()), dim=-1)
                if self.vary_attention and timesteps[0] < 0.6:
                    features = features.flip(1)
            for layer in range(3):
                attn_capture(layer, features * (layer + 1))
        return velocity, info


def run_tiny_edit(vis_path, attention=True, constant_attention=False, captured_steps=None,
                  front=0, schedule=None, latent="cross", signal="attention", **mask_options):
    # A nonuniform schedule catches incorrectly paired inversion intervals.
    if schedule is None:
        schedule = [1.0, 0.8, 0.55, 0.3, 0.1, 0.0]
    num_steps = len(schedule) - 1
    inject_list = build_inject_list(len(schedule), inject_step=1, front_pad=front, tail_pad=1)
    info = {"inject_step": 1, "feature": {}}
    if vis_path is not None:
        info["vis_path"] = str(vis_path)
    inputs = dict(
        img=torch.zeros(1, 6, 4), img_ids=torch.zeros(1, 6, 3),
        txt=torch.ones(1, 4, 8), txt_ids=torch.zeros(1, 4, 3), vec=torch.ones(1, 4),
    )
    if attention:  # Attention Difference always runs with its per-step masks.
        steps = range(num_steps) if captured_steps is None else captured_steps
        # The target prompt is twice the source one; a same-latent difference needs the other of the two.
        other = (2 if latent == "source" else 1) * inputs["txt"]
        info["attn_diff"] = MidpointAttentionDifference(
            [0, 2], steps, num_blocks=3, latent=latent,
            cond=dict(txt=other, txt_ids=inputs["txt_ids"], y=inputs["vec"]), signal=signal,
        )
        info["dynamic_mask"] = PerStepAttentionMask(num_steps, front, num_steps - 4, tail=1, **mask_options)
    model = TinyProbeModel(constant_attention, vary_attention=attention)
    noise, info = denoise(model, **inputs, timesteps=schedule, inverse=True, info=info,
                          inject_list=inject_list, guidance=1)
    inputs["img"] = noise
    inputs["txt"] = 2 * inputs["txt"]
    # Retain edit.py's existing height-first width/height argument convention.
    result, info = denoise_with_TDM(
        model, **inputs, timesteps=schedule, inverse=False, info=info,
        inject_list=inject_list, guidance=2, width=32, height=48, front_pad=front, tail_pad=1,
    )
    return result, info, model, schedule


def test_midpoint_pairing_of_inversion_and_target_probes(tmp_path):
    result, info, model, schedule = run_tiny_edit(tmp_path)
    num_steps = len(schedule) - 1
    assert len(model.calls) == 6 * num_steps  # Original solver/probe call count.
    source_captures = [call for call in model.calls if call["capture"] and call["inverse"]]
    target_captures = [call for call in model.calls if call["capture"] and not call["inverse"]]
    expected_midpoints = [(a + b) / 2 for a, b in zip(schedule[:-1], schedule[1:])]
    assert [call["time"] for call in source_captures] == pytest.approx(expected_midpoints[::-1])
    assert [call["time"] for call in target_captures] == pytest.approx(expected_midpoints)
    assert all(not call["controlled"] for call in target_captures)
    assert not info["attn_diff"].source
    assert torch.isfinite(result).all()
    config = json.loads((tmp_path / "attn_diff_config.json").read_text())
    assert config["evaluation"] == "midpoint"
    assert config["signal"] == "attention"
    assert config["block_indices_zero_based"] == [0, 2]
    assert config["soft_mask"] == {"gaussian_sigma": 0.7, "percentiles": [50, 98], "center": 0.3, "steepness": 15}


@pytest.mark.parametrize("latent", ["source", "target"])
def test_same_latent_compares_one_latent_under_both_prompts(tmp_path, latent):
    result, info, model, schedule = run_tiny_edit(tmp_path, latent=latent)
    num_steps = len(schedule) - 1
    assert torch.isfinite(result).all()
    assert len(model.calls) == 7 * num_steps  # One extra evaluation per captured step.
    assert not info["attn_diff"].source and not info["attn_diff"].delta
    inversion_calls = (3 if latent == "source" else 2) * num_steps
    captured = [index for index, call in enumerate(model.calls) if call["capture"]]
    assert len(captured) == 2 * num_steps
    assert all((index < inversion_calls) == (latent == "source") for index in captured)
    midpoints = [(a + b) / 2 for a, b in zip(schedule[:-1], schedule[1:])]
    conditions = (1.0, 2.0) if latent == "source" else (2.0, 1.0)  # Trajectory prompt first, then the other.
    for first, second, midpoint in zip(captured[::2], captured[1::2], midpoints[::-1] if latent == "source" else midpoints):
        assert second == first + 1
        first, second = model.calls[first], model.calls[second]
        assert first["time"] == second["time"] == pytest.approx(midpoint)
        torch.testing.assert_close(first["latent"], second["latent"], rtol=0, atol=0)
        assert (first["condition"], second["condition"]) == conditions
        assert first["controlled"] == (latent == "source") and not second["controlled"]
    # Layer scales 1 and 3 average to 2; patch p differs by (p, p^2) between the two prompts.
    diagnostics = json.loads((tmp_path / "mask_diagnostics.json").read_text())
    for item in diagnostics:
        assert item["raw_divergence"]["min"] == pytest.approx(2 * 2 ** 0.5)
        assert item["raw_divergence"]["max"] == pytest.approx(2 * (36 + 1296) ** 0.5)
    assert json.loads((tmp_path / "attn_diff_config.json").read_text())["latent"] == latent


def test_same_latent_window_only_capture_and_validation():
    for latent in ("source", "target"):
        _, info, model, schedule = run_tiny_edit(None, captured_steps=[0, 1], freeze=True, latent=latent)
        assert len(model.calls) == 6 * (len(schedule) - 1) + 2
        assert sum(call["capture"] for call in model.calls) == 4
        assert info["dynamic_mask"].mask_step == 1 and not info["attn_diff"].delta
    with pytest.raises(ValueError, match="cross, source or target"):
        MidpointAttentionDifference([0], [0], num_blocks=1, latent="both")
    with pytest.raises(ValueError, match="conditioning"):
        MidpointAttentionDifference([0], [0], num_blocks=1, latent="source")


def test_default_velocity_path_does_not_collect_attention(tmp_path):
    result, info, model, _ = run_tiny_edit(tmp_path, attention=False)
    assert not any(call["capture"] for call in model.calls)
    assert torch.isfinite(result).all()
    assert info["edit_map"] is not None
    assert (tmp_path / "edit_map.png").is_file()
    assert not (tmp_path / "attn_diff_config.json").exists()


def test_velocity_signal_builds_per_step_masks_from_the_original_tdm_map(tmp_path):
    reference, _, _, schedule = run_tiny_edit(tmp_path / "tdm", attention=False)
    result, info, model, _ = run_tiny_edit(tmp_path / "velocity", signal="velocity")
    num_steps = len(schedule) - 1
    assert len(model.calls) == 6 * num_steps
    assert not any(call["capture"] for call in model.calls)
    assert not info["attn_diff"].source
    # The tiny model's velocity ignores injection, so both runs follow one trajectory and share their maps.
    torch.testing.assert_close(result, reference, rtol=0, atol=0)
    for step in range(num_steps):
        name = f"delta/delta_map_{step}.png"
        assert np.array_equal(np.array(Image.open(tmp_path / "velocity" / name)),
                              np.array(Image.open(tmp_path / "tdm" / name)))
    config = json.loads((tmp_path / "velocity" / "attn_diff_config.json").read_text())
    assert config["signal"] == "velocity"
    assert config["injection_steps_zero_based"] == list(range(num_steps - 1))
    diagnostics = json.loads((tmp_path / "velocity" / "mask_diagnostics.json").read_text())
    assert [item["mask_updated"] for item in diagnostics] == [True] * (num_steps - 1) + [False]
    assert all(item["raw_divergence"]["max"] > 0 for item in diagnostics)
    with pytest.raises(ValueError, match="attention or velocity"):
        MidpointAttentionDifference([0], [0], num_blocks=1, signal="both")
    with pytest.raises(ValueError, match="attention signal"):
        MidpointAttentionDifference([0], [0], num_blocks=1, latent="source", cond={}, signal="velocity")

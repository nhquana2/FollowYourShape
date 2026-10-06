"""Per-step soft-mask scheduling and real KV blending checks, all on CPU."""

import json

import numpy as np
from PIL import Image
import pytest
import torch

import flux.modules.layers as layers
from flux.attn_diff import MidpointAttentionDifference, PerStepAttentionMask
from test_attn_diff import run_tiny_edit


@pytest.mark.parametrize("constant_attention", [False, True])
def test_same_step_injection_freezing_and_saved_applied_masks(tmp_path, constant_attention):
    result, info, model, schedule = run_tiny_edit(
        tmp_path, front=1, constant_attention=constant_attention,
        schedule=[1.0, 0.9, 0.75, 0.6, 0.4, 0.25, 0.1, 0.0], freeze=True,
    )
    num_steps = len(schedule) - 1
    assert torch.isfinite(result).all()
    assert len(model.calls) == num_steps * 6  # No additional model evaluations.
    assert not model.kv_cache  # Every needed source evaluation was cached and consumed.
    assert not info["attn_diff"].source
    assert not info["map"]  # No temporal accumulation in this mode.

    controlled = [call for call in model.calls if call["controlled"] and not call["inverse"]]
    probes = [call for call in model.calls[2 * num_steps:] if not call["controlled"]]
    assert len(probes) == 2 * num_steps
    assert all(not call["inject"] and call["edit_indices"] is None for call in probes)
    masks = [np.load(tmp_path / "masks" / f"edit_map_{i}.npy") for i in range(num_steps)]
    for step, mask in enumerate(masks):
        start, midpoint = controlled[2 * step:2 * step + 2]
        assert not start["second_order"] and midpoint["second_order"]
        assert start["inject"] == midpoint["inject"] == (step < num_steps - 1)
        assert start["edit_indices"] == midpoint["edit_indices"]
        assert start["edit_weight"] == midpoint["edit_weight"]
        if 1 <= step < num_steps - 1:
            assert start["edit_indices"] == np.flatnonzero(mask).tolist()
            # The soft mask weights V; the patches above 0.5 are the ones that switch K.
            assert np.flatnonzero(np.array(start["edit_weight"]) > 0.5).tolist() == start["edit_indices"]
        else:  # Front and tail steps use no soft weights.
            assert start["edit_weight"] is None
        with Image.open(tmp_path / "masks" / f"edit_map_{step}.png") as image:
            assert image.size == (3, 2)
        assert (tmp_path / "delta" / f"delta_map_{step}.png").is_file()

    assert not masks[0].any()  # Original front stabilization.
    assert masks[-1].all()  # Original uninjected tail, all target KV.
    np.testing.assert_array_equal(masks[3], masks[4])  # Gap after the mask window.
    np.testing.assert_array_equal(masks[3], masks[5])  # Original late injection stage.
    np.testing.assert_array_equal(masks[3], np.load(tmp_path / "edit_map.npy"))
    soft = np.load(tmp_path / "soft_mask.npy")
    np.testing.assert_array_equal(soft > 0.5, masks[3].astype(bool))
    assert soft.min() > 0 and soft.max() <= 1
    assert info["edit_map"].tolist() == np.flatnonzero(masks[3]).tolist()
    if constant_attention:
        assert all(not mask.any() for mask in masks[1:-1])
    else:
        assert not np.array_equal(masks[1], masks[3])  # Fresh maps actually change the applied mask.
        assert all(mask.any() and not mask.all() for mask in masks[1:-1])
    for name in ("edit_map.png", "soft_edit_map.png"):
        with Image.open(tmp_path / name) as image:
            image.verify()

    diagnostics = json.loads((tmp_path / "mask_diagnostics.json").read_text())
    assert [item["step"] for item in diagnostics if item["mask_updated"]] == [1, 2, 3]
    assert [item["step"] for item in diagnostics if item["frozen"]] == [4, 5]
    assert diagnostics[4]["changed_patches"] == 0
    assert all(item["raw_divergence"] is not None for item in diagnostics)
    config = json.loads((tmp_path / "attn_diff_config.json").read_text())
    assert config["mask_update_steps_zero_based"] == [1, 2, 3]
    assert config["freeze_step_zero_based"] == 3
    assert config["injection_steps_zero_based"] == list(range(6))


def test_masks_update_at_every_injected_step_by_default(tmp_path):
    _, info, model, schedule = run_tiny_edit(tmp_path, front=1, schedule=[1.0, 0.9, 0.75, 0.6, 0.4, 0.25, 0.1, 0.0])
    num_steps = len(schedule) - 1
    assert len(model.calls) == num_steps * 6  # No additional model evaluations.
    assert not model.kv_cache and not info["attn_diff"].source
    controlled = [call for call in model.calls if call["controlled"] and not call["inverse"]]
    masks = [np.load(tmp_path / "masks" / f"edit_map_{i}.npy") for i in range(num_steps)]
    for step in range(1, num_steps - 1):  # Every injected step after the front applies its own mask.
        assert controlled[2 * step]["edit_indices"] == np.flatnonzero(masks[step]).tolist()
        assert (tmp_path / "masks" / f"soft_edit_map_{step}.png").is_file()
    np.testing.assert_array_equal(masks[5], np.load(tmp_path / "edit_map.npy"))
    assert info["edit_map"].tolist() == np.flatnonzero(masks[5]).tolist()
    diagnostics = json.loads((tmp_path / "mask_diagnostics.json").read_text())
    assert [item["step"] for item in diagnostics if item["mask_updated"]] == [1, 2, 3, 4, 5]
    assert not any(item["frozen"] for item in diagnostics)
    config = json.loads((tmp_path / "attn_diff_config.json").read_text())
    assert config["mask_update_steps_zero_based"] == [1, 2, 3, 4, 5]
    assert config["freeze_step_zero_based"] is None


def test_soft_mask_rescales_between_percentiles_before_the_sigmoid():
    values = np.arange(101, dtype=np.float64).reshape(1, 101)  # The percentile p of this map is the value p.
    policy = PerStepAttentionMask(5, front=0, cut=1)
    assert (policy.percentiles, policy.center, policy.steepness, policy.sigma) == ((50, 98), 0.3, 15, 0.7)
    soft = policy.soft_mask(values)[0]
    sigmoid = lambda u: 1 / (1 + np.exp(-15 * (u - 0.3)))
    np.testing.assert_allclose(soft[:51], sigmoid(0), rtol=1e-6)  # At or below the lower percentile.
    np.testing.assert_allclose(soft[98:], sigmoid(1), rtol=1e-6)  # At or above the upper percentile.
    np.testing.assert_allclose(soft[74], sigmoid(0.5), rtol=1e-6)
    # The mask is the part above the centre: 50 + 0.3 * (98 - 50) = 64.4.
    np.testing.assert_array_equal(np.flatnonzero(soft > 0.5), np.arange(65, 101))

    loose = PerStepAttentionMask(5, front=0, cut=1, percentiles=(10, 90), center=0.5, steepness=10).soft_mask(values)[0]
    np.testing.assert_array_equal(np.flatnonzero(loose > 0.5), np.arange(51, 101))
    flat = policy.soft_mask(np.zeros((2, 3)))  # A map without signal selects nothing.
    assert not (flat > 0.5).any()


def test_dynamic_window_only_capture_without_visualization():
    _, info, model, _ = run_tiny_edit(None, captured_steps=[0, 1], freeze=True)
    assert not model.kv_cache
    assert info["dynamic_mask"].mask_step == 1
    assert sum(call["capture"] for call in model.calls) == 4
    _, info, model, _ = run_tiny_edit(None, captured_steps=[0, 1, 2, 3])
    assert info["dynamic_mask"].mask_step == 3
    assert sum(call["capture"] for call in model.calls) == 8


def test_dynamic_schedule_validation():
    with pytest.raises(ValueError, match="nonempty"):
        PerStepAttentionMask(5, front=2, cut=1)
    policy = PerStepAttentionMask(5, front=0, cut=1)
    with pytest.raises(ValueError, match="require attention"):
        policy.validate(None, num_steps=5)
    with pytest.raises(ValueError, match="every mask update"):
        policy.validate(MidpointAttentionDifference([0], [1], num_blocks=1), num_steps=5)
    assert list(policy.update_steps) == [0, 1, 2, 3]  # Every injected step; frozen masks stop at the cut.
    assert list(PerStepAttentionMask(5, front=0, cut=1, freeze=True).update_steps) == [0, 1]
    for percentiles in ((50,), (98, 50), (-1, 98), (50, 101)):
        with pytest.raises(ValueError, match="Percentiles"):
            PerStepAttentionMask(5, front=0, cut=1, percentiles=percentiles)
    with pytest.raises(ValueError, match="Steepness"):
        PerStepAttentionMask(5, front=0, cut=1, steepness=0)
    with pytest.raises(ValueError, match="schedule"):
        policy.validate(MidpointAttentionDifference([0], [0, 1], num_blocks=1), num_steps=6)


@pytest.mark.parametrize("second_order", [False, True])
@pytest.mark.parametrize("edit_indices", [[], [1, 4], list(range(6))])
def test_real_block_blends_current_mask_and_consumes_immutable_source_cache(monkeypatch, second_order, edit_indices):
    torch.manual_seed(17)
    block = layers.SingleStreamBlock(hidden_size=32, num_heads=4, mlp_ratio=1).eval()
    pe = layers.EmbedND(dim=8, theta=10000, axes_dim=[2, 2, 4])(torch.zeros(1, 518, 3))
    source = torch.randn(1, 518, 32)
    target = torch.randn_like(source)
    vec = torch.randn(1, 32)
    captured = []
    attention = layers.attention

    def observe(q, k, v, pe):
        captured.append((k.detach().clone(), v.detach().clone()))
        return attention(q, k, v, pe)

    monkeypatch.setattr(layers, "attention", observe)
    info = dict(inverse=True, inject=True, id=20, t=0.8, second_order=second_order,
                type="single", feature={}, edit_map=None,
                dynamic_mask=PerStepAttentionMask(5, front=0, cut=1))
    with torch.no_grad():
        block(source, vec, pe, info)
        source_references = dict(info["feature"])
        source_copies = {key: value.clone() for key, value in source_references.items()}
        block(target, vec, pe, None)  # Native target KV.
        info["inverse"] = False
        info["edit_map"] = torch.tensor(edit_indices, dtype=torch.long)
        block(target, vec, pe, info)
    assert not info["feature"]
    for key, value in source_references.items():
        torch.testing.assert_close(value, source_copies[key], rtol=0, atol=0)
    for source_kv, target_kv, blended in zip(captured[0], captured[1], captured[2]):
        expected = source_kv.clone()
        expected[:, :, :512] = target_kv[:, :, :512]
        indices = 512 + torch.tensor(edit_indices, dtype=torch.long)
        expected[:, :, indices] = target_kv[:, :, indices]
        torch.testing.assert_close(blended, expected, rtol=0, atol=0)


@pytest.mark.parametrize("second_order", [False, True])
def test_real_block_blends_values_by_soft_weight_and_switches_keys_by_mask(monkeypatch, second_order):
    torch.manual_seed(23)
    block = layers.SingleStreamBlock(hidden_size=32, num_heads=4, mlp_ratio=1).eval()
    pe = layers.EmbedND(dim=8, theta=10000, axes_dim=[2, 2, 4])(torch.zeros(1, 518, 3))
    source = torch.randn(1, 518, 32)
    target = torch.randn_like(source)
    vec = torch.randn(1, 32)
    captured = []
    attention = layers.attention

    def observe(q, k, v, pe):
        captured.append((k.detach().clone(), v.detach().clone()))
        return attention(q, k, v, pe)

    monkeypatch.setattr(layers, "attention", observe)
    weight = torch.tensor([0.0, 0.9, 0.25, 0.5, 1.0, 0.6])
    edit_indices = torch.nonzero(weight > 0.5).squeeze(1)
    info = dict(inverse=True, inject=True, id=20, t=0.8, second_order=second_order,
                type="single", feature={}, edit_map=None,
                dynamic_mask=PerStepAttentionMask(5, front=0, cut=1))
    with torch.no_grad():
        block(source, vec, pe, info)
        block(target, vec, pe, None)  # Native target KV.
        info.update(inverse=False, edit_map=edit_indices, edit_weight=weight)
        block(target, vec, pe, info)
    assert not info["feature"]
    (source_k, source_v), (target_k, target_v), (blended_k, blended_v) = captured
    expected_k = source_k.clone()
    expected_k[:, :, :512] = target_k[:, :, :512]
    expected_k[:, :, 512 + edit_indices] = target_k[:, :, 512 + edit_indices]
    torch.testing.assert_close(blended_k, expected_k, rtol=0, atol=0)
    expected_v = target_v.clone()
    expected_v[:, :, 512:] = torch.lerp(source_v[:, :, 512:], target_v[:, :, 512:], weight[None, None, :, None])
    torch.testing.assert_close(blended_v, expected_v)

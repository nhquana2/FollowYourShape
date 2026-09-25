"""Dynamic-mask scheduling and real KV blending checks, all on CPU."""

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
        tmp_path, dynamic=True, front=1, constant_attention=constant_attention,
        schedule=[1.0, 0.9, 0.75, 0.6, 0.4, 0.25, 0.1, 0.0],
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
        if 1 <= step < num_steps - 1:
            assert start["edit_indices"] == np.flatnonzero(mask).tolist()
        with Image.open(tmp_path / "masks" / f"edit_map_{step}.png") as image:
            assert image.size == (3, 2)
        assert (tmp_path / "delta" / f"delta_map_{step}.png").is_file()

    assert not masks[0].any()  # Original front stabilization.
    assert masks[-1].all()  # Original uninjected tail, all target KV.
    np.testing.assert_array_equal(masks[3], masks[4])  # Gap after the mask window.
    np.testing.assert_array_equal(masks[3], masks[5])  # Original late injection stage.
    np.testing.assert_array_equal(masks[3], np.load(tmp_path / "edit_map.npy"))
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
    assert config["mask_mode"] == "per_step"
    assert config["aggregation_steps_zero_based"] == []
    assert config["mask_update_steps_zero_based"] == [1, 2, 3]
    assert config["injection_steps_zero_based"] == list(range(6))
    assert config["postprocessing"]["temporal_softmax_scale"] is None


def test_dynamic_window_only_capture_without_visualization():
    _, info, model, _ = run_tiny_edit(None, dynamic=True, captured_steps=[0, 1])
    assert not model.kv_cache
    assert info["dynamic_mask"].mask_step == 1
    assert sum(call["capture"] for call in model.calls) == 4


def test_dynamic_schedule_validation():
    with pytest.raises(ValueError, match="nonempty"):
        PerStepAttentionMask(5, front=2, cut=1)
    with pytest.raises(ValueError, match="require attention"):
        run_tiny_edit(None, dynamic=True, attention=False)
    policy = PerStepAttentionMask(5, front=0, cut=1)
    with pytest.raises(ValueError, match="every dynamic"):
        policy.validate(MidpointAttentionDifference([0], [1], num_blocks=1), num_steps=5)
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

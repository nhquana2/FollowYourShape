"""Midpoint attention collection and optional per-step mask control."""

import json
import math
from pathlib import Path
from typing import Callable, Iterable

import torch
from torch import Tensor


def parse_attn_layers(value: str) -> tuple[int, ...]:
    """Parse comma-separated, zero-based double-stream block indices."""
    try:
        layers = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise ValueError("Expected comma-separated block indices, e.g. 13,14,15,16,17,18") from exc
    if not layers or any(layer < 0 for layer in layers) or len(set(layers)) != len(layers):
        raise ValueError("Attention block indices must be nonnegative and unique")
    return layers


class StepAttentionDifference:
    """Reduce each selected block immediately, without caching target features."""

    def __init__(self, source: dict[int, Tensor]):
        self.pending = source
        self.num_layers = len(source)
        self.total: Tensor | None = None

    def __call__(self, layer: int, output: Tensor) -> None:
        if layer not in self.pending:
            return
        source = self.pending.pop(layer)
        if source.shape != output.shape:
            raise ValueError(f"Attention shape mismatch at block {layer}: {source.shape} vs {output.shape}")
        # The source cache stays in its original dtype. Only the current block's
        # subtraction and channel reduction use FP32 on the target device.
        difference = output.detach().float() - source.to(device=output.device, dtype=torch.float32)
        distance = torch.linalg.vector_norm(difference, dim=-1)
        self.total = distance if self.total is None else self.total + distance

    def result(self) -> Tensor:
        if self.pending or self.total is None:
            raise RuntimeError(f"Missing target attention outputs for blocks {sorted(self.pending)}")
        return self.total / self.num_layers


class MidpointAttentionDifference:
    """Cache source features by denoising interval index, not float timestep keys.

    Inversion visits intervals in reverse order. Its midpoint outputs can be
    paired with the uninjected target probe at the same interval midpoint.
    """

    def __init__(self, layers: Iterable[int], steps: Iterable[int], num_blocks: int):
        self.layers = tuple(layers)
        if (
            not self.layers
            or len(set(self.layers)) != len(self.layers)
            or any(layer < 0 or layer >= num_blocks for layer in self.layers)
        ):
            raise ValueError(f"Attention layers must be unique indices in [0, {num_blocks - 1}]")
        self.steps = frozenset(steps)
        self.source: dict[int, dict[int, Tensor]] = {}
        self.midpoints: dict[int, float] = {}

    def source_collector(self, step: int, midpoint: float) -> Callable[[int, Tensor], None] | None:
        if step not in self.steps:
            return None
        if step in self.source:
            raise RuntimeError(f"Source attention already recorded for interval {step}")
        features: dict[int, Tensor] = {}
        self.source[step] = features
        self.midpoints[step] = midpoint

        def collect(layer: int, output: Tensor) -> None:
            if layer in self.layers:
                # Copy even on CPU so the cache neither aliases model activations
                # nor retains the storage for the text part of joint attention.
                features[layer] = output.detach().to(device="cpu", copy=True)

        return collect

    def validate_source(self, step: int) -> None:
        if step in self.steps:
            missing = set(self.layers) - self.source.get(step, {}).keys()
            if missing:
                raise RuntimeError(f"Missing source attention at interval {step}, blocks {sorted(missing)}")

    def target_collector(self, step: int, midpoint: float) -> StepAttentionDifference | None:
        if step not in self.steps:
            return None
        if step not in self.source:
            raise RuntimeError(f"No inversion attention cache for interval {step}; run inversion first")
        if not math.isclose(midpoint, self.midpoints[step], rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError(
                f"Midpoint mismatch at interval {step}: inversion={self.midpoints[step]}, target={midpoint}"
            )
        self.validate_source(step)
        # The collector consumes each layer as it is compared, releasing its CPU
        # storage instead of keeping all inversion features through denoising.
        return StepAttentionDifference(self.source.pop(step))


class PerStepAttentionMask:
    """Update a mask before each controlled solver step, then freeze it.

    All indices refer to the forward denoising schedule. The original front
    stabilization and uninjected tail are retained; middle steps now inject KV.
    """

    def __init__(self, num_steps: int, front: int, cut: int, tail: int = 1):
        self.num_steps = num_steps
        self.front = front
        self.freeze_step = cut
        self.inject_end = num_steps - tail
        if not 0 <= front <= cut < self.inject_end <= num_steps:
            raise ValueError("Dynamic attention difference requires a nonempty mask window before the uninjected tail")
        self.update_steps = range(front, self.freeze_step + 1)
        self.mask = None
        self.mask_step = None
        self.previous_applied = None
        self.diagnostics = []

    def injects(self, step: int) -> bool:
        return 0 <= step < self.inject_end

    def validate(self, attn_diff: MidpointAttentionDifference | None, num_steps: int) -> None:
        if attn_diff is None:
            raise ValueError("Dynamic masks require attention difference")
        if num_steps != self.num_steps:
            raise ValueError("Dynamic mask schedule does not match the sampler")
        if not set(self.update_steps).issubset(attn_diff.steps):
            raise ValueError("Attention capture must cover every dynamic mask update step")

    def process(self, step: int, delta_map: Tensor | None, raw_delta: Tensor | None,
                grid_shape: tuple[int, int], initial_edit_indices: Tensor | None = None,
                vis_dir: str | None = None) -> Tensor | None:
        """Return new edit indices only when updating; record the applied mask."""
        import numpy as np
        from scipy.ndimage import gaussian_filter
        from skimage.filters import threshold_otsu

        updated = step in self.update_steps
        edit_indices = None
        smoothed = None
        if updated:
            if delta_map is None:
                raise RuntimeError(f"Missing attention map for dynamic mask step {step}")
            smoothed = gaussian_filter(delta_map.detach().float().cpu().numpy(), sigma=0.7)
            self.mask = (smoothed > threshold_otsu(smoothed)).astype(np.uint8)
            self.mask_step = step
            edit_indices = torch.from_numpy(np.flatnonzero(self.mask)).to(delta_map.device)

        if not self.injects(step):
            # No source injection means all patches retain target KV.
            applied = np.ones(grid_shape, dtype=np.uint8)
        elif step < self.front:
            applied = np.zeros(grid_shape, dtype=np.uint8)
            if initial_edit_indices is not None:
                applied.flat[initial_edit_indices.detach().cpu().numpy()] = 1
        elif self.mask is None:
            raise RuntimeError("No dynamic mask available before selective injection")
        else:
            applied = self.mask.copy()

        stats = None
        if raw_delta is not None:
            values = raw_delta.detach().float()
            stats = {"min": values.min().item(), "max": values.max().item(),
                     "mean": values.mean().item(), "std": values.std(unbiased=False).item()}
        changed = None if self.previous_applied is None else int(np.count_nonzero(applied != self.previous_applied))
        self.diagnostics.append({
            "step": step, "injection": self.injects(step), "mask_updated": updated,
            "mask_source_step": self.mask_step,
            "frozen": self.injects(step) and step > self.freeze_step,
            "editable_patches": int(applied.sum()), "editable_fraction": float(applied.mean()),
            "changed_patches": changed, "raw_divergence": stats,
        })
        self.previous_applied = applied.copy()

        if vis_dir:
            import matplotlib.pyplot as plt

            root = Path(vis_dir)
            mask_dir = root / "masks"
            mask_dir.mkdir(parents=True, exist_ok=True)
            plt.imsave(mask_dir / f"edit_map_{step}.png", applied, cmap="viridis", vmin=0, vmax=1)
            np.save(mask_dir / f"edit_map_{step}.npy", applied)
            if updated:
                plt.imsave(mask_dir / f"soft_edit_map_{step}.png", smoothed, cmap="viridis", vmin=0, vmax=1)
            if step == self.freeze_step:
                plt.figure()
                plt.imshow(self.mask, cmap="viridis", vmin=0, vmax=1)
                plt.colorbar()
                plt.title("Edit Map")
                plt.savefig(root / "edit_map.png")
                plt.close()
                plt.imsave(root / "soft_edit_map.png", smoothed, cmap="viridis", vmin=0, vmax=1)
                np.save(root / "edit_map.npy", self.mask)
            (root / "mask_diagnostics.json").write_text(json.dumps(self.diagnostics, indent=2), encoding="utf-8")
        return edit_indices

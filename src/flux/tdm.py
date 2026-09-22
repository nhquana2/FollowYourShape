"""Read-only midpoint attention collection for the optional attention TDM."""

import math
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


class AttentionDifference:
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


class MidpointAttentionTDM:
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

    def target_collector(self, step: int, midpoint: float) -> AttentionDifference | None:
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
        return AttentionDifference(self.source.pop(step))

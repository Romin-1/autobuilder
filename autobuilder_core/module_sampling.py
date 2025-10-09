"""Utilities for sampling and adapting PlugNPlay modules for random pipelines."""

from __future__ import annotations

import inspect
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
from torch import nn

from .module_loader import load_class

SAMPLE_CHANNEL = 256
SAMPLE_SIZES = (128, 64, 32, 16)
SAMPLE_FEATURES = [
    torch.randn(1, SAMPLE_CHANNEL, size, size) for size in SAMPLE_SIZES
]


def _infer_default(name: str) -> object | None:
    """Best-effort guess for required constructor parameters.

    Returns ``None`` when the argument cannot be inferred.
    """

    normalized = name.lower()
    if normalized in {"in_channels", "out_channels", "in_ch", "out_ch", "channels", "channel"}:
        return SAMPLE_CHANNEL
    if normalized in {"in_features", "out_features", "dim", "inc", "outc"}:
        return SAMPLE_CHANNEL
    if normalized in {"kernel_size", "k"}:
        return 3
    if normalized in {"stride", "s"}:
        return 1
    if normalized in {"padding", "p"}:
        return 1
    if normalized in {"dilation", "d"}:
        return 1
    if normalized in {"group", "groups"}:
        return 1
    if normalized in {"expend_ratio", "expand_ratio"}:
        return 2
    if normalized in {"sa_num_heads", "ca_num_heads", "num_heads"}:
        return 4
    if normalized == "bias":
        return False
    if normalized in {"width", "height"}:
        return 32
    return None


def _instantiate_with_heuristics(klass: type[nn.Module]) -> nn.Module | None:
    """Instantiate ``klass`` if all required arguments can be inferred."""

    signature = inspect.signature(klass.__init__)
    kwargs = {}
    for parameter in list(signature.parameters.values())[1:]:
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if parameter.default is inspect._empty:
            inferred = _infer_default(parameter.name)
            if inferred is None:
                return None
            kwargs[parameter.name] = inferred
    try:
        return klass(**kwargs)
    except Exception:
        return None


@dataclass
class ModuleCandidate:
    name: str
    adapter: "ModuleAdapter"
    path: Path


class ModuleAdapter(nn.Module):
    """Adapts arbitrary modules to a common multi-scale feature interface."""

    def __init__(self, name: str, module: nn.Module) -> None:
        super().__init__()
        self.name = name
        self.module = module
        self.mode = self._infer_mode()
        if self.mode is None:
            raise ValueError(f"Unable to determine how to use module '{name}'.")
        self.eval()
        with torch.no_grad():
            try:
                self.forward(SAMPLE_FEATURES)
            except Exception as exc:
                raise ValueError(
                    f"Module '{name}' could not process sample feature maps."
                ) from exc

    def _infer_mode(self) -> str | None:
        module = self.module
        module.eval()
        with torch.no_grad():
            try:
                result = module(list(SAMPLE_FEATURES))
            except Exception:
                pass
            else:
                if isinstance(result, (list, tuple)) and len(result) == len(SAMPLE_FEATURES):
                    return "pyramid"
        forward_signature = inspect.signature(module.forward)
        parameters = [
            p
            for p in list(forward_signature.parameters.values())[1:]
            if p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        required = [p for p in parameters if p.default is inspect._empty]
        if not required:
            return "per_level"
        if len(required) == 1:
            return "per_level"
        if len(required) == 2:
            return "dual"
        if len(required) >= 3:
            return "triple"
        return None

    def forward(self, features: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        tensors = list(features)
        if self.mode == "pyramid":
            outputs = self.module(tensors)
            if isinstance(outputs, dict):
                outputs = list(outputs.values())
            if isinstance(outputs, torch.Tensor):
                outputs = [outputs]
            outputs_list = [*outputs]
            if len(outputs_list) != len(tensors):
                raise ValueError(
                    f"Module '{self.name}' changed feature count from {len(tensors)} to {len(outputs_list)}."
                )
            return outputs_list
        if self.mode == "per_level":
            return [self.module(tensor) for tensor in tensors]
        if self.mode == "dual":
            return [self.module(tensor, tensor) for tensor in tensors]
        if self.mode == "triple":
            fused: List[torch.Tensor] = []
            for index, tensor in enumerate(tensors):
                low = tensors[index - 1] if index > 0 else None
                high = tensors[index + 1] if index + 1 < len(tensors) else None
                fused.append(self.module(tensor, low, high))
            return fused
        raise RuntimeError(f"Unsupported adapter mode '{self.mode}'.")


class RandomModulePipeline(nn.Module):
    """Sequentially applies a list of module adapters to feature maps."""

    def __init__(self, adapters: Iterable[ModuleAdapter]) -> None:
        super().__init__()
        adapters = list(adapters)
        self.adapters = nn.ModuleList(adapters)
        self.module_names = [adapter.name for adapter in adapters]

    def forward(self, features: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        tensors = list(features)
        for adapter in self.adapters:
            tensors = adapter(tensors)
        return tensors


def discover_candidates(module_root: Path) -> List[ModuleCandidate]:
    """Enumerate modules that can be instantiated and adapted."""

    candidates: List[ModuleCandidate] = []
    for path in sorted(module_root.glob("*.py")):
        if path.name.startswith("__"):
            continue
        class_name = path.stem
        try:
            klass = load_class(path, class_name)
        except (FileNotFoundError, ImportError, ModuleNotFoundError, AttributeError):
            continue
        module = _instantiate_with_heuristics(klass)
        if module is None:
            continue
        try:
            adapter = ModuleAdapter(class_name, module)
        except ValueError:
            continue
        candidates.append(ModuleCandidate(name=class_name, adapter=adapter, path=path))
    return candidates


def sample_random_pipeline(
    module_root: Path,
    *,
    min_count: int,
    max_count: int,
    seed: int | None = None,
) -> Tuple[RandomModulePipeline, List[str]]:
    """Randomly select modules and assemble them into a pipeline."""

    min_count = max(min_count, 0)
    max_count = max(max_count, 0)
    if max_count < min_count:
        raise ValueError("max_count must be greater than or equal to min_count")

    candidates = discover_candidates(module_root)
    if not candidates:
        raise ValueError(f"No adaptable modules found in '{module_root}'.")

    available = len(candidates)
    min_count = min(min_count, available)
    max_count = min(max_count, available)
    rng = random.Random(seed)
    num_to_pick = rng.randint(min_count, max_count) if max_count > 0 else 0
    if num_to_pick == 0:
        pipeline = RandomModulePipeline([])
        return pipeline, []

    chosen = rng.sample(candidates, k=num_to_pick)
    adapters = [candidate.adapter for candidate in chosen]
    pipeline = RandomModulePipeline(adapters)
    names = [candidate.name for candidate in chosen]
    return pipeline, names

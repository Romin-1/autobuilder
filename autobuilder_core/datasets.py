"""Synthetic datasets used to validate automatic module wiring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class DetectionSample:
    """Container describing a single detection training sample."""

    image: torch.Tensor
    target: Dict[str, torch.Tensor]


class SyntheticDetectionDataset(Dataset[DetectionSample]):
    """Toy dataset with coloured squares/rectangles for object detection.

    Each sample contains a single object whose colour channel encodes its class.
    The dataset is purposely lightweight so it can be used to quickly validate
    that the assembled detection pipeline converges.
    """

    def __init__(
        self,
        num_samples: int = 64,
        image_size: int = 128,
        num_classes: int = 2,
        seed: int = 0,
    ) -> None:
        self._num_samples = num_samples
        self._image_size = image_size
        self._num_classes = num_classes
        self._rng = np.random.RandomState(seed)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self._num_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        rng = np.random.RandomState(self._rng.randint(0, 2**32 - 1) + index)
        image = np.zeros((3, self._image_size, self._image_size), dtype=np.float32)

        size = rng.randint(self._image_size // 8, self._image_size // 3)
        x1 = rng.randint(0, self._image_size - size - 1)
        y1 = rng.randint(0, self._image_size - size - 1)
        x2 = x1 + size
        y2 = y1 + size

        label = rng.randint(1, self._num_classes + 1)
        channel = label - 1
        image[channel, y1:y2, x1:x2] = 1.0

        tensor_image = torch.from_numpy(image)
        boxes = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
        labels = torch.tensor([label], dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}
        return tensor_image, target


def detection_collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """Custom collate function that keeps detection targets as lists."""

    images, targets = zip(*batch)
    return list(images), list(targets)

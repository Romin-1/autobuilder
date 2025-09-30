"""Task-specific trainers that stitch PlugNPlay modules into working models."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torchvision.models.detection import FasterRCNN

from .datasets import SyntheticDetectionDataset, detection_collate_fn
from .module_loader import load_instance


@dataclass
class TrainStats:
    epoch: int
    train_loss: float
    val_loss: float


class TaskTrainer:
    """Base class for auto-assembled training tasks."""

    def run(self) -> TrainStats:
        raise NotImplementedError


class ObjectDetectionTrainer(TaskTrainer):
    """Auto-configured training pipeline for object detection."""

    def __init__(
        self,
        dataset: SyntheticDetectionDataset,
        batch_size: int,
        target_loss: float,
        max_epochs: int,
        device: torch.device,
    ) -> None:
        self.device = device
        self.dataset = dataset
        self.batch_size = batch_size
        self.target_loss = target_loss
        self.max_epochs = max_epochs

        self.model = self._build_model(dataset).to(device)
        self.optimizer: Optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.5)

        num_val = max(4, len(dataset) // 5)
        num_train = len(dataset) - num_val
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset, [num_train, num_val], generator=torch.Generator().manual_seed(42)
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=detection_collate_fn,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=detection_collate_fn,
        )

    def _build_model(self, dataset: SyntheticDetectionDataset) -> FasterRCNN:
        afpn_path = Path("PlugNPlay-Modules/目标检测/AFPN.py")
        afpn = load_instance(afpn_path, "AFPN", in_channels=[256, 512, 1024, 2048], out_channels=256)

        resnet = resnet50(weights=None)
        modules = list(resnet.children())
        stem = nn.Sequential(*modules[:4])  # conv1, bn1, relu, maxpool
        layer1, layer2, layer3, layer4 = modules[4:8]

        class BackboneWithAFPN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.stem = stem
                self.layer1 = layer1
                self.layer2 = layer2
                self.layer3 = layer3
                self.layer4 = layer4
                self.fpn = afpn
                self.out_channels = 256

            def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
                x = self.stem(x)
                c2 = self.layer1(x)
                c3 = self.layer2(c2)
                c4 = self.layer3(c3)
                c5 = self.layer4(c4)

                p2, p3, p4, p5 = self.fpn([c2, c3, c4, c5])
                return OrderedDict({
                    "p2": p2,
                    "p3": p3,
                    "p4": p4,
                    "p5": p5,
                })

        backbone = BackboneWithAFPN()
        num_classes = dataset.num_classes + 1  # background + objects
        model = FasterRCNN(backbone, num_classes=num_classes)
        return model

    def _step(
        self,
        batch: Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]],
    ) -> torch.Tensor:
        images, targets = batch
        images = [img.to(self.device) for img in images]
        targets = [
            {key: value.to(self.device) for key, value in target.items()}
            for target in targets
        ]
        loss_dict = self.model(images, targets)
        loss = sum(loss_dict.values())
        return loss

    def run(self) -> TrainStats:
        history: List[TrainStats] = []
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            running_loss = 0.0
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                loss = self._step(batch)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            avg_train = running_loss / max(len(self.train_loader), 1)

            self.model.train()
            with torch.no_grad():
                val_loss = 0.0
                for batch in self.val_loader:
                    loss = self._step(batch)
                    val_loss += loss.item()
            avg_val = val_loss / max(len(self.val_loader), 1)

            history.append(TrainStats(epoch=epoch, train_loss=avg_train, val_loss=avg_val))
            self.scheduler.step()

            if avg_val <= self.target_loss:
                break

        return history[-1]


TASK_REGISTRY = {
    "object_detection": ObjectDetectionTrainer,
}


def build_trainer(args: argparse.Namespace) -> TaskTrainer:
    task = args.task
    if task not in TASK_REGISTRY:
        raise KeyError(f"Unsupported task '{task}'. Available: {sorted(TASK_REGISTRY)}")

    if task == "object_detection":
        dataset = SyntheticDetectionDataset(
            num_samples=args.num_samples,
            image_size=args.image_size,
            num_classes=args.num_classes,
            seed=args.seed,
        )
        return ObjectDetectionTrainer(
            dataset=dataset,
            batch_size=args.batch_size,
            target_loss=args.target_loss,
            max_epochs=args.max_epochs,
            device=args.device,
        )

    raise AssertionError("Unhandled task registration")

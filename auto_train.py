"""Command line interface for automatically wiring PlugNPlay modules into training runs."""

from __future__ import annotations

import argparse
from typing import Any

import torch

from autobuilder_core.tasks import build_trainer


def parse_args(argv: Any = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["object_detection"], required=True, help="Task type to train.")
    parser.add_argument("--target-loss", type=float, default=0.75, help="Stop when validation loss <= this value.")
    parser.add_argument("--max-epochs", type=int, default=12, help="Maximum epochs to run.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training and validation.")
    parser.add_argument("--num-samples", type=int, default=80, help="Total synthetic samples to generate.")
    parser.add_argument("--image-size", type=int, default=128, help="Side length of synthetic images.")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of foreground classes in synthetic data.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data generation.")
    parser.add_argument(
        "--module-root",
        default="PlugNPlay-Modules/目标检测",
        help="Directory containing task-specific PlugNPlay modules to sample.",
    )
    parser.add_argument(
        "--min-modules",
        type=int,
        default=1,
        help="Minimum number of modules to chain together when building the model.",
    )
    parser.add_argument(
        "--max-modules",
        type=int,
        default=3,
        help="Maximum number of modules to chain together when building the model.",
    )
    parser.add_argument(
        "--module-seed",
        type=int,
        help="Optional seed controlling random module selection (defaults to --seed).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to run on (default automatically chooses CUDA when available).",
    )
    return parser.parse_args(argv)


def main(argv: Any = None) -> None:
    args = parse_args(argv)
    args.device = torch.device(args.device)
    trainer = build_trainer(args)
    if getattr(trainer, "selected_module_names", None):
        modules = ", ".join(trainer.selected_module_names)
    else:
        modules = "<none>"
    print(f"Selected modules: {modules}")
    stats = trainer.run()

    print(
        f"Task: {args.task}\n"
        f"Epochs ran: {stats.epoch}\n"
        f"Final training loss: {stats.train_loss:.4f}\n"
        f"Final validation loss: {stats.val_loss:.4f}"
    )


if __name__ == "__main__":
    main()

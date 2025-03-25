#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import npfl138
npfl138.require_version("2425.4")
from npfl138.datasets.cifar10 import CIFAR10
from npfl138.trainable_module import TrainableModule

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--epochs", default=49, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--threads", default=0, type=int)
parser.add_argument("--decay", default=0.0005, type=float)
parser.add_argument("--lr", default=0.01, type=float)


class SimpleVGGClassifier(TrainableModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.args = args

        # Simple Sequential feature extractor + classifier
        self.model = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),

            # Conv Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),

            # Conv Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),

            # Flatten and FC layers
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 10)
        )

        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.configure()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.decay)

    def _prepare_batch(self, batch, labels=True):
        x = batch["image"]
        if x.ndim == 4 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
        x = x.float() / 255.0
        x = x.to(self.device)
        if labels:
            y = batch["label"].to(self.device)
            return x, y
        return x, None

    def training_step(self, batch):
        x, y = self._prepare_batch(batch)
        predictions = self(x)
        return self.loss_fn(predictions, y)

    def validation_step(self, batch):
        x, y = self._prepare_batch(batch)
        predictions = self(x)
        accuracy = (predictions.argmax(dim=1) == y).float().mean()
        return {"accuracy": accuracy.item()}

    def test_step(self, batch):
        x, _ = self._prepare_batch(batch, labels=False)
        return self(x)


def main(args: argparse.Namespace):
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v)
                  for k, v in sorted(vars(args).items())))
    ))

    # Load CIFAR10 dataset
    cifar = CIFAR10()
    train_loader = torch.utils.data.DataLoader(cifar.train, batch_size=args.batch_size, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(cifar.dev, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(cifar.test, batch_size=args.batch_size)

    model = SimpleVGGClassifier(args)
    model.fit(train_loader, epochs=args.epochs)
    model.evaluate(dev_loader)

    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as out:
        for batch in test_loader:
            predictions = model.test_step(batch)
            for pred in predictions.argmax(dim=1).cpu().numpy():
                out.write(f"{pred}\n")


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)

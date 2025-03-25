#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import npfl138
npfl138.require_version("2425.4")
from npfl138.datasets.cifar10 import CIFAR10

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay.")

class LeNet5(nn.Module):
    """
    Classic LeNet-5 adapted for CIFAR-10 (3 input channels).
    Original: input 1x28x28 -> output 10, but we now have 3x32x32 -> 10.
    """
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        # 3x32x32 -> 6x28x28
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        # 6x28x28 -> 6x14x14
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 6x14x14 -> 16x10x10
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # 16x10x10 -> 16x5x5
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 16x5x5 -> 120x1x1
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        # Flatten -> 120
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # 3x32x32 -> 6x28x28
        x = self.pool1(x)          # 6x28x28 -> 6x14x14
        x = F.relu(self.conv2(x))   # 6x14x14 -> 16x10x10
        x = self.pool2(x)          # 16x10x10 -> 16x5x5
        x = F.relu(self.conv3(x))   # 16x5x5 -> 120x1x1
        x = x.view(-1, 120)        # Flatten
        x = F.relu(self.fc1(x))    # 120 -> 84
        x = self.fc2(x)            # 84 -> 10
        return x

    def predict(self, dataloader):
        self.eval()
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"]
                if not torch.is_tensor(images):
                    images = torch.tensor(images)
                if images.ndim == 4 and images.shape[-1] == 3:
                    images = images.permute(0, 3, 1, 2)
                images = images.float() / 255.0
                images = images.to(next(self.parameters()).device)
                outputs = self.forward(images)
                for output in outputs:
                    predictions.append(output.cpu().numpy())
        return predictions

def main(args: argparse.Namespace) -> None:
    # Set random seed and thread options.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a unique log directory.
    args.logdir = os.path.join(
        "logs",
        "{}-{}-{}".format(
            os.path.basename(globals().get("__file__", "notebook")),
            datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            ",".join(
                (
                    "{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v)
                    for k, v in sorted(vars(args).items())
                )
            ),
        ),
    )

    # Load the CIFAR10 dataset.
    cifar = CIFAR10()
    train_loader = torch.utils.data.DataLoader(
        cifar.train, batch_size=args.batch_size, shuffle=True
    )
    dev_loader = torch.utils.data.DataLoader(
        cifar.dev, batch_size=args.batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        cifar.test, batch_size=args.batch_size, shuffle=False
    )

    # Create the model and move it to the GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet5(num_classes=10).to(device)

    # Define loss function and optimizer.
    criterion = nn.CrossEntropyLoss()  # No label smoothing here by default
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Training loop.
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch["image"]
            labels = batch["label"]
            if not torch.is_tensor(images):
                images = torch.tensor(images)
            if images.ndim == 4 and images.shape[-1] == 3:
                images = images.permute(0, 3, 1, 2)
            images = images.float() / 255.0
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        avg_loss = running_loss / len(cifar.train)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")

        # Evaluate on the development set.
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in dev_loader:
                images = batch["image"]
                labels = batch["label"]
                if not torch.is_tensor(images):
                    images = torch.tensor(images)
                if images.ndim == 4 and images.shape[-1] == 3:
                    images = images.permute(0, 3, 1, 2)
                images = images.float() / 255.0
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        dev_acc = correct / total * 100
        print(f"Dev Accuracy: {dev_acc:.2f}%")

    # Generate test set predictions and save them in args.logdir.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as predictions_file:
        for prediction in model.predict(test_loader):
            predictions_file.write(f"{int(np.argmax(prediction))}\n")

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)

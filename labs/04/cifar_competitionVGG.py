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

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--lr", default=0.0001, type=float, help="Learning rate.")
parser.add_argument("--weight_decay", default=0.0001, type=float, help="Weight decay.")
parser.add_argument("--batch_norm", action="store_true", help="Use batch normalization in conv layers.")

###############################################################################
# A "scaled-down VGG" for CIFAR-10:
# Four conv blocks (instead of five), each with up to 3 conv layers, then pool.
# Final feature map is 2x2 -> flattened to 2048 -> smaller FC layers.
###############################################################################
cfg_vgg_cifar = [
    64, 64, "M",         # Block 1
    128, 128, "M",       # Block 2
    256, 256, 256, "M",  # Block 3
    512, 512, 512, "M"   # Block 4
]

def make_layers(cfg, batch_norm=True):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGGScaledDown(nn.Module):
    def __init__(self, features, num_classes=10, init_weights=True):
        super(VGGScaledDown, self).__init__()
        self.features = features
        # After 4 pools on a 32x32 input, the feature map is 512 x 2 x 2 -> 2048.
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        if init_weights:
            self._initialize_weights()
            
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten from (N, 512, 2, 2) to (N, 2048).
        x = self.classifier(x)
        return x

    def predict(self, dataloader):
        self.eval()
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"]
                if not torch.is_tensor(images):
                    images = torch.tensor(images)
                # If images are (N, H, W, C), permute to (N, C, H, W).
                if images.ndim == 4 and images.shape[-1] == 3:
                    images = images.permute(0, 3, 1, 2)
                images = images.float() / 255.0
                images = images.to(next(self.parameters()).device)
                outputs = self.forward(images)
                for output in outputs:
                    predictions.append(output.cpu().numpy())
        return predictions

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for conv layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

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

    # Build the scaled-down VGG features.
    features = make_layers(cfg_vgg_cifar, batch_norm=args.batch_norm)
    # Create the scaled-down VGG model and move it to the GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGGScaledDown(features).to(device)

    # Define loss function and optimizer.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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

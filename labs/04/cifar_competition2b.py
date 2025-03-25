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

# Define a simple CNN model.
class CNN(nn.Module):
    def __init__(self, kernel_size=3, padding=1):
        super(CNN, self).__init__()
        # First block: two conv layers followed by max pooling.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        # Second block: two conv layers with increased channels.
        self.conv3 = nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=kernel_size, padding=padding)
        self.bn4 = nn.BatchNorm2d(128)
        # Fully connected layers.
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    # Predict method to yield outputs (one per example) from a given dataloader.
    def predict(self, dataloader):
        self.eval()
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"]
                # Convert to tensor if necessary.
                if not torch.is_tensor(images):
                    images = torch.tensor(images)
                # If images come as (N, H, W, C), rearrange to (N, C, H, W).
                if images.ndim == 4 and images.shape[-1] == 3:
                    images = images.permute(0, 3, 1, 2)
                images = images.float() / 255.0
                images = images.to(next(self.parameters()).device)
                outputs = self.forward(images)
                for output in outputs:
                    predictions.append(output.cpu().numpy())
        return predictions

def main(args: argparse.Namespace) -> None:
    # Set the random seed and number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a unique log directory.
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v)
                  for k, v in sorted(vars(args).items())))
    ))

    # Load the data.
    cifar = CIFAR10()
    
    # Use the CIFAR10 dataset objects directly with DataLoader.
    train_loader = torch.utils.data.DataLoader(cifar.train, batch_size=args.batch_size, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(cifar.dev, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(cifar.test, batch_size=args.batch_size, shuffle=False)

    # Create the model and move it to the appropriate device.
    device = torch.device("cuda")
    model = CNN().to(device)

    # Define loss function and optimizer.
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)

    # Training loop.
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            # Each batch is assumed to be a dict with keys "image" and "label".
            images = batch["image"]
            labels = batch["label"]
            if not torch.is_tensor(images):
                images = torch.tensor(images)
            # Rearrange images if needed.
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

        # Evaluate on the dev set.
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
            # Write the predicted class (using argmax on raw logits).
            predictions_file.write(f"{int(np.argmax(prediction))}\n")

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)

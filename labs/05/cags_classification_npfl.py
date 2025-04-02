#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from torchvision.transforms import v2
import timm

import npfl138
npfl138.require_version("2425.5")
from npfl138.datasets.cags import CAGS

# Define reasonable defaults.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=3, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
# added argument
parser.add_argument("--f_epochs", default=10, type=int, help="Number of epochs for fine-tuning (entire model).")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--flr", default=0.0001, type=float, help="Fine tuning learning rate.")

# Dataset wrapper based on npfl138.TransformedDataset.
class Dataset(npfl138.TransformedDataset):
    def __init__(self, dataset, transform=None):
        super().__init__(dataset)
        self._transform = transform

    def transform(self, example):
        # Convert the image to float and scale to [0,1].
        image = example["image"].to(torch.float32) / 255.0
        if self._transform:
            image = self._transform(image)
        label = example["label"]
        return image, label

# Define the classification model using EfficientNetV2-B0.
class CAGSClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        # Backbone returns a 1280-dimensional feature vector.
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create logdir name.
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the CAGS dataset.
    cags = CAGS(decode_on_demand=False)
    # Assume cags has train, dev, and test splits.
    train_data = cags.train
    dev_data = cags.dev if hasattr(cags, "dev") else cags.train
    test_data = cags.test

    # Load the EfficientNetV2-B0 backbone without its classification layer.
    backbone = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True, num_classes=0)

    model_module = CAGSClassifier(backbone, num_classes=cags.LABELS)

    # Freeze the backbone so only the classifier is trained initially.
    for param in model_module.backbone.parameters():
        param.requires_grad = False

    # Wrap the model with npfl138.TrainableModule.
    model = npfl138.TrainableModule(model_module)

    # Define transformations.
    # Training transformation: add a random horizontal flip before normalization.
    train_transform = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.Normalize(mean=backbone.pretrained_cfg["mean"], std=backbone.pretrained_cfg["std"]),
    ])
    # Dev/test transformation: only normalization.
    test_transform = v2.Compose([
        v2.Normalize(mean=backbone.pretrained_cfg["mean"], std=backbone.pretrained_cfg["std"]),
    ])

    # Create DataLoaders.
    train_loader = torch.utils.data.DataLoader(
        Dataset(train_data, transform=train_transform),
        batch_size=args.batch_size, shuffle=True
    )
    dev_loader = torch.utils.data.DataLoader(
        Dataset(dev_data, transform=test_transform),
        batch_size=args.batch_size
    )
    test_loader = torch.utils.data.DataLoader(
        Dataset(test_data, transform=test_transform),
        batch_size=args.batch_size
    )

    # Configure the model.
    # Since the backbone is frozen, filter for parameters requiring gradients.
    model.configure(
        # optimizer=torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.01),
        optimizer=torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr),
        loss=torch.nn.CrossEntropyLoss(),
        metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=cags.LABELS)},
        logdir=args.logdir,
    )

    model.to(device="cuda")

    # Train the model.
    model.fit(train_loader, dev=dev_loader, epochs=args.epochs)


    # fine tuning
    for param in model_module.backbone.parameters():
        param.requires_grad = True

    optimizer_ft = torch.optim.AdamW(model.parameters(), lr=args.flr, weight_decay=0.01)

    model.configure(
        optimizer=optimizer_ft,
        loss=torch.nn.CrossEntropyLoss(),
        metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=cags.LABELS)},
        logdir=args.logdir,
    )
    model.fit(train_loader, dev=dev_loader, epochs=args.f_epochs)

    # Generate test set annotations.
    os.makedirs(args.logdir, exist_ok=True)
    predictions_path = os.path.join(args.logdir, "cags_classification.txt")
    with open(predictions_path, "w", encoding="utf-8") as predictions_file:
        for prediction in model.predict(test_loader, data_with_labels=True):
            # Each prediction is a vector of logits; use argmax to obtain the label.
            print(np.argmax(prediction), file=predictions_file)

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)

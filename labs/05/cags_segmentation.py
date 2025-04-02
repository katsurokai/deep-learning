#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2

import npfl138
npfl138.require_version("2425.5")
from npfl138 import TransformedDataset
from npfl138.datasets.cags import CAGS

# Define reasonable defaults.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

# Create a dataset wrapper for segmentation.
class SegmentationDataset(TransformedDataset):
    def __init__(self, dataset, transform=None):
        super().__init__(dataset)
        self._transform = transform

    def transform(self, example):
        # Convert image to float and apply preprocessing.
        image = example["image"].to(torch.float32)
        if self._transform:
            image = self._transform(image)
        # The mask is already a [1,224,224] float32 tensor with values in [0,1].
        mask = example["mask"]
        return image, mask

# Define a segmentation model with a pretrained backbone and a segmentation head.
class CAGSSegmentationModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        # The backbone outputs features of shape [N,1280,7,7]. We then upsample:
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),  # 7x7 -> 14x14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),  # 14x14 -> 28x28
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 28x28 -> 56x56
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),   # 56x56 -> 112x112
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),     # 112x112 -> 224x224
            nn.Sigmoid()  # Produces output values between 0 and 1.
        )
    def forward(self, x):
        # Get features from the backbone.
        # Obtain the feature map before global pooling.
        output, intermediates = self.backbone.forward_intermediates(x)
        # Use the output which should have shape [batch, 1280, 7, 7].
        mask = self.segmentation_head(output)
        return mask

def main(args: argparse.Namespace) -> None:
    # Set random seed and threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create log directory.
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v)
                  for k, v in sorted(vars(args).items())))
    ))

    # Load the CAGS dataset.
    cags = CAGS(decode_on_demand=False)

    # Load the EfficientNetV2-B0 backbone without its classification layer.
    efficientnetv2_b0 = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True, num_classes=0)

    # Create preprocessing: convert image to float, scale to [0,1] and normalize.
    preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=efficientnetv2_b0.pretrained_cfg["mean"],
                     std=efficientnetv2_b0.pretrained_cfg["std"]),
    ])

    # Prepare datasets.
    train_dataset = SegmentationDataset(cags.train, transform=preprocessing)
    dev_dataset = SegmentationDataset(cags.dev if hasattr(cags, "dev") else cags.train,
                                      transform=preprocessing)
    test_dataset = SegmentationDataset(cags.test, transform=preprocessing)

    # Create DataLoaders.
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

    # Create the segmentation model.
    segmentation_model = CAGSSegmentationModel(efficientnetv2_b0)
    model = npfl138.TrainableModule(segmentation_model)

    # Configure training: using binary cross entropy loss for the mask,
    # AdamW optimizer, and MaskIoUMetric to measure segmentation performance.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    loss_fn = torch.nn.BCELoss()
    metrics = {"iou": CAGS.MaskIoUMetric()}
    model.configure(optimizer=optimizer, loss=loss_fn, metrics=metrics, logdir=args.logdir)

    model.to(device="cuda")

    # Train the model.
    model.fit(train_loader, dev=dev_loader, epochs=args.epochs)

    # Generate test set annotations in the required run-length encoding format.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cags_segmentation.txt"), "w", encoding="utf-8") as predictions_file:
        # Here we assume that model.predict yields a predicted mask per test example.
        for mask in model.predict(dev_loader, data_with_labels=True):
            zeros, ones, runs = 0, 0, []
            # Flatten the mask after thresholding at 0.5.
            for pixel in np.reshape(mask >= 0.5, [-1]):
                if pixel:
                    if zeros or (not zeros and not ones):
                        runs.append(zeros)
                        zeros = 0
                    ones += 1
                else:
                    if ones:
                        runs.append(ones)
                        ones = 0
                    zeros += 1
            runs.append(zeros + ones)
            print(*runs, file=predictions_file)

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)

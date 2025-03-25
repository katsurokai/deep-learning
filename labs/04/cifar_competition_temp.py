#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import torch
import torchmetrics
from torchvision.transforms import v2

import npfl138
npfl138.require_version("2425.4")
from npfl138.datasets.cifar10 import CIFAR10

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--epochs", default=49, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")

class Dataset(npfl138.TransformedDataset):
    def __init__(self, dataset, transform=None):
        super().__init__(dataset)
        self._transform = transform

    def transform(self, example):
        image = example["image"].to(torch.float32) / 255.0
        if self._transform:
            image = self._transform(image)
        label = example["label"]
        return image, label

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

    # Load the data.
    cifar = CIFAR10()

    # TODO: Create the model and train it.
    model = npfl138.TrainableModule(torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, 3, padding=1), 
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 64, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(64, 128, 3, padding=1), 
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Flatten(),
        torch.nn.Linear(8 * 8 * 128, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, CIFAR10.LABELS)
    ))
    
    model.configure(
        optimizer=torch.optim.Adam(model.parameters()),
        loss=torch.nn.CrossEntropyLoss(),
        metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=CIFAR10.LABELS)},
        logdir=args.logdir,
    )
    
    model.to(device="cuda")
    
    transform = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomCrop(32, padding=4),
    ])

    train = torch.utils.data.DataLoader(Dataset(cifar.train, transform=transform), batch_size=args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(Dataset(cifar.dev),batch_size=args.batch_size)
    test = torch.utils.data.DataLoader(Dataset(cifar.test), batch_size=args.batch_size)
    model.fit(train, dev=dev, epochs=args.epochs)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Perform the prediction on the test data. The line below assumes you have
        # a dataloader `test` where the individual examples are `(image, target)` pairs.
        for prediction in model.predict(test, data_with_labels=True):
            print(np.argmax(prediction), file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)

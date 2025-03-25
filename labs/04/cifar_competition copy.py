#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import torch

import npfl138
npfl138.require_version("2425.4")
from npfl138.datasets.cifar10 import CIFAR10

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.

class SequentialModel(torch.nn.Sequential):
    def predict(self, dataloader, data_with_labels=False):
        self.eval()
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                if data_with_labels:
                    images, _ = batch
                else:
                    images = batch
                images = images.to(next(self.parameters()).device)
                outputs = self.forward(images)
                for output in outputs:
                    predictions.append(output.cpu().numpy())

def build_model():
    model = SequentialModel(
        # first
        torch.nn.Conv2d(3, 64, kernel_size= 3, padding=1),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2,2),
        # second
        torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(),
        torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2),
        # flatten 
        torch.nn.Flatten(),
        torch.nn.Linear(128 * 8 * 8, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(256, 10)
    )
    return model

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=..., type=int, help="Batch size.")
parser.add_argument("--epochs", default=..., type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


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
    model = ...

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

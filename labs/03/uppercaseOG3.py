#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch
import torch.nn.functional as F

import npfl138
npfl138.require_version("2425.3.1")
from npfl138.datasets.uppercase_data import UppercaseData

# Set reasonable default hyperparameters.
parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size", default=100, type=int, help="If given, use this many most frequent chars.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use (0 means all cores).")
parser.add_argument("--window", default=3, type=int, help="Window size to use (number of characters to the left/right).")


class BatchGenerator:
    """A simple batch generator, optionally with shuffling.

    Similar in functionality to torch.utils.data.DataLoader for datasets stored as tensors.
    """
    def __init__(self, inputs: torch.Tensor, outputs: torch.Tensor, batch_size: int, shuffle: bool):
        self._inputs = inputs.to(device="cuda")
        self._outputs = outputs.to(device="cuda")
        self._batch_size = batch_size
        self._shuffle = shuffle

    def __len__(self):
        return (len(self._inputs) + self._batch_size - 1) // self._batch_size

    def __iter__(self):
        indices = torch.randperm(len(self._inputs), device="cuda") if self._shuffle else torch.arange(len(self._inputs), device="cuda")
        while len(indices):
            batch = indices[:self._batch_size]
            indices = indices[self._batch_size:]
            yield self._inputs[batch], self._outputs[batch]


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self._args = args
        # Use an embedding layer to map character indices to embeddings.
        # Note: We assume that the dataset reserves index 0 for padding if needed.
        self.embedding = torch.nn.Embedding(num_embeddings=args.alphabet_size, embedding_dim=16)
        # The input dimension to the classifier is (2*window + 1) * embedding_dim.
        input_dim = (2 * args.window + 1) * 16
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 1)  # Single output for binary classification.
        self.relu = torch.nn.ReLU()

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        # windows: [batch_size, 2*window+1] containing character indices.
        x = self.embedding(windows)  # [batch_size, 2*window+1, 16]
        x = x.view(x.size(0), -1)     # flatten to [batch_size, (2*window+1)*16]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  # Raw logits; use BCEWithLogitsLoss during training.


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
    # The label_dtype is torch.float32, which is appropriate for binary classification using BCEWithLogitsLoss.
    uppercase_data = UppercaseData(args.window, args.alphabet_size, label_dtype=torch.float32)

    # Create batch generators.
    train = BatchGenerator(uppercase_data.train.windows, uppercase_data.train.labels, args.batch_size, shuffle=True)
    dev = BatchGenerator(uppercase_data.dev.windows, uppercase_data.dev.labels, args.batch_size, shuffle=False)
    test = BatchGenerator(uppercase_data.test.windows, uppercase_data.test.labels, args.batch_size, shuffle=False)

    # Instantiate the model.
    model = Model(args)
    model = model.to(device="cuda")

    # Set up the loss function and optimizer.
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop.
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch_windows, batch_labels in train:
            optimizer.zero_grad()
            outputs = model(batch_windows)
            # Squeeze outputs to match shape of labels.
            loss = loss_fn(outputs.squeeze(-1), batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train)
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_loss:.4f}")

        # Evaluate on the development set.
        model.eval()
        total_dev_loss = 0.0
        with torch.no_grad():
            for batch_windows, batch_labels in dev:
                outputs = model(batch_windows)
                loss = loss_fn(outputs.squeeze(-1), batch_labels)
                total_dev_loss += loss.item()
        avg_dev_loss = total_dev_loss / len(dev)
        print(f"Epoch {epoch+1}/{args.epochs}, Dev Loss: {avg_dev_loss:.4f}")

    # Generate predictions on the test set.
    # We assume that the order of windows in uppercase_data.test.windows corresponds
    # to the order of characters in uppercase_data.test.text.
    all_preds = []
    model.eval()
    with torch.no_grad():
        for batch_windows, _ in test:
            outputs = model(batch_windows)
            # Apply sigmoid to get probabilities and threshold at 0.5.
            preds = (torch.sigmoid(outputs) > 0.5).squeeze(-1).long()
            all_preds.append(preds)
    all_preds = torch.cat(all_preds, dim=0)

    # Use the predictions to transform the original test text.
    # For each character, if the corresponding prediction is 1, output its uppercase version.
    result_chars = []
    # Note: We assume that the number of windows equals the number of characters in the text.
    for ch, pred in zip(uppercase_data.test.text, all_preds):
        if pred.item() == 1:
            result_chars.append(ch.upper())
        else:
            result_chars.append(ch)
    result_text = "".join(result_chars)

    # Write the transformed text to predictions file.
    os.makedirs(args.logdir, exist_ok=True)
    predictions_path = os.path.join(args.logdir, "uppercase_test.txt")
    with open(predictions_path, "w", encoding="utf-8") as predictions_file:
        predictions_file.write(result_text)
    print(f"Predictions written to {predictions_path}")


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)

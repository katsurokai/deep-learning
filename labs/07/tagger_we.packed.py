#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch
import torchmetrics

import npfl138
npfl138.require_version("2425.7.2")
from npfl138.datasets.morpho_dataset import MorphoDataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--rnn", default="LSTM", choices=["LSTM", "GRU"], help="RNN layer type.")
parser.add_argument("--rnn_dim", default=64, type=int, help="RNN layer dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--we_dim", default=128, type=int, help="Word embedding dimension.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        super().__init__()

        # Create all needed layers.
        # TODO: Create a `torch.nn.Embedding` layer, embedding the word ids
        # from `train.words.string_vocab` to dimensionality `args.we_dim`.
        self._word_embedding = ...

        # TODO: Create an RNN layer, either `torch.nn.LSTM` or `torch.nn.GRU` depending
        # on `args.rnn`. The layer should be bidirectional (`bidirectional=True`) with
        # dimensionality `args.rnn_dim`. During the model computation, the layer will
        # process the word embeddings generated by the `self._word_embedding` layer,
        # and we will sum the outputs of forward and backward directions.
        self._word_rnn = ...

        # TODO: Create an output linear layer (`torch.nn.Linear`) processing the RNN output,
        # producing logits for tag prediction; `train.tags.string_vocab` is the tag vocabulary.
        self._output_layer = ...

    def forward(self, word_ids: torch.nn.utils.rnn.PackedSequence) -> torch.nn.utils.rnn.PackedSequence:
        # The input `word_ids` is a `PackedSequence` object. It allows us to:
        # - get the flattened data using `word_ids.data`; these are the data without
        #   padding elements, i.e., a 1D vector of shape `[sum_of_sentence_lengths]`;
        # - replace the data while keeping the sizes of the original sequences
        #   by calling `word_ids._replace(data=...)` and getting a new `PackedSequence`.
        # Therefore, depending on the context, we need to use either the flattened
        # data or the `PackedSequence` object.

        # TODO: Start by embedding the `word_ids` using the word embedding layer.
        hidden = ...

        # TODO: Process the embedded words through the RNN layer, utilizing
        # the `PackedSequence` structure of `word_ids` (i.e., the same sentence lengths).
        hidden = ...

        # TODO: Sum the outputs of forward and backward directions.
        hidden = ...

        # TODO: Pass the RNN output through the output layer.
        hidden = ...

        # TODO: Finally, produce output predictions as a `PackedSequence` with the same
        # `PackedSequence` structure as `word_ids` (same sentence lengths).
        hidden = ...

        return hidden

    def compute_loss(self, y_pred, y_true, *xs):
        # Because the `y_pred` and `y_true` are `PackedSequence` objects, we take
        # just their raw data and pass them to the loss function.
        return super().compute_loss(y_pred.data, y_true.data, *xs)

    def compute_metrics(self, y_pred, y_true, *xs):
        # Because the `y_pred` and `y_true` are `PackedSequence` objects, we take
        # just their raw data and pass them to the metric computation.
        return super().compute_metrics(y_pred.data, y_true.data, *xs)


class TrainableDataset(npfl138.TransformedDataset):
    def transform(self, example):
        # TODO: Construct a single example, each consisting of the following pair:
        # - a PyTorch tensor of integer ids of input words as input,
        # - a PyTorch tensor of integer tag ids as targets.
        # To create the ids, use `string_vocab` of `self.dataset.words` and `self.dataset.tags`.
        word_ids = ...
        tag_ids = ...
        return word_ids, tag_ids

    def collate(self, batch):
        # Construct a single batch, where `data` is a list of examples
        # generated by `transform`.
        word_ids, tag_ids = zip(*batch)
        # TODO: Combine `word_ids` into a `PackedSequence` by calling
        # `torch.nn.utils.rnn.pack_sequence` with `enforce_sorted=False`.
        word_ids = ...
        # TODO: Process `tag_ids` analogously to `word_ids`.
        tag_ids = ...
        return word_ids, tag_ids


def main(args: argparse.Namespace) -> dict[str, float]:
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
    morpho = MorphoDataset("czech_cac", max_sentences=args.max_sentences)

    # Prepare the data for training.
    train = TrainableDataset(morpho.train).dataloader(batch_size=args.batch_size, shuffle=True)
    dev = TrainableDataset(morpho.dev).dataloader(batch_size=args.batch_size)

    # Create the model and train.
    model = Model(args, morpho.train)

    model.configure(
        # TODO: Create the Adam optimizer.
        optimizer=...,
        # TODO: Use the usual `torch.nn.CrossEntropyLoss` loss function.
        loss=...,
        # TODO: Create a multiclass `torchmetrics.Accuracy` metric, with `num_classes`
        # set to the number of unique tags.
        metrics={"accuracy": torchmetrics.Accuracy(...)},
        logdir=args.logdir,
    )

    logs = model.fit(train, dev=dev, epochs=args.epochs)

    # Return development metrics for ReCodEx to validate.
    return {metric: value for metric, value in logs.items() if metric.startswith("dev_")}


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)

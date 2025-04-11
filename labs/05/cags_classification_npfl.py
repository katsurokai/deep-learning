#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import timm
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.v2 as v2
import torchvision.transforms.functional as TF

import bboxes_utils
import npfl138
npfl138.require_version("2425.6.1")
from npfl138.datasets.svhn import SVHN
from npfl138.transformed_dataset import TransformedDataset
from trainable_module import TrainableModule

# --- Command-line arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

# --- Detector definition ---
class SVHNDetector(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        in_channels = 1280  # EfficientNetV2-B0 outputs 1280 channels.
        # Classification head: 11 channels (0: background, 1-10: digits 0-9)
        self.cls_head = torch.nn.Conv2d(in_channels, 11, kernel_size=3, padding=1)
        # Regression head: 4 numbers per anchor.
        self.reg_head = torch.nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)

    def forward(self, x):
        # Get the final feature map from the backbone.
        output, _ = self.backbone.forward_intermediates(x)
        cls_logits = self.cls_head(output)  # [N, 11, H_out, W_out]
        bbox_reg = self.reg_head(output)      # [N, 4, H_out, W_out]
        N, _, H_out, W_out = cls_logits.shape
        # Reshape to get one prediction per spatial location.
        cls_logits = cls_logits.view(N, -1, 11)  # [N, num_anchors, 11]
        bbox_reg = bbox_reg.view(N, -1, 4)         # [N, num_anchors, 4]

        # Generate anchors for the feature map.
        _, _, H, W = x.shape
        stride_y = H / H_out
        stride_x = W / W_out
        device = x.device
        ys = (torch.arange(H_out, device=device) + 0.5) * stride_y
        xs = (torch.arange(W_out, device=device) + 0.5) * stride_x
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        half_h = stride_y / 2
        half_w = stride_x / 2
        anchors = torch.stack([
            grid_y - half_h,
            grid_x - half_w,
            grid_y + half_h,
            grid_x + half_w,
        ], dim=-1).view(-1, 4)  # [num_anchors, 4]
        return cls_logits, bbox_reg, anchors

# --- TrainableModule wrapper ---
class TrainableSVHNDetector(TrainableModule):
    def __init__(self, backbone):
        super().__init__()
        self.detector = SVHNDetector(backbone)

    def forward(self, x):
        return self.detector(x)

    def compute_loss(self, y_pred, y, *args):
        """
        Compute the loss for object detection.
        y_pred: tuple (cls_logits, bbox_reg, anchors)
        y: a batch of dictionaries with keys "classes" and "bboxes"
        """
        cls_logits, bbox_reg, anchors = y_pred
        total_loss = 0.0
        batch_size = cls_logits.shape[0]

        # Process each sample individually.
        for i in range(batch_size):
            gold = y[i]
            gold_classes = gold["classes"].to(self.device)
            gold_bboxes = gold["bboxes"].to(self.device)
            # Assign ground-truth boxes to anchors.
            target_cls, target_bbox = bboxes_utils.bboxes_training(
                anchors, gold_classes, gold_bboxes, iou_threshold=0.5)
            # Classification loss (cross entropy over all anchors).
            loss_cls = F.cross_entropy(cls_logits[i], target_cls)
            # Regression loss (smooth L1 only on positive anchors).
            pos = target_cls > 0
            if pos.sum() > 0:
                loss_reg = F.smooth_l1_loss(bbox_reg[i][pos], target_bbox[pos])
            else:
                loss_reg = 0.0
            total_loss += loss_cls + loss_reg
        return total_loss / batch_size

# --- Transform function using TransformedDataset ---
def transform(sample):
    """
    Resizes the image to 224x224 and scales bounding boxes accordingly.
    Assumes sample is a dict with keys "image", "classes", and "bboxes".
    """
    original_size = sample["image"].shape[1:]  # (H, W)
    new_size = (224, 224)
    # Resize the image.
    sample["image"] = TF.resize(sample["image"], new_size)
    # Compute scaling factors.
    scale_y = new_size[0] / original_size[0]
    scale_x = new_size[1] / original_size[1]
    # Adjust bounding boxes.
    sample["bboxes"] = sample["bboxes"].float()
    sample["bboxes"][:, [0, 2]] *= scale_y  # top and bottom
    sample["bboxes"][:, [1, 3]] *= scale_x  # left and right
    return sample

def main(args: argparse.Namespace) -> None:
    # Set random seed and number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a logging directory.
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the SVHN dataset.
    svhn = SVHN(decode_on_demand=False)

    # Wrap train and dev datasets in TransformedDataset to ensure consistent image size.
    train_dataset = TransformedDataset(svhn.train)
    dev_dataset = TransformedDataset(svhn.dev)
    train_dataset.transform = transform
    dev_dataset.transform = transform

    # Create DataLoaders with the collate function from TransformedDataset.
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size,
                                             collate_fn=dev_dataset.collate_fn)

    # Load the EfficientNetV2-B0 backbone.
    efficientnetv2_b0 = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True, num_classes=0)
    # Define additional preprocessing for inference (scaling and normalization).
    preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=efficientnetv2_b0.pretrained_cfg["mean"],
                     std=efficientnetv2_b0.pretrained_cfg["std"]),
    ])

    # Instantiate the trainable detector.
    model = TrainableSVHNDetector(efficientnetv2_b0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.configure(optimizer=optimizer, loss=model.compute_loss, device=device, logdir=args.logdir)

    # --- Train the model using TrainableModule.fit ---
    model.fit(train_loader, dev=dev_loader, epochs=args.epochs)

    # After training, generate test predictions.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "svhn_competition.txt"), "w", encoding="utf-8") as predictions_file:
        model.eval()
        with torch.no_grad():
            for example in svhn.test:
                # Apply the same transform to the test sample.
                sample = transform(example)
                image = sample["image"]
                # Preprocess: normalize and add batch dimension.
                image_preprocessed = preprocessing(image).unsqueeze(0).to(device)
                cls_logits, bbox_reg, anchors = model(image_preprocessed)
                cls_probs = torch.softmax(cls_logits, dim=-1)[0]
                scores, labels = cls_probs.max(dim=-1)
                from bboxes_utils import bboxes_from_rcnn
                decoded_boxes = bboxes_from_rcnn(anchors, bbox_reg[0])
                # Filter out background and low-confidence detections.
                keep = (labels != 0) & (scores > 0.5)
                if keep.sum() == 0:
                    predicted_classes = torch.empty((0,), dtype=torch.long)
                    predicted_bboxes = torch.empty((0, 4), dtype=torch.float32)
                else:
                    boxes = decoded_boxes[keep]
                    kept_labels = labels[keep]
                    kept_scores = scores[keep]
                    final_indices = []
                    # Apply non-maximum suppression for each detected class.
                    for cls in kept_labels.unique():
                        cls_mask = (kept_labels == cls)
                        cls_boxes = boxes[cls_mask]
                        cls_scores = kept_scores[cls_mask]
                        nms_indices = torchvision.ops.nms(cls_boxes, cls_scores, 0.3)
                        kept_idx = torch.nonzero(cls_mask, as_tuple=False).view(-1)
                        final_indices.append(kept_idx[nms_indices])
                    final_indices = torch.cat(final_indices)
                    # Convert labels from {1,...,10} to actual digit labels {0,...,9}.
                    predicted_classes = kept_labels[final_indices] - 1
                    predicted_bboxes = boxes[final_indices]
                # Write predictions in the required format: "label top left bottom right".
                output = []
                for label, bbox in zip(predicted_classes.tolist(), predicted_bboxes.tolist()):
                    output += [int(label)] + list(map(float, bbox))
                print(*output, file=predictions_file)

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)

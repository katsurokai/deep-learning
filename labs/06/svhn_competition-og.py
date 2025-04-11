#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.v2 as v2
import torchvision.transforms.functional as TF

import bboxes_utils
import npfl138
npfl138.require_version("2425.6.1")
from npfl138.datasets.svhn import SVHN
from npfl138.transformed_dataset import TransformedDataset
from npfl138.trainable_module import TrainableModule
import torchvision.ops as ops

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
parser.add_argument("--weight_decay", default=1e-5, type=float, help="Weight decay.")

# --- Modified Detection Head using a conv tower and focal loss ---
class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # Classification tower: two conv layers with ReLU.
        self.cls_tower = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Final classification layer producing num_classes channels (for digits 0-9).
        self.cls_head = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        
        # Regression tower: similar structure.
        self.reg_tower = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Final regression layer produces 4 numbers per anchor.
        self.reg_head = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)

    def forward(self, x):
        cls_features = self.cls_tower(x)
        cls_logits = self.cls_head(cls_features)
        reg_features = self.reg_tower(x)
        bbox_reg = self.reg_head(reg_features)
        return cls_logits, bbox_reg

def train_transform(sample):
    original_size = sample["image"].shape[1:]  # (H, W)
    new_size = (224, 224)
    # Resize image.
    image = TF.resize(sample["image"], new_size)
    # Convert image to float in [0,1].
    image = image.to(torch.float32).div(255.0)
    # Normalize the image.
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    image = TF.normalize(image, mean=mean, std=std)
    
    # Adjust bounding boxes.
    scale_y = new_size[0] / original_size[0]
    scale_x = new_size[1] / original_size[1]
    bboxes = sample["bboxes"].float().clone()
    bboxes[:, [0, 2]] *= scale_y  # adjust top and bottom
    bboxes[:, [1, 3]] *= scale_x  # adjust left and right
    return image, (sample["classes"], bboxes)

def transform(sample):
    new_size = (224, 224)
    image = TF.resize(sample["image"], new_size)
    image = image.to(torch.float32).div(255.0)
    return image

def custom_collate(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    lengths = [item[1][0].shape[0] for item in batch]
    max_len = max(lengths)
    padded_classes = []
    padded_bboxes = []
    for (cls, bboxes) in [item[1] for item in batch]:
        num = cls.shape[0]
        if num < max_len:
            pad = torch.full((max_len - num,), -1, dtype=cls.dtype)
            cls = torch.cat([cls, pad], dim=0)
            pad_boxes = torch.zeros((max_len - num, 4), dtype=bboxes.dtype)
            bboxes = torch.cat([bboxes, pad_boxes], dim=0)
        padded_classes.append(cls)
        padded_bboxes.append(bboxes)
    padded_classes = torch.stack(padded_classes, dim=0)  # [B, max_len]
    padded_bboxes = torch.stack(padded_bboxes, dim=0)        # [B, max_len, 4]
    return images, (padded_classes, padded_bboxes)

class SVHNDetector(nn.Module):
    def __init__(self, backbone, fixed_anchor_size=50):
        super().__init__()
        self.backbone = backbone
        self.fixed_anchor_size = fixed_anchor_size  # Fixed anchor side length in network input (224x224) coordinates.
        in_channels = 1280  # For EfficientNetV2-B0.
        self.detection_head = DetectionHead(in_channels, 10)

    def forward(self, x):
        output, _ = self.backbone.forward_intermediates(x)
        cls_logits, bbox_reg = self.detection_head(output)
        N, _, H_out, W_out = cls_logits.shape
        cls_logits = cls_logits.view(N, -1, 10)  # [N, num_anchors, 10]
        bbox_reg   = bbox_reg.view(N, -1, 4)       # [N, num_anchors, 4]
        # Instead of calculating an anchor size from the spatial resolution,
        # we use a fixed anchor size.
        _, _, H, W = x.shape  # network input size (usually 224x224)
        stride_y = H / H_out
        stride_x = W / W_out
        device = x.device
        ys = (torch.arange(H_out, device=device) + 0.5) * stride_y
        xs = (torch.arange(W_out, device=device) + 0.5) * stride_x
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        half_size = self.fixed_anchor_size / 2
        anchors = torch.stack([
            grid_y - half_size,
            grid_x - half_size,
            grid_y + half_size,
            grid_x + half_size,
        ], dim=-1).view(-1, 4)
        return cls_logits, bbox_reg, anchors


class TrainableSVHNDetector(TrainableModule):
    def __init__(self, backbone):
        super().__init__()
        self.detector = SVHNDetector(backbone)

    def forward(self, x):
        return self.detector(x)

    def compute_loss(self, y_pred, y, *args):
        """
        Compute loss using focal loss for classification and smooth L1 for regression.
        y_pred: tuple (cls_logits, bbox_reg, anchors)
        y: a tuple (padded_classes, padded_bboxes) with shapes [B, max_len] and [B, max_len, 4].
        """
        cls_logits, bbox_reg, anchors = y_pred
        total_loss = 0.0
        batch_size = cls_logits.shape[0]
        padded_classes, padded_bboxes = y
        loss_cls_weight = 1.0  # weight for the classification loss
        loss_reg_weight = 0.5  # weight for the regression loss
        for i in range(batch_size):
            valid = padded_classes[i] != -1
            gold_classes = padded_classes[i][valid]
            gold_bboxes = padded_bboxes[i][valid]
            target_cls, target_bbox = bboxes_utils.bboxes_training(
                anchors, gold_classes.to(self.device), gold_bboxes.to(self.device), iou_threshold=0.5)
            onehot_target = torch.zeros(target_cls.size(0), 10, device=self.device)
            pos_mask = target_cls > 0
            if pos_mask.sum() > 0:
                onehot_target[pos_mask] = F.one_hot((target_cls[pos_mask] - 1).long(), num_classes=10).float()
            loss_cls = ops.sigmoid_focal_loss(cls_logits[i], onehot_target, alpha=0.25, gamma=2.0, reduction='sum')
            loss_cls = loss_cls / (pos_mask.sum().float() + 1e-6)
            pos = target_cls > 0
            if pos.sum() > 0:
                loss_reg = F.smooth_l1_loss(bbox_reg[i][pos], target_bbox[pos])
            else:
                loss_reg = 0.0
            total_loss += loss_cls_weight * loss_cls + loss_reg_weight * loss_reg
        return total_loss / batch_size

    # def predict(self, dataset, preprocessing, device):
    #     """
    #     Predict on a dataset and return a dictionary with:
    #       - 'class_output': a list of 1D numpy arrays containing the predicted digits.
    #       - 'bboxes_output': a list of [N, 4] numpy arrays with bounding boxes.
    #     """
    #     self.eval()
    #     all_class_outputs = []
    #     all_bboxes_outputs = []
    #     with torch.no_grad():
    #         for example in dataset:
    #             image = transform(example)
    #             image_preprocessed = preprocessing(image).unsqueeze(0).to(device)
    #             cls_logits, bbox_reg, anchors = self(image_preprocessed)
    #             cls_probs = torch.sigmoid(cls_logits)[0]
    #             scores, labels = torch.max(cls_probs, dim=-1)
    #             decoded_boxes = bboxes_utils.bboxes_from_rcnn(anchors, bbox_reg[0])
    #             keep = scores > 0.5
    #             if keep.sum() == 0:
    #                 predicted_classes = torch.empty((0,), dtype=torch.long)
    #                 predicted_bboxes = torch.empty((0, 4), dtype=torch.float32)
    #             else:
    #                 boxes = decoded_boxes[keep]
    #                 kept_labels = labels[keep]
    #                 kept_scores = scores[keep]
    #                 final_indices = []
    #                 # Apply NMS for each unique predicted class.
    #                 for cls in kept_labels.unique():
    #                     cls_mask = (kept_labels == cls)
    #                     cls_boxes = boxes[cls_mask]
    #                     cls_scores = kept_scores[cls_mask]
    #                     nms_indices = torchvision.ops.nms(cls_boxes, cls_scores, 0.3)
    #                     kept_idx = torch.nonzero(cls_mask, as_tuple=False).view(-1)
    #                     final_indices.append(kept_idx[nms_indices])
    #                 final_indices = torch.cat(final_indices)
    #                 predicted_classes = kept_labels[final_indices]
    #                 predicted_bboxes = boxes[final_indices]
    #             all_class_outputs.append(predicted_classes.cpu().numpy())
    #             all_bboxes_outputs.append(predicted_bboxes.cpu().numpy())
    #     return {'class_output': all_class_outputs, 'bboxes_output': all_bboxes_outputs}

def main(args: argparse.Namespace) -> None:
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v)
                  for k, v in sorted(vars(args).items())))
    ))

    svhn = SVHN(decode_on_demand=False)

    train_dataset = TransformedDataset(svhn.train)
    dev_dataset = TransformedDataset(svhn.dev)
    train_dataset.transform = train_transform
    dev_dataset.transform = train_transform
    train_dataset.collate = custom_collate
    dev_dataset.collate = custom_collate

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)

    efficientnetv2_b0 = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True, num_classes=0)
    preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=efficientnetv2_b0.pretrained_cfg["mean"],
                     std=efficientnetv2_b0.pretrained_cfg["std"]),
    ])

    model = TrainableSVHNDetector(efficientnetv2_b0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.configure(optimizer=optimizer, loss=model.compute_loss, device=device, logdir=args.logdir)

    # Training
    model.fit(train_loader, dev=dev_loader, epochs=args.epochs)

    os.makedirs(args.logdir, exist_ok=True)
        # Inference on DEV set
    dev_predictions_file = os.path.join(args.logdir, "svhn_competition_dev.txt")
    with open(dev_predictions_file, "w", encoding="utf-8") as f:
        model.eval()
        with torch.no_grad():
            for example in svhn.dev:
                image = transform(example)
                image_preprocessed = preprocessing(image).unsqueeze(0).to(device)
                cls_logits, bbox_reg, anchors = model(image_preprocessed)
                cls_probs = torch.sigmoid(cls_logits)[0]
                scores, labels = torch.max(cls_probs, dim=-1)
                decoded_boxes = bboxes_utils.bboxes_from_rcnn(anchors, bbox_reg[0])
                keep = scores > 0.5
                if keep.sum() == 0:
                    f.write("\n")
                else:
                    boxes = decoded_boxes[keep]
                    kept_labels = labels[keep]
                    kept_scores = scores[keep]
                    final_indices = []
                    for cls in kept_labels.unique():
                        cls_mask = (kept_labels == cls)
                        cls_boxes = boxes[cls_mask]
                        cls_scores = kept_scores[cls_mask]
                        nms_indices = torchvision.ops.nms(cls_boxes, cls_scores, 0.3)
                        kept_idx = torch.nonzero(cls_mask, as_tuple=False).view(-1)
                        final_indices.append(kept_idx[nms_indices])
                    final_indices = torch.cat(final_indices)
                    predicted_classes = kept_labels[final_indices]
                    predicted_bboxes = boxes[final_indices]
                    # Rescale predicted bboxes from the network input (224x224) back to the original image size.
                    orig_size = example["image"].shape[1]  # assuming square images
                    scale_factor = orig_size / 224.0
                    predicted_bboxes = predicted_bboxes * scale_factor
                    output = []
                    for pcls, bbox in zip(predicted_classes.tolist(), predicted_bboxes.tolist()):
                        output += [int(pcls)] + [float(x) for x in bbox]
                    f.write(" ".join(map(str, output)) + "\n")

    # --- Evaluate the dev predictions ---
    with open(dev_predictions_file, "r", encoding="utf-8-sig") as f:
        dev_accuracy = SVHN.evaluate_file(svhn.dev, f)
    print("Dev set accuracy: {:.2f}%".format(dev_accuracy))

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)

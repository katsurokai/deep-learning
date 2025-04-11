#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import timm
import torch
import torchvision.transforms.v2 as v2

import bboxes_utils
import npfl138
npfl138.require_version("2425.6.1")
from npfl138.datasets.svhn import SVHN
import torchvision.ops as ops

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

class SVHNDataset(npfl138.TransformedDataset):
    def __init__(self, dataset, transform):
        super().__init__(dataset)
        self._transform = transform

    def transform(self, example):
        original_h, original_w = example["image"].shape[1:]
        image = self._transform(example["image"])
        new_h, new_w = image.shape[1:]

        classes = example["classes"]
        bboxes = example["bboxes"].clone().float()

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * new_h / original_h 
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * new_w / original_w  

        return image, classes, bboxes, (original_h, original_w)
    
def detection_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    classes = [item[1] for item in batch]
    bboxes = [item[2] for item in batch]
    sizes = torch.tensor([item[3] for item in batch])

    max_len = max(len(c) for c in classes)
    padded_classes = torch.full((len(classes), max_len), fill_value=-1, dtype=torch.long)
    padded_bboxes = torch.zeros((len(bboxes), max_len, 4), dtype=torch.float32)

    for i, (cls, box) in enumerate(zip(classes, bboxes)):
        padded_classes[i, :len(cls)] = cls
        padded_bboxes[i, :len(box)] = box

    return (images, sizes), (padded_classes, padded_bboxes)

def generate_anchors(image_size=224, feature_size=7, box_size=32):
    step = image_size // feature_size
    anchors = []
    for y in range(feature_size):
        for x in range(feature_size):
            center_y = step * (y + 0.5)
            center_x = step * (x + 0.5)
            half_size = box_size / 2
            top = center_y - half_size
            left = center_x - half_size
            bottom = center_y + half_size
            right = center_x + half_size
            anchors.append([top, left, bottom, right])
            
    return torch.tensor(anchors, dtype=torch.float32)

class RetinaNetModel(npfl138.TrainableModule):
    def __init__(self, backbone, num_classes, device, freeze_backbone=True):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.backbone = backbone
        self.out_channels = self.backbone.feature_info[-1]['num_chs']
        self.feature_size = 256 
        
        self.fpn = torch.nn.Sequential(
            torch.nn.Conv2d(self.out_channels, self.feature_size, kernel_size=1, stride=1),
            torch.nn.Conv2d(self.feature_size, self.feature_size, kernel_size=3, padding=1)
        )

        self.cls_head = torch.nn.Sequential(
            torch.nn.Conv2d(self.feature_size, self.feature_size, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.feature_size, self.feature_size, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.feature_size, (num_classes + 1), 3, padding=1),
        )

        self.box_head = torch.nn.Sequential(
            torch.nn.Conv2d(self.feature_size, self.feature_size, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.feature_size, self.feature_size, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.feature_size, 4, 3, padding=1),
        )
        
        if freeze_backbone:
            self.freeze_backbone()

    def forward(self, x):
        features = self.backbone(x)
        C5 = features[-1]
        
        P5 = self.fpn(C5)
        anchors = generate_anchors().to(x.device)
        
   
        cls_logits = self.cls_head(P5)
        box_deltas = self.box_head(P5)

        B, C, H, W = cls_logits.shape
        cls_logits = cls_logits.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes + 1)
        box_deltas = box_deltas.permute(0, 2, 3, 1).reshape(B, -1, 4)

        return (cls_logits, box_deltas), anchors
    
    
    def compute_loss(self, y_pred, y, *xs):
        (cls_logits, box_deltas), anchors = y_pred  
        gold_classes, gold_bboxes = y 

                
        B, _, _ = cls_logits.shape
        
        anchor_classes_all = []
        anchor_bboxes_all = []

        for b in range(B):
            anchor_classes, anchor_bboxes = bboxes_utils.bboxes_training(
                anchors, gold_classes[b], gold_bboxes[b], iou_threshold=0.5
            )
            
            anchor_classes_all.append(anchor_classes)
            anchor_bboxes_all.append(anchor_bboxes)
            


        anchor_classes = torch.stack(anchor_classes_all)
        anchor_bboxes = torch.stack(anchor_bboxes_all)
        
        target_cls = torch.nn.functional.one_hot(anchor_classes, num_classes=self.num_classes + 1).float()
        cls_loss = ops.sigmoid_focal_loss(cls_logits, target_cls, reduction="mean")

        reg_loss = torch.tensor(0.0, device=self.device)
        positive = anchor_classes > 0
        
        if positive.any():
            reg_loss = torch.nn.functional.smooth_l1_loss(
                box_deltas[positive], anchor_bboxes[positive], reduction="mean"
            )

        return cls_loss + reg_loss
    
    def train_step(self, xs, y):
        """An overridable method performing a single training step, returning the logs."""
        self.optimizer.zero_grad()
        images, sizes = xs
        y_pred = self(images)
        loss = self.compute_loss(y_pred, y)
        loss.backward()
        with torch.no_grad():
            self.optimizer.step()
            self.scheduler is not None and self.scheduler.step()
            return {"loss": self.loss_tracker(loss)} \
                | ({"lr": self.scheduler.get_last_lr()[0]} if self.scheduler else {}) \
                    
    def test_step(self, xs, y):
        """An overridable method performing a single evaluation step, returning the logs."""
        input, sizes = xs
        with torch.no_grad():
            y_pred = self(input)
            loss = self.compute_loss(y_pred, y, input)
            return {"loss": self.loss_tracker(loss)} | self.compute_metrics(y_pred, y, *xs)
                
    
    def predict(self, images, score_thresh=0.5, iou_thresh=0.5):
        self.eval()
        with torch.no_grad():
            (cls_logits, box_deltas), anchors = self(images)
            probs = torch.sigmoid(cls_logits)  # [B, 49, 11]
            results = []
            for i in range(images.size(0)):
                scores, labels = probs[i][:, 1:].max(dim=-1)
                keep = scores > score_thresh
                if keep.any():
                    decoded = bboxes_utils.bboxes_from_rcnn(anchors, box_deltas[i])
                    boxes = decoded[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    
                    keep_nms = ops.batched_nms(boxes, scores, labels, iou_thresh)
                    results.append((labels[keep_nms], boxes[keep_nms]))
                else:
                    results.append((torch.empty(0, dtype=torch.long), torch.empty(0, 4)))
            return results
        
    def train(self, mode=True):
        self.backbone.eval()
        self.backbone.train(False)
        self.cls_head.train(mode)
        self.box_head.train(mode)
        
        return self
        
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        
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

    # Load the data. The individual examples are dictionaries with the keys:
    # - "image", a `[3, SIZE, SIZE]` tensor of `torch.uint8` values in [0-255] range,
    # - "classes", a `[num_digits]` PyTorch vector with classes of image digits,
    # - "bboxes", a `[num_digits, 4]` PyTorch vector with bounding boxes of image digits.
    # The `decode_on_demand` argument can be set to `True` to save memory and decode
    # each image only when accessed, but it will most likely slow down training.
    svhn = SVHN(decode_on_demand=False)

    # Load the EfficientNetV2-B0 model without the classification layer.
    # Apart from calling the model as in the classification task, you can call it using
    #   output, features = efficientnetv2_b0.forward_intermediates(batch_of_images)
    # obtaining (assuming the input images have 224x224 resolution):
    # - `output` is a `[N, 1280, 7, 7]` tensor with the final features before global average pooling,
    # - `features` is a list of intermediate features with resolution 112x112, 56x56, 28x28, 14x14, 7x7.
    efficientnetv2_b0 = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True, features_only=True ,num_classes=0)

    # Create a simple preprocessing performing necessary normalization.
    preprocessing = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToDtype(torch.float32, scale=True),  # The `scale=True` also rescales the image to [0, 1].
        v2.Normalize(mean=efficientnetv2_b0.pretrained_cfg["mean"], std=efficientnetv2_b0.pretrained_cfg["std"]),
    ])
        
    train = torch.utils.data.DataLoader(SVHNDataset(svhn.train, preprocessing), batch_size=args.batch_size, collate_fn=detection_collate_fn, num_workers=0, shuffle=True)
    dev = torch.utils.data.DataLoader(SVHNDataset(svhn.dev, preprocessing), batch_size=args.batch_size, collate_fn=detection_collate_fn, num_workers=0, shuffle=True)
    dev_test = torch.utils.data.DataLoader(SVHNDataset(svhn.dev, preprocessing), batch_size=args.batch_size, collate_fn=detection_collate_fn, shuffle=False)
    test = torch.utils.data.DataLoader(SVHNDataset(svhn.test, preprocessing), batch_size=args.batch_size, collate_fn=detection_collate_fn, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RetinaNetModel(backbone=efficientnetv2_b0, num_classes=svhn.LABELS, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    # schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    model.configure(
        optimizer=optimizer,
        # scheduler=schedular,
        loss=lambda y_pred, y, *xs: model.compute_loss(y_pred, y, *xs),
        logdir=args.logdir,
    )
    
    checkpoint_path = 'retinanet_svhn_base.pth'
    state_dict = torch.load('path_to_pretrained_weights.pth')
    model.load_state_dict(state_dict, strict=False)

    # model.load_state_dict(torch.load("checkpoints/retinanet_svhn_base.pth", map_location=device))

    model.fit(train, dev=dev, epochs=args.epochs)

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("checkpoints", checkpoint_path))
            
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "svhn_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the digits and their bounding boxes on the test set.
        # Assume that for a single test image we get
        # - `predicted_classes`: a 1D array with the predicted digits,
        # - `predicted_bboxes`: a [len(predicted_classes), 4] array with bboxes;
        batch_j = 0
        for input, _, in dev_test:
            print("Evaluate", batch_j, "/", len(dev_test))
            images, sizes = input
            images = images.to(device)
            sizes = sizes.to(device)
            predictions = model.predict(images, score_thresh=0.25)  
            for i, (predicted_classes, predicted_bboxes) in enumerate(predictions):
                orig_h, orig_w = sizes[i]
                scale_h = float(orig_h) / 224.0
                scale_w = float(orig_w) / 224.0


                predicted_bboxes[:, [0, 2]] *= scale_h
                predicted_bboxes[:, [1, 3]] *= scale_w
                
                output = []
                for label, bbox in zip(predicted_classes, predicted_bboxes):
                    output += [int(label)] + list(map(float, bbox))
                print(*output, file=predictions_file)
                
            
            batch_j += 1
            
    print("python3 -m npfl138.datasets.svhn --visualize="+os.path.join(args.logdir, "svhn_competition.txt")+" --dataset=dev")
                


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)

import argparse
import datetime
import os
import re

import numpy as np
import timm
import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF

import bboxes_utils
import npfl138
npfl138.require_version("2425.6.1")
from npfl138.datasets.svhn import SVHN

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")

class SVHNWrapper(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        image = TF.resize(sample["image"], [224, 224])
        if self.transform:
            image = self.transform(image)
        labels = sample["classes"].clone().detach().long()
        bboxes = sample["bboxes"].clone().detach().float()
        return image, labels, bboxes

def main(args: argparse.Namespace) -> None:
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    svhn = SVHN(decode_on_demand=False)

    efficientnetv2_b0 = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True, num_classes=0, features_only=True)
    feature_channels = efficientnetv2_b0.feature_info.channels()[-1]

    class FeatureExtractor(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, x):
            feats = self.base(x)[-1]
            return feats

    backbone = FeatureExtractor(efficientnetv2_b0)

    preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class SVHNModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.classifier = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(feature_channels, 10)
            )
            self.bbox_regressor = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(feature_channels, 4)
            )

        def forward(self, x):
            features = self.backbone(x)
            class_logits = self.classifier(features)
            bbox_coords = self.bbox_regressor(features)
            return class_logits, bbox_coords

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SVHNModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_class = torch.nn.CrossEntropyLoss()
    loss_bbox = torch.nn.MSELoss()

    train_dataset = SVHNWrapper(svhn.train, transform=preprocessing)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = SVHNWrapper(svhn.test, transform=preprocessing)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    for epoch in range(args.epochs):
        model.train()
        for images, labels, bboxes in train_loader:
            images = images.to(device)
            bboxes = bboxes[:, 0].to(device)  # Use bbox for first digit only
            label = labels[:, 0].to(device)   # Use label for first digit only

            pred_class, pred_bbox = model(images)
            loss = loss_class(pred_class, label) + loss_bbox(pred_bbox, bboxes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    os.makedirs(args.logdir, exist_ok=True)
    predictions = []
    with open(os.path.join(args.logdir, "svhn_competition.txt"), "w", encoding="utf-8") as predictions_file:
        with torch.no_grad():
            for images, labels, bboxes in test_loader:
                images = images.to(device)
                batch_size = images.shape[0]
                pred_class, pred_bbox = model(images)

                pred_labels = torch.argmax(pred_class, dim=1).cpu().numpy()
                pred_bboxes = pred_bbox.cpu().numpy()

                for i in range(batch_size):
                    output = [int(pred_labels[i])] + list(map(float, pred_bboxes[i]))
                    print(*output, file=predictions_file)
                    predictions.append(([int(pred_labels[i])], [pred_bboxes[i].tolist()]))

    acc = SVHN.evaluate(svhn.test, predictions)
    print(f"Test accuracy (IoU >= 0.5): {acc:.2f}%")

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)

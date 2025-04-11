#!/usr/bin/env python3
import argparse
from math import log
from typing import Callable
import unittest

import torch

# Bounding boxes and anchors are expected to be PyTorch tensors,
# where the last dimension has size 4.

# For bounding boxes in pixel coordinates, the 4 values correspond to
# (top, left, bottom, right) with top <= bottom and left <= right.
TOP: int = 0
LEFT: int = 1
BOTTOM: int = 2
RIGHT: int = 3


def bboxes_area(bboxes: torch.Tensor) -> torch.Tensor:
    return torch.relu(bboxes[..., BOTTOM] - bboxes[..., TOP]) \
        * torch.relu(bboxes[..., RIGHT] - bboxes[..., LEFT])


def bboxes_iou(xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    intersections = torch.stack([
        torch.maximum(xs[..., TOP], ys[..., TOP]),
        torch.maximum(xs[..., LEFT], ys[..., LEFT]),
        torch.minimum(xs[..., BOTTOM], ys[..., BOTTOM]),
        torch.minimum(xs[..., RIGHT], ys[..., RIGHT]),
    ], dim=-1)

    xs_area, ys_area, intersections_area = bboxes_area(xs), bboxes_area(ys), bboxes_area(intersections)

    return intersections_area / (xs_area + ys_area - intersections_area)


def bboxes_to_rcnn(anchors: torch.Tensor, bboxes: torch.Tensor) -> torch.Tensor:
    # TODO: Implement according to the docstring.
    # Compute anchor centers and sizes.
    anchor_height = anchors[..., BOTTOM] - anchors[..., TOP]
    anchor_width = anchors[..., RIGHT] - anchors[..., LEFT]
    anchor_center_y = (anchors[..., TOP] + anchors[..., BOTTOM]) / 2.0
    anchor_center_x = (anchors[..., LEFT] + anchors[..., RIGHT]) / 2.0

    # Compute bbox centers and sizes.
    bbox_height = bboxes[..., BOTTOM] - bboxes[..., TOP]
    bbox_width = bboxes[..., RIGHT] - bboxes[..., LEFT]
    bbox_center_y = (bboxes[..., TOP] + bboxes[..., BOTTOM]) / 2.0
    bbox_center_x = (bboxes[..., LEFT] + bboxes[..., RIGHT]) / 2.0

    # Compute the transformation.
    dy = (bbox_center_y - anchor_center_y) / anchor_height
    dx = (bbox_center_x - anchor_center_x) / anchor_width
    dh = torch.log(bbox_height / anchor_height)
    dw = torch.log(bbox_width / anchor_width)

    return torch.stack((dy, dx, dh, dw), dim=-1)


def bboxes_from_rcnn(anchors: torch.Tensor, rcnns: torch.Tensor) -> torch.Tensor:
    # TODO: Implement according to the docstring.
    # Compute anchor centers and sizes.
    anchor_height = anchors[..., BOTTOM] - anchors[..., TOP]
    anchor_width = anchors[..., RIGHT] - anchors[..., LEFT]
    anchor_center_y = (anchors[..., TOP] + anchors[..., BOTTOM]) / 2.0
    anchor_center_x = (anchors[..., LEFT] + anchors[..., RIGHT]) / 2.0

    # Extract R-CNN deltas.
    dy = rcnns[..., 0]
    dx = rcnns[..., 1]
    dh = rcnns[..., 2]
    dw = rcnns[..., 3]

    # Recover bbox center and size.
    bbox_center_y = dy * anchor_height + anchor_center_y
    bbox_center_x = dx * anchor_width + anchor_center_x
    bbox_height = torch.exp(dh) * anchor_height
    bbox_width = torch.exp(dw) * anchor_width

    # Compute bbox coordinates.
    top = bbox_center_y - bbox_height / 2.0
    left = bbox_center_x - bbox_width / 2.0
    bottom = bbox_center_y + bbox_height / 2.0
    right = bbox_center_x + bbox_width / 2.0

    return torch.stack((top, left, bottom, right), dim=-1)


def bboxes_training(
    anchors: torch.Tensor, gold_classes: torch.Tensor, gold_bboxes: torch.Tensor, iou_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO: First, for each gold object, assign it to an anchor with the
    # largest IoU (the anchor with smaller index if there are several). In case
    # several gold objects are assigned to a single anchor, use the gold object
    # with smaller index.

    num_anchors = anchors.shape[0]
    num_gold = gold_bboxes.shape[0]
    assigned_gold = torch.full((num_anchors,), -1, dtype=torch.long)

    # Compute IoU matrix: shape (num_anchors, num_gold)
    iou_matrix = bboxes_iou(anchors.unsqueeze(1), gold_bboxes.unsqueeze(0))

    for g in range(num_gold):
        best_anchor = torch.argmax(iou_matrix[:, g]).item()  # smallest index in case of ties
        if assigned_gold[best_anchor] == -1:
            assigned_gold[best_anchor] = g

    # TODO: For each unused anchor, find the gold object with the largest IoU
    # (again the gold object with smaller index if there are several), and if
    # the IoU is >= threshold, assign the object to the anchor.

    for a in range(num_anchors):
        if assigned_gold[a] == -1:
            best_iou, best_gold = torch.max(iou_matrix[a, :], dim=0)
            if best_iou >= iou_threshold:
                assigned_gold[a] = best_gold.item()

    anchor_classes, anchor_bboxes = torch.zeros(num_anchors, dtype=torch.long, device="cuda"), torch.zeros((num_anchors, 4), dtype=torch.float32, device="cuda")

    assigned_mask = assigned_gold != -1
    if assigned_mask.any():
        gold_idx = assigned_gold[assigned_mask]
        anchor_classes[assigned_mask] = 1 + gold_classes[gold_idx]
        anchor_bboxes[assigned_mask] = bboxes_to_rcnn(anchors[assigned_mask], gold_bboxes[gold_idx])

    return anchor_classes, anchor_bboxes


def main(args: argparse.Namespace) -> tuple[Callable, Callable, Callable]:
    return bboxes_to_rcnn, bboxes_from_rcnn, bboxes_training


class Tests(unittest.TestCase):
    def test_bboxes_to_from_rcnn(self):
        data = [
            [[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 0, 0]],
            [[0, 0, 10, 10], [5, 0, 15, 10], [.5, 0, 0, 0]],
            [[0, 0, 10, 10], [0, 5, 10, 15], [0, .5, 0, 0]],
            [[0, 0, 10, 10], [0, 0, 20, 30], [.5, 1, log(2), log(3)]],
            [[0, 9, 10, 19], [2, 10, 5, 16], [-0.15, -0.1, -1.20397, -0.51083]],
            [[5, 3, 15, 13], [7, 7, 10, 9], [-0.15, 0, -1.20397, -1.60944]],
            [[7, 6, 17, 16], [9, 10, 12, 13], [-0.15, 0.05, -1.20397, -1.20397]],
            [[5, 6, 15, 16], [7, 7, 10, 10], [-0.15, -0.25, -1.20397, -1.20397]],
            [[6, 3, 16, 13], [8, 5, 12, 8], [-0.1, -0.15, -0.91629, -1.20397]],
            [[5, 2, 15, 12], [9, 6, 12, 8], [0.05, 0, -1.20397, -1.60944]],
            [[2, 10, 12, 20], [6, 11, 8, 17], [0, -0.1, -1.60944, -0.51083]],
            [[10, 9, 20, 19], [12, 13, 17, 16], [-0.05, 0.05, -0.69315, -1.20397]],
            [[6, 7, 16, 17], [10, 11, 12, 14], [0, 0.05, -1.60944, -1.20397]],
            [[2, 2, 12, 12], [3, 5, 8, 8], [-0.15, -0.05, -0.69315, -1.20397]],
        ]
        # First run on individual anchors, and then on all together
        for anchors, bboxes, rcnns in [map(lambda x: [x], row) for row in data] + [zip(*data)]:
            anchors, bboxes, rcnns = [torch.tensor(data, dtype=torch.float32) for data in [anchors, bboxes, rcnns]]
            torch.testing.assert_close(bboxes_to_rcnn(anchors, bboxes), rcnns, atol=1e-3, rtol=1e-3)
            torch.testing.assert_close(bboxes_from_rcnn(anchors, rcnns), bboxes, atol=1e-3, rtol=1e-3)

    def test_bboxes_training(self):
        anchors = torch.tensor([[0, 0, 10, 10], [0, 10, 10, 20], [10, 0, 20, 10], [10, 10, 20, 20]])
        for gold_classes, gold_bboxes, anchor_classes, anchor_bboxes, iou in [
                [[1], [[14, 14, 16, 16]], [0, 0, 0, 2], [[0, 0, 0, 0]] * 3 + [[0, 0, log(.2), log(.2)]], 0.5],
                [[2], [[0, 0, 20, 20]], [3, 0, 0, 0], [[.5, .5, log(2), log(2)]] + [[0, 0, 0, 0]] * 3, 0.26],
                [[2], [[0, 0, 20, 20]], [3, 3, 3, 3],
                 [[y, x, log(2), log(2)] for y in [.5, -.5] for x in [.5, -.5]], 0.24],
                [[0, 1], [[3, 3, 20, 18], [10, 1, 18, 21]], [0, 0, 0, 1],
                 [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [-0.35, -0.45, 0.53062, 0.40546]], 0.5],
                [[0, 1], [[3, 3, 20, 18], [10, 1, 18, 21]], [0, 0, 2, 1],
                 [[0, 0, 0, 0], [0, 0, 0, 0], [-0.1, 0.6, -0.22314, 0.69314], [-0.35, -0.45, 0.53062, 0.40546]], 0.3],
                [[0, 1], [[3, 3, 20, 18], [10, 1, 18, 21]], [0, 1, 2, 1],
                 [[0, 0, 0, 0], [0.65, -0.45, 0.53062, 0.40546], [-0.1, 0.6, -0.22314, 0.69314],
                  [-0.35, -0.45, 0.53062, 0.40546]], 0.17],
        ]:
            gold_classes, anchor_classes = torch.tensor(gold_classes), torch.tensor(anchor_classes)
            gold_bboxes, anchor_bboxes = torch.tensor(gold_bboxes), torch.tensor(anchor_bboxes)
            computed_classes, computed_bboxes = bboxes_training(anchors, gold_classes, gold_bboxes, iou)
            torch.testing.assert_close(computed_classes, anchor_classes, atol=1e-3, rtol=1e-3)
            torch.testing.assert_close(computed_bboxes, anchor_bboxes, atol=1e-3, rtol=1e-3)


if __name__ == '__main__':
    unittest.main()

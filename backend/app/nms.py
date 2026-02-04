from __future__ import annotations

from typing import Iterable, List, Mapping, Tuple, Union


BBox = Tuple[int, int, int, int]
BoxLike = Union[BBox, Mapping[str, float]]


def _as_box_tuple(box: BoxLike) -> BBox:
    if isinstance(box, tuple):
        if len(box) != 4:
            raise ValueError("bbox tuple must have 4 elements")
        return int(box[0]), int(box[1]), int(box[2]), int(box[3])
    return int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])


def compute_iou(box1: BoxLike, box2: BoxLike) -> float:
    x1, y1, w1, h1 = _as_box_tuple(box1)
    x2, y2, w2, h2 = _as_box_tuple(box2)

    if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
        return 0.0

    ax1, ay1, ax2, ay2 = x1, y1, x1 + w1, y1 + h1
    bx1, by1, bx2, by2 = x2, y2, x2 + w2, y2 + h2

    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0

    inter_area = inter_w * inter_h
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area / union)


def nms(bboxes: List[BoxLike], scores: List[float], iou_threshold: float) -> List[int]:
    if len(bboxes) != len(scores):
        raise ValueError("bboxes and scores must have same length")
    if not bboxes:
        return []

    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    keep: List[int] = []

    while order:
        current = order.pop(0)
        keep.append(current)
        if not order:
            break
        remaining: List[int] = []
        current_box = bboxes[current]
        for idx in order:
            if compute_iou(current_box, bboxes[idx]) <= iou_threshold:
                remaining.append(idx)
        order = remaining

    return keep

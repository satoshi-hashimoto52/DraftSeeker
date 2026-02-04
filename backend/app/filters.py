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


def _extract_bbox_score(item: object) -> Tuple[BBox, float]:
    if hasattr(item, "bbox") and hasattr(item, "score"):
        bbox = getattr(item, "bbox")
        score = getattr(item, "score")
        return _as_box_tuple(bbox), float(score)
    if isinstance(item, Mapping):
        if "bbox" in item and "score" in item:
            return _as_box_tuple(item["bbox"]), float(item["score"])
        if {"x", "y", "w", "h", "score"}.issubset(item.keys()):
            bbox = {"x": item["x"], "y": item["y"], "w": item["w"], "h": item["h"]}
            return _as_box_tuple(bbox), float(item["score"])
    if isinstance(item, tuple) and len(item) == 2:
        bbox, score = item
        return _as_box_tuple(bbox), float(score)
    raise ValueError("unsupported bbox item; expected bbox+score")


def filter_bboxes(
    bboxes: Iterable[object],
    roi_size: int,
    score_threshold: float,
) -> List[object]:
    min_size = roi_size * 0.05
    max_area = roi_size * roi_size * 0.8
    filtered: List[object] = []
    for item in bboxes:
        bbox, score = _extract_bbox_score(item)
        _, _, w, h = bbox
        if w < min_size or h < min_size:
            continue
        if (w * h) > max_area:
            continue
        if score < score_threshold:
            continue
        filtered.append(item)
    return filtered

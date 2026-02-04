from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple


def _clamp01(value: float) -> float:
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return float(value)


def normalize_bbox(bbox: dict, image_w: int, image_h: int) -> Tuple[float, float, float, float]:
    x = float(bbox["x"])
    y = float(bbox["y"])
    w = float(bbox["w"])
    h = float(bbox["h"])
    cx = (x + w / 2.0) / float(image_w)
    cy = (y + h / 2.0) / float(image_h)
    nw = w / float(image_w)
    nh = h / float(image_h)
    return _clamp01(cx), _clamp01(cy), _clamp01(nw), _clamp01(nh)


def normalize_polygon(poly: Sequence[Sequence[float]], image_w: int, image_h: int) -> List[float]:
    coords: List[float] = []
    for pt in poly:
        if len(pt) < 2:
            continue
        x = _clamp01(float(pt[0]) / float(image_w))
        y = _clamp01(float(pt[1]) / float(image_h))
        coords.extend([x, y])
    return coords


def make_yolo_lines(
    annotations: Iterable[dict],
    class_to_id: Dict[str, int],
    image_w: int,
    image_h: int,
) -> List[str]:
    lines: List[str] = []
    for ann in annotations:
        class_name = ann.get("class_name")
        if class_name not in class_to_id:
            continue
        class_id = class_to_id[class_name]
        poly = ann.get("segPolygon") or ann.get("seg_polygon")
        use_poly = isinstance(poly, list) and len(poly) >= 3
        if use_poly:
            coords = normalize_polygon(poly, image_w, image_h)
            if len(coords) >= 6:
                parts = [str(class_id)] + [f"{v:.6f}" for v in coords]
                lines.append(" ".join(parts))
                continue
        bbox = ann.get("bbox")
        if not bbox:
            continue
        cx, cy, w, h = normalize_bbox(bbox, image_w, image_h)
        parts = [str(class_id), f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
        lines.append(" ".join(parts))
    return lines

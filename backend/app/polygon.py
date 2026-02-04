from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def mask_to_polygon(mask: np.ndarray, eps: float) -> List[List[int]]:
    if mask is None or mask.size == 0:
        return []
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() <= 1:
        mask = mask * 255

    contour_data = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contour_data[0] if len(contour_data) == 2 else contour_data[1]
    if not contours:
        return []

    contour = max(contours, key=cv2.contourArea)
    epsilon = max(0.0, float(eps))
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return [[int(pt[0][0]), int(pt[0][1])] for pt in approx]


def polygon_to_bbox(poly: List[List[int]]) -> dict:
    if not poly:
        return {"x": 0, "y": 0, "w": 0, "h": 0}
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    return {"x": min_x, "y": min_y, "w": max_x - min_x, "h": max_y - min_y}

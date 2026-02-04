from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class ContourCandidate:
    bbox: Tuple[int, int, int, int]
    contour: List[Tuple[int, int]]
    score: float


def _clip_roi(x: int, y: int, roi_size: int, width: int, height: int) -> Tuple[int, int, int, int]:
    half = roi_size // 2
    x0 = max(0, x - half)
    y0 = max(0, y - half)
    x1 = min(width, x + half)
    y1 = min(height, y + half)
    if x1 <= x0:
        x1 = min(width, x0 + 1)
    if y1 <= y0:
        y1 = min(height, y0 + 1)
    return x0, y0, x1, y1


def find_roi_contours(
    image_bgr: np.ndarray,
    x: int,
    y: int,
    roi_size: int,
) -> List[ContourCandidate]:
    if image_bgr is None:
        return []
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    height, width = image_gray.shape[:2]
    x0, y0, x1, y1 = _clip_roi(x, y, roi_size, width, height)
    roi = image_gray[y0:y1, x0:x1]

    roi_blur = cv2.GaussianBlur(roi, (3, 3), 0)
    _threshold, binary = cv2.threshold(
        roi_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contour_data = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contour_data[0] if len(contour_data) == 2 else contour_data[1]

    px = x - x0
    py = y - y0
    roi_area = max(1, (x1 - x0) * (y1 - y0))

    candidates: List[ContourCandidate] = []
    for contour in contours:
        if contour is None or len(contour) == 0:
            continue
        inside = cv2.pointPolygonTest(contour, (float(px), float(py)), False)
        if inside < 0:
            continue

        bx, by, bw, bh = cv2.boundingRect(contour)
        contour_points = [
            (int(pt[0][0] + x0), int(pt[0][1] + y0)) for pt in contour
        ]
        score = float(cv2.contourArea(contour)) / float(roi_area)
        candidates.append(
            ContourCandidate(
                bbox=(x0 + bx, y0 + by, bw, bh),
                contour=contour_points,
                score=score,
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates

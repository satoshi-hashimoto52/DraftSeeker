from __future__ import annotations

from typing import Dict, List

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import numpy as np
from pathlib import Path

from .config import (
    DEFAULT_SCALE_MAX,
    DEFAULT_SCALE_MIN,
    DEFAULT_SCALE_STEPS,
    DEFAULT_TOPK,
    IMAGES_DIR,
    TEMPLATES_ROOT,
)
from .contours import find_roi_contours
from .filters import filter_bboxes
from .matching import MatchResult, match_templates
from .nms import nms
from .schemas import (
    DetectFullRequest,
    DetectFullResponse,
    DetectFullResult,
    DetectPointRequest,
    DetectPointResponse,
    DetectResult,
    SegmentCandidateRequest,
    SegmentCandidateResponse,
    SegmentMeta,
    ExportYoloRequest,
    ExportYoloResponse,
    ProjectInfo,
    TemplateInfo,
    UploadResponse,
)
from .storage import RUNS_DIR, resolve_image_path, save_upload
from .templates import scan_templates
from .sam_service import get_sam_predictor
from .sam_device import get_sam_device
from .polygon import mask_to_polygon, polygon_to_bbox
from .export_yolo import make_yolo_lines


app = FastAPI(title="Annotator MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates_cache = scan_templates(TEMPLATES_ROOT)


@app.get("/templates", response_model=List[ProjectInfo])
def list_templates() -> List[ProjectInfo]:
    projects: List[ProjectInfo] = []
    for project, classes in templates_cache.items():
        templates: List[TemplateInfo] = [
            TemplateInfo(class_name=class_name, count=len(items))
            for class_name, items in classes.items()
        ]
        templates.sort(key=lambda t: t.class_name)
        projects.append(ProjectInfo(name=project, classes=templates))
    projects.sort(key=lambda p: p.name)
    return projects


@app.get("/projects", response_model=List[str])
def list_projects() -> List[str]:
    return sorted(templates_cache.keys())


@app.post("/image/upload", response_model=UploadResponse)
def upload_image(file: UploadFile = File(...)) -> UploadResponse:
    try:
        image_id, width, height = save_upload(file, IMAGES_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return UploadResponse(image_id=image_id, width=width, height=height)


@app.post("/detect/point", response_model=DetectPointResponse)
def detect_point(payload: DetectPointRequest) -> DetectPointResponse:
    try:
        image_path = resolve_image_path(IMAGES_DIR, payload.image_id)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="invalid image_id")

    image = cv2.imread(str(image_path))
    if image is None:
        raise HTTPException(status_code=400, detail="failed to read image")

    if payload.template_off:
        candidates = find_roi_contours(image, payload.x, payload.y, payload.roi_size)
        candidates = candidates[: payload.topk]
        results: List[DetectResult] = [
            DetectResult(
                class_name="contour",
                score=candidate.score,
                bbox={
                    "x": candidate.bbox[0],
                    "y": candidate.bbox[1],
                    "w": candidate.bbox[2],
                    "h": candidate.bbox[3],
                },
                template_name="contour",
                scale=1.0,
                contour=[{"x": pt[0], "y": pt[1]} for pt in candidate.contour],
            )
            for candidate in candidates
        ]
        return DetectPointResponse(results=results)

    project_templates = templates_cache.get(payload.project)
    if project_templates is None:
        raise HTTPException(status_code=400, detail="invalid project")

    matches = match_templates(
        image_bgr=image,
        x=payload.x,
        y=payload.y,
        roi_size=payload.roi_size,
        templates=project_templates,
        scale_min=payload.scale_min or DEFAULT_SCALE_MIN,
        scale_max=payload.scale_max or DEFAULT_SCALE_MAX,
        scale_steps=payload.scale_steps or DEFAULT_SCALE_STEPS,
    )

    filtered_matches = filter_bboxes(matches, payload.roi_size, payload.score_threshold)

    grouped: Dict[str, List[int]] = {}
    for idx, match in enumerate(filtered_matches):
        grouped.setdefault(match.class_name, []).append(idx)

    kept_matches: List = []
    for indices in grouped.values():
        class_boxes = [filtered_matches[i].bbox for i in indices]
        class_scores = [filtered_matches[i].score for i in indices]
        kept_indices = nms(class_boxes, class_scores, payload.iou_threshold)
        kept_matches.extend(filtered_matches[indices[i]] for i in kept_indices)

    best_per_class: Dict[str, MatchResult] = {}
    for match in kept_matches:
        current = best_per_class.get(match.class_name)
        if current is None or match.score > current.score:
            best_per_class[match.class_name] = match

    representative = list(best_per_class.values())
    representative.sort(key=lambda r: r.score, reverse=True)
    representative = representative[: payload.topk]

    results: List[DetectResult] = [
        DetectResult(
            class_name=match.class_name,
            score=match.score,
            bbox={"x": match.bbox[0], "y": match.bbox[1], "w": match.bbox[2], "h": match.bbox[3]},
            template_name=match.template_name,
            scale=match.scale,
        )
        for match in representative
    ]

    return DetectPointResponse(results=results)


@app.post("/detect/full", response_model=DetectFullResponse)
def detect_full(payload: DetectFullRequest) -> DetectFullResponse:
    try:
        image_path = resolve_image_path(IMAGES_DIR, payload.image_id)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="invalid image_id")

    image = cv2.imread(str(image_path))
    if image is None:
        raise HTTPException(status_code=400, detail="failed to read image")

    project_templates = templates_cache.get(payload.project)
    if project_templates is None:
        raise HTTPException(status_code=400, detail="invalid project")

    tile_size = 1024
    height, width = image.shape[:2]
    matches: List[MatchResult] = []

    for y0 in range(0, height, tile_size):
        y1 = min(height, y0 + tile_size)
        for x0 in range(0, width, tile_size):
            x1 = min(width, x0 + tile_size)
            tile = image[y0:y1, x0:x1]
            tile_w = x1 - x0
            tile_h = y1 - y0
            if tile_w <= 0 or tile_h <= 0:
                continue
            cx = tile_w // 2
            cy = tile_h // 2
            roi_size = max(tile_w, tile_h)
            tile_matches = match_templates(
                image_bgr=tile,
                x=cx,
                y=cy,
                roi_size=roi_size,
                templates=project_templates,
                scale_min=payload.scale_min or DEFAULT_SCALE_MIN,
                scale_max=payload.scale_max or DEFAULT_SCALE_MAX,
                scale_steps=payload.scale_steps or DEFAULT_SCALE_STEPS,
            )
            for match in tile_matches:
                matches.append(
                    MatchResult(
                        class_name=match.class_name,
                        template_name=match.template_name,
                        score=match.score,
                        scale=match.scale,
                        bbox=(
                            match.bbox[0] + x0,
                            match.bbox[1] + y0,
                            match.bbox[2],
                            match.bbox[3],
                        ),
                    )
                )

    filtered_matches = filter_bboxes(matches, tile_size, payload.score_threshold)

    grouped: Dict[str, List[int]] = {}
    for idx, match in enumerate(filtered_matches):
        grouped.setdefault(match.class_name, []).append(idx)

    kept_matches: List[MatchResult] = []
    for indices in grouped.values():
        class_boxes = [filtered_matches[i].bbox for i in indices]
        class_scores = [filtered_matches[i].score for i in indices]
        kept_indices = nms(class_boxes, class_scores, payload.iou_threshold)
        kept_matches.extend(filtered_matches[indices[i]] for i in kept_indices)

    best_per_class: Dict[str, MatchResult] = {}
    for match in kept_matches:
        current = best_per_class.get(match.class_name)
        if current is None or match.score > current.score:
            best_per_class[match.class_name] = match

    representative = list(best_per_class.values())
    representative.sort(key=lambda r: r.score, reverse=True)
    representative = representative[: payload.topk]

    results: List[DetectFullResult] = [
        DetectFullResult(
            class_name=match.class_name,
            score=match.score,
            bbox={"x": match.bbox[0], "y": match.bbox[1], "w": match.bbox[2], "h": match.bbox[3]},
        )
        for match in representative
    ]

    return DetectFullResponse(results=results)


@app.post("/segment/candidate", response_model=SegmentCandidateResponse)
def segment_candidate(payload: SegmentCandidateRequest) -> SegmentCandidateResponse:
    try:
        image_path = resolve_image_path(IMAGES_DIR, payload.image_id)
    except FileNotFoundError:
        return SegmentCandidateResponse(ok=False, error="invalid image_id")

    image = cv2.imread(str(image_path))
    if image is None:
        return SegmentCandidateResponse(ok=False, error="failed to read image")

    height, width = image.shape[:2]
    bx = payload.bbox.x
    by = payload.bbox.y
    bw = payload.bbox.w
    bh = payload.bbox.h
    if bw <= 0 or bh <= 0:
        return SegmentCandidateResponse(ok=False, error="invalid bbox size")

    expand_w = int(round(bw * payload.expand))
    expand_h = int(round(bh * payload.expand))
    x0 = max(0, bx - expand_w)
    y0 = max(0, by - expand_h)
    x1 = min(width, bx + bw + expand_w)
    y1 = min(height, by + bh + expand_h)
    if x1 <= x0 or y1 <= y0:
        return SegmentCandidateResponse(ok=False, error="invalid expanded bbox")

    roi = image[y0:y1, x0:x1]
    if roi.size == 0:
        return SegmentCandidateResponse(ok=False, error="empty roi")

    box = np.array([[bx - x0, by - y0, bx - x0 + bw, by - y0 + bh]])
    point_coords = None
    point_labels = None
    if payload.click is not None:
        px = payload.click.x - x0
        py = payload.click.y - y0
        point_coords = np.array([[px, py]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)

    def fallback_segment() -> SegmentCandidateResponse | None:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _th, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)

        contour_data = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contour_data[0] if len(contour_data) == 2 else contour_data[1]
        if not contours:
            return None

        target = None
        if payload.click is not None:
            px = payload.click.x - x0
            py = payload.click.y - y0
            for contour in contours:
                if cv2.pointPolygonTest(contour, (float(px), float(py)), False) >= 0:
                    target = contour
                    break
        if target is None:
            cx = (bx + bw / 2) - x0
            cy = (by + bh / 2) - y0
            for contour in contours:
                if cv2.pointPolygonTest(contour, (float(cx), float(cy)), False) >= 0:
                    target = contour
                    break
        if target is None:
            target = max(contours, key=cv2.contourArea)

        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.drawContours(mask, [target], -1, 255, thickness=-1)
        polygon_local = mask_to_polygon(mask, payload.simplify_eps)
        if not polygon_local:
            return None
        polygon = [{"x": pt[0] + x0, "y": pt[1] + y0} for pt in polygon_local]
        area = int(cv2.contourArea(target))
        return SegmentCandidateResponse(
            ok=True,
            polygon=polygon,
            bbox={"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0},
            meta=SegmentMeta(device=get_sam_device(), method="fallback", area=area),
        )

    try:
        predictor = get_sam_predictor()
        predictor.set_image(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        masks, _scores, _logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=False,
        )
        if masks is None or len(masks) == 0:
            raise RuntimeError("sam returned no mask")

        mask = masks[0].astype(np.uint8)
        polygon_local = mask_to_polygon(mask, payload.simplify_eps)
        if not polygon_local:
            raise RuntimeError("no contour found")
        polygon = [{"x": pt[0] + x0, "y": pt[1] + y0} for pt in polygon_local]
        local_bbox = polygon_to_bbox(polygon_local)
        area = int(local_bbox["w"] * local_bbox["h"])

        return SegmentCandidateResponse(
            ok=True,
            polygon=polygon,
            bbox={"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0},
            meta=SegmentMeta(device=get_sam_device(), method="sam", area=area),
        )
    except Exception as exc:
        fallback = fallback_segment()
        if fallback is not None:
            return fallback
        return SegmentCandidateResponse(ok=False, error=str(exc))


@app.post("/export/yolo", response_model=ExportYoloResponse)
def export_yolo(payload: ExportYoloRequest) -> ExportYoloResponse:
    try:
        image_path = resolve_image_path(IMAGES_DIR, payload.image_id)
    except FileNotFoundError:
        return ExportYoloResponse(ok=False, error="invalid image_id")

    image = cv2.imread(str(image_path))
    if image is None:
        return ExportYoloResponse(ok=False, error="failed to read image")
    height, width = image.shape[:2]

    project_templates = templates_cache.get(payload.project)
    if project_templates is None:
        return ExportYoloResponse(ok=False, error="invalid project")
    class_names = sorted(project_templates.keys())
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    annotations = [ann.model_dump() for ann in payload.annotations]
    lines = make_yolo_lines(annotations, class_to_id, width, height)

    project_dir = RUNS_DIR / payload.project
    project_dir.mkdir(parents=True, exist_ok=True)
    output_path = project_dir / f"{payload.image_id}.txt"
    output_path.write_text("\n".join(lines), encoding="utf-8")

    preview_lines = lines[:5]
    preview = "\n".join(preview_lines)
    return ExportYoloResponse(ok=True, saved_path=str(output_path), text_preview=preview)


@app.get("/export/yolo/download")
def download_yolo(path: str) -> FileResponse:
    target = Path(path).expanduser().resolve()
    runs_root = RUNS_DIR.resolve()
    if not str(target).startswith(str(runs_root)):
        raise HTTPException(status_code=400, detail="invalid path")
    if not target.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(str(target), media_type="text/plain", filename=target.name)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class TemplateInfo(BaseModel):
    class_name: str
    count: int


class ProjectInfo(BaseModel):
    name: str
    classes: List[TemplateInfo]


class UploadResponse(BaseModel):
    image_id: str
    width: int
    height: int


class DetectPointRequest(BaseModel):
    image_id: str
    project: str
    x: int
    y: int
    roi_size: int = Field(..., gt=0)
    scale_min: float = Field(0.5, gt=0)
    scale_max: float = Field(1.5, gt=0)
    scale_steps: int = Field(12, gt=0)
    score_threshold: float = Field(-1.0, ge=-1.0, le=1.0)
    iou_threshold: float = Field(0.4, ge=0, le=1)
    topk: int = Field(3, gt=0)
    template_off: bool = False


class BBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class DetectResult(BaseModel):
    class_name: str
    score: float
    bbox: BBox
    template_name: str
    scale: float
    contour: Optional[List[Point]] = None


class DetectPointResponse(BaseModel):
    results: List[DetectResult]


class DetectFullRequest(BaseModel):
    image_id: str
    project: str
    scale_min: float = Field(0.5, gt=0)
    scale_max: float = Field(1.5, gt=0)
    scale_steps: int = Field(12, gt=0)
    score_threshold: float = Field(-1.0, ge=-1.0, le=1.0)
    iou_threshold: float = Field(0.4, ge=0, le=1)
    topk: int = Field(20, gt=0)


class DetectFullResult(BaseModel):
    class_name: str
    score: float
    bbox: BBox


class DetectFullResponse(BaseModel):
    results: List[DetectFullResult]


class Point(BaseModel):
    x: int
    y: int


class SegmentCandidateRequest(BaseModel):
    image_id: str
    bbox: BBox
    click: Point | None = None
    expand: float = Field(0.2, ge=0)
    simplify_eps: float = Field(2.0, ge=0)


class SegmentMeta(BaseModel):
    device: str
    method: str
    area: int


class SegmentCandidateResponse(BaseModel):
    ok: bool
    polygon: List[Point] | None = None
    bbox: BBox | None = None
    meta: SegmentMeta | None = None
    error: str | None = None


class ExportAnnotation(BaseModel):
    class_name: str
    bbox: BBox
    segPolygon: List[Point] | None = None


class ExportYoloRequest(BaseModel):
    project: str
    image_id: str
    annotations: List[ExportAnnotation]


class ExportYoloResponse(BaseModel):
    ok: bool
    saved_path: str | None = None
    text_preview: str | None = None
    error: str | None = None

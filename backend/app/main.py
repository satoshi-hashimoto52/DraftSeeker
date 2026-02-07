from __future__ import annotations

from typing import Dict, List

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import tempfile
import zipfile
from PIL import Image
import numpy as np
import json
from pathlib import Path
from uuid import uuid4
import io
import shutil
from datetime import datetime
import random

from .config import (
    DEFAULT_SCALE_MAX,
    DEFAULT_SCALE_MIN,
    DEFAULT_SCALE_STEPS,
    DEFAULT_TOPK,
    DATASETS_DIR,
    IMAGES_DIR,
    TEMPLATES_ROOT,
)
from .contours import find_roi_contours
from .filters import exclude_confirmed_candidates, filter_bboxes
from .matching import (
    MatchResult,
    apply_vertical_padding,
    filter_overlapping_matches,
    match_templates,
    refine_match_bboxes,
)
from .nms import nms
from .schemas import (
    DetectFullRequest,
    DetectFullResponse,
    DetectFullResult,
    DetectPointRequest,
    DetectPointResponse,
    DetectResult,
    DatasetImportResponse,
    DatasetInfo,
    DatasetSelectRequest,
    ProjectCreateRequest,
    LoadAnnotationsResponse,
    ExportDatasetBBoxRequest,
    ExportDatasetBBoxResponse,
    ExportDatasetSegRequest,
    ExportDatasetSegResponse,
    SaveAnnotationsRequest,
    SegmentCandidateRequest,
    SegmentCandidateResponse,
    SegmentMeta,
    ExportYoloRequest,
    ExportYoloResponse,
    ProjectInfo,
    TemplateInfo,
    UploadResponse,
)
from .storage import IMAGE_EXTS, RUNS_DIR, resolve_image_path
from .templates import scan_templates
from .sam_service import get_sam_predictor
from .sam_device import get_sam_device
from .polygon import mask_to_polygon, polygon_to_bbox
from .export_yolo import make_yolo_lines
from .export_yolo import normalize_bbox


app = FastAPI(title="Annotator MVP")

DATASET_IMAGE_PREFIX = "dataset::"
MEMORY_IMAGE_PREFIX = "mem::"
IN_MEMORY_IMAGES: Dict[str, bytes] = {}

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
BBOX_PAD_DEFAULT_TOP = 0
BBOX_PAD_DEFAULT_BOTTOM = 0
BBOX_PAD_MAP: Dict[str, Dict[str, int]] = {}


def _get_project_class_names(project: str) -> List[str]:
    project_dir = TEMPLATES_ROOT / project
    if project_dir.exists() and project_dir.is_dir():
        class_names = sorted([p.name for p in project_dir.iterdir() if p.is_dir()])
        if class_names:
            return class_names
    project_templates = templates_cache.get(project)
    if project_templates:
        return sorted(project_templates.keys())
    return []


def _split_counts(total: int, ratios: List[int]) -> List[int]:
    ratio_sum = sum(max(0, r) for r in ratios)
    if total <= 0 or ratio_sum <= 0:
        return [0 for _ in ratios]
    raw = [(r / ratio_sum) * total for r in ratios]
    floors = [int(v) for v in raw]
    remaining = total - sum(floors)
    fracs = sorted(
        [(i, raw[i] - floors[i]) for i in range(len(raw))],
        key=lambda t: t[1],
        reverse=True,
    )
    counts = floors[:]
    for i in range(len(fracs)):
        if remaining <= 0:
            break
        idx = fracs[i][0]
        counts[idx] += 1
        remaining -= 1
    return counts


def _load_matching_table(dataset_dir: Path) -> List[dict]:
    table_path = dataset_dir / "matching_table.json"
    if not table_path.exists():
        return []
    try:
        data = json.loads(table_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception:
        return []
    return []


def _save_matching_table(dataset_dir: Path, rows: List[dict]) -> None:
    table_path = dataset_dir / "matching_table.json"
    table_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_export_index(dataset_dir: Path) -> Dict[str, str]:
    index_path = dataset_dir / "exports_index.json"
    if not index_path.exists():
        return {}
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        return {}
    return {}


def _save_export_index(dataset_dir: Path, index: Dict[str, str]) -> None:
    index_path = dataset_dir / "exports_index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


def _project_dir(name: str) -> Path:
    safe = Path(name).name
    return DATASETS_DIR / safe


def _project_images_dir(project_name: str) -> Path:
    return _project_dir(project_name) / "images"


def _project_annotations_dir(project_name: str) -> Path:
    return _project_dir(project_name) / "annotations"


def _project_meta_path(project_name: str) -> Path:
    return _project_dir(project_name) / "meta.json"


def _resolve_any_image_path(image_id: str) -> Path:
    if image_id.startswith(DATASET_IMAGE_PREFIX):
        rest = image_id[len(DATASET_IMAGE_PREFIX) :]
        if "::" not in rest:
            raise FileNotFoundError(image_id)
        project_name, filename = rest.split("::", 1)
        safe_name = Path(filename).name
        return _project_images_dir(project_name) / safe_name
    return resolve_image_path(IMAGES_DIR, image_id)


def _read_image_bgr(image_id: str) -> np.ndarray:
    if image_id.startswith(MEMORY_IMAGE_PREFIX):
        data = IN_MEMORY_IMAGES.get(image_id)
        if not data:
            raise FileNotFoundError(image_id)
        arr = np.frombuffer(data, np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("failed to read image")
        return image
    image_path = _resolve_any_image_path(image_id)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("failed to read image")
    return image


def _save_upload_memory(image_file: UploadFile) -> UploadResponse:
    suffix = Path(image_file.filename or "").suffix.lower()
    if suffix not in IMAGE_EXTS:
        raise ValueError("unsupported file type")
    data = image_file.file.read()
    if not data:
        raise ValueError("empty file")
    try:
        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
    except Exception as exc:
        raise ValueError("invalid image") from exc
    image_id = f"{MEMORY_IMAGE_PREFIX}{uuid4().hex}{suffix}"
    IN_MEMORY_IMAGES[image_id] = data
    return UploadResponse(image_id=image_id, width=width, height=height)


def _parse_internal_id(value: object, fallback: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return fallback


def _load_meta_entries(project_name: str) -> List[Dict[str, object]]:
    meta_path = _project_meta_path(project_name)
    if not meta_path.exists():
        return []
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    images = meta.get("images", [])
    if not isinstance(images, list):
        return []
    if not images:
        return []
    if isinstance(images[0], str):
        entries = []
        for idx, name in enumerate(images, start=1):
            entries.append(
                {
                    "original_filename": str(name),
                    "internal_id": f"{idx:03d}",
                    "import_order": idx,
                }
            )
        return entries
    entries = []
    for idx, item in enumerate(images, start=1):
        if not isinstance(item, dict):
            continue
        name = str(item.get("original_filename") or item.get("filename") or item.get("name") or "")
        if not name:
            continue
        import_order = int(item.get("import_order") or idx)
        internal_raw = item.get("internal_id")
        internal_num = _parse_internal_id(internal_raw, import_order)
        entries.append(
            {
                "original_filename": name,
                "internal_id": f"{internal_num:03d}",
                "import_order": import_order,
            }
        )
    return entries


def _entries_to_filenames(entries: List[Dict[str, object]]) -> List[str]:
    ordered = sorted(entries, key=lambda e: int(e.get("import_order") or 0))
    return [str(e.get("original_filename")) for e in ordered if e.get("original_filename")]


def _entries_to_api(entries: List[Dict[str, object]]) -> List[Dict[str, object]]:
    ordered = sorted(entries, key=lambda e: int(e.get("import_order") or 0))
    return [
        {
            "original_filename": str(e.get("original_filename")),
            "filename": str(e.get("original_filename")),
            "internal_id": str(e.get("internal_id")),
            "import_order": int(e.get("import_order") or 0),
        }
        for e in ordered
        if e.get("original_filename")
    ]


def _project_stats(project_name: str) -> Dict[str, object]:
    images_dir = _project_images_dir(project_name)
    annotations_dir = _project_annotations_dir(project_name)
    meta_entries = _load_meta_entries(project_name)
    images = _entries_to_api(meta_entries)
    if not images:
        images = (
            [
                {
                    "original_filename": p.name,
                    "internal_id": f"{idx:03d}",
                    "import_order": idx,
                }
                for idx, p in enumerate(
                    sorted([p for p in images_dir.iterdir() if p.is_file()]), start=1
                )
            ]
            if images_dir.exists()
            else []
        )
    total_images = len(images)
    annotated_images = 0
    bbox_count = 0
    seg_count = 0
    latest_ts = 0.0
    if annotations_dir.exists():
        for ann_path in annotations_dir.glob("*.json"):
            try:
                data = json.loads(ann_path.read_text(encoding="utf-8"))
                if data:
                    annotated_images += 1
                for ann in data or []:
                    if ann.get("bbox"):
                        bbox_count += 1
                    poly = ann.get("segPolygon")
                    if isinstance(poly, list) and len(poly) >= 3:
                        seg_count += 1
            except Exception:
                continue
            latest_ts = max(latest_ts, ann_path.stat().st_mtime)
    if images_dir.exists():
        for p in images_dir.iterdir():
            try:
                latest_ts = max(latest_ts, p.stat().st_mtime)
            except Exception:
                continue
    updated_at = datetime.fromtimestamp(latest_ts).isoformat() if latest_ts > 0 else None
    return {
        "images": images,
        "total_images": total_images,
        "annotated_images": annotated_images,
        "bbox_count": bbox_count,
        "seg_count": seg_count,
        "updated_at": updated_at,
    }


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


@app.get("/dataset/projects", response_model=List[DatasetInfo])
def list_dataset_projects() -> List[DatasetInfo]:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    projects: List[DatasetInfo] = []
    for p in sorted([d for d in DATASETS_DIR.iterdir() if d.is_dir()]):
        stats = _project_stats(p.name)
        projects.append(
            DatasetInfo(
                project_name=p.name,
                images=stats["images"],
                total_images=stats["total_images"],
                annotated_images=stats["annotated_images"],
                bbox_count=stats["bbox_count"],
                seg_count=stats["seg_count"],
                updated_at=stats["updated_at"],
            )
        )
    return projects


@app.post("/dataset/projects", response_model=DatasetInfo)
def create_dataset_project(payload: ProjectCreateRequest) -> DatasetInfo:
    project_name = Path(payload.project_name).name
    if not project_name:
        raise HTTPException(status_code=400, detail="invalid project_name")
    project_dir = _project_dir(project_name)
    if project_dir.exists():
        raise HTTPException(status_code=400, detail="project already exists")
    project_dir.mkdir(parents=True, exist_ok=True)
    _project_images_dir(project_name).mkdir(parents=True, exist_ok=True)
    _project_annotations_dir(project_name).mkdir(parents=True, exist_ok=True)
    meta = {"project_name": project_name, "images": []}
    _project_meta_path(project_name).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return DatasetInfo(project_name=project_name, images=[], total_images=0, annotated_images=0, bbox_count=0, seg_count=0, updated_at=None)


@app.delete("/dataset/projects/{project_name}")
def delete_dataset_project(project_name: str) -> Dict[str, bool]:
    project_dir = _project_dir(project_name)
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="project not found")
    shutil.rmtree(project_dir)
    return {"ok": True}


@app.post("/dataset/import", response_model=DatasetImportResponse)
def import_dataset(
    project_name: str = Form(...),
    files: List[UploadFile] = File(...),
) -> DatasetImportResponse:
    if not files:
        raise HTTPException(status_code=400, detail="no files")
    project_name = Path(project_name).name
    project_dir = _project_dir(project_name)
    if not project_dir.exists():
        raise HTTPException(status_code=400, detail="project not found")
    images_dir = _project_images_dir(project_name)
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir = _project_annotations_dir(project_name)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    meta_entries = _load_meta_entries(project_name)
    prev_images = [e["original_filename"] for e in meta_entries if e.get("original_filename")]
    prev_set = set(prev_images)

    incoming_order: List[str] = []
    incoming_set: set[str] = set()
    new_files: List[str] = []
    for f in files:
        original = Path(f.filename or "").name
        if not original:
            continue
        suffix = Path(original).suffix.lower()
        if suffix not in IMAGE_EXTS:
            continue
        if original in prev_set:
            if original not in incoming_set:
                incoming_order.append(original)
                incoming_set.add(original)
            continue
        data = f.file.read()
        if not data:
            continue
        dest_name = original
        with (images_dir / dest_name).open("wb") as out:
            out.write(data)
        new_files.append(dest_name)
        if dest_name not in incoming_order:
            incoming_order.append(dest_name)
            incoming_set.add(dest_name)

    prev_filtered = [name for name in prev_images if name in incoming_set]
    appended = [name for name in incoming_order if name not in prev_set]

    max_order = max([int(e.get("import_order") or 0) for e in meta_entries], default=0)
    max_internal = max(
        [_parse_internal_id(e.get("internal_id"), int(e.get("import_order") or 0)) for e in meta_entries],
        default=0,
    )
    kept_entries = [e for e in meta_entries if e.get("original_filename") in incoming_set]
    for name in appended:
        max_order += 1
        max_internal += 1
        kept_entries.append(
            {
                "original_filename": name,
                "internal_id": f"{max_internal:03d}",
                "import_order": max_order,
            }
        )

    # remove files/annotations not present in the new import set
    for path in images_dir.iterdir():
        if not path.is_file():
            continue
        if path.name not in incoming_set:
            path.unlink(missing_ok=True)
    for ann_path in annotations_dir.iterdir():
        if not ann_path.is_file():
            continue
        if not ann_path.name.endswith(".json"):
            continue
        img_name = ann_path.name[: -len(".json")]
        if img_name in incoming_set:
            continue
        ann_path.unlink(missing_ok=True)

    meta = {"project_name": project_name, "images": _entries_to_api(kept_entries)}
    _project_meta_path(project_name).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return DatasetImportResponse(project_name=project_name, count=len(new_files))


@app.get("/dataset/{project_name}", response_model=DatasetInfo)
def get_dataset(project_name: str) -> DatasetInfo:
    project_dir = _project_dir(project_name)
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="project not found")
    stats = _project_stats(project_name)
    return DatasetInfo(
        project_name=project_name,
        images=stats["images"],
        total_images=stats["total_images"],
        annotated_images=stats["annotated_images"],
        bbox_count=stats["bbox_count"],
        seg_count=stats["seg_count"],
        updated_at=stats["updated_at"],
    )


@app.get("/dataset/{project_name}/image/{filename}")
def get_dataset_image(project_name: str, filename: str) -> FileResponse:
    safe_name = Path(filename).name
    image_path = _project_images_dir(project_name) / safe_name
    if not image_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"image not found: project={project_name} filename={safe_name}",
        )
    return FileResponse(image_path)


@app.post("/dataset/select", response_model=UploadResponse)
def select_dataset_image(payload: DatasetSelectRequest) -> UploadResponse:
    project_name = payload.project_name or payload.dataset_id
    if not project_name:
        raise HTTPException(status_code=400, detail="project_name or dataset_id is required")

    filename = payload.filename
    if not filename:
        entries = _load_meta_entries(project_name)
        if entries:
            entries.sort(key=lambda e: int(e.get("import_order") or 0))
            filename = entries[0].get("original_filename")
    if not filename:
        raise HTTPException(status_code=400, detail="filename is required or dataset has no images")

    safe_name = Path(filename).name
    image_path = _project_images_dir(project_name) / safe_name
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="image not found")
    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid image") from exc
    image_id = f"{DATASET_IMAGE_PREFIX}{project_name}::{safe_name}"
    return UploadResponse(image_id=image_id, width=width, height=height, filename=safe_name)


@app.post("/annotations/save")
def save_annotations(payload: SaveAnnotationsRequest) -> Dict[str, bool]:
    project_dir = _project_dir(payload.project_name)
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="project not found")
    annotations_dir = _project_annotations_dir(payload.project_name)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    safe_key = Path(payload.image_key).name
    out_path = annotations_dir / f"{safe_key}.json"
    data = [ann.model_dump() for ann in payload.annotations]
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True}


@app.get("/annotations/load", response_model=LoadAnnotationsResponse)
def load_annotations(project_name: str, image_key: str) -> LoadAnnotationsResponse:
    project_dir = _project_dir(project_name)
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="project not found")
    safe_key = Path(image_key).name
    path = _project_annotations_dir(project_name) / f"{safe_key}.json"
    if not path.exists():
        return LoadAnnotationsResponse(ok=True, annotations=[])
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail="invalid annotations") from exc
    return LoadAnnotationsResponse(ok=True, annotations=data)


@app.post("/export/dataset/bbox", response_model=ExportDatasetBBoxResponse)
def export_dataset_bbox(payload: ExportDatasetBBoxRequest) -> ExportDatasetBBoxResponse:
    project_dir = _project_dir(payload.project_name)
    meta_path = _project_meta_path(payload.project_name)
    if not meta_path.exists():
        return ExportDatasetBBoxResponse(ok=False, error="project not found")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return ExportDatasetBBoxResponse(ok=False, error="invalid meta")

    meta_entries = _load_meta_entries(payload.project_name)
    images = _entries_to_filenames(meta_entries)
    if not images:
        return ExportDatasetBBoxResponse(ok=False, error="no images")
    parent_name = meta.get("project_name", payload.project_name)

    class_names = _get_project_class_names(payload.project)
    if not class_names:
        return ExportDatasetBBoxResponse(ok=False, error="invalid project")
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    ratios = [payload.split_train, payload.split_val, payload.split_test]
    counts = _split_counts(len(images), ratios)
    rng = random.Random(payload.seed)
    shuffled = images[:]
    rng.shuffle(shuffled)

    train_count, val_count, test_count = counts
    train_images = shuffled[:train_count]
    val_images = shuffled[train_count : train_count + val_count]
    test_images = shuffled[train_count + val_count : train_count + val_count + test_count]

    output_dir = Path(payload.output_dir).expanduser()
    if not output_dir.is_absolute():
        return ExportDatasetBBoxResponse(ok=False, error="output_dir must be absolute")
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    output_root = output_dir / f"dataset_{parent_name}_{date_str}"
    if output_root.exists():
        shutil.rmtree(output_root)
    for split in ["train", "val", "test"]:
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=True)
        (output_root / split / "classes.txt").write_text(
            "\n".join(class_names), encoding="utf-8"
        )
        notes = {
            "categories": [{"id": idx, "name": name} for name, idx in class_to_id.items()],
            "info": {"year": datetime.now().year, "version": "1.0", "contributor": "DraftSeeker"},
        }
        (output_root / split / "notes.json").write_text(
            json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    table_rows = _load_matching_table(project_dir)
    def export_split(split_name: str, split_images: List[str], start_idx: int) -> int:
        idx = start_idx
        for image_key in split_images:
            src = _project_images_dir(payload.project_name) / image_key
            if not src.exists():
                continue
            suffix = src.suffix.lower()
            out_name = f"{idx:03d}{suffix}"
            out_img = output_root / split_name / "images" / out_name
            out_lbl = output_root / split_name / "labels" / f"{idx:03d}.txt"
            out_img.write_bytes(src.read_bytes())
            rel_out = out_img.relative_to(output_root.parent).as_posix()
            rel_out = out_img.relative_to(output_root.parent).as_posix()

            ann_path = _project_annotations_dir(payload.project_name) / f"{Path(image_key).name}.json"
            annotations = []
            if ann_path.exists():
                try:
                    annotations = json.loads(ann_path.read_text(encoding="utf-8"))
                except Exception:
                    annotations = []

            if not annotations and payload.include_negatives:
                out_lbl.write_text("", encoding="utf-8")
                table_rows.append(
                    {
                        "image_name": image_key,
                        "index": idx,
                        "split": split_name,
                        "dataset_type": "bbox",
                        "output_path": rel_out,
                    }
                )
                idx += 1
                continue
            if not annotations and not payload.include_negatives:
                idx += 1
                continue

            image = cv2.imread(str(src))
            if image is None:
                out_lbl.write_text("", encoding="utf-8")
                idx += 1
                continue
            height, width = image.shape[:2]
            lines: List[str] = []
            for ann in annotations:
                class_name = ann.get("class_name")
                if class_name not in class_to_id:
                    continue
                bbox = ann.get("bbox")
                if not bbox:
                    continue
                cx, cy, w, h = normalize_bbox(bbox, width, height)
                parts = [
                    str(class_to_id[class_name]),
                    f"{cx:.6f}",
                    f"{cy:.6f}",
                    f"{w:.6f}",
                    f"{h:.6f}",
                ]
                lines.append(" ".join(parts))
            out_lbl.write_text("\n".join(lines), encoding="utf-8")
            table_rows.append(
                {
                    "image_name": image_key,
                    "index": idx,
                    "split": split_name,
                    "dataset_type": "bbox",
                    "output_path": rel_out,
                }
            )
            idx += 1
        return idx

    idx = 1
    idx = export_split("train", train_images, idx)
    idx = export_split("val", val_images, idx)
    idx = export_split("test", test_images, idx)

    _save_matching_table(project_dir, table_rows)
    index = _load_export_index(project_dir)
    index[output_root.name] = str(output_root)
    _save_export_index(project_dir, index)

    return ExportDatasetBBoxResponse(
        ok=True,
        output_dir=str(output_root),
        export_id=output_root.name,
        counts={"train": len(train_images), "val": len(val_images), "test": len(test_images)},
    )


@app.post("/export/dataset/seg", response_model=ExportDatasetSegResponse)
def export_dataset_seg(payload: ExportDatasetSegRequest) -> ExportDatasetSegResponse:
    project_dir = _project_dir(payload.project_name)
    meta_path = _project_meta_path(payload.project_name)
    if not meta_path.exists():
        return ExportDatasetSegResponse(ok=False, error="project not found")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return ExportDatasetSegResponse(ok=False, error="invalid meta")

    meta_entries = _load_meta_entries(payload.project_name)
    images = _entries_to_filenames(meta_entries)
    if not images:
        return ExportDatasetSegResponse(ok=False, error="no images")
    parent_name = meta.get("project_name", payload.project_name)

    class_names = _get_project_class_names(payload.project)
    if not class_names:
        return ExportDatasetSegResponse(ok=False, error="invalid project")
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    ratios = [payload.split_train, payload.split_val, payload.split_test]
    rng = random.Random(payload.seed)
    shuffled = images[:]
    rng.shuffle(shuffled)

    # filter images that have at least one segPolygon
    seg_images: List[str] = []
    annotations_dir = _project_annotations_dir(payload.project_name)
    for image_key in shuffled:
        ann_path = annotations_dir / f"{Path(image_key).name}.json"
        if not ann_path.exists():
            continue
        try:
            anns = json.loads(ann_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if any(isinstance(a.get("segPolygon"), list) and len(a.get("segPolygon")) >= 3 for a in anns):
            seg_images.append(image_key)

    counts = _split_counts(len(seg_images), ratios)
    train_count, val_count, test_count = counts
    train_images = seg_images[:train_count]
    val_images = seg_images[train_count : train_count + val_count]
    test_images = seg_images[train_count + val_count : train_count + val_count + test_count]

    output_dir = Path(payload.output_dir).expanduser()
    if not output_dir.is_absolute():
        return ExportDatasetSegResponse(ok=False, error="output_dir must be absolute")
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    output_root = output_dir / f"dataset_{parent_name}_{date_str}_seg"
    if output_root.exists():
        shutil.rmtree(output_root)
    for split in ["train", "val", "test"]:
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=True)
        (output_root / split / "classes.txt").write_text(
            "\n".join(class_names), encoding="utf-8"
        )
        notes = {
            "categories": [{"id": idx, "name": name} for name, idx in class_to_id.items()],
            "info": {"year": datetime.now().year, "version": "1.0", "contributor": "DraftSeeker"},
        }
        (output_root / split / "notes.json").write_text(
            json.dumps(notes, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def export_split(split_name: str, split_images: List[str], start_idx: int) -> int:
        idx = start_idx
        for image_key in split_images:
            src = _project_images_dir(payload.project_name) / image_key
            if not src.exists():
                continue
            suffix = src.suffix.lower()
            out_name = f"{idx:03d}{suffix}"
            out_img = output_root / split_name / "images" / out_name
            out_lbl = output_root / split_name / "labels" / f"{idx:03d}.txt"
            out_img.write_bytes(src.read_bytes())

            ann_path = annotations_dir / f"{Path(image_key).name}.json"
            try:
                annotations = json.loads(ann_path.read_text(encoding="utf-8"))
            except Exception:
                annotations = []

            image = cv2.imread(str(src))
            if image is None:
                out_lbl.write_text("", encoding="utf-8")
                idx += 1
                continue
            height, width = image.shape[:2]
            lines: List[str] = []
            for ann in annotations:
                poly = ann.get("segPolygon")
                if not (isinstance(poly, list) and len(poly) >= 3):
                    continue
                class_name = ann.get("class_name")
                if class_name not in class_to_id:
                    continue
                coords = []
                for pt in poly:
                    if len(pt) < 2:
                        continue
                    x = max(0.0, min(1.0, float(pt["x"]) / float(width)))
                    y = max(0.0, min(1.0, float(pt["y"]) / float(height)))
                    coords.extend([x, y])
                if len(coords) < 6:
                    continue
                parts = [str(class_to_id[class_name])] + [f"{v:.6f}" for v in coords]
                lines.append(" ".join(parts))
            out_lbl.write_text("\n".join(lines), encoding="utf-8")
            table_rows.append(
                {
                    "image_name": image_key,
                    "index": idx,
                    "split": split_name,
                    "dataset_type": "seg",
                    "output_path": rel_out,
                }
            )
            idx += 1
        return idx

    idx = 1
    idx = export_split("train", train_images, idx)
    idx = export_split("val", val_images, idx)
    idx = export_split("test", test_images, idx)

    _save_matching_table(project_dir, table_rows)
    index = _load_export_index(project_dir)
    index[output_root.name] = str(output_root)
    _save_export_index(project_dir, index)

    return ExportDatasetSegResponse(
        ok=True,
        output_dir=str(output_root),
        export_id=output_root.name,
        counts={"train": len(train_images), "val": len(val_images), "test": len(test_images)},
    )


@app.get("/dataset/export/download")
def download_dataset_export(project_name: str, export_id: str) -> FileResponse:
    project_dir = _project_dir(project_name)
    index = _load_export_index(project_dir)
    export_path = index.get(export_id)
    if not export_path:
        raise HTTPException(status_code=404, detail="export not found")
    export_dir = Path(export_path)
    if not export_dir.exists() or not export_dir.is_dir():
        raise HTTPException(status_code=404, detail="export not found")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp.close()
    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in export_dir.rglob("*"):
            if path.is_dir():
                continue
            rel = path.relative_to(export_dir.parent).as_posix()
            zf.write(path, rel)
    return FileResponse(
        tmp.name,
        media_type="application/zip",
        filename=f"{export_id}.zip",
    )


@app.post("/image/upload", response_model=UploadResponse)
def upload_image(file: UploadFile = File(...)) -> UploadResponse:
    try:
        result = _save_upload_memory(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result


@app.post("/detect/point", response_model=DetectPointResponse)
def detect_point(payload: DetectPointRequest) -> DetectPointResponse:
    try:
        image = _read_image_bgr(payload.image_id)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="invalid image_id")
    except ValueError:
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
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if payload.refine_contour:
        matches = refine_match_bboxes(
            image_gray,
            matches,
            click_xy=(payload.x, payload.y),
            pad=12,
        )
    matches = apply_vertical_padding(
        matches,
        image_gray.shape[0],
        default_top=BBOX_PAD_DEFAULT_TOP,
        default_bottom=BBOX_PAD_DEFAULT_BOTTOM,
        class_pad_map=BBOX_PAD_MAP,
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
    if payload.exclude_enabled:
        confirmed = []
        if payload.confirmed_annotations:
            confirmed = [ann.model_dump() for ann in payload.confirmed_annotations]
        elif payload.confirmed_boxes:
            confirmed = [{"bbox": b.model_dump() if hasattr(b, "model_dump") else b} for b in payload.confirmed_boxes]
        representative = exclude_confirmed_candidates(
            representative,
            confirmed,
            exclude_mode=payload.exclude_mode,
            center_check=payload.exclude_center,
            iou_threshold=payload.exclude_iou_threshold,
        )
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
        image = _read_image_bgr(payload.image_id)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="invalid image_id")
    except ValueError:
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

    matches = apply_vertical_padding(
        matches,
        image.shape[0],
        default_top=BBOX_PAD_DEFAULT_TOP,
        default_bottom=BBOX_PAD_DEFAULT_BOTTOM,
        class_pad_map=BBOX_PAD_MAP,
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
    if payload.exclude_enabled:
        confirmed = []
        if payload.confirmed_annotations:
            confirmed = [ann.model_dump() for ann in payload.confirmed_annotations]
        elif payload.confirmed_boxes:
            confirmed = [{"bbox": b.model_dump() if hasattr(b, "model_dump") else b} for b in payload.confirmed_boxes]
        representative = exclude_confirmed_candidates(
            representative,
            confirmed,
            exclude_mode=payload.exclude_mode,
            center_check=payload.exclude_center,
            iou_threshold=payload.exclude_iou_threshold,
        )
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
        image = _read_image_bgr(payload.image_id)
    except FileNotFoundError:
        return SegmentCandidateResponse(ok=False, error="invalid image_id")
    except ValueError:
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
        image = _read_image_bgr(payload.image_id)
    except FileNotFoundError:
        return ExportYoloResponse(ok=False, error="invalid image_id")
    except ValueError:
        return ExportYoloResponse(ok=False, error="failed to read image")

    output_dir = Path(payload.output_dir).expanduser()
    if not output_dir.is_absolute():
        return ExportYoloResponse(ok=False, error="output_dir must be absolute")
    output_dir.mkdir(parents=True, exist_ok=True)

    height, width = image.shape[:2]

    class_names = _get_project_class_names(payload.project)
    if not class_names:
        return ExportYoloResponse(ok=False, error="invalid project")
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    annotations = [ann.model_dump() for ann in payload.annotations]
    lines = make_yolo_lines(annotations, class_to_id, width, height)

    parent_name = "images"
    if payload.project_name and payload.image_key:
        meta_path = _project_meta_path(payload.project_name)
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                parent_name = meta.get("project_name", parent_name)
            except Exception:
                pass
    date_str = datetime.now().strftime("%Y%m%d")
    output_root = output_dir / f"dataset_{parent_name}_{date_str}"
    output_root.mkdir(parents=True, exist_ok=True)

    classes_json_path = output_root / "classes.json"
    classes_json_path.write_text(
        json.dumps(class_to_id, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    classes_txt_path = output_root / "classes.txt"
    classes_txt_path.write_text("\n".join(class_names), encoding="utf-8")
    notes_path = output_root / "notes.json"
    notes_path.write_text(
        json.dumps(
            {"class_names": class_names, "class_to_id": class_to_id},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    output_path = output_root / f"{payload.image_id}.txt"
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

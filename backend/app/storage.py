from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple
from uuid import uuid4

from fastapi import UploadFile
from PIL import Image

from .config import RUNS_DIR


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def get_runs_dir() -> Path:
    return RUNS_DIR


def save_upload(image_file: UploadFile, images_dir: Path) -> Tuple[str, int, int]:
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

    image_id = f"{uuid4().hex}{suffix}"
    images_dir.mkdir(parents=True, exist_ok=True)
    dest = images_dir / image_id

    with dest.open("wb") as f:
        f.write(data)

    return image_id, width, height


def resolve_image_path(images_dir: Path, image_id: str) -> Path:
    path = images_dir / image_id
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(image_id)
    return path

"""Minimal REST API for neural OCR with EasyOCR."""

import io
import os
import time
from functools import lru_cache
from typing import Annotated, Any

import easyocr
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "40000000"))
USE_GPU = os.getenv("EASYOCR_GPU", "false").lower() == "true"


class DetectionResult(BaseModel):
    text: str
    confidence: float = Field(ge=0, le=100)
    bounding_box: list[list[float]]


class OCRResponse(BaseModel):
    engine: str = "easyocr"
    filename: str
    languages: list[str]
    text: str
    confidence: float | None
    detection_count: int
    duration_ms: int
    detections: list[DetectionResult]


app = FastAPI(
    title="EasyOCR API",
    version="1.0.0",
    description="Neural image OCR with EasyOCR, PyTorch, and optional GPU execution.",
)


def parse_languages(value: str) -> tuple[str, ...]:
    languages = tuple(
        dict.fromkeys(item.strip() for item in value.split(",") if item.strip())
    )
    if not languages:
        raise ValueError("At least one EasyOCR language code is required.")
    return languages


@lru_cache(maxsize=4)
def get_reader(languages: tuple[str, ...], gpu: bool) -> easyocr.Reader:
    """Load each language model once and reuse it across requests."""
    return easyocr.Reader(list(languages), gpu=gpu)


def normalize_box(raw_box: Any) -> list[list[float]]:
    return [[round(float(x), 2), round(float(y), 2)] for x, y in raw_box]


def read_image(
    image: Image.Image,
    languages: tuple[str, ...],
) -> tuple[str, float | None, list[DetectionResult]]:
    reader = get_reader(languages, USE_GPU)
    results = reader.readtext(np.asarray(image), detail=1, paragraph=False)

    detections: list[DetectionResult] = []
    confidences: list[float] = []
    for raw_box, raw_text, raw_confidence in results:
        text = str(raw_text).strip()
        if not text:
            continue
        confidence = max(0.0, min(100.0, float(raw_confidence) * 100))
        confidences.append(confidence)
        detections.append(
            DetectionResult(
                text=text,
                confidence=round(confidence, 2),
                bounding_box=normalize_box(raw_box),
            )
        )

    full_text = "\n".join(item.text for item in detections)
    mean_confidence = (
        round(sum(confidences) / len(confidences), 2) if confidences else None
    )
    return full_text, mean_confidence, detections


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {"status": "ok", "engine": "easyocr", "gpu_enabled": USE_GPU}


@app.post("/extract", response_model=OCRResponse)
async def extract(
    file: Annotated[UploadFile, File()],
    languages: Annotated[str, Query(min_length=2, max_length=64)] = "en,pt",
) -> OCRResponse:
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()

    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The uploaded file is too large.")

    try:
        selected_languages = parse_languages(languages)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    started = time.perf_counter()
    try:
        with Image.open(io.BytesIO(content)) as source:
            width, height = source.size
            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413, detail="The image dimensions are too large."
                )
            image = ImageOps.exif_transpose(source).convert("RGB")
        text, confidence, detections = await run_in_threadpool(
            read_image,
            image,
            selected_languages,
        )
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=415, detail="Upload a PNG, JPEG, TIFF, or BMP image."
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503, detail=f"EasyOCR failed to initialize: {exc}"
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"EasyOCR rejected the language selection: {exc}"
        ) from exc

    return OCRResponse(
        filename=file.filename or "image",
        languages=list(selected_languages),
        text=text,
        confidence=confidence,
        detection_count=len(detections),
        duration_ms=round((time.perf_counter() - started) * 1000),
        detections=detections,
    )

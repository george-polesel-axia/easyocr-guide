# EasyOCR API

A minimal REST API for neural text recognition with
[EasyOCR](https://github.com/JaidedAI/EasyOCR), PyTorch, and optional GPU
execution.

This repository reconstructs one of the OCR components used in the former
ProWatsom document-ingestion backend. It contains no customer documents,
credentials, or proprietary business rules.

## What it demonstrates

- Neural OCR for natural-scene and document images.
- More than 80 supported languages.
- Multiple-language recognition, using English and Portuguese by default.
- Detection confidence and polygon bounding boxes.
- Cached model loading instead of rebuilding the reader for every request.
- CPU execution by default and optional GPU execution.
- FastAPI, Docker, automated tests, and GitHub Actions.

## Run with Docker

The image downloads the English and Portuguese models during the build:

```bash
docker build -t easyocr-api .
docker run --rm -p 8000:8000 easyocr-api
```

Open `http://localhost:8000/docs`.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements-dev.txt
uvicorn main:app --reload
```

On the first execution, EasyOCR may download the selected language models.
Installing PyTorch from its CPU index avoids downloading CUDA packages on
machines that do not have a supported GPU.

## Extract text

```bash
curl -X POST "http://localhost:8000/extract?languages=en,pt" \
  -F "file=@sample.png"
```

Example response:

```json
{
  "engine": "easyocr",
  "filename": "sample.png",
  "languages": ["en", "pt"],
  "text": "Recognized text",
  "confidence": 91.73,
  "detection_count": 2,
  "duration_ms": 812,
  "detections": [
    {
      "text": "Recognized",
      "confidence": 93.4,
      "bounding_box": [[10, 20], [140, 20], [140, 50], [10, 50]]
    }
  ]
}
```

## GPU mode

Set the environment variable below only when PyTorch and the host support the
appropriate accelerator:

```bash
EASYOCR_GPU=true uvicorn main:app --reload
```

## Operational note

EasyOCR is heavier than Tesseract because it depends on PyTorch and language
models. The API caches each configured reader, but production systems should
limit the number of accepted language combinations to control memory usage.

## Tests

```bash
pytest -q
```

## License

MIT © 2026 George Hamilton Buzzi Polesel.

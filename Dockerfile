FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EASYOCR_MODULE_PATH=/opt/easyocr-models

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install -r requirements.txt \
    && python -c "import easyocr; easyocr.Reader(['en', 'pt'], gpu=False)"
COPY main.py .

RUN useradd --create-home --uid 10001 apiuser \
    && chown -R apiuser:apiuser /app /opt/easyocr-models
USER apiuser

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

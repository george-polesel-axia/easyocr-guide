import io
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from main import app, parse_languages

client = TestClient(app)


class FakeReader:
    def readtext(self, image, detail, paragraph):
        assert image.shape == (40, 100, 3)
        assert detail == 1
        assert paragraph is False
        return [
            ([[0, 0], [50, 0], [50, 20], [0, 20]], "Hello", 0.95),
            ([[51, 0], [99, 0], [99, 20], [51, 20]], "OCR", 0.85),
        ]


def image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (100, 40), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_parse_languages_removes_duplicates() -> None:
    assert parse_languages("en,pt,en") == ("en", "pt")


@patch("main.get_reader", return_value=FakeReader())
def test_extract(mock_reader) -> None:
    response = client.post(
        "/extract?languages=en,pt",
        files={"file": ("sample.png", image_bytes(), "image/png")},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["engine"] == "easyocr"
    assert result["text"] == "Hello\nOCR"
    assert result["confidence"] == 90.0
    assert result["detection_count"] == 2
    mock_reader.assert_called_once_with(("en", "pt"), False)


def test_rejects_non_image() -> None:
    response = client.post(
        "/extract",
        files={"file": ("sample.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415

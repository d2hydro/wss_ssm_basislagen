from pathlib import Path

import geopandas as gpd
import pytest
import requests
from shapely.geometry import Point

from waterlagen._downloads import (
    DownloadPayloadError,
    GeoServerExceptionError,
    download_geopackage_with_metadata,
    validate_geopackage,
)
from waterlagen.dijkringen.download import (
    DEFAULT_FILENAME,
    DIJKRINGEN_URL,
    download_dijkringen_historie,
)


class FakeResponse:
    def __init__(
        self,
        payload: bytes,
        *,
        headers: dict[str, str] | None = None,
        chunks: list[bytes] | None = None,
        stream_error: Exception | None = None,
        status_error: Exception | None = None,
    ):
        self.payload = payload
        self.headers = headers or {}
        self.chunks = chunks
        self.stream_error = stream_error
        self.status_error = status_error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def raise_for_status(self):
        if self.status_error is not None:
            raise self.status_error

    def iter_content(self, chunk_size: int):
        chunks = self.chunks
        if chunks is None:
            chunks = [
                self.payload[start : start + chunk_size]
                for start in range(0, len(self.payload), chunk_size)
            ]
        for index, chunk in enumerate(chunks):
            if self.stream_error is not None and index == 1:
                raise self.stream_error
            yield chunk


def _valid_gpkg_bytes(path: Path, value: int = 1) -> bytes:
    gpkg_path = path / f"valid_{value}.gpkg"
    gdf = gpd.GeoDataFrame(
        {"id": [value]},
        geometry=[Point(value, value)],
        crs="EPSG:4326",
    )
    gdf.to_file(gpkg_path, driver="GPKG", layer="sample")
    payload = gpkg_path.read_bytes()
    validate_geopackage(gpkg_path)
    return payload


def _temp_downloads_for(target: Path) -> list[Path]:
    return list(target.parent.glob(f".{target.name}.*.gpkg"))


def _mock_download_response(monkeypatch, response: FakeResponse):
    def fake_get(url, **kwargs):
        assert kwargs == {"stream": True, "allow_redirects": True, "timeout": 30}
        return response

    monkeypatch.setattr("waterlagen._downloads.requests.get", fake_get)


def test_download_geopackage_with_known_content_length(monkeypatch, tmp_path):
    payload = _valid_gpkg_bytes(tmp_path)
    target = tmp_path / "known.gpkg"
    _mock_download_response(
        monkeypatch,
        FakeResponse(
            payload,
            headers={
                "Content-Length": str(len(payload)),
                "Content-Type": "application/geopackage+sqlite3",
            },
        ),
    )

    result = download_geopackage_with_metadata("https://example.com/known", target)

    assert result.source_url == "https://example.com/known"
    assert result.target_path == target
    assert result.downloaded_bytes == len(payload)
    assert result.total_size_known is True
    assert result.total_bytes == len(payload)
    assert result.content_type == "application/geopackage+sqlite3"
    validate_geopackage(target)
    assert _temp_downloads_for(target) == []


def test_download_geopackage_without_content_length(monkeypatch, tmp_path):
    payload = _valid_gpkg_bytes(tmp_path)
    target = tmp_path / "unknown.gpkg"
    _mock_download_response(monkeypatch, FakeResponse(payload))

    result = download_geopackage_with_metadata("https://example.com/unknown", target)

    assert result.downloaded_bytes == len(payload)
    assert result.total_size_known is False
    assert result.total_bytes is None
    validate_geopackage(target)


def test_download_geopackage_chunked_streamed_response(monkeypatch, tmp_path):
    payload = _valid_gpkg_bytes(tmp_path)
    target = tmp_path / "chunked.gpkg"
    chunks = [payload[:7], b"", payload[7:100], payload[100:]]
    _mock_download_response(monkeypatch, FakeResponse(payload, chunks=chunks))

    result = download_geopackage_with_metadata("https://example.com/chunked", target)

    assert result.downloaded_bytes == len(payload)
    assert target.read_bytes() == payload


def test_progress_without_total_reports_bytes_only(monkeypatch, tmp_path, capsys):
    payload = _valid_gpkg_bytes(tmp_path)
    target = tmp_path / "progress_unknown.gpkg"
    _mock_download_response(monkeypatch, FakeResponse(payload))

    download_geopackage_with_metadata(
        "https://example.com/progress-unknown",
        target,
        chunk_size=len(payload),
    )

    captured = capsys.readouterr()
    assert "MB" in captured.out
    assert "%" not in captured.out


def test_progress_with_total_reports_percentage(monkeypatch, tmp_path, capsys):
    payload = _valid_gpkg_bytes(tmp_path)
    target = tmp_path / "progress_known.gpkg"
    _mock_download_response(
        monkeypatch,
        FakeResponse(payload, headers={"Content-Length": str(len(payload))}),
    )

    download_geopackage_with_metadata(
        "https://example.com/progress-known",
        target,
        chunk_size=len(payload),
    )

    captured = capsys.readouterr()
    assert "%" in captured.out


def test_invalid_content_length_is_treated_as_unknown(monkeypatch, tmp_path, capsys):
    payload = _valid_gpkg_bytes(tmp_path)
    target = tmp_path / "invalid_length.gpkg"
    _mock_download_response(
        monkeypatch,
        FakeResponse(payload, headers={"Content-Length": "chunked"}),
    )

    result = download_geopackage_with_metadata(
        "https://example.com/invalid-length",
        target,
        chunk_size=len(payload),
    )

    captured = capsys.readouterr()
    assert result.total_size_known is False
    assert "%" not in captured.out


def test_download_dijkringen_historie_success(monkeypatch, tmp_path):
    payload = _valid_gpkg_bytes(tmp_path)
    _mock_download_response(
        monkeypatch,
        FakeResponse(payload, headers={"Content-Length": str(len(payload))}),
    )

    result = download_dijkringen_historie(download_dir=tmp_path, progress=False)

    assert result.source_url == DIJKRINGEN_URL
    assert result.target_path == tmp_path / DEFAULT_FILENAME
    assert result.downloaded_bytes == len(payload)
    assert result.total_size_known is True
    validate_geopackage(result.target_path)


def test_download_dijkringen_historie_allows_target_path_override(
    monkeypatch, tmp_path
):
    payload = _valid_gpkg_bytes(tmp_path)
    target = tmp_path / "custom.gpkg"
    _mock_download_response(monkeypatch, FakeResponse(payload))

    result = download_dijkringen_historie(target_path=target, progress=False)

    assert result.target_path == target
    assert target.exists()


def test_html_error_payload_is_rejected(monkeypatch, tmp_path):
    target = tmp_path / "html.gpkg"
    _mock_download_response(
        monkeypatch,
        FakeResponse(
            b"<html><body>GeoServer error</body></html>",
            headers={"Content-Type": "text/html"},
        ),
    )

    with pytest.raises(DownloadPayloadError, match="not a valid GeoPackage"):
        download_geopackage_with_metadata("https://example.com/html", target)

    assert not target.exists()
    assert _temp_downloads_for(target) == []


def test_ogc_exception_payload_raises_useful_message(monkeypatch, tmp_path):
    target = tmp_path / "exception.gpkg"
    payload = b"""<?xml version="1.0" encoding="UTF-8"?>
<ows:ExceptionReport xmlns:ows="http://www.opengis.net/ows/1.1">
  <ows:Exception exceptionCode="InvalidParameterValue" locator="typeName">
    <ows:ExceptionText>Feature type dijkring_v_2012 is unknown</ows:ExceptionText>
  </ows:Exception>
</ows:ExceptionReport>
"""
    _mock_download_response(
        monkeypatch,
        FakeResponse(payload, headers={"Content-Type": "application/xml"}),
    )

    with pytest.raises(GeoServerExceptionError) as exc_info:
        download_geopackage_with_metadata("https://example.com/exception", target)

    assert "Feature type dijkring_v_2012 is unknown" in str(exc_info.value)
    assert "InvalidParameterValue" in str(exc_info.value)
    assert not target.exists()
    assert _temp_downloads_for(target) == []


def test_http_error_response_preserves_existing_target(monkeypatch, tmp_path):
    target = tmp_path / "http_error.gpkg"
    existing = b"existing target"
    target.write_bytes(existing)
    _mock_download_response(
        monkeypatch,
        FakeResponse(
            b"not found",
            status_error=requests.HTTPError("404 Client Error"),
        ),
    )

    with pytest.raises(requests.HTTPError, match="404"):
        download_geopackage_with_metadata("https://example.com/http-error", target)

    assert target.read_bytes() == existing
    assert _temp_downloads_for(target) == []


def test_interrupted_stream_preserves_existing_target(monkeypatch, tmp_path):
    payload = _valid_gpkg_bytes(tmp_path)
    target = tmp_path / "interrupted.gpkg"
    existing = b"existing target"
    target.write_bytes(existing)
    _mock_download_response(
        monkeypatch,
        FakeResponse(
            payload,
            chunks=[payload[:10], payload[10:]],
            stream_error=ConnectionError("interrupted"),
        ),
    )

    with pytest.raises(ConnectionError, match="interrupted"):
        download_geopackage_with_metadata("https://example.com/interrupted", target)

    assert target.read_bytes() == existing
    assert _temp_downloads_for(target) == []


def test_invalid_geopackage_preserves_existing_target(monkeypatch, tmp_path):
    target = tmp_path / "invalid.gpkg"
    existing = b"existing target"
    target.write_bytes(existing)
    _mock_download_response(monkeypatch, FakeResponse(b"not a geopackage"))

    with pytest.raises(DownloadPayloadError, match="not a valid GeoPackage"):
        download_geopackage_with_metadata("https://example.com/invalid", target)

    assert target.read_bytes() == existing
    assert _temp_downloads_for(target) == []


def test_atomic_replacement_after_success(monkeypatch, tmp_path):
    payload = _valid_gpkg_bytes(tmp_path, value=2)
    target = tmp_path / "replace.gpkg"
    target.write_bytes(b"old target")
    _mock_download_response(monkeypatch, FakeResponse(payload))

    download_geopackage_with_metadata("https://example.com/replace", target)

    assert target.read_bytes() == payload
    assert _temp_downloads_for(target) == []

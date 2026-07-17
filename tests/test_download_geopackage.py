from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from waterlagen._downloads import download_geopackage, validate_geopackage
from waterlagen.bag.download import download_bag_light
from waterlagen.top10nl.download import download_top10nl


class FakeResponse:
    def __init__(self, payload: bytes, *, error: Exception | None = None):
        self.payload = payload
        self.error = error
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size: int):
        midpoint = max(1, len(self.payload) // 2)
        yield self.payload[:midpoint]
        if self.error is not None:
            raise self.error
        yield self.payload[midpoint:]


def _valid_gpkg_bytes(path: Path) -> bytes:
    gpkg_path = path / "valid.gpkg"
    gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    )
    gdf.to_file(gpkg_path, driver="GPKG", layer="sample")
    payload = gpkg_path.read_bytes()
    validate_geopackage(gpkg_path)
    return payload


def _temp_downloads_for(target: Path) -> list[Path]:
    return list(target.parent.glob(f".{target.name}.*.gpkg"))


def test_download_geopackage_success(monkeypatch, tmp_path):
    payload = _valid_gpkg_bytes(tmp_path)
    target = tmp_path / "download.gpkg"
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse(payload)

    monkeypatch.setattr("waterlagen._downloads.requests.get", fake_get)

    result = download_geopackage("https://example.com/download.gpkg", target)

    assert result == target
    assert target.exists()
    validate_geopackage(target)
    assert target.read_bytes() == payload
    assert _temp_downloads_for(target) == []
    assert calls == [
        (
            "https://example.com/download.gpkg",
            {"stream": True, "allow_redirects": True, "timeout": 30},
        )
    ]


def test_download_geopackage_rejects_invalid_gpkg(monkeypatch, tmp_path):
    old_payload = _valid_gpkg_bytes(tmp_path)
    target = tmp_path / "download.gpkg"
    target.write_bytes(old_payload)

    def fake_get(url, **kwargs):
        return FakeResponse(b"<html>not a geopackage</html>")

    monkeypatch.setattr("waterlagen._downloads.requests.get", fake_get)

    with pytest.raises(ValueError, match="not a valid GeoPackage"):
        download_geopackage("https://example.com/download.gpkg", target)

    assert target.read_bytes() == old_payload
    assert _temp_downloads_for(target) == []


def test_download_geopackage_interrupted_download_cleans_temp(monkeypatch, tmp_path):
    payload = _valid_gpkg_bytes(tmp_path)
    target = tmp_path / "download.gpkg"

    def fake_get(url, **kwargs):
        return FakeResponse(payload, error=ConnectionError("interrupted"))

    monkeypatch.setattr("waterlagen._downloads.requests.get", fake_get)

    with pytest.raises(ConnectionError, match="interrupted"):
        download_geopackage("https://example.com/download.gpkg", target)

    assert not target.exists()
    assert _temp_downloads_for(target) == []


def test_download_geopackage_existing_target_without_overwrite(monkeypatch, tmp_path):
    payload = _valid_gpkg_bytes(tmp_path)
    target = tmp_path / "download.gpkg"
    target.write_bytes(payload)

    def fake_get(url, **kwargs):
        raise AssertionError("download should not be requested")

    monkeypatch.setattr("waterlagen._downloads.requests.get", fake_get)

    result = download_geopackage(
        "https://example.com/download.gpkg",
        target,
        overwrite=False,
    )

    assert result == target
    assert target.read_bytes() == payload
    assert _temp_downloads_for(target) == []


def test_download_bag_light_uses_shared_downloader(monkeypatch, tmp_path):
    calls = []

    def fake_download_geopackage(*, url, target_path, overwrite, logger, expected_crs):
        calls.append((url, target_path, overwrite, logger, expected_crs))
        return target_path

    monkeypatch.setattr(
        "waterlagen.bag.download.download_geopackage",
        fake_download_geopackage,
    )

    result = download_bag_light(download_dir=tmp_path, overwrite=False)

    assert result == tmp_path / "bag-light.gpkg"
    assert (
        calls[0][0]
        == "https://service.pdok.nl/lv/bag/atom/downloads/bag-light.gpkg"
    )
    assert calls[0][1] == tmp_path / "bag-light.gpkg"
    assert calls[0][2] is False
    assert calls[0][4] == "EPSG:28992"


def test_download_top10nl_uses_shared_downloader(monkeypatch, tmp_path):
    calls = []

    def fake_download_geopackage(*, url, target_path, overwrite, logger):
        calls.append((url, target_path, overwrite, logger))
        return target_path

    monkeypatch.setattr(
        "waterlagen.top10nl.download.download_geopackage",
        fake_download_geopackage,
    )

    result = download_top10nl(download_dir=tmp_path, overwrite=False)

    assert result == tmp_path / "top10nl_Compleet.gpkg"
    assert (
        calls[0][0]
        == "https://service.pdok.nl/kadaster/brt-topnl/atom/downloads/top10nl_Compleet.gpkg"
    )
    assert calls[0][1] == tmp_path / "top10nl_Compleet.gpkg"
    assert calls[0][2] is False

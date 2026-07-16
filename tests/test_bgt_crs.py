import io
import zipfile

import geopandas as gpd
import pyogrio
import pytest
from shapely.geometry import Point

from waterlagen._crs import MissingCRSError
from waterlagen.bgt.download import download_to_geopackage


class FakeResponse:
    def __init__(self, payload: bytes):
        self.content = payload

    def raise_for_status(self):
        return None


def _zip_with_gml() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("bgt_sample.gml", "<xml />")
    return buffer.getvalue()


def test_bgt_download_reprojects_written_geopackage(monkeypatch, tmp_path):
    def fake_get(url):
        return FakeResponse(_zip_with_gml())

    def fake_read_file(path):
        return gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[Point(5, 52)],
            crs="EPSG:4326",
        )

    monkeypatch.setattr("waterlagen.bgt.download.requests.get", fake_get)
    monkeypatch.setattr("waterlagen.bgt.download.gpd.read_file", fake_read_file)

    download_to_geopackage("https://example.com/bgt.zip", tmp_path)

    gpkg = tmp_path / "bgt_sample.gpkg"
    assert gpkg.exists()
    assert pyogrio.read_info(gpkg, layer="bgt_sample")["crs"] == "EPSG:28992"


def test_bgt_download_missing_crs_raises_without_replacing_existing(
    monkeypatch,
    tmp_path,
):
    existing = tmp_path / "bgt_sample.gpkg"
    gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[Point(100000, 450000)],
        crs="EPSG:28992",
    ).to_file(existing, driver="GPKG", layer="bgt_sample")
    before = existing.read_bytes()

    def fake_get(url):
        return FakeResponse(_zip_with_gml())

    def fake_read_file(path):
        return gpd.GeoDataFrame({"id": [1]}, geometry=[Point(5, 52)], crs=None)

    monkeypatch.setattr("waterlagen.bgt.download.requests.get", fake_get)
    monkeypatch.setattr("waterlagen.bgt.download.gpd.read_file", fake_read_file)

    with pytest.raises(MissingCRSError, match="have no CRS"):
        download_to_geopackage("https://example.com/bgt.zip", tmp_path)

    assert existing.read_bytes() == before
    assert pyogrio.read_info(existing, layer="bgt_sample")["crs"] == "EPSG:28992"

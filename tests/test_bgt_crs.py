import io
import zipfile

import geopandas as gpd
import pyogrio
import pytest
from shapely.geometry import Point

from waterlagen._crs import MissingCRSError
from waterlagen.bgt.download import (
    bgt_custom_download,
    download_to_geopackage,
    get_bgt_features,
)


class FakeResponse:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    @property
    def content(self):
        raise AssertionError("download_to_geopackage should stream the ZIP to disk")

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size: int):
        midpoint = max(1, len(self.payload) // 2)
        yield self.payload[:midpoint]
        yield self.payload[midpoint:]


def _zip_with_gml() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("bgt_sample.gml", "<xml />")
    return buffer.getvalue()


def test_bgt_download_reprojects_written_geopackage(monkeypatch, tmp_path):
    def fake_get(url, **kwargs):
        assert kwargs == {"stream": True, "allow_redirects": True, "timeout": 30}
        return FakeResponse(_zip_with_gml())

    def fail_read_file(*args, **kwargs):
        raise AssertionError("GML conversion should not load a GeoDataFrame")

    def fake_translate(gml_path, target_path, *, layer_name, expected_crs, **kwargs):
        gdf = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[Point(5, 52)],
            crs="EPSG:4326",
        ).to_crs(expected_crs)
        gdf.to_file(target_path, driver="GPKG", layer=layer_name)

    monkeypatch.setattr("waterlagen._downloads.requests.get", fake_get)
    monkeypatch.setattr("geopandas.read_file", fail_read_file)
    monkeypatch.setattr(
        "waterlagen.bgt.download._translate_gml_layer_to_geopackage",
        fake_translate,
    )

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

    def fake_get(url, **kwargs):
        assert kwargs == {"stream": True, "allow_redirects": True, "timeout": 30}
        return FakeResponse(_zip_with_gml())

    def fake_translate(*args, **kwargs):
        raise MissingCRSError(
            "Dataset layer(s) have no CRS: bgt_sample. "
            "Cannot reproject to EPSG:28992 without a source CRS."
        )

    monkeypatch.setattr("waterlagen._downloads.requests.get", fake_get)
    monkeypatch.setattr(
        "waterlagen.bgt.download._translate_gml_layer_to_geopackage",
        fake_translate,
    )

    with pytest.raises(MissingCRSError, match="have no CRS"):
        download_to_geopackage("https://example.com/bgt.zip", tmp_path)

    assert existing.read_bytes() == before
    assert pyogrio.read_info(existing, layer="bgt_sample")["crs"] == "EPSG:28992"


def test_bgt_custom_download_orchestrates_request_poll_and_download(
    monkeypatch, tmp_path
):
    calls = []

    def fake_request_download(*, featuretypes, poly_mask):
        calls.append(("request", featuretypes, poly_mask))
        return "request-1"

    def fake_poll_downloadstatus(*, download_request_id, poll_interval_s):
        calls.append(("poll", download_request_id, poll_interval_s))
        return "https://example.com/bgt.zip"

    def fake_download_to_geopackage(*, download_url, download_dir):
        calls.append(("download", download_url, download_dir))
        return download_dir

    monkeypatch.setattr(
        "waterlagen.bgt.download.request_download",
        fake_request_download,
    )
    monkeypatch.setattr(
        "waterlagen.bgt.download.poll_downloadstatus",
        fake_poll_downloadstatus,
    )
    monkeypatch.setattr(
        "waterlagen.bgt.download.download_to_geopackage",
        fake_download_to_geopackage,
    )

    result = bgt_custom_download(
        featuretypes=["waterdeel", "pand"],
        download_dir=tmp_path,
        poll_interval_s=1,
    )

    assert result.download_dir == tmp_path
    assert result.featuretypes == ("waterdeel", "pand")
    assert result.request_id == "request-1"
    assert result.download_url == "https://example.com/bgt.zip"
    assert calls == [
        ("request", ("waterdeel", "pand"), None),
        ("poll", "request-1", 1),
        ("download", "https://example.com/bgt.zip", tmp_path),
    ]


def test_bgt_custom_download_without_overwrite_skips_existing_outputs(
    monkeypatch, tmp_path
):
    (tmp_path / "bgt_waterdeel.gpkg").write_bytes(b"existing")
    (tmp_path / "bgt_pand.gpkg").write_bytes(b"existing")

    def fail_request_download(*args, **kwargs):
        raise AssertionError("download should not be requested")

    monkeypatch.setattr(
        "waterlagen.bgt.download.request_download",
        fail_request_download,
    )

    result = bgt_custom_download(
        featuretypes=["waterdeel", "pand"],
        download_dir=tmp_path,
        overwrite=False,
    )

    assert result.download_dir == tmp_path
    assert result.featuretypes == ("waterdeel", "pand")
    assert result.request_id is None
    assert result.download_url is None


def test_bgt_custom_download_uses_default_featuretypes(monkeypatch, tmp_path):
    calls = []

    def fake_request_download(*, featuretypes, poly_mask):
        calls.append(("request", featuretypes, poly_mask))
        return "request-1"

    monkeypatch.setattr(
        "waterlagen.bgt.download.request_download",
        fake_request_download,
    )
    monkeypatch.setattr(
        "waterlagen.bgt.download.poll_downloadstatus",
        lambda **kwargs: "https://example.com/bgt.zip",
    )
    monkeypatch.setattr(
        "waterlagen.bgt.download.download_to_geopackage",
        lambda **kwargs: tmp_path,
    )

    result = bgt_custom_download(download_dir=tmp_path)

    assert result.featuretypes == ("waterdeel", "pand")
    assert calls == [("request", ("waterdeel", "pand"), None)]


def test_get_bgt_features_keeps_backward_compatible_return(monkeypatch, tmp_path):
    calls = []

    def fake_bgt_custom_download(*, featuretypes, poly_mask, download_dir):
        calls.append((featuretypes, poly_mask, download_dir))
        return type(
            "Download",
            (),
            {"download_dir": download_dir},
        )()

    monkeypatch.setattr(
        "waterlagen.bgt.download.bgt_custom_download",
        fake_bgt_custom_download,
    )

    result = get_bgt_features(
        featuretypes=["waterdeel"],
        poly_mask=None,
        download_dir=tmp_path,
    )

    assert result == tmp_path
    assert calls == [(["waterdeel"], None, tmp_path)]

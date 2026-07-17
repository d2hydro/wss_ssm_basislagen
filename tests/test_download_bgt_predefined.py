import io
import zipfile
from pathlib import Path

import geopandas as gpd
import pyogrio
import pytest
from shapely.geometry import Point

import waterlagen.bgt.download as bgt_download_module
from waterlagen._crs import format_crs, read_layer_crs_info, same_crs
from waterlagen._downloads import DownloadPayloadError, validate_geopackage
from waterlagen.bgt.download import (
    BGT_PREDEFINED_URL,
    _relax_gfs_template,
    _translate_gml_layer_to_geopackage,
    bgt_download,
    download_bgt,
)


class FakeStreamResponse:
    def __init__(
        self,
        payload: bytes,
        *,
        headers: dict[str, str] | None = None,
        stream_error: Exception | None = None,
    ):
        self.payload = payload
        self.headers = headers or {}
        self.stream_error = stream_error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size: int):
        midpoint = max(1, len(self.payload) // 2)
        yield self.payload[:midpoint]
        if self.stream_error is not None:
            raise self.stream_error
        yield self.payload[midpoint:]


def _zip_with_gml(*names: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name in names:
            zf.writestr(name, "<gml />")
    return buffer.getvalue()


def _mock_download(monkeypatch, response: FakeStreamResponse):
    def fake_get(url, **kwargs):
        assert url == BGT_PREDEFINED_URL
        assert kwargs == {"stream": True, "allow_redirects": True, "timeout": 30}
        return response

    monkeypatch.setattr("waterlagen._downloads.requests.get", fake_get)


def _mock_gml_reader(monkeypatch, crs_by_layer: dict[str, str | None]):
    def fake_translate(gml_path, target_path, *, layer_name, expected_crs, append, **kwargs):
        layer = Path(gml_path).stem
        crs = crs_by_layer[layer]
        if crs == "EPSG:4326":
            point = Point(5, 52)
        elif crs == "EPSG:3857":
            point = Point(556597, 6800125)
        else:
            point = Point(100000, 450000)
        if crs is None:
            bgt_download_module.logger.warning(
                f"Layer {layer_name} has no CRS. Assigning {format_crs(expected_crs)}."
            )
            gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[point], crs=expected_crs)
        else:
            if not same_crs(crs, expected_crs):
                bgt_download_module.logger.warning(
                    f"Layer {layer_name} CRS is {format_crs(crs)}, "
                    f"expected {format_crs(expected_crs)}. Reprojecting layer."
                )
            gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[point], crs=crs).to_crs(
                expected_crs
            )
        pyogrio.write_dataframe(
            gdf,
            target_path,
            layer=layer_name,
            driver="GPKG",
            append=append,
        )

    monkeypatch.setattr(
        "waterlagen.bgt.download._translate_gml_layer_to_geopackage",
        fake_translate,
    )


def _valid_gpkg(path: Path, *, layer: str = "old") -> bytes:
    gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[Point(100000, 450000)],
        crs="EPSG:28992",
    ).to_file(path, driver="GPKG", layer=layer)
    validate_geopackage(path)
    return path.read_bytes()


def _temp_files_for(target: Path) -> list[Path]:
    return list(target.parent.glob(f".{target.name}.*"))


def _spatial_crs_values(path: Path) -> dict[str, str | None]:
    return {
        info.layer: info.crs for info in read_layer_crs_info(path) if info.is_spatial
    }


def test_bgt_download_alias_points_to_download_bgt(monkeypatch):
    calls = []

    def fake_download_bgt(*args, **kwargs):
        calls.append((args, kwargs))
        return "download"

    monkeypatch.setattr(
        "waterlagen.bgt.download.download_bgt",
        fake_download_bgt,
    )

    assert bgt_download("dir", overwrite=False) == "download"
    assert calls == [(("dir",), {"overwrite": False})]


def test_gdal_translation_does_not_read_gml_into_geodataframe(monkeypatch, tmp_path):
    calls = []
    gml_path = tmp_path / "bgt_waterdeel.gml"
    gml_path.write_text("<gml />")

    def fail_read_file(*args, **kwargs):
        raise AssertionError("GML conversion should not load a GeoDataFrame")

    def fake_vector_translate(dest, src, **kwargs):
        calls.append((dest, src, kwargs))
        return object()

    monkeypatch.setattr("geopandas.read_file", fail_read_file)
    monkeypatch.setattr(
        "waterlagen.bgt.download.pyogrio.read_info",
        lambda path: {"crs": "EPSG:28992"},
    )
    monkeypatch.setattr(
        "waterlagen.bgt.download.gdal.VectorTranslate",
        fake_vector_translate,
    )

    _translate_gml_layer_to_geopackage(
        gml_path,
        tmp_path / "bgt.gpkg",
        layer_name="bgt_waterdeel",
        expected_crs="EPSG:28992",
        append=False,
    )

    assert len(calls) == 1
    assert calls[0][0] == str(tmp_path / "bgt.gpkg")
    assert calls[0][1] == str(gml_path)


def test_gdal_translation_formats_explicit_source_crs(monkeypatch, tmp_path):
    gml_path = tmp_path / "bgt_waterdeel.gml"
    gml_path.write_text("<gml />")
    calls = []

    def fail_read_info(*args, **kwargs):
        raise AssertionError("source_crs should skip GML CRS probing")

    def fake_options(**kwargs):
        return kwargs

    def fake_vector_translate(dest, src, **kwargs):
        calls.append((dest, src, kwargs))
        return object()

    monkeypatch.setattr(
        "waterlagen.bgt.download.pyogrio.read_info",
        fail_read_info,
    )
    monkeypatch.setattr(
        "waterlagen.bgt.download.gdal.VectorTranslateOptions",
        fake_options,
    )
    monkeypatch.setattr(
        "waterlagen.bgt.download.gdal.VectorTranslate",
        fake_vector_translate,
    )

    _translate_gml_layer_to_geopackage(
        gml_path,
        tmp_path / "bgt.gpkg",
        layer_name="bgt_waterdeel",
        expected_crs=28992,
        append=False,
        source_crs=28992,
    )

    assert calls[0][2]["options"]["srcSRS"] == "EPSG:28992"
    assert calls[0][2]["options"]["dstSRS"] == "EPSG:28992"


def test_gfs_template_is_relaxed_for_full_bgt_file():
    gfs = """<GMLFeatureClassList>
  <GMLFeatureClass>
    <Name>Waterdeel</Name>
    <DatasetSpecificInfo><FeatureCount>1</FeatureCount></DatasetSpecificInfo>
    <PropertyDefn>
      <Name>bgt-type</Name>
      <Type>String</Type>
      <Width>9</Width>
    </PropertyDefn>
  </GMLFeatureClass>
</GMLFeatureClassList>"""

    relaxed = _relax_gfs_template(gfs)

    assert "DatasetSpecificInfo" not in relaxed
    assert "<Width>0</Width>" in relaxed


def test_successful_predefined_bgt_download(monkeypatch, tmp_path):
    payload = _zip_with_gml("bgt_waterdeel.gml")
    _mock_download(
        monkeypatch,
        FakeStreamResponse(
            payload,
            headers={
                "Content-Length": str(len(payload)),
                "Content-Type": "application/zip",
            },
        ),
    )
    _mock_gml_reader(monkeypatch, {"bgt_waterdeel": "EPSG:28992"})

    result = download_bgt(download_dir=tmp_path, progress=False)

    assert result.source_url == BGT_PREDEFINED_URL
    assert result.target_path == tmp_path / "bgt.gpkg"
    assert result.downloaded_bytes == len(payload)
    assert result.total_size_known is True
    assert result.layer_count == 1
    assert result.final_crs == "EPSG:28992"
    validate_geopackage(result.target_path)
    assert _temp_files_for(result.target_path) == []


def test_predefined_download_with_content_length_reports_progress(
    monkeypatch, tmp_path, capsys
):
    payload = _zip_with_gml("bgt_waterdeel.gml")
    _mock_download(
        monkeypatch,
        FakeStreamResponse(payload, headers={"Content-Length": str(len(payload))}),
    )
    _mock_gml_reader(monkeypatch, {"bgt_waterdeel": "EPSG:28992"})

    result = download_bgt(download_dir=tmp_path)

    captured = capsys.readouterr()
    assert result.total_size_known is True
    assert "%" in captured.out


def test_predefined_download_without_content_length_reports_bytes_only(
    monkeypatch, tmp_path, capsys
):
    payload = _zip_with_gml("bgt_waterdeel.gml")
    _mock_download(monkeypatch, FakeStreamResponse(payload))
    _mock_gml_reader(monkeypatch, {"bgt_waterdeel": "EPSG:28992"})

    result = download_bgt(download_dir=tmp_path)

    captured = capsys.readouterr()
    assert result.total_size_known is False
    assert "MB" in captured.out
    assert "%" not in captured.out


def test_invalid_zip_is_rejected(monkeypatch, tmp_path):
    target = tmp_path / "bgt.gpkg"
    existing = _valid_gpkg(target)
    _mock_download(monkeypatch, FakeStreamResponse(b"not a zip"))

    with pytest.raises(DownloadPayloadError, match="valid ZIP"):
        download_bgt(download_dir=tmp_path, progress=False)

    assert target.read_bytes() == existing
    assert _temp_files_for(target) == []


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"<html><body>error</body></html>", "HTML error"),
        (b"<?xml version='1.0'?><error>bad</error>", "XML/HTML error|XML error"),
        (b'{"error": "bad"}', "JSON error"),
    ],
)
def test_error_payloads_are_rejected(monkeypatch, tmp_path, payload, message):
    target = tmp_path / "bgt.gpkg"
    existing = _valid_gpkg(target)
    _mock_download(monkeypatch, FakeStreamResponse(payload))

    with pytest.raises(DownloadPayloadError, match=message):
        download_bgt(download_dir=tmp_path, progress=False)

    assert target.read_bytes() == existing
    assert _temp_files_for(target) == []


def test_interrupted_download_cleans_temp_and_preserves_existing(monkeypatch, tmp_path):
    target = tmp_path / "bgt.gpkg"
    existing = _valid_gpkg(target)
    payload = _zip_with_gml("bgt_waterdeel.gml")
    _mock_download(
        monkeypatch,
        FakeStreamResponse(payload, stream_error=ConnectionError("interrupted")),
    )

    with pytest.raises(ConnectionError, match="interrupted"):
        download_bgt(download_dir=tmp_path, progress=False)

    assert target.read_bytes() == existing
    assert _temp_files_for(target) == []


def test_multiple_gml_layers_and_layer_names_preserved(monkeypatch, tmp_path):
    payload = _zip_with_gml("bgt_waterdeel.gml", "nested/bgt_wegdeel.gml")
    _mock_download(monkeypatch, FakeStreamResponse(payload))
    _mock_gml_reader(
        monkeypatch,
        {"bgt_waterdeel": "EPSG:28992", "bgt_wegdeel": "EPSG:28992"},
    )

    result = download_bgt(
        download_dir=tmp_path,
        featuretypes=["waterdeel", "wegdeel"],
        progress=False,
    )

    assert result.layer_count == 2
    assert set(pyogrio.list_layers(result.target_path)[:, 0]) == {
        "bgt_waterdeel",
        "bgt_wegdeel",
    }


def test_predefined_download_filters_requested_feature_types(monkeypatch, tmp_path):
    payload = _zip_with_gml(
        "bgt_waterdeel.gml",
        "bgt_pand.gml",
        "bgt_wegdeel.gml",
    )
    _mock_download(monkeypatch, FakeStreamResponse(payload))
    _mock_gml_reader(
        monkeypatch,
        {
            "bgt_waterdeel": "EPSG:28992",
            "bgt_pand": "EPSG:28992",
            "bgt_wegdeel": "EPSG:28992",
        },
    )

    result = download_bgt(
        download_dir=tmp_path,
        featuretypes=["waterdeel", "pand"],
        progress=False,
    )

    assert result.layer_count == 2
    assert set(pyogrio.list_layers(result.target_path)[:, 0]) == {
        "bgt_waterdeel",
        "bgt_pand",
    }


def test_predefined_download_warns_about_missing_feature_types(
    monkeypatch,
    tmp_path,
    caplog,
):
    payload = _zip_with_gml("bgt_waterdeel.gml")
    _mock_download(monkeypatch, FakeStreamResponse(payload))
    _mock_gml_reader(monkeypatch, {"bgt_waterdeel": "EPSG:28992"})

    result = download_bgt(
        download_dir=tmp_path,
        featuretypes=["waterdeel", "pand"],
        progress=False,
    )

    assert result.layer_count == 1
    assert "pand (expected bgt_pand.gml)" in caplog.text


def test_layer_already_in_expected_crs(monkeypatch, tmp_path, caplog):
    payload = _zip_with_gml("bgt_waterdeel.gml")
    _mock_download(monkeypatch, FakeStreamResponse(payload))
    _mock_gml_reader(monkeypatch, {"bgt_waterdeel": "EPSG:28992"})

    result = download_bgt(download_dir=tmp_path, progress=False)

    assert _spatial_crs_values(result.target_path) == {"bgt_waterdeel": "EPSG:28992"}
    assert "Reprojecting layer" not in caplog.text


def test_layer_in_another_crs_is_reprojected(monkeypatch, tmp_path, caplog):
    payload = _zip_with_gml("bgt_waterdeel.gml")
    _mock_download(monkeypatch, FakeStreamResponse(payload))
    _mock_gml_reader(monkeypatch, {"bgt_waterdeel": "EPSG:4326"})

    result = download_bgt(download_dir=tmp_path, progress=False)

    assert _spatial_crs_values(result.target_path) == {"bgt_waterdeel": "EPSG:28992"}
    assert (
        "Layer bgt_waterdeel CRS is EPSG:4326, expected EPSG:28992. Reprojecting layer."
    ) in caplog.text


def test_missing_crs_assigns_expected_crs_with_warning(monkeypatch, tmp_path, caplog):
    payload = _zip_with_gml("bgt_waterdeel.gml")
    _mock_download(monkeypatch, FakeStreamResponse(payload))
    _mock_gml_reader(monkeypatch, {"bgt_waterdeel": None})

    result = download_bgt(download_dir=tmp_path, progress=False)

    assert _spatial_crs_values(result.target_path) == {"bgt_waterdeel": "EPSG:28992"}
    assert "Layer bgt_waterdeel has no CRS. Assigning EPSG:28992." in caplog.text


def test_final_geopackage_valid_and_all_layers_expected_crs(monkeypatch, tmp_path):
    payload = _zip_with_gml("bgt_waterdeel.gml", "bgt_wegdeel.gml")
    _mock_download(monkeypatch, FakeStreamResponse(payload))
    _mock_gml_reader(
        monkeypatch,
        {"bgt_waterdeel": "EPSG:4326", "bgt_wegdeel": "EPSG:3857"},
    )

    result = download_bgt(
        download_dir=tmp_path,
        featuretypes=["waterdeel", "wegdeel"],
        progress=False,
    )

    validate_geopackage(result.target_path)
    assert set(_spatial_crs_values(result.target_path).values()) == {"EPSG:28992"}


def test_atomic_replacement_after_success(monkeypatch, tmp_path):
    target = tmp_path / "custom.gpkg"
    _valid_gpkg(target, layer="old")
    payload = _zip_with_gml("bgt_waterdeel.gml")
    _mock_download(monkeypatch, FakeStreamResponse(payload))
    _mock_gml_reader(monkeypatch, {"bgt_waterdeel": "EPSG:28992"})

    result = download_bgt(target_path=target, progress=False)

    assert result.target_path == target
    assert set(pyogrio.list_layers(target)[:, 0]) == {"bgt_waterdeel"}
    assert _temp_files_for(target) == []

from pathlib import Path
import logging

import geopandas as gpd
import pandas as pd
import pyogrio
import pytest
from shapely.geometry import Point

from waterlagen._crs import MissingCRSError, ensure_dataset_crs, read_layer_crs_info
from waterlagen._downloads import validate_geopackage


EXPECTED_CRS = "EPSG:28992"


def _write_spatial_layer(
    path: Path,
    *,
    layer: str,
    crs: str | None,
    x: float = 0,
    y: float = 0,
) -> None:
    gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[Point(x, y)],
        crs=crs,
    )
    pyogrio.write_dataframe(
        gdf,
        path,
        layer=layer,
        driver="GPKG",
        append=path.exists(),
    )


def _write_table(path: Path, *, layer: str = "metadata") -> None:
    df = pd.DataFrame({"id": [1], "name": ["preserved"]})
    pyogrio.write_dataframe(
        df,
        path,
        layer=layer,
        driver="GPKG",
        append=path.exists(),
    )


def _layer_crs(path: Path, layer: str) -> str | None:
    return pyogrio.read_info(path, layer=layer)["crs"]


def _spatial_crs_values(path: Path) -> dict[str, str | None]:
    return {
        info.layer: info.crs
        for info in read_layer_crs_info(path)
        if info.is_spatial
    }


def test_dataset_already_in_expected_crs_is_unchanged(tmp_path):
    gpkg = tmp_path / "ok.gpkg"
    _write_spatial_layer(gpkg, layer="sample", crs=EXPECTED_CRS, x=100000, y=450000)
    before = gpkg.read_bytes()

    ensure_dataset_crs(gpkg, expected_crs=EXPECTED_CRS)

    assert gpkg.read_bytes() == before
    assert _layer_crs(gpkg, "sample") == EXPECTED_CRS


def test_dataset_in_another_crs_is_reprojected(tmp_path, caplog):
    gpkg = tmp_path / "wrong.gpkg"
    _write_spatial_layer(gpkg, layer="sample", crs="EPSG:4326", x=5, y=52)

    logger = logging.getLogger("waterlagen.tests.crs")
    ensure_dataset_crs(gpkg, expected_crs=EXPECTED_CRS, logger=logger)

    assert _layer_crs(gpkg, "sample") == EXPECTED_CRS
    assert (
        "Dataset CRS is EPSG:4326, expected EPSG:28992. Reprojecting dataset."
        in caplog.text
    )


def test_missing_crs_raises_clear_error(tmp_path):
    gpkg = tmp_path / "missing.gpkg"
    _write_spatial_layer(gpkg, layer="sample", crs=None)

    with pytest.raises(MissingCRSError, match="have no CRS"):
        ensure_dataset_crs(gpkg, expected_crs=EXPECTED_CRS)

    assert _layer_crs(gpkg, "sample") is None


def test_multiple_geopackage_layers_are_checked_and_reprojected(tmp_path):
    gpkg = tmp_path / "multi.gpkg"
    _write_spatial_layer(gpkg, layer="wgs84", crs="EPSG:4326", x=5, y=52)
    _write_spatial_layer(gpkg, layer="rd", crs=EXPECTED_CRS, x=100000, y=450000)
    _write_table(gpkg)

    ensure_dataset_crs(gpkg, expected_crs=EXPECTED_CRS)

    assert _spatial_crs_values(gpkg) == {
        "wgs84": EXPECTED_CRS,
        "rd": EXPECTED_CRS,
    }
    assert pyogrio.read_dataframe(gpkg, layer="metadata", read_geometry=False).loc[
        0, "name"
    ] == "preserved"


def test_conversion_failure_preserves_existing_file(tmp_path, monkeypatch):
    gpkg = tmp_path / "preserved.gpkg"
    _write_spatial_layer(gpkg, layer="sample", crs="EPSG:4326", x=5, y=52)
    before = gpkg.read_bytes()

    def fail_write(*args, **kwargs):
        raise RuntimeError("conversion failed")

    monkeypatch.setattr("waterlagen._crs.pyogrio.write_dataframe", fail_write)

    with pytest.raises(RuntimeError, match="conversion failed"):
        ensure_dataset_crs(gpkg, expected_crs=EXPECTED_CRS)

    assert gpkg.read_bytes() == before
    assert _layer_crs(gpkg, "sample") == "EPSG:4326"


def test_converted_file_is_valid_geopackage(tmp_path):
    gpkg = tmp_path / "valid_after_conversion.gpkg"
    _write_spatial_layer(gpkg, layer="sample", crs="EPSG:4326", x=5, y=52)

    ensure_dataset_crs(gpkg, expected_crs=EXPECTED_CRS)

    validate_geopackage(gpkg)


def test_all_spatial_layers_end_in_expected_crs(tmp_path):
    gpkg = tmp_path / "all_layers.gpkg"
    _write_spatial_layer(gpkg, layer="a", crs="EPSG:4326", x=5, y=52)
    _write_spatial_layer(gpkg, layer="b", crs="EPSG:3857", x=556597, y=6800125)

    ensure_dataset_crs(gpkg, expected_crs=EXPECTED_CRS)

    assert set(_spatial_crs_values(gpkg).values()) == {EXPECTED_CRS}

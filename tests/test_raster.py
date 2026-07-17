from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio

import waterlagen.raster.vrt as vrt_mod


def test_create_vrt_file_builds_from_tifs(tmp_path, monkeypatch):
    tif_a = tmp_path / "a.tif"
    tif_b = tmp_path / "b.tif"
    tif_a.write_text("a")
    tif_b.write_text("b")
    vrt = tmp_path / "combined.vrt"

    calls = {}
    flushed = {"value": False}

    class DummyDataset:
        def FlushCache(self):
            flushed["value"] = True

    def _fake_options(**kwargs):
        calls["options_kwargs"] = kwargs
        return {"options": kwargs}

    def _fake_build_vrt(destName, srcDSOrSrcDSTab, options):
        calls["destName"] = destName
        calls["srcDSOrSrcDSTab"] = srcDSOrSrcDSTab
        calls["options"] = options
        return DummyDataset()

    monkeypatch.setattr(vrt_mod, "gdal", SimpleNamespace(
        BuildVRTOptions=_fake_options,
        BuildVRT=_fake_build_vrt,
    ))

    out = vrt_mod.create_vrt_file(vrt_file=vrt, directory=tmp_path)

    assert out == vrt
    assert calls["destName"] == vrt.as_posix()
    assert set(calls["srcDSOrSrcDSTab"]) == {
        tif_a.absolute().resolve().as_posix(),
        tif_b.absolute().resolve().as_posix(),
    }
    assert calls["options_kwargs"]["resolution"] == "average"
    assert flushed["value"] is True


def test_create_vrt_file_warns_when_no_tifs(tmp_path, monkeypatch):
    warnings = []
    monkeypatch.setattr(vrt_mod.logger, "warning", warnings.append)
    vrt = tmp_path / "empty.vrt"

    out = vrt_mod.create_vrt_file(vrt_file=vrt, directory=tmp_path)

    assert out == vrt
    assert len(warnings) == 1
    assert "No vrt-file created" in warnings[0]


def test_list_tif_files_in_vrt_file_filters_vrt_self(tmp_path, monkeypatch):
    vrt = tmp_path / "tiles.vrt"
    files = [vrt.as_posix(), "C:/tmp/one.tif", "C:/tmp/two.tif"]

    monkeypatch.setattr(
        vrt_mod,
        "gdal",
        SimpleNamespace(Info=lambda path, format: {"files": files}),
    )

    listed = vrt_mod.list_tif_files_in_vrt_file(vrt_file=vrt)

    assert listed == [Path("C:/tmp/one.tif"), Path("C:/tmp/two.tif")]


class _DummyBand:
    DataType = 1

    def GetNoDataValue(self):
        return 0


class _DummyDataset:
    def GetRasterBand(self, index):
        return _DummyBand()

    def FlushCache(self):
        pass


def test_create_cog_file_translates_vrt_directly_to_cog(tmp_path, monkeypatch):
    vrt = tmp_path / "functioneel_landgebruik.vrt"
    vrt.write_text("vrt")
    cog = tmp_path / "functioneel_landgebruik.tif"
    calls = {}

    def fake_options(**kwargs):
        calls["options_kwargs"] = kwargs
        return {"options": kwargs}

    def fake_translate(destName, srcDS, options):
        calls["destName"] = destName
        calls["srcDS"] = srcDS
        calls["options"] = options
        Path(destName).write_text("cog")
        return _DummyDataset()

    monkeypatch.setattr(vrt_mod, "_open_gdal_dataset", lambda path: _DummyDataset())
    monkeypatch.setattr(vrt_mod, "_validate_cog_file", lambda *args: None)
    monkeypatch.setattr(vrt_mod.gdal, "TranslateOptions", fake_options)
    monkeypatch.setattr(vrt_mod.gdal, "Translate", fake_translate)

    out = vrt_mod.create_cog_file(vrt, cog, show_progress=False)

    assert out == cog
    assert cog.read_text() == "cog"
    assert calls["destName"] == (tmp_path / "functioneel_landgebruik.tmp.tif").as_posix()
    assert calls["options_kwargs"]["format"] == "COG"
    assert calls["options_kwargs"]["creationOptions"] == list(vrt_mod.COG_CREATION_OPTIONS)
    assert "OVERVIEW_RESAMPLING=MODE" in calls["options_kwargs"]["creationOptions"]
    assert calls["options_kwargs"]["noData"] == 0
    assert "callback" not in calls["options_kwargs"]


def test_create_cog_file_skips_existing_output_without_opening_vrt(tmp_path, monkeypatch):
    cog = tmp_path / "functioneel_landgebruik.tif"
    cog.write_text("existing")

    def fail_open(path):
        raise AssertionError("existing COG should be returned without opening VRT")

    monkeypatch.setattr(vrt_mod, "_open_gdal_dataset", fail_open)

    out = vrt_mod.create_cog_file(
        tmp_path / "missing.vrt",
        cog,
        overwrite=False,
    )

    assert out == cog
    assert cog.read_text() == "existing"


def test_create_cog_file_requires_existing_vrt(tmp_path):
    with pytest.raises(FileNotFoundError, match="VRT file does not exist"):
        vrt_mod.create_cog_file(
            tmp_path / "missing.vrt",
            tmp_path / "functioneel_landgebruik.tif",
        )


def test_create_cog_file_rejects_invalid_vrt(tmp_path):
    vrt = tmp_path / "invalid.vrt"
    vrt.write_text("not a vrt")

    with pytest.raises(ValueError, match="Could not open raster with GDAL"):
        vrt_mod.create_cog_file(
            vrt,
            tmp_path / "functioneel_landgebruik.tif",
            show_progress=False,
        )


def test_create_cog_file_removes_temp_and_keeps_existing_output_on_failure(
    tmp_path,
    monkeypatch,
):
    vrt = tmp_path / "functioneel_landgebruik.vrt"
    vrt.write_text("vrt")
    cog = tmp_path / "functioneel_landgebruik.tif"
    cog.write_text("existing")
    tmp = tmp_path / "functioneel_landgebruik.tmp.tif"

    def fake_translate(destName, srcDS, options):
        Path(destName).write_text("temporary")
        return _DummyDataset()

    def fail_validation(*args):
        raise ValueError("validation failed")

    monkeypatch.setattr(vrt_mod, "_open_gdal_dataset", lambda path: _DummyDataset())
    monkeypatch.setattr(vrt_mod, "_validate_cog_file", fail_validation)
    monkeypatch.setattr(vrt_mod.gdal, "TranslateOptions", lambda **kwargs: kwargs)
    monkeypatch.setattr(vrt_mod.gdal, "Translate", fake_translate)

    with pytest.raises(ValueError, match="validation failed"):
        vrt_mod.create_cog_file(vrt, cog, overwrite=True, show_progress=False)

    assert cog.read_text() == "existing"
    assert not tmp.exists()


def test_create_cog_file_writes_valid_cog_from_vrt(tmp_path):
    source = tmp_path / "source.tif"
    vrt = tmp_path / "functioneel_landgebruik.vrt"
    cog = tmp_path / "functioneel_landgebruik.tif"
    tmp = tmp_path / "functioneel_landgebruik.tmp.tif"
    data = (np.arange(1024 * 1024, dtype=np.uint32) % 5).astype("uint8").reshape(
        1024,
        1024,
    )

    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        width=1024,
        height=1024,
        count=1,
        dtype="uint8",
        crs="EPSG:28992",
        transform=rasterio.transform.from_origin(0, 1024, 1, 1),
        nodata=0,
        tiled=True,
        blockxsize=512,
        blockysize=512,
    ) as dst:
        dst.write(data, 1)

    vrt_mod.create_vrt_file(vrt_file=vrt, directory=tmp_path)
    out = vrt_mod.create_cog_file(vrt_file=vrt, cog_file=cog, show_progress=False)

    assert out == cog
    assert cog.exists()
    assert not tmp.exists()

    vrt_ds = vrt_mod.gdal.Open(vrt.as_posix())
    cog_ds = vrt_mod.gdal.Open(cog.as_posix())
    try:
        cog_band = cog_ds.GetRasterBand(1)
        vrt_band = vrt_ds.GetRasterBand(1)
        assert cog_ds.GetDriver().ShortName == "GTiff"
        assert tuple(cog_band.GetBlockSize()) == (512, 512)
        assert cog_band.GetOverviewCount() >= 1
        assert cog_band.DataType == vrt_band.DataType
        assert cog_band.GetNoDataValue() == 0
        assert vrt_mod._same_crs(vrt_ds.GetProjectionRef(), cog_ds.GetProjectionRef())
    finally:
        vrt_ds = None
        cog_ds = None

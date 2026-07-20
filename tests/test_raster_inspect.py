import logging
from types import SimpleNamespace

import pytest

import waterlagen.raster.inspect as inspect_mod


class _FakeSpatialRef:
    def GetAuthorityName(self, target):
        return "EPSG"

    def GetAuthorityCode(self, target):
        return "28992"

    def ExportToPrettyWkt(self):
        return "unused"


class _FakeOverview:
    XSize = 30000
    YSize = 35000


class _FakeBand:
    DataType = 1

    def GetNoDataValue(self):
        return 0

    def GetBlockSize(self):
        return [512, 512]

    def GetOverviewCount(self):
        return 1

    def GetOverview(self, index):
        assert index == 0
        return _FakeOverview()

    def ReadAsArray(self):
        raise AssertionError("inspect_raster must not read raster values")


class _FakeDataset:
    RasterXSize = 60000
    RasterYSize = 70000
    RasterCount = 1

    def GetDriver(self):
        return SimpleNamespace(ShortName="GTiff")

    def GetSpatialRef(self):
        return _FakeSpatialRef()

    def GetProjectionRef(self):
        return ""

    def GetGeoTransform(self, can_return_null=True):
        return (0, 1, 0, 70000, 0, -1)

    def GetMetadata(self, domain=None):
        if domain == "IMAGE_STRUCTURE":
            return {"COMPRESSION": "ZSTD", "LAYOUT": "COG"}
        return {"AREA_OR_POINT": "Area"}

    def GetRasterBand(self, index):
        assert index == 1
        return _FakeBand()

    def ReadAsArray(self):
        raise AssertionError("inspect_raster must not read raster values")


def test_inspect_raster_logs_lazy_metadata(tmp_path, monkeypatch, caplog):
    raster = tmp_path / "national.tif"
    raster.write_text("not read")
    opened = []

    def fake_open(path, access):
        opened.append((path, access))
        return _FakeDataset()

    monkeypatch.setattr(inspect_mod.gdal, "Open", fake_open)
    monkeypatch.setattr(inspect_mod.gdal, "GetDataTypeName", lambda data_type: "Byte")
    caplog.set_level(logging.INFO, logger=inspect_mod.logger.name)

    inspect_mod.inspect_raster(raster)

    assert opened == [(raster.as_posix(), inspect_mod.gdal.GA_ReadOnly)]
    assert f"File: {raster}" in caplog.text
    assert "Driver: GTiff" in caplog.text
    assert "Size: 60000 x 70000" in caplog.text
    assert "CRS: EPSG:28992" in caplog.text
    assert "GeoTransform: (0, 1, 0, 70000, 0, -1)" in caplog.text
    assert "Number of bands: 1" in caplog.text
    assert "Default metadata:" in caplog.text
    assert "AREA_OR_POINT: Area" in caplog.text
    assert "IMAGE_STRUCTURE metadata:" in caplog.text
    assert "COMPRESSION: ZSTD" in caplog.text
    assert "Band: 1" in caplog.text
    assert "Type: Byte" in caplog.text
    assert "NoData: 0" in caplog.text
    assert "Block size: 512 x 512" in caplog.text
    assert "Overviews: 1" in caplog.text
    assert "Overview 1: 30000 x 35000" in caplog.text


def test_inspect_raster_requires_existing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Raster file does not exist"):
        inspect_mod.inspect_raster(tmp_path / "missing.tif")


def test_inspect_raster_rejects_unopenable_raster(tmp_path, monkeypatch):
    raster = tmp_path / "broken.tif"
    raster.write_text("broken")
    monkeypatch.setattr(inspect_mod.gdal, "Open", lambda path, access: None)

    with pytest.raises(ValueError, match="Could not open raster with GDAL"):
        inspect_mod.inspect_raster(raster)

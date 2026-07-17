from pathlib import Path
from types import SimpleNamespace

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

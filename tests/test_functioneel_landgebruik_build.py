import inspect

import numpy as np
import rasterio

from waterlagen.functioneel_landgebruik import build as build_mod
from waterlagen.functioneel_landgebruik import bouw_functioneel_landgebruik
from waterlagen.raster.config import RasterOutputConfig


def _patch_sources(monkeypatch, prepared_sources=()):
    monkeypatch.setattr(build_mod, "_download_missing_sources", lambda *args: None)
    monkeypatch.setattr(build_mod, "_validate_sources_exist", lambda sources: None)
    monkeypatch.setattr(build_mod, "_read_dike_area", lambda *args: object())
    monkeypatch.setattr(
        build_mod,
        "_prepare_priority_sources",
        lambda *args, **kwargs: list(prepared_sources),
    )


def test_build_does_not_use_block_windows():
    source = inspect.getsource(build_mod.bouw_functioneel_landgebruik)
    assert "block_windows" not in source


def test_build_allocates_one_full_tile_raster_and_reuses_it(tmp_path, monkeypatch):
    _patch_sources(monkeypatch, prepared_sources=["a", "b", "c"])
    full_calls = []
    raster_ids = []
    original_full = np.full

    def fake_full(shape, fill_value, dtype):
        full_calls.append((shape, fill_value, dtype))
        return original_full(shape, fill_value, dtype=dtype)

    def fake_rasterize(raster, data, transform):
        raster_ids.append(id(raster))
        raster[:, :] = len(raster_ids)
        return raster

    monkeypatch.setattr(build_mod.np, "full", fake_full)
    monkeypatch.setattr(build_mod, "rasterize_features", fake_rasterize)

    target = tmp_path / "landgebruik.tif"
    bouw_functioneel_landgebruik(
        target_path=target,
        bounds=(0, 0, 16, 16),
        resolution_m=1,
        output_config=RasterOutputConfig(block_size=16, overview_factors=(2,)),
        download_missing=False,
    )

    assert full_calls == [((16, 16), 0, np.uint8)]
    assert len(set(raster_ids)) == 1
    with rasterio.open(target) as src:
        assert src.read(1).min() == 3
        assert src.read(1).max() == 3


def test_build_default_output_is_tiled_512_with_mode_overviews(tmp_path, monkeypatch):
    _patch_sources(monkeypatch)

    target = tmp_path / "landgebruik_default.tif"
    bouw_functioneel_landgebruik(
        target_path=target,
        bounds=(0, 0, 1024, 1024),
        resolution_m=1,
        download_missing=False,
    )

    with rasterio.open(target) as src:
        assert src.profile["tiled"] is True
        assert src.block_shapes == [(512, 512)]
        assert src.overviews(1) == [4, 8, 16, 32, 64, 128, 256]
        assert src.tags(ns="rio_overview")["resampling"] == "mode"
        assert src.nodata == 0
        assert src.descriptions == ("Landgebruik",)


def test_build_accepts_custom_output_config(tmp_path, monkeypatch):
    _patch_sources(monkeypatch)

    target = tmp_path / "landgebruik_custom.tif"
    bouw_functioneel_landgebruik(
        target_path=target,
        bounds=(0, 0, 64, 64),
        resolution_m=1,
        output_config=RasterOutputConfig(block_size=16, overview_factors=(2, 4)),
        download_missing=False,
    )

    with rasterio.open(target) as src:
        assert src.block_shapes == [(16, 16)]
        assert src.overviews(1) == [2, 4]

import numpy as np
import pytest
import rasterio
from rasterio.enums import Resampling

from waterlagen.raster.config import RasterOutputConfig
from waterlagen.raster.overviews import build_raster_overviews


def _write_base_raster(path, *, width=16, height=16):
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:28992",
        "transform": rasterio.transform.from_origin(0, height, 1, 1),
        "tiled": True,
        "blockxsize": 16,
        "blockysize": 16,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.ones((height, width), dtype=np.uint8), 1)


@pytest.mark.parametrize(
    ("factors", "error"),
    [
        ((1,), ValueError),
        ((2, 2), ValueError),
        ((4, 2), ValueError),
        ((2.0,), TypeError),
    ],
)
def test_invalid_overview_factors_raise_clear_errors(factors, error):
    with pytest.raises(error, match="Overview factors"):
        RasterOutputConfig(overview_factors=factors)


def test_overview_helper_skips_factors_too_large_for_raster(tmp_path):
    path = tmp_path / "small.tif"
    _write_base_raster(path, width=16, height=16)

    with rasterio.open(path, "r+") as dst:
        build_raster_overviews(
            dst,
            factors=(2, 4, 64),
            resampling=Resampling.nearest,
        )

    with rasterio.open(path) as src:
        assert src.overviews(1) == [2, 4]
        assert src.tags(ns="rio_overview")["resampling"] == "nearest"


def test_raster_output_config_validates_block_size():
    with pytest.raises(ValueError, match="block_size"):
        RasterOutputConfig(block_size=0)
    with pytest.raises(TypeError, match="block_size"):
        RasterOutputConfig(block_size=512.0)
    with pytest.raises(ValueError, match="multiple of 16"):
        RasterOutputConfig(block_size=500)

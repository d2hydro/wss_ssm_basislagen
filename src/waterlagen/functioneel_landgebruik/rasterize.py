from typing import Any

import geopandas as gpd
import numpy as np
from affine import Affine
from rasterio import enums, features


def rasterize_features(
    raster: np.ndarray,
    data: gpd.GeoDataFrame,
    transform: Affine,
    *,
    value_column: str = "code",
    all_touched: bool = True,
) -> np.ndarray:
    """Rasterize vector features into an existing array."""
    if data.empty:
        return raster

    valid = data.dropna(subset=[value_column, "geometry"])
    if valid.empty:
        return raster

    shapes: list[tuple[Any, int]] = [
        (row.geometry, int(getattr(row, value_column)))
        for row in valid[["geometry", value_column]].itertuples()
        if row.geometry is not None and not row.geometry.is_empty
    ]
    if not shapes:
        return raster

    features.rasterize(
        shapes,
        out=raster,
        transform=transform,
        all_touched=all_touched,
        merge_alg=enums.MergeAlg.replace,
    )
    return raster

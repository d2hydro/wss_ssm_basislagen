import tempfile
import os
from dataclasses import dataclass
from importlib import resources
from math import ceil, floor
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd
import pyogrio
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from waterlagen import _geopandas as wgpd
from waterlagen import datastore
from waterlagen._crs import format_crs, same_crs
from waterlagen.settings import settings

TILES_LAYER = "tiles"
BOUNDARY_RESOURCE = "landsgrens.gpkg"
EXPECTED_CRS = "EPSG:28992"


@dataclass(frozen=True, slots=True)
class Tile:
    tile_id: str
    column: int
    row: int
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    @property
    def width(self) -> int:
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        return self.ymax - self.ymin


def _default_tiles_path() -> Path:
    return datastore.processed_data_dir / "tiles" / "tiles.gpkg"


def _validate_inputs(
    target_path: Path,
    *,
    tile_size_m: int,
    origin_x: int,
    origin_y: int,
) -> None:
    if not isinstance(tile_size_m, int) or isinstance(tile_size_m, bool):
        raise TypeError("tile_size_m must be an integer")
    if tile_size_m <= 0:
        raise ValueError("tile_size_m must be greater than 0")
    if not isinstance(origin_x, int) or isinstance(origin_x, bool):
        raise TypeError("origin_x must be an integer")
    if not isinstance(origin_y, int) or isinstance(origin_y, bool):
        raise TypeError("origin_y must be an integer")
    if target_path.suffix.lower() != ".gpkg":
        raise ValueError("target_path suffix must be .gpkg")


def _boundary_resource_path():
    return resources.files("waterlagen.resources").joinpath(BOUNDARY_RESOURCE)


def _read_boundary() -> gpd.GeoDataFrame:
    resource = _boundary_resource_path()
    with resources.as_file(resource) as path:
        layers = pyogrio.list_layers(path)
        if len(layers) == 0:
            raise ValueError(f"No layers found in boundary resource: {BOUNDARY_RESOURCE}")
        layer_name = layers[0][0]
        return wgpd.read_file(path, layer=layer_name)


def _prepare_boundary() -> tuple[gpd.GeoDataFrame, BaseGeometry]:
    gdf = _read_boundary()
    if gdf.empty:
        raise ValueError("Boundary dataset is empty")
    if "geometry" not in gdf or gdf.geometry.isna().all():
        raise ValueError("Boundary dataset contains no geometries")
    if gdf.crs is None:
        raise ValueError("Boundary dataset has no CRS")
    if not same_crs(settings.crs, EXPECTED_CRS):
        raise ValueError(
            f"Expected configured CRS to be {EXPECTED_CRS}, got {format_crs(settings.crs)}"
        )
    if not same_crs(gdf.crs, settings.crs):
        gdf = gdf.to_crs(settings.crs)

    boundary = gdf.geometry.make_valid().union_all()
    if boundary.is_empty:
        raise ValueError("Boundary geometry is empty after merging")
    return gdf, boundary


def _snap_bounds(
    bounds: tuple[float, float, float, float],
    *,
    tile_size_m: int,
    origin_x: int,
    origin_y: int,
) -> tuple[int, int, int, int]:
    minx, miny, maxx, maxy = bounds
    xmin = origin_x + floor((minx - origin_x) / tile_size_m) * tile_size_m
    ymin = origin_y + floor((miny - origin_y) / tile_size_m) * tile_size_m
    xmax = origin_x + ceil((maxx - origin_x) / tile_size_m) * tile_size_m
    ymax = origin_y + ceil((maxy - origin_y) / tile_size_m) * tile_size_m
    return int(xmin), int(ymin), int(xmax), int(ymax)


def _format_coordinate(value: int) -> str:
    if value < 0 or value > 999999:
        raise ValueError(
            f"Tile coordinate {value} cannot be represented as exactly six digits"
        )
    return f"{value:06d}"


def _tile_id(xmin: int, ymin: int, xmax: int, ymax: int) -> str:
    return (
        f"{_format_coordinate(xmin)}_"
        f"{_format_coordinate(ymin)}_"
        f"{_format_coordinate(xmax)}_"
        f"{_format_coordinate(ymax)}"
    )


def _iter_tiles(
    boundary: BaseGeometry,
    *,
    tile_size_m: int,
    origin_x: int,
    origin_y: int,
) -> Iterable[dict]:
    xmin, ymin, xmax, ymax = _snap_bounds(
        boundary.bounds,
        tile_size_m=tile_size_m,
        origin_x=origin_x,
        origin_y=origin_y,
    )
    for y in range(ymin, ymax, tile_size_m):
        for x in range(xmin, xmax, tile_size_m):
            geom = box(x, y, x + tile_size_m, y + tile_size_m)
            if not geom.intersects(boundary):
                continue
            yield {
                "tile_id": _tile_id(x, y, x + tile_size_m, y + tile_size_m),
                "column": (x - origin_x) // tile_size_m,
                "row": (y - origin_y) // tile_size_m,
                "xmin": x,
                "ymin": y,
                "xmax": x + tile_size_m,
                "ymax": y + tile_size_m,
                "geometry": geom,
            }


def _build_tiles_gdf(
    boundary: BaseGeometry,
    *,
    tile_size_m: int,
    origin_x: int,
    origin_y: int,
) -> gpd.GeoDataFrame:
    records = list(
        _iter_tiles(
            boundary,
            tile_size_m=tile_size_m,
            origin_x=origin_x,
            origin_y=origin_y,
        )
    )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=EXPECTED_CRS)


def _validate_tiles(
    tiles: gpd.GeoDataFrame,
    boundary: BaseGeometry,
    *,
    tile_size_m: int,
    origin_x: int,
    origin_y: int,
) -> None:
    if tiles.empty:
        raise ValueError("Generated tile dataset is empty")
    if not tiles.geom_type.eq("Polygon").all():
        raise ValueError("Every tile geometry must be a polygon")
    if not tiles.geometry.is_valid.all():
        raise ValueError("Every tile geometry must be valid")
    if not tiles["tile_id"].is_unique:
        raise ValueError("Tile IDs must be unique")

    width = tiles["xmax"] - tiles["xmin"]
    height = tiles["ymax"] - tiles["ymin"]
    if not width.eq(tile_size_m).all():
        raise ValueError(f"Every tile width must be {tile_size_m}")
    if not height.eq(tile_size_m).all():
        raise ValueError(f"Every tile height must be {tile_size_m}")

    aligned = (
        ((tiles["xmin"] - origin_x) % tile_size_m).eq(0)
        & ((tiles["xmax"] - origin_x) % tile_size_m).eq(0)
        & ((tiles["ymin"] - origin_y) % tile_size_m).eq(0)
        & ((tiles["ymax"] - origin_y) % tile_size_m).eq(0)
    )
    if not aligned.all():
        raise ValueError("Every tile bound must align to the configured origin")
    if not tiles.geometry.intersects(boundary).all():
        raise ValueError("Every retained tile must intersect the national boundary")


def build_tiles(
    target_path: Path | None = None,
    *,
    tile_size_m: int = 2000,
    origin_x: int = 0,
    origin_y: int = 0,
    overwrite: bool = False,
) -> Path:
    """Build the reusable national tile index GeoPackage."""
    target_path = Path(target_path) if target_path is not None else _default_tiles_path()
    _validate_inputs(
        target_path,
        tile_size_m=tile_size_m,
        origin_x=origin_x,
        origin_y=origin_y,
    )

    if target_path.exists() and not overwrite:
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    _, boundary = _prepare_boundary()
    tiles = _build_tiles_gdf(
        boundary,
        tile_size_m=tile_size_m,
        origin_x=origin_x,
        origin_y=origin_y,
    )
    tiles = tiles.sort_values(["ymin", "xmin"], kind="stable").reset_index(drop=True)
    int_columns = ["column", "row", "xmin", "ymin", "xmax", "ymax"]
    tiles[int_columns] = tiles[int_columns].astype("int64")
    _validate_tiles(
        tiles,
        boundary,
        tile_size_m=tile_size_m,
        origin_x=origin_x,
        origin_y=origin_y,
    )

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.",
        suffix=".gpkg",
        dir=target_path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    tmp_path.unlink(missing_ok=True)
    try:
        tiles.to_file(tmp_path, layer=TILES_LAYER, driver="GPKG")
        tmp_path.replace(target_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return target_path


def read_tiles(path: Path | None = None) -> gpd.GeoDataFrame:
    """Read an existing tile index GeoPackage."""
    path = Path(path) if path is not None else _default_tiles_path()
    return wgpd.read_file(path, layer=TILES_LAYER)


def tile_from_row(row: pd.Series) -> Tile:
    """Convert a GeoDataFrame row to an immutable Tile."""
    return Tile(
        tile_id=str(row["tile_id"]),
        column=int(row["column"]),
        row=int(row["row"]),
        xmin=int(row["xmin"]),
        ymin=int(row["ymin"]),
        xmax=int(row["xmax"]),
        ymax=int(row["ymax"]),
    )


def tile_filename(
    layer_name: str,
    tile: Tile,
    *,
    suffix: str = ".tif",
) -> str:
    """Create a stable tile output filename for a layer."""
    if not suffix.startswith("."):
        raise ValueError("suffix must start with '.'")
    return f"{layer_name}_{tile.tile_id}{suffix}"

# %%
from pathlib import Path

import geopandas as gpd
import rasterio
from affine import Affine
from numpy import ndarray
from rasterio.enums import Resampling
from rasterio.fill import fillnodata
from rasterio.io import DatasetReader
from rasterio.windows import Window, from_bounds
from rasterio.windows import transform as window_transform
from shapely.geometry import base, box

from waterlagen import datastore
from waterlagen.logger import get_logger
from waterlagen.raster.vrt import create_vrt_file, list_tif_files_in_vrt_file

logger = get_logger(__name__)


def _tiles_series_from_vrt(
    vrt_file: Path, indices: list[str] | None = None
) -> gpd.GeoSeries:
    """Return a GeoSeries with tiles from a vrt-file"""

    def _get_box(tif_file: Path):
        with rasterio.open(tif_file) as src:
            return box(*src.bounds)

    # read crs for GeoSeries
    with rasterio.open(vrt_file) as src:
        crs = src.crs

    # list ahn_tifs and limit bij indices if not None
    ahn_tifs = list_tif_files_in_vrt_file(vrt_file=vrt_file)
    if indices is not None:
        ahn_tifs = [i for i in ahn_tifs if i.stem in indices]

    # return GeoSeries
    return gpd.GeoSeries({i.stem: _get_box(i) for i in ahn_tifs}, crs=crs)


def interpolate_within_geometry(
    src: DatasetReader,
    geometry: base.BaseGeometry,
    max_search_distance: float = 100,
    band: int = 1,
) -> tuple[ndarray, Affine]:
    """Interpolate raster within geometry

    Parameters
    ----------
    src : DatasetReader
        Opened rasterio DataSet
    geometry : base.BaseGeometry
        A box within geometry.bounds will be interpolated
    max_search_distance : float, optional
        Distance to fill raster ánd to buffer around geometry, by default 100
    band: int, optional
        Band in src to use, 1 by default

    Returns
    -------
    tuple[ndarray, Affine]
        data, transform to use for writing result to a new rasterio DataSet

    Raises
    ------
    ValueError
        Empty window if geometry is not within src.bounds
    """
    # fill window by buffered polygon
    fill_window = from_bounds(
        *geometry.buffer(max_search_distance).bounds, transform=src.transform
    )
    fill_window = fill_window.intersection(
        Window(col_off=0, row_off=0, width=src.width, height=src.height)
    )

    if fill_window.width <= 0 or fill_window.height <= 0:
        raise ValueError("Resulting window is empty after clipping to dataset extent.")

    # read data
    fill_data = src.read(band, window=fill_window, masked=True)

    # fill nodata
    fill_data = fillnodata(
        fill_data,
        max_search_distance=max_search_distance,
    )

    # data window by polygon
    window = from_bounds(*geometry.bounds, transform=src.transform).intersection(
        Window(col_off=0, row_off=0, width=src.width, height=src.height)
    )

    # Compute offsets of small window relative to big window.
    row0 = int(round(window.row_off - fill_window.row_off))
    col0 = int(round(window.col_off - fill_window.col_off))
    h = int(round(window.height))
    w = int(round(window.width))

    # clip data to polygon window
    data = fill_data[row0 : row0 + h, col0 : col0 + w]
    transform = window_transform(window, src.transform)

    return data, transform


def interpolate_ahn_tiles(
    ahn_vrt_file: Path,
    indices: list[str] | None = None,
    max_search_distance: float = 100,
    dir_name: str = "ahn_filled",
    create_vrt: bool = True,
    missing_only: bool = False,
) -> Path:
    """Interpolate ahn raster tiles

    Parameters
    ----------
    ahn_vrt_file : Path
        VRT file referring to AHN tiles
    indices : list[str] | None, optional
        Optional selection of AHN tiles to interpolate, by default None
    max_search_distance: float, optional
        Set search_distance for neighbor search and interpolation distance. By default set to 100
    dir_name : str, optional
        Optional output dir-name within datastore to store result, by default "ahn_filled"
    create_vrt : bool, optional
        Switch for generating a vrt-file for resulting rasters, by default True
    missing_only: bool, optional
        Interpolate only rasters that haven't been interpolated yet, by default False

    Returns
    -------
    Path
        Path to VRT-file (create_vrt=True) or destination directory
    """
    # make destination directory
    dst_dir = datastore.processed_data_dir.joinpath(dir_name)
    dst_dir.mkdir(exist_ok=True, parents=True)
    with rasterio.Env():
        tiles = _tiles_series_from_vrt(vrt_file=ahn_vrt_file, indices=indices)
        if missing_only:
            indices = [
                i
                for i in tiles.index.values
                if not dst_dir.joinpath(f"{dst_dir.name}.vrt").exists()
            ]

            tiles = tiles[indices]
        with rasterio.open(ahn_vrt_file) as src:
            # get base profile for writing
            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                compress="deflate",
                predictor=2,
                tiled=True,
            )
            scales = src.scales
            raster_cell_size = abs(src.res[0])

            # iter polygons
            for idx, (index, geometry) in enumerate(tiles.items()):
                # interpolate
                logger.info(f"start interpolating {index} ({idx + 1}/{len(tiles)})")
                data, transform = interpolate_within_geometry(
                    src=src, geometry=geometry, max_search_distance=max_search_distance
                )

                # write data to GeoTiff
                dst_file = dst_dir.joinpath(f"{index}.tif")
                logger.info(f"writing {dst_file}")
                # updating profile with size and transform
                profile.update(
                    height=data.shape[0], width=data.shape[1], transform=transform
                )
                with rasterio.open(dst_file, "w", **profile) as dst:
                    # writing data and scale if present
                    dst.write(data, 1)
                    dst.scales = scales

                    # add overviews to 5m and 25m
                    factors = [
                        int(size / raster_cell_size)
                        for size in [5, 25]
                        if size > raster_cell_size
                    ]
                    dst.build_overviews(factors, Resampling.average)
                    dst.update_tags(ns="rio_overview", resampling="average")

    if create_vrt:
        vrt_file = dst_dir.joinpath(f"{dst_dir.name}.vrt")
        create_vrt_file(vrt_file=vrt_file, directory=dst_dir)
        return vrt_file
    else:
        return dst_dir

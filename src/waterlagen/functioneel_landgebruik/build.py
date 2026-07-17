import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio.enums import Resampling

from waterlagen import _geopandas as wgpd
from waterlagen import datastore
from waterlagen.bag import download_bag_light
from waterlagen.bgt import download_bgt
from waterlagen.brp import download_brp
from waterlagen.dijkringen import download_dijkringen
from waterlagen.functioneel_landgebruik.legend import COLORMAP
from waterlagen.functioneel_landgebruik.rasterize import rasterize_features
from waterlagen.functioneel_landgebruik.sources import (
    prepare_bag,
    prepare_brp,
    prepare_functionele_gebieden,
    prepare_water,
    prepare_wegen,
)
from waterlagen.raster.config import RasterOutputConfig
from waterlagen.raster.grid import RasterGrid
from waterlagen.raster.overviews import build_raster_overviews
from waterlagen.settings import settings
from waterlagen.top10nl import download_top10nl


@dataclass(frozen=True)
class FunctioneelLandgebruikSources:
    bgt_gpkg: Path = datastore.bgt_dir / "bgt.gpkg"
    bag_gpkg: Path = datastore.bag_dir / "bag-light.gpkg"
    brp_gpkg: Path = datastore.brp_dir / "brpgewaspercelen_definitief_2025.gpkg"
    top10nl_gpkg: Path = datastore.top10nl_dir / "top10nl_Compleet.gpkg"
    dijkringen_gpkg: Path = datastore.dijkringen_dir / "dijkringen_historie_2012.gpkg"


@dataclass(frozen=True)
class FunctioneelLandgebruikLayers:
    bgt_water: str = "bgt_waterdeel"
    bgt_wegdeel: str = "bgt_wegdeel"
    bag_pand: str = "pand"
    bag_verblijfsobject: str = "verblijfsobject"
    brp: str = "brp_gewas"
    top10nl_functioneel_gebied: str = "top10nl_functioneel_gebied_vlak"
    dijkringen: str = "dijkring_v_2012"


def _profile_for_grid(grid: RasterGrid, *, output_config: RasterOutputConfig) -> dict:
    return {
        "driver": "GTiff",
        "count": 1,
        "dtype": "uint8",
        "nodata": 0,
        "width": grid.width,
        "height": grid.height,
        "transform": grid.transform,
        "crs": grid.crs,
        "tiled": True,
        "blockxsize": output_config.block_size,
        "blockysize": output_config.block_size,
        "compress": "ZSTD",
        "zstd_level": 9,
        "predictor": 2,
        "interleave": "band",
        "bigtiff": "IF_SAFER",
    }


def _download_missing_sources(
    sources: FunctioneelLandgebruikSources,
    layers: FunctioneelLandgebruikLayers,
) -> None:
    if not sources.bgt_gpkg.exists():
        download_bgt(
            download_dir=sources.bgt_gpkg.parent,
            target_path=sources.bgt_gpkg,
            featuretypes=[
                layers.bgt_water.removeprefix("bgt_"),
                layers.bgt_wegdeel.removeprefix("bgt_"),
            ],
            overwrite=False,
        )

    if not sources.bag_gpkg.exists():
        download_bag_light(download_dir=sources.bag_gpkg.parent, overwrite=False)

    if not sources.brp_gpkg.exists():
        download_brp(
            download_dir=sources.brp_gpkg.parent,
            filename=sources.brp_gpkg.name,
            overwrite=False,
        )

    if not sources.top10nl_gpkg.exists():
        download_top10nl(download_dir=sources.top10nl_gpkg.parent, overwrite=False)

    if not sources.dijkringen_gpkg.exists():
        download_dijkringen(
            download_dir=sources.dijkringen_gpkg.parent,
            target_path=sources.dijkringen_gpkg,
            overwrite=False,
        )


def _validate_sources_exist(sources: FunctioneelLandgebruikSources) -> None:
    missing = [path for path in sources.__dict__.values() if not Path(path).exists()]
    if missing:
        labels = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing source dataset(s): {labels}")


def _read_dike_area(
    sources: FunctioneelLandgebruikSources,
    layers: FunctioneelLandgebruikLayers,
):
    dijkringen = wgpd.read_file(sources.dijkringen_gpkg, layer=layers.dijkringen)
    return dijkringen.geometry.make_valid().union_all()


def _prepare_priority_sources(
    sources: FunctioneelLandgebruikSources,
    layers: FunctioneelLandgebruikLayers,
    *,
    bounds: tuple[float, float, float, float],
    dike_area,
) -> list[gpd.GeoDataFrame]:
    return [
        prepare_functionele_gebieden(
            sources.top10nl_gpkg,
            layer=layers.top10nl_functioneel_gebied,
            bounds=bounds,
            dike_area=dike_area,
        ),
        prepare_brp(
            sources.brp_gpkg,
            layer=layers.brp,
            bounds=bounds,
            dike_area=dike_area,
        ),
        prepare_water(
            sources.bgt_gpkg,
            layer=layers.bgt_water,
            bounds=bounds,
        ),
        prepare_wegen(
            sources.bgt_gpkg,
            layer=layers.bgt_wegdeel,
            bounds=bounds,
            dike_area=dike_area,
        ),
        prepare_bag(
            sources.bag_gpkg,
            pand_layer=layers.bag_pand,
            verblijfsobject_layer=layers.bag_verblijfsobject,
            bounds=bounds,
            dike_area=dike_area,
        ),
    ]


def bouw_functioneel_landgebruik(
    target_path: Path,
    *,
    bounds: tuple[float, float, float, float],
    resolution_m: float = 0.5,
    crs: str = settings.crs,
    sources: FunctioneelLandgebruikSources | None = None,
    layers: FunctioneelLandgebruikLayers | None = None,
    download_missing_sources: bool | None = None,
    download_missing: bool | None = None,
    overwrite: bool = True,
    output_config: RasterOutputConfig | None = None,
) -> Path:
    """Build the functional land-use GeoTIFF for a requested extent."""
    target_path = Path(target_path)
    sources = sources or FunctioneelLandgebruikSources()
    layers = layers or FunctioneelLandgebruikLayers()
    output_config = output_config or RasterOutputConfig()
    if download_missing_sources is None:
        download_missing_sources = True if download_missing is None else download_missing

    if target_path.exists() and not overwrite:
        return target_path

    if download_missing_sources:
        _download_missing_sources(sources, layers)
    _validate_sources_exist(sources)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    grid = RasterGrid.from_bounds(bounds, resolution=resolution_m, crs=crs)
    profile = _profile_for_grid(grid, output_config=output_config)
    dike_area = _read_dike_area(sources, layers)

    nodata = 0
    raster = np.full(
        (grid.height, grid.width),
        fill_value=nodata,
        dtype=np.uint8,
    )
    for data in _prepare_priority_sources(
        sources,
        layers,
        bounds=grid.bounds,
        dike_area=dike_area,
    ):
        rasterize_features(raster, data, grid.transform)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.",
        suffix=target_path.suffix,
        dir=target_path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    tmp_path.unlink(missing_ok=True)
    try:
        with rio.open(tmp_path, "w", **profile) as dst:
            dst.write(raster, 1)
            build_raster_overviews(
                dst,
                factors=output_config.overview_factors,
                resampling=Resampling.mode,
            )
            dst.set_band_description(1, "Landgebruik")
            dst.colorinterp = (rio.enums.ColorInterp.palette,)
            dst.write_colormap(1, COLORMAP)

        tmp_path.replace(target_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return target_path

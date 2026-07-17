import io
import sys
from pathlib import Path
from typing import Literal, Union

import geopandas as gpd
import requests

from waterlagen import datastore
from waterlagen._downloads import download_geopackage
from waterlagen.logger import get_logger
from waterlagen.settings import settings

logger = get_logger(name=__name__)

ROOT_URL = "https://service.pdok.nl/lv/bag"
MAX_WFS_FEATURES = 50000
WFS_COUNT = 1000
WFS_TYPE_NAME_BY_LAYER: dict[
    Literal["pand", "verblijfsobject"],
    Literal["bag:pand", "bag:verblijfsobject"],
] = {
    "pand": "bag:pand",
    "verblijfsobject": "bag:verblijfsobject",
}


def get_bag_features_from_wfs(
    bbox: tuple[float, float, float, float],
    type_name: Literal["bag:pand", "bag:verblijfsobject"] = "bag:pand",
    crs: int = 28992,
) -> gpd.GeoDataFrame:
    """Download BAG for an extent via WFS to a GeoDataFrame

    Parameters
    ----------
    type_names: Literal["bag:pand", "bag:verblijfsobject"], optional
        WFS typeName to download, by default "bag:pand"
    bbox : tuple[float, float, float, float] | None, optional
        Extent with (xmin, ymin, xmax, ymax). BAG will be downloaded for this extent only.
    crs: int, optional
        Download CRS EPSG-code, by default 28992

    Returns
    -------
    GeoDataFrame
        GeoDataFrame with BAG features
    """

    # specify request url and params
    url = f"{ROOT_URL}/wfs/v2_0"
    page_num = 1

    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": type_name,
        "srsName": f"EPSG:{crs}",
        "bbox": ",".join(map(str, (*bbox, f"EPSG:{crs}"))),
        "count": WFS_COUNT,
        "outputFormat": "application/json",
    }

    # init iter
    features = []
    start_index = 0
    features_sum = 0
    logger.info(
        f"Start downloading BAG {type_name} using WFS, {WFS_COUNT} features per page (download)"
    )
    while True:
        # echo page number to console
        msg = f"Downloading page: {page_num:02d}"
        sys.stdout.write("\r" + msg)
        sys.stdout.flush()

        # request by start-index
        params["startIndex"] = start_index
        response = requests.get(url, params=params)
        response.raise_for_status()

        # read to GeoPackage
        gdf_page = gpd.read_file(io.BytesIO(response.content))

        if gdf_page.empty:
            break

        # append to list
        features.append(gdf_page)
        features_sum += len(gdf_page)
        if features_sum >= MAX_WFS_FEATURES:
            logger.warning(
                f"Your download is incomplete!. Requested features more than {MAX_WFS_FEATURES}. Use a smaller bbox of use `download_bag_nl()` to download BAG for The Netherlands"
            )
            break

        # next page
        start_index += len(gdf_page)
        page_num += 1
        if len(gdf_page) < WFS_COUNT:
            break

    if not features:
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    # concat features and set CRS
    gdf = gpd.pd.concat(features, ignore_index=True)
    gdf = gdf.set_crs(crs)

    return gdf


def download_bag_light(
    download_dir: Path = datastore.bag_dir, overwrite: bool = True
) -> Path:
    """Download BAG for The Netherlands

    Parameters
    ----------
    download_dir : Path, optional
        Download dir to store GeoPackages. By default datastore.bag_dir
    overwrite : bool, optional
        If not True BAG will only be downloaded if not existing. Default is True

    Returns
    -------
    Path
        Path to BAG GeoPackage
    """
    download_dir = Path(download_dir)

    url = f"{ROOT_URL}/atom/downloads/bag-light.gpkg"
    bag_gpkg = download_dir / Path(url).name
    return download_geopackage(
        url=url,
        target_path=bag_gpkg,
        overwrite=overwrite,
        logger=logger,
        expected_crs=settings.crs,
    )


def get_bag_features(
    bbox: tuple[float, float, float, float],
    *,
    layer: Literal["pand", "verblijfsobject"] = "pand",
    source: Literal["bag-light", "wfs"] = "wfs",
    bag_light_gpkg: Union[Path, str] = datastore.bag_dir.joinpath("bag-light.gpkg"),
    wfs_crs: int = 28992,
) -> gpd.GeoDataFrame:
    """Get BAG features

    Use `source=wfs` to use the BAG wfs. Note (!) BAG only allows you to extract 50000 features. If your bbox is too large, you'll get an incomplete result
    Use `source=bag-light` to extract from an existing bag-light GeoPackage. This file can be downloaded with the `download_bag_light` function in the `waterlagen.bag` module

    Parameters
    ----------
    bbox : tuple[float, float, float, float] | None, optional
        Extent with (xmin, ymin, xmax, ymax). BAG will be downloaded for this extent only.
    layer : Literal["pand", "verblijfsobject"], optional
        Layer to extract/download, by default "pand"
    source : Literal["bag-light", "wfs"], optional
        Source to extract/download features from, by default "wfs"
    bag_light_gpkg : Union[Path, str], optional
        bag_light GeoPackage to extract features from with `source=bag-light`, by default it will be found in the datastore at `datastore.bag_dir.joinpath("bag-light.gpkg")`
    wfs_crs : int, optional
        CRS to use for extracting BAG features from WFS, by default 28992

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with BAG-features
    """
    # Download bag via WFS
    if source == "wfs":
        # get type_name and download via WFS
        type_name = WFS_TYPE_NAME_BY_LAYER[layer]
        return get_bag_features_from_wfs(bbox, type_name=type_name, crs=wfs_crs)
    else:  # get features from bag-light GeoPackage
        bag_light_gpkg = Path(bag_light_gpkg)
        if not bag_light_gpkg.exists():
            raise FileNotFoundError(
                f"{bag_light_gpkg} does not exist. Specify a bag-light GPKG in `bag_light_gpkg` or download one with `waterlagen.bag.download_bag_light()`"
            )
        logger.info(f"reading features from {bag_light_gpkg}")

        return gpd.read_file(bag_light_gpkg, bbox=tuple(bbox), layer=layer)

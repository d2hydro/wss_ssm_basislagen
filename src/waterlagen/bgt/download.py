import io
import logging
import os
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Iterable, Optional

import geopandas as gpd
import requests
from shapely.geometry import Polygon, box, shape

from waterlagen._crs import ensure_dataset_crs
from waterlagen.settings import settings

logger = logging.getLogger(__name__)

ROOT_URL = "https://api.pdok.nl"


def request_download(featuretypes: Iterable[str], poly_mask: Optional[shape]) -> str:
    """Make a request for a BGT download. Will respond a download request id that can be used for download

    Parameters
    ----------
    featuretypes : Iterable[str]
        BGT feature-types, e.g. ["waterdeel", "pand"]
    poly_mask : shape | None, optional
        Optional polygon-mask to use as geofilter. If not Polygon, shape-bounding box will be used.
        By default None

    Returns
    -------
    str
        BGT downloadRequestId
    """
    url = f"{ROOT_URL}/lv/bgt/download/v1_0/full/custom"
    body = {
        "featuretypes": featuretypes,
        "format": "citygml",
    }

    # add poly-mask
    if poly_mask is not None:
        if not isinstance(poly_mask, Polygon):
            poly_mask = box(*poly_mask.bounds)
        body["geofilter"] = poly_mask.wkt

    # post download request
    response = requests.post(url, json=body)
    response.raise_for_status()

    # return download request-id
    download_request_id = response.json()["downloadRequestId"]
    logger.debug(r"downloadRequestid: {download_request_id}")
    return download_request_id


def poll_downloadstatus(
    download_request_id: str,
    poll_interval_s: int = 5,
) -> str:
    """Poll bgt download status

    Parameters
    ----------
    download_request_id : str
        BGT downloadRequestId as response from`request_bgt_download()`
    poll_interval_s : int, optional
        Interval to poll status (seconds), by default 5

    Returns
    -------
    str
        BGT download URL
    """
    status_url = (
        f"{ROOT_URL}/lv/bgt/download/v1_0/full/custom/{download_request_id}/status"
    )

    waiting = True

    while waiting:
        response = requests.get(status_url, timeout=60)
        response.raise_for_status()

        data = response.json()
        status = data["status"]
        if status == "COMPLETED":
            waiting = False
            download_url = f"{ROOT_URL}{data['_links']['download']['href']}"
            logger.debug(f"status: {status}. download_url: {download_url}")
        else:
            logger.debug(f"status: {status}. progress: {data['progress']}")
            time.sleep(poll_interval_s)

    return download_url


def _temp_gpkg_path(target_path: Path) -> Path:
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.",
        suffix=".gpkg",
        dir=target_path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    tmp_path.unlink(missing_ok=True)
    return tmp_path


def download_to_geopackage(
    download_url: str, download_dir: Path, crs: int | str = settings.crs
) -> Path:
    """Download BGT and safe as GPKG files

    Parameters
    ----------
    download_url : str
        BGT download url
    download_dir : Path
        Download dir
    crs : int | str, optional
        Expected CRS for downloaded BGT layers, by default settings.crs

    Returns
    -------
    Path
        download_dir
    """
    # 0) make sure download-dir exists
    download_dir = Path(download_dir)
    download_dir.mkdir(exist_ok=True, parents=True)

    # 1) download content
    response = requests.get(download_url)
    response.raise_for_status()
    zip_bytes = response.content

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        gml_names = [n for n in zf.namelist() if n.lower().endswith(".gml")]
        if not gml_names:
            raise ValueError("No GMLs in download")

        for gml_name in gml_names:
            gml_bytes = zf.read(gml_name)

            # Windows-safe temp file: close it before reading with GDAL
            tmp_path = None
            try:
                fd, tmp_path = tempfile.mkstemp(suffix=".gml")
                with os.fdopen(fd, "wb") as f:
                    f.write(gml_bytes)

                gdf = gpd.read_file(tmp_path)

                layer = Path(gml_name).stem

                gpkg_out = download_dir / f"{layer}.gpkg"
                tmp_gpkg = _temp_gpkg_path(gpkg_out)
                try:
                    logger.debug(f"writing {gpkg_out}")
                    gdf.to_file(tmp_gpkg, driver="GPKG", layer=layer)
                    ensure_dataset_crs(
                        tmp_gpkg,
                        expected_crs=crs,
                        logger=logger,
                    )
                    tmp_gpkg.replace(gpkg_out)
                except Exception:
                    tmp_gpkg.unlink(missing_ok=True)
                    raise

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

    return download_dir


def get_bgt_features(
    featuretypes: Iterable[str], poly_mask: Optional[shape], download_dir: Path
) -> Path:
    """Download BGT features in GeoPackages

    Parameters
    ----------
    featuretypes : Iterable[str]
        BGT feature-types, e.g. ["waterdeel", "pand"]
    poly_mask : shape | None, optional
        Optional polygon-mask to use as geofilter. If not Polygon, shape-bounding box will be used.
        By default None
    download_dir : Path
        Download dir to store GeoPackages

    Returns
    -------
    Path
        Download dir to store GeoPackages
    """

    logger.info("Requesting a BGT download")
    download_request_id = request_download(
        featuretypes=featuretypes, poly_mask=poly_mask
    )

    logger.info("Polling status of BGT download")
    download_url = poll_downloadstatus(download_request_id=download_request_id)

    logger.info("Downloading result to GeoPackage")
    download_dir = download_to_geopackage(
        download_url=download_url, download_dir=download_dir
    )

    return download_dir

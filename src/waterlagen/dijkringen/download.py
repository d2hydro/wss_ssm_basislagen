from pathlib import Path

from waterlagen import datastore
from waterlagen._downloads import GeoPackageDownload, download_geopackage_with_metadata
from waterlagen.logger import get_logger
from waterlagen.settings import settings

logger = get_logger(name=__name__)

DIJKRINGEN_URL = (
    "https://geo.rijkswaterstaat.nl/services/ogc/gdr/dijkringen_historie/ows"
    "?service=WFS&version=2.0.0&request=GetFeature"
    "&typeName=dijkring_v_2012&outputFormat=geopackage"
)
DEFAULT_FILENAME = "dijkringen_historie_2012.gpkg"


def download_dijkringen(
    download_dir: Path = datastore.dijkringen_dir,
    *,
    target_path: Path | None = None,
    overwrite: bool = True,
    progress: bool = True,
) -> GeoPackageDownload:
    """Download the historical Rijkswaterstaat dike-ring GeoPackage."""
    if target_path is None:
        target_path = Path(download_dir) / DEFAULT_FILENAME
    else:
        target_path = Path(target_path)

    return download_geopackage_with_metadata(
        url=DIJKRINGEN_URL,
        target_path=target_path,
        overwrite=overwrite,
        logger=logger,
        progress=progress,
        expected_crs=settings.crs,
    )


def download_dijkringen_historie(
    download_dir: Path = datastore.dijkringen_dir,
    *,
    target_path: Path | None = None,
    overwrite: bool = True,
    progress: bool = True,
) -> GeoPackageDownload:
    """Download the historical Rijkswaterstaat dike-ring GeoPackage."""
    return download_dijkringen(
        download_dir=download_dir,
        target_path=target_path,
        overwrite=overwrite,
        progress=progress,
    )

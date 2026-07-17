from pathlib import Path

from waterlagen import datastore
from waterlagen._downloads import download_geopackage
from waterlagen.logger import get_logger

logger = get_logger(name=__name__)

ROOT_URL = "https://service.pdok.nl/kadaster/brt-topnl"
TOP10NL_FILENAME = "top10nl_Compleet.gpkg"


def download_top10nl(
    download_dir: Path = datastore.top10nl_dir, overwrite: bool = True
) -> Path:
    """Download TOP10NL for The Netherlands.

    Parameters
    ----------
    download_dir : Path, optional
        Download dir to store GeoPackages. By default datastore.top10nl_dir
    overwrite : bool, optional
        If not True TOP10NL will only be downloaded if not existing. Default is True

    Returns
    -------
    Path
        Path to TOP10NL GeoPackage
    """
    download_dir = Path(download_dir)
    url = f"{ROOT_URL}/atom/downloads/{TOP10NL_FILENAME}"
    top10nl_gpkg = download_dir / TOP10NL_FILENAME
    return download_geopackage(
        url=url,
        target_path=top10nl_gpkg,
        overwrite=overwrite,
        logger=logger,
    )

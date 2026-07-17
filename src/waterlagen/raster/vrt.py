from pathlib import Path

from osgeo import gdal

from waterlagen.logger import get_logger

logger = get_logger(__name__)

gdal.UseExceptions()


def create_vrt_file(vrt_file: Path, directory: Path | list[Path]):
    """Create a vrt-file from tif-files in (a list of) directory(s)."""
    if isinstance(directory, Path):
        directory = [directory]

    vrt_file = Path(vrt_file)

    tif_files = []
    for dir in directory:
        tif_files += [i.absolute().resolve().as_posix() for i in dir.glob("*.tif")]

    if len(tif_files) > 0:
        vrt_options = gdal.BuildVRTOptions(
            resolution="average",
            separate=False,
            addAlpha=False,
            bandList=[1],
        )

        ds = gdal.BuildVRT(
            destName=vrt_file.as_posix(),
            srcDSOrSrcDSTab=tif_files,
            options=vrt_options,
        )
        ds.FlushCache()
        logger.info(f"VRT file created {vrt_file}")
    else:
        logger.warning(f"No vrt-file created as no files exist in {directory}")

    return vrt_file


def list_tif_files_in_vrt_file(vrt_file: Path):
    """Return a list of files within a vrt-file."""
    vrt_file = Path(vrt_file)

    info = gdal.Info(vrt_file.as_posix(), format="json")

    return [Path(i) for i in info["files"] if i != vrt_file.as_posix()]

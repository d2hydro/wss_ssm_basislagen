from pathlib import Path

from osgeo import gdal, osr
from tqdm.auto import tqdm

from waterlagen.logger import get_logger

logger = get_logger(__name__)

gdal.UseExceptions()

COG_CREATION_OPTIONS = (
    "COMPRESS=ZSTD",
    "LEVEL=9",
    "BLOCKSIZE=512",
    "OVERVIEW_RESAMPLING=MODE",
    "BIGTIFF=IF_SAFER",
    "NUM_THREADS=ALL_CPUS",
)


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


class _GdalProgressBar:
    def __init__(self, *, desc: str) -> None:
        self._progress = tqdm(total=100, desc=desc, unit="%")
        self._current = 0

    def callback(self, complete: float, message: str, data: object) -> int:
        value = max(0, min(100, int(round(complete * 100))))
        if value > self._current:
            self._progress.update(value - self._current)
            self._current = value
        return 1

    def finish(self) -> None:
        if self._current < 100:
            self._progress.update(100 - self._current)
            self._current = 100

    def close(self) -> None:
        self._progress.close()


def _open_gdal_dataset(path: Path):
    try:
        dataset = gdal.Open(path.as_posix(), getattr(gdal, "GA_ReadOnly", 0))
    except Exception as exc:
        raise ValueError(f"Could not open raster with GDAL: {path}") from exc
    if dataset is None:
        raise ValueError(f"Could not open raster with GDAL: {path}")
    return dataset


def _same_crs(left_wkt: str, right_wkt: str) -> bool:
    if not left_wkt and not right_wkt:
        return True
    if not left_wkt or not right_wkt:
        return False

    left = osr.SpatialReference()
    right = osr.SpatialReference()
    if left.ImportFromWkt(left_wkt) != 0 or right.ImportFromWkt(right_wkt) != 0:
        return left_wkt == right_wkt
    return bool(left.IsSame(right))


def _run_cog_validator(cog_file: Path) -> None:
    try:
        from osgeo_utils.samples.validate_cloud_optimized_geotiff import validate
    except ImportError:
        logger.warning("GDAL COG validation utility is not available")
        return

    try:
        validation_result = validate(cog_file.as_posix(), check_tiled=True)
    except Exception as exc:
        raise ValueError(f"COG validation failed for {cog_file}: {exc}") from exc
    errors = validation_result[0]
    if errors:
        message = "; ".join(str(error) for error in errors)
        raise ValueError(f"COG validation failed for {cog_file}: {message}")


def _validate_cog_file(cog_file: Path, vrt_dataset) -> None:
    if not cog_file.exists():
        raise ValueError(f"COG was not created: {cog_file}")

    cog_dataset = _open_gdal_dataset(cog_file)
    try:
        driver = cog_dataset.GetDriver()
        driver_name = driver.ShortName if driver is not None else None
        if driver_name != "GTiff":
            raise ValueError(
                f"COG validation failed for {cog_file}: expected GTiff driver, "
                f"got {driver_name!r}"
            )

        vrt_band = vrt_dataset.GetRasterBand(1)
        cog_band = cog_dataset.GetRasterBand(1)
        block_size = tuple(cog_band.GetBlockSize())
        if block_size != (512, 512):
            raise ValueError(
                f"COG validation failed for {cog_file}: expected 512 x 512 "
                f"blocks, got {block_size[0]} x {block_size[1]}"
            )

        if not _same_crs(vrt_dataset.GetProjectionRef(), cog_dataset.GetProjectionRef()):
            raise ValueError(f"COG validation failed for {cog_file}: CRS differs from VRT")

        if cog_band.DataType != vrt_band.DataType:
            raise ValueError(
                f"COG validation failed for {cog_file}: dtype differs from VRT"
            )

        if cog_band.GetNoDataValue() != vrt_band.GetNoDataValue():
            raise ValueError(
                f"COG validation failed for {cog_file}: nodata differs from VRT"
            )

        if cog_band.GetOverviewCount() < 1:
            raise ValueError(
                f"COG validation failed for {cog_file}: internal overviews are missing"
            )

        _run_cog_validator(cog_file)
    finally:
        cog_dataset = None


def create_cog_file(
    vrt_file: Path,
    cog_file: Path,
    *,
    overwrite: bool = False,
    show_progress: bool = True,
) -> Path:
    """Create a Cloud Optimized GeoTIFF directly from a VRT file."""
    vrt_file = Path(vrt_file)
    cog_file = Path(cog_file)
    tmp_file = cog_file.with_name(f"{cog_file.stem}.tmp{cog_file.suffix}")

    if cog_file.exists() and not overwrite:
        logger.info("Skipping existing COG file %s", cog_file)
        return cog_file

    if not vrt_file.exists():
        raise FileNotFoundError(f"VRT file does not exist: {vrt_file}")

    cog_file.parent.mkdir(parents=True, exist_ok=True)
    if tmp_file.exists():
        tmp_file.unlink()

    vrt_dataset = _open_gdal_dataset(vrt_file)
    progress = _GdalProgressBar(desc="VRT to COG") if show_progress else None
    try:
        vrt_band = vrt_dataset.GetRasterBand(1)
        translate_kwargs = {
            "format": "COG",
            "creationOptions": list(COG_CREATION_OPTIONS),
        }
        nodata = vrt_band.GetNoDataValue()
        if nodata is not None:
            translate_kwargs["noData"] = nodata
        if progress is not None:
            translate_kwargs["callback"] = progress.callback

        options = gdal.TranslateOptions(**translate_kwargs)
        dataset = gdal.Translate(
            destName=tmp_file.as_posix(),
            srcDS=vrt_dataset,
            options=options,
        )
        if dataset is None:
            raise RuntimeError(f"GDAL failed to create COG file: {tmp_file}")
        dataset.FlushCache()
        dataset = None
        if progress is not None:
            progress.finish()

        _validate_cog_file(tmp_file, vrt_dataset)
        tmp_file.replace(cog_file)
        logger.info("COG file created %s", cog_file)
        return cog_file
    except Exception:
        if tmp_file.exists():
            tmp_file.unlink()
        raise
    finally:
        if progress is not None:
            progress.close()
        vrt_dataset = None

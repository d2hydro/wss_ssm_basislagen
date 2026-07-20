from pathlib import Path

from osgeo import gdal

from waterlagen.logger import get_logger

logger = get_logger(__name__)


def _open_raster(file: Path):
    try:
        dataset = gdal.Open(file.as_posix(), getattr(gdal, "GA_ReadOnly", 0))
    except Exception as exc:
        raise ValueError(f"Could not open raster with GDAL: {file}") from exc
    if dataset is None:
        raise ValueError(f"Could not open raster with GDAL: {file}")
    return dataset


def _format_crs(dataset) -> str:
    if hasattr(dataset, "GetSpatialRef"):
        spatial_ref = dataset.GetSpatialRef()
        if spatial_ref is not None:
            authority_name = spatial_ref.GetAuthorityName(None)
            authority_code = spatial_ref.GetAuthorityCode(None)
            if authority_name and authority_code:
                return f"{authority_name}:{authority_code}"
            pretty_wkt = spatial_ref.ExportToPrettyWkt()
            if pretty_wkt:
                return pretty_wkt

    projection = dataset.GetProjectionRef()
    return projection if projection else "<none>"


def _format_geotransform(dataset) -> str:
    try:
        geotransform = dataset.GetGeoTransform(can_return_null=True)
    except TypeError:
        geotransform = dataset.GetGeoTransform()
    return str(geotransform) if geotransform is not None else "<none>"


def _add_metadata(lines: list[str], title: str, metadata: dict[str, str]) -> None:
    if not metadata:
        return

    lines.append(f"{title}:")
    for key, value in sorted(metadata.items()):
        lines.append(f"  {key}: {value}")


def inspect_raster(file: Path) -> None:
    """Log lazy GDAL raster metadata for debugging."""
    file = Path(file)
    if not file.exists():
        raise FileNotFoundError(f"Raster file does not exist: {file}")

    dataset = _open_raster(file)
    try:
        driver = dataset.GetDriver()
        driver_name = driver.ShortName if driver is not None else "<unknown>"
        lines = [
            f"File: {file}",
            f"Driver: {driver_name}",
            f"Size: {dataset.RasterXSize} x {dataset.RasterYSize}",
            f"CRS: {_format_crs(dataset)}",
            f"GeoTransform: {_format_geotransform(dataset)}",
            f"Number of bands: {dataset.RasterCount}",
        ]

        _add_metadata(lines, "Default metadata", dataset.GetMetadata() or {})
        _add_metadata(
            lines,
            "IMAGE_STRUCTURE metadata",
            dataset.GetMetadata("IMAGE_STRUCTURE") or {},
        )

        for band_index in range(1, dataset.RasterCount + 1):
            band = dataset.GetRasterBand(band_index)
            block_x_size, block_y_size = band.GetBlockSize()
            overview_count = band.GetOverviewCount()
            lines.extend(
                [
                    f"Band: {band_index}",
                    f"  Type: {gdal.GetDataTypeName(band.DataType)}",
                    f"  Block size: {block_x_size} x {block_y_size}",
                    f"  NoData: {band.GetNoDataValue()}",
                    f"  Overviews: {overview_count}",
                ]
            )

            for overview_index in range(overview_count):
                overview = band.GetOverview(overview_index)
                lines.append(
                    "    "
                    f"Overview {overview_index + 1}: "
                    f"{overview.XSize} x {overview.YSize}"
                )

        logger.info("\n".join(lines))
    finally:
        dataset = None

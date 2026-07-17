from collections.abc import Sequence

from rasterio.enums import Resampling
from rasterio.io import DatasetWriter


def validate_overview_factors(factors: Sequence[int]) -> tuple[int, ...]:
    """Validate overview factors that are independent of raster dimensions."""
    result = tuple(factors)
    if any(not isinstance(factor, int) or isinstance(factor, bool) for factor in result):
        raise TypeError("Overview factors must be integers")
    if any(factor <= 1 for factor in result):
        raise ValueError("Overview factors must be larger than 1")
    if len(set(result)) != len(result):
        raise ValueError("Overview factors must be unique")
    if result != tuple(sorted(result)):
        raise ValueError("Overview factors must be sorted ascending")
    return result


def usable_overview_factors(
    factors: Sequence[int],
    *,
    width: int,
    height: int,
) -> tuple[int, ...]:
    """Return overview factors that sensibly reduce at least one raster dimension."""
    validated = validate_overview_factors(factors)
    return tuple(
        factor for factor in validated if not (factor > width and factor > height)
    )


def build_raster_overviews(
    dataset: DatasetWriter,
    *,
    factors: Sequence[int],
    resampling: Resampling,
) -> None:
    """Build internal raster overviews and store the resampling metadata tag."""
    usable_factors = usable_overview_factors(
        factors,
        width=dataset.width,
        height=dataset.height,
    )
    if not usable_factors:
        return

    dataset.build_overviews(usable_factors, resampling)
    dataset.update_tags(ns="rio_overview", resampling=resampling.name)

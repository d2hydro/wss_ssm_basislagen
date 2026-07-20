from dataclasses import dataclass
from math import ceil

from affine import Affine
from rasterio.transform import from_bounds


@dataclass(frozen=True)
class RasterGrid:
    """Regular raster grid definition."""

    bounds: tuple[float, float, float, float]
    resolution: float
    crs: str
    width: int
    height: int
    transform: Affine

    @classmethod
    def from_bounds(
        cls,
        bounds: tuple[float, float, float, float],
        *,
        resolution: float,
        crs: str,
    ) -> "RasterGrid":
        minx, miny, maxx, maxy = bounds
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        if minx >= maxx or miny >= maxy:
            raise ValueError("bounds must be ordered as minx, miny, maxx, maxy")

        width = ceil((maxx - minx) / resolution)
        height = ceil((maxy - miny) / resolution)
        transform = from_bounds(minx, miny, maxx, maxy, width, height)
        return cls(
            bounds=bounds,
            resolution=resolution,
            crs=crs,
            width=width,
            height=height,
            transform=transform,
        )

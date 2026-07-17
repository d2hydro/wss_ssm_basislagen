from dataclasses import dataclass

from waterlagen.raster.overviews import validate_overview_factors


@dataclass(frozen=True, slots=True)
class RasterOutputConfig:
    block_size: int = 512
    overview_factors: tuple[int, ...] = (4, 8, 16, 32, 64, 128, 256)

    def __post_init__(self) -> None:
        if not isinstance(self.block_size, int) or isinstance(self.block_size, bool):
            raise TypeError("block_size must be an integer")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.block_size % 16 != 0:
            raise ValueError("block_size must be a multiple of 16 for GeoTIFF tiling")
        object.__setattr__(
            self,
            "overview_factors",
            validate_overview_factors(self.overview_factors),
        )

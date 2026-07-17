# %%
import os
from multiprocessing import freeze_support
from pathlib import Path

from waterlagen import datastore
from waterlagen.functioneel_landgebruik import bouw_functioneel_landgebruik_tiles
from waterlagen.logger import get_logger
from waterlagen.raster.inspect import inspect_raster
from waterlagen.raster.tiles import build_tiles
from waterlagen.raster.vrt import create_cog_file, create_vrt_file

logger = get_logger(__name__)


def _safe_workers(max_workers: int = 3) -> int:
    return max(1, min(max_workers, os.cpu_count() or 1))


def main() -> Path:
    tiles_path = build_tiles(
        tile_size_m=5000,
        overwrite=False,
    )

    workers = _safe_workers()
    print(f"attempt to build with #workers: {workers}")

    data_dir = datastore.processed_data_dir / "functioneel_landgebruik"
    tiles_dir = data_dir / "tiles"
    tile_files = bouw_functioneel_landgebruik_tiles(
        target_dir=tiles_dir,
        tiles_path=tiles_path,
        workers=workers,
        overwrite=False,
    )

    print(f"Tiles built: {len(tile_files)}")

    print("Create VRT-file")
    vrt_file = create_vrt_file(
        vrt_file=data_dir / "functioneel_landgebruik.vrt", directory=tiles_dir
    )

    print(f"Create cog_file from vrt_file: {vrt_file}")
    cog_file = create_cog_file(
        vrt_file=vrt_file,
        cog_file=data_dir / "functioneel_landgebruik.tif",
        overwrite=False,
    )

    inspect_raster(cog_file)
    return cog_file


if __name__ == "__main__":
    freeze_support()
    main()

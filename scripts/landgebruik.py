# %%
import os
from multiprocessing import freeze_support
from pathlib import Path

from waterlagen import datastore
from waterlagen.functioneel_landgebruik import bouw_functioneel_landgebruik_tiles
from waterlagen.raster.tiles import build_tiles
from waterlagen.raster.vrt import create_vrt_file


def _safe_workers(max_workers: int = 3) -> int:
    return max(1, min(max_workers, os.cpu_count() or 1))


def main() -> list[Path]:
    tiles_path = build_tiles(
        tile_size_m=5000,
        overwrite=False,
    )

    workers = _safe_workers()
    print(f"attempt to build with #workers: {workers}")

    data_dir = datastore.processed_data_dir / "functioneel_landgebruik"
    tiles_dir = data_dir / "tiles"
    paths = bouw_functioneel_landgebruik_tiles(
        target_dir=tiles_dir,
        tiles_path=tiles_path,
        workers=workers,
        overwrite=False,
    )

    print(f"Built or skipped {len(paths)} tiles")

    print("Create VRT-file")
    vrt_path = create_vrt_file(
        vrt_file=data_dir / "functioneel_landgebruik.vrt", directory=tiles_dir
    )
    return vrt_path


if __name__ == "__main__":
    freeze_support()
    main()

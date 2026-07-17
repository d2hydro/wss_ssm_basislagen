from __future__ import annotations

import os
from collections.abc import Collection
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
from tqdm.auto import tqdm

from waterlagen.functioneel_landgebruik.build import (
    FunctioneelLandgebruikLayers,
    FunctioneelLandgebruikSources,
    _download_missing_sources,
    _validate_sources_exist,
    bouw_functioneel_landgebruik,
)
from waterlagen.logger import get_logger
from waterlagen.raster.config import RasterOutputConfig
from waterlagen.raster.tiles import Tile, read_tiles, tile_filename
from waterlagen.settings import settings

logger = get_logger(__name__)

REQUIRED_TILE_COLUMNS = {"tile_id", "xmin", "ymin", "xmax", "ymax", "geometry"}
LAYER_NAME = "functioneel_landgebruik"


class TileBuildError(RuntimeError):
    """Raised when one or more functioneel-landgebruik tiles failed."""

    def __init__(self, failures: dict[str, BaseException]) -> None:
        self.failures = failures
        details = "; ".join(
            f"{tile_id}: {type(error).__name__}: {error}"
            for tile_id, error in failures.items()
        )
        super().__init__(f"Failed to build functioneel landgebruik tile(s): {details}")


@dataclass(frozen=True, slots=True)
class FunctioneelLandgebruikTileJob:
    tile_id: str
    bounds: tuple[int, int, int, int]
    target_path: Path
    overwrite: bool
    resolution_m: float
    crs: str
    sources: FunctioneelLandgebruikSources
    layers: FunctioneelLandgebruikLayers
    output_config: RasterOutputConfig


def _build_tile_worker(job: FunctioneelLandgebruikTileJob) -> Path:
    return bouw_functioneel_landgebruik(
        target_path=job.target_path,
        bounds=job.bounds,
        resolution_m=job.resolution_m,
        crs=job.crs,
        sources=job.sources,
        layers=job.layers,
        output_config=job.output_config,
        overwrite=job.overwrite,
        download_missing_sources=False,
    )


def _resolve_workers(workers: int | None) -> int:
    if workers is None:
        return min(4, os.cpu_count() or 1)
    if not isinstance(workers, int) or isinstance(workers, bool):
        raise TypeError("workers must be an integer or None")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    return workers


def _validate_tile_index(tiles: gpd.GeoDataFrame) -> None:
    missing = REQUIRED_TILE_COLUMNS - set(tiles.columns)
    if missing:
        labels = ", ".join(sorted(missing))
        raise ValueError(f"Tile index is missing required column(s): {labels}")
    if tiles.empty:
        raise ValueError("Tile index is empty")


def _select_tiles(
    tiles: gpd.GeoDataFrame,
    tile_ids: Collection[str] | None,
) -> gpd.GeoDataFrame:
    if tile_ids is None:
        return tiles
    if isinstance(tile_ids, (str, bytes)):
        raise TypeError("tile_ids must be a collection of strings, not a string")

    requested = {str(tile_id) for tile_id in tile_ids}
    available = set(tiles["tile_id"].astype(str))
    missing = sorted(requested - available)
    if missing:
        labels = ", ".join(missing)
        raise ValueError(f"Unknown tile ID(s): {labels}")

    selected = tiles[tiles["tile_id"].astype(str).isin(requested)]
    return selected.reset_index(drop=True)


def _tile_from_row(row: pd.Series) -> Tile:
    return Tile(
        tile_id=str(row["tile_id"]),
        column=int(row["column"]) if "column" in row else 0,
        row=int(row["row"]) if "row" in row else 0,
        xmin=int(row["xmin"]),
        ymin=int(row["ymin"]),
        xmax=int(row["xmax"]),
        ymax=int(row["ymax"]),
    )


def _job_from_row(
    row: pd.Series,
    *,
    target_dir: Path,
    overwrite: bool,
    resolution_m: float,
    crs: str,
    sources: FunctioneelLandgebruikSources,
    layers: FunctioneelLandgebruikLayers,
    output_config: RasterOutputConfig,
) -> FunctioneelLandgebruikTileJob:
    tile = _tile_from_row(row)
    return FunctioneelLandgebruikTileJob(
        tile_id=tile.tile_id,
        bounds=tile.bounds,
        target_path=target_dir / tile_filename(LAYER_NAME, tile),
        overwrite=overwrite,
        resolution_m=resolution_m,
        crs=crs,
        sources=sources,
        layers=layers,
        output_config=output_config,
    )


def _prepare_sources_once(
    sources: FunctioneelLandgebruikSources,
    layers: FunctioneelLandgebruikLayers,
    *,
    download_missing_sources: bool,
) -> None:
    if download_missing_sources:
        _download_missing_sources(sources, layers)
    _validate_sources_exist(sources)


def bouw_functioneel_landgebruik_tiles(
    target_dir: Path,
    *,
    tiles_path: Path | None = None,
    workers: int | None = None,
    overwrite: bool = False,
    tile_ids: Collection[str] | None = None,
    resolution_m: float = 0.5,
    crs: str = settings.crs,
    sources: FunctioneelLandgebruikSources | None = None,
    layers: FunctioneelLandgebruikLayers | None = None,
    output_config: RasterOutputConfig | None = None,
    download_missing_sources: bool = True,
    show_progress: bool = True,
) -> list[Path]:
    """Build functioneel-landgebruik GeoTIFFs for tiles from the tile index."""
    worker_count = _resolve_workers(workers)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    sources = sources or FunctioneelLandgebruikSources()
    layers = layers or FunctioneelLandgebruikLayers()
    output_config = output_config or RasterOutputConfig()

    tiles = read_tiles(tiles_path)
    _validate_tile_index(tiles)
    selected = _select_tiles(tiles, tile_ids)
    logger.info("Selected %s functioneel-landgebruik tile(s)", len(selected))

    jobs = [
        _job_from_row(
            row,
            target_dir=target_dir,
            overwrite=overwrite,
            resolution_m=resolution_m,
            crs=crs,
            sources=sources,
            layers=layers,
            output_config=output_config,
        )
        for _, row in selected.iterrows()
    ]

    results_by_tile_id: dict[str, Path] = {}
    jobs_to_submit: list[FunctioneelLandgebruikTileJob] = []
    for job in jobs:
        if job.target_path.exists() and not overwrite:
            logger.info("Skipping existing functioneel-landgebruik tile %s", job.tile_id)
            results_by_tile_id[job.tile_id] = job.target_path
            continue
        jobs_to_submit.append(job)

    logger.info(
        "Skipped %s existing tile(s), submitting %s tile(s) with %s worker(s)",
        len(results_by_tile_id),
        len(jobs_to_submit),
        worker_count,
    )

    failures: dict[str, BaseException] = {}
    progress = None
    if show_progress:
        progress = tqdm(
            total=len(jobs),
            initial=len(results_by_tile_id),
            desc="Functioneel landgebruik",
            unit="tile",
        )
    try:
        if jobs_to_submit:
            _prepare_sources_once(
                sources,
                layers,
                download_missing_sources=download_missing_sources,
            )

            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(_build_tile_worker, job): job
                    for job in jobs_to_submit
                }
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        results_by_tile_id[job.tile_id] = future.result()
                        logger.info(
                            "Completed functioneel-landgebruik tile %s",
                            job.tile_id,
                        )
                    except Exception as exc:
                        failures[job.tile_id] = exc
                        if show_progress:
                            tqdm.write(
                                f"Failed functioneel-landgebruik tile "
                                f"{job.tile_id}: {exc}"
                            )
                        logger.exception(
                            "Failed functioneel-landgebruik tile %s",
                            job.tile_id,
                        )
                    finally:
                        if progress is not None:
                            progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    if failures:
        raise TileBuildError(failures)

    logger.info(
        "Completed %s functioneel-landgebruik tile(s)",
        len(results_by_tile_id),
    )
    return [results_by_tile_id[job.tile_id] for job in jobs]

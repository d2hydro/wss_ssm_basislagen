import pytest
import geopandas as gpd
from shapely.geometry import box

from waterlagen.functioneel_landgebruik import parallel as parallel_mod
from waterlagen.functioneel_landgebruik.parallel import (
    TileBuildError,
    bouw_functioneel_landgebruik_tiles,
)


def _tiles_gdf() -> gpd.GeoDataFrame:
    records = [
        {
            "tile_id": "000000_000000_002000_002000",
            "column": 0,
            "row": 0,
            "xmin": 0,
            "ymin": 0,
            "xmax": 2000,
            "ymax": 2000,
            "geometry": box(0, 0, 2000, 2000),
        },
        {
            "tile_id": "002000_000000_004000_002000",
            "column": 1,
            "row": 0,
            "xmin": 2000,
            "ymin": 0,
            "xmax": 4000,
            "ymax": 2000,
            "geometry": box(2000, 0, 4000, 2000),
        },
        {
            "tile_id": "000000_002000_002000_004000",
            "column": 0,
            "row": 1,
            "xmin": 0,
            "ymin": 2000,
            "xmax": 2000,
            "ymax": 4000,
            "geometry": box(0, 2000, 2000, 4000),
        },
    ]
    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:28992")


class _FakeFuture:
    def __init__(self, result=None, exception=None):
        self._result = result
        self._exception = exception

    def result(self):
        if self._exception is not None:
            raise self._exception
        return self._result


def _patch_executor(monkeypatch, *, reverse_completed: bool = False):
    class FakeExecutor:
        max_workers_seen = []
        submitted_jobs = []

        def __init__(self, *, max_workers):
            self.max_workers = max_workers
            self.max_workers_seen.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def submit(self, fn, job):
            self.submitted_jobs.append(job)
            try:
                return _FakeFuture(result=fn(job))
            except Exception as exc:
                return _FakeFuture(exception=exc)

    def fake_as_completed(futures):
        items = list(futures)
        if reverse_completed:
            items.reverse()
        return items

    monkeypatch.setattr(parallel_mod, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(parallel_mod, "as_completed", fake_as_completed)
    return FakeExecutor


def _patch_read_tiles(monkeypatch, tiles):
    calls = []

    def fake_read_tiles(path=None):
        calls.append(path)
        return tiles

    monkeypatch.setattr(parallel_mod, "read_tiles", fake_read_tiles)
    return calls


def _patch_sources(monkeypatch):
    monkeypatch.setattr(parallel_mod, "_download_missing_sources", lambda *args: None)
    monkeypatch.setattr(parallel_mod, "_validate_sources_exist", lambda sources: None)


def _patch_builder(monkeypatch, *, failing_tile_ids=()):
    calls = []
    failures = set(failing_tile_ids)

    def fake_builder(**kwargs):
        calls.append(kwargs)
        tile_id = kwargs["target_path"].stem.removeprefix("functioneel_landgebruik_")
        if tile_id in failures:
            raise RuntimeError(f"failed {tile_id}")
        kwargs["target_path"].write_text("built")
        return kwargs["target_path"]

    monkeypatch.setattr(parallel_mod, "bouw_functioneel_landgebruik", fake_builder)
    return calls


def _patch_progress(monkeypatch):
    class FakeTqdm:
        instances = []
        writes = []

        def __init__(self, *, total, initial, desc, unit):
            self.total = total
            self.initial = initial
            self.desc = desc
            self.unit = unit
            self.updates = []
            self.closed = False
            self.instances.append(self)

        def update(self, count):
            self.updates.append(count)

        def close(self):
            self.closed = True

        @classmethod
        def write(cls, message):
            cls.writes.append(message)

    monkeypatch.setattr(parallel_mod, "tqdm", FakeTqdm)
    return FakeTqdm


def test_parallel_build_reads_tiles_and_creates_jobs_with_bounds_and_filenames(
    tmp_path,
    monkeypatch,
):
    tiles = _tiles_gdf()
    read_calls = _patch_read_tiles(monkeypatch, tiles)
    _patch_sources(monkeypatch)
    executor = _patch_executor(monkeypatch)
    builder_calls = _patch_builder(monkeypatch)
    tiles_path = tmp_path / "tiles.gpkg"

    result = bouw_functioneel_landgebruik_tiles(
        target_dir=tmp_path,
        tiles_path=tiles_path,
        workers=1,
    )

    assert read_calls == [tiles_path]
    assert [call["bounds"] for call in builder_calls] == [
        (0, 0, 2000, 2000),
        (2000, 0, 4000, 2000),
        (0, 2000, 2000, 4000),
    ]
    assert [path.name for path in result] == [
        "functioneel_landgebruik_000000_000000_002000_002000.tif",
        "functioneel_landgebruik_002000_000000_004000_002000.tif",
        "functioneel_landgebruik_000000_002000_002000_004000.tif",
    ]
    assert executor.max_workers_seen == [1]
    assert [job.tile_id for job in executor.submitted_jobs] == list(tiles["tile_id"])


def test_parallel_build_filters_tile_ids_preserving_index_order(tmp_path, monkeypatch):
    tiles = _tiles_gdf()
    _patch_read_tiles(monkeypatch, tiles)
    _patch_sources(monkeypatch)
    _patch_executor(monkeypatch)
    builder_calls = _patch_builder(monkeypatch)

    result = bouw_functioneel_landgebruik_tiles(
        target_dir=tmp_path,
        workers=1,
        tile_ids={
            "000000_002000_002000_004000",
            "000000_000000_002000_002000",
        },
    )

    assert [path.name for path in result] == [
        "functioneel_landgebruik_000000_000000_002000_002000.tif",
        "functioneel_landgebruik_000000_002000_002000_004000.tif",
    ]
    assert [call["bounds"] for call in builder_calls] == [
        (0, 0, 2000, 2000),
        (0, 2000, 2000, 4000),
    ]


def test_parallel_build_rejects_unknown_tile_ids(tmp_path, monkeypatch):
    _patch_read_tiles(monkeypatch, _tiles_gdf())

    with pytest.raises(ValueError, match="Unknown tile ID"):
        bouw_functioneel_landgebruik_tiles(
            target_dir=tmp_path,
            tile_ids={"missing"},
        )


def test_parallel_build_skips_existing_outputs(tmp_path, monkeypatch):
    tiles = _tiles_gdf().iloc[:2].copy()
    _patch_read_tiles(monkeypatch, tiles)
    _patch_sources(monkeypatch)
    executor = _patch_executor(monkeypatch)
    builder_calls = _patch_builder(monkeypatch)
    existing = tmp_path / "functioneel_landgebruik_000000_000000_002000_002000.tif"
    existing.write_text("existing")

    result = bouw_functioneel_landgebruik_tiles(
        target_dir=tmp_path,
        workers=1,
        overwrite=False,
    )

    assert [path.name for path in result] == [
        "functioneel_landgebruik_000000_000000_002000_002000.tif",
        "functioneel_landgebruik_002000_000000_004000_002000.tif",
    ]
    assert existing.read_text() == "existing"
    assert len(builder_calls) == 1
    assert [job.tile_id for job in executor.submitted_jobs] == [
        "002000_000000_004000_002000"
    ]


def test_parallel_build_rebuilds_existing_outputs_when_overwrite_true(
    tmp_path,
    monkeypatch,
):
    tiles = _tiles_gdf().iloc[:1].copy()
    _patch_read_tiles(monkeypatch, tiles)
    _patch_sources(monkeypatch)
    executor = _patch_executor(monkeypatch)
    _patch_builder(monkeypatch)
    existing = tmp_path / "functioneel_landgebruik_000000_000000_002000_002000.tif"
    existing.write_text("existing")

    result = bouw_functioneel_landgebruik_tiles(
        target_dir=tmp_path,
        workers=2,
        overwrite=True,
    )

    assert result == [existing]
    assert existing.read_text() == "built"
    assert executor.max_workers_seen == [2]
    assert [job.tile_id for job in executor.submitted_jobs] == [
        "000000_000000_002000_002000"
    ]


@pytest.mark.parametrize("workers", [0, -1])
def test_parallel_build_rejects_invalid_worker_count(tmp_path, workers):
    with pytest.raises(ValueError, match="workers"):
        bouw_functioneel_landgebruik_tiles(target_dir=tmp_path, workers=workers)


def test_parallel_build_uses_requested_multiple_workers(tmp_path, monkeypatch):
    _patch_read_tiles(monkeypatch, _tiles_gdf().iloc[:1].copy())
    _patch_sources(monkeypatch)
    executor = _patch_executor(monkeypatch)
    _patch_builder(monkeypatch)

    bouw_functioneel_landgebruik_tiles(target_dir=tmp_path, workers=3)

    assert executor.max_workers_seen == [3]


def test_parallel_build_returns_paths_in_tile_index_order(tmp_path, monkeypatch):
    tiles = _tiles_gdf()
    _patch_read_tiles(monkeypatch, tiles)
    _patch_sources(monkeypatch)
    _patch_executor(monkeypatch, reverse_completed=True)
    _patch_builder(monkeypatch)

    result = bouw_functioneel_landgebruik_tiles(target_dir=tmp_path, workers=2)

    assert [path.name for path in result] == [
        "functioneel_landgebruik_000000_000000_002000_002000.tif",
        "functioneel_landgebruik_002000_000000_004000_002000.tif",
        "functioneel_landgebruik_000000_002000_002000_004000.tif",
    ]


def test_parallel_build_reports_all_failed_tiles_and_keeps_successful_outputs(
    tmp_path,
    monkeypatch,
):
    tiles = _tiles_gdf()
    _patch_read_tiles(monkeypatch, tiles)
    _patch_sources(monkeypatch)
    _patch_executor(monkeypatch)
    _patch_builder(
        monkeypatch,
        failing_tile_ids={
            "000000_000000_002000_002000",
            "000000_002000_002000_004000",
        },
    )

    with pytest.raises(TileBuildError) as exc:
        bouw_functioneel_landgebruik_tiles(target_dir=tmp_path, workers=2)

    message = str(exc.value)
    assert "000000_000000_002000_002000" in message
    assert "000000_002000_002000_004000" in message
    assert (
        tmp_path / "functioneel_landgebruik_002000_000000_004000_002000.tif"
    ).exists()


def test_parallel_build_prepares_sources_once_before_pool_starts(
    tmp_path,
    monkeypatch,
):
    events = []

    class FakeExecutor:
        def __init__(self, *, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            events.append("pool")
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def submit(self, fn, job):
            return _FakeFuture(result=fn(job))

    _patch_read_tiles(monkeypatch, _tiles_gdf().iloc[:2].copy())
    monkeypatch.setattr(parallel_mod, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(parallel_mod, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(
        parallel_mod,
        "_download_missing_sources",
        lambda *args: events.append("download"),
    )
    monkeypatch.setattr(
        parallel_mod,
        "_validate_sources_exist",
        lambda sources: events.append("validate"),
    )
    _patch_builder(monkeypatch)

    bouw_functioneel_landgebruik_tiles(target_dir=tmp_path, workers=1)

    assert events[:3] == ["download", "validate", "pool"]


def test_parallel_build_progress_starts_with_skipped_tiles(tmp_path, monkeypatch):
    tiles = _tiles_gdf().iloc[:2].copy()
    _patch_read_tiles(monkeypatch, tiles)
    _patch_sources(monkeypatch)
    _patch_executor(monkeypatch)
    _patch_builder(monkeypatch)
    progress = _patch_progress(monkeypatch)
    existing = tmp_path / "functioneel_landgebruik_000000_000000_002000_002000.tif"
    existing.write_text("existing")

    bouw_functioneel_landgebruik_tiles(
        target_dir=tmp_path,
        workers=1,
        overwrite=False,
    )

    assert len(progress.instances) == 1
    bar = progress.instances[0]
    assert bar.total == 2
    assert bar.initial == 1
    assert bar.desc == "Functioneel landgebruik"
    assert bar.unit == "tile"
    assert bar.updates == [1]
    assert bar.closed is True


def test_parallel_build_progress_updates_for_successful_and_failed_tiles(
    tmp_path,
    monkeypatch,
):
    _patch_read_tiles(monkeypatch, _tiles_gdf())
    _patch_sources(monkeypatch)
    _patch_executor(monkeypatch)
    _patch_builder(
        monkeypatch,
        failing_tile_ids={
            "000000_000000_002000_002000",
            "000000_002000_002000_004000",
        },
    )
    progress = _patch_progress(monkeypatch)

    with pytest.raises(TileBuildError):
        bouw_functioneel_landgebruik_tiles(target_dir=tmp_path, workers=2)

    bar = progress.instances[0]
    assert bar.updates == [1, 1, 1]
    assert bar.closed is True
    assert len(progress.writes) == 2
    assert "000000_000000_002000_002000" in progress.writes[0]
    assert "000000_002000_002000_004000" in progress.writes[1]


def test_parallel_build_can_disable_progress(tmp_path, monkeypatch):
    _patch_read_tiles(monkeypatch, _tiles_gdf().iloc[:1].copy())
    _patch_sources(monkeypatch)
    _patch_executor(monkeypatch)
    _patch_builder(monkeypatch)

    def fail_tqdm(*args, **kwargs):
        raise AssertionError("progress should not be created")

    monkeypatch.setattr(parallel_mod, "tqdm", fail_tqdm)

    bouw_functioneel_landgebruik_tiles(
        target_dir=tmp_path,
        workers=1,
        show_progress=False,
    )

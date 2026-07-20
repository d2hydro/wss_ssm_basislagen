import geopandas as gpd
import pytest
from shapely.geometry import box

from waterlagen.raster import tiles as tiles_mod
from waterlagen.raster.tiles import (
    Tile,
    build_tiles,
    read_tiles,
    tile_filename,
    tile_from_row,
)


def _patch_boundary(monkeypatch, boundary):
    monkeypatch.setattr(
        tiles_mod,
        "_prepare_boundary",
        lambda: (
            gpd.GeoDataFrame(geometry=[boundary], crs=tiles_mod.EXPECTED_CRS),
            boundary,
        ),
    )


def test_build_tiles_snaps_bounds_to_origin_and_keeps_intersecting_squares(
    tmp_path,
    monkeypatch,
):
    boundary = box(100, 100, 4100, 2100)
    _patch_boundary(monkeypatch, boundary)

    path = build_tiles(tmp_path / "tiles.gpkg", tile_size_m=2000)
    gdf = read_tiles(path)

    assert len(gdf) == 6
    assert set(gdf["xmin"]) == {0, 2000, 4000}
    assert set(gdf["ymin"]) == {0, 2000}
    assert ((gdf["xmax"] - gdf["xmin"]) == 2000).all()
    assert ((gdf["ymax"] - gdf["ymin"]) == 2000).all()
    assert gdf.geom_type.eq("Polygon").all()
    assert gdf.geometry.intersects(boundary).all()
    assert list(gdf[["ymin", "xmin"]].itertuples(index=False, name=None)) == sorted(
        gdf[["ymin", "xmin"]].itertuples(index=False, name=None)
    )


def test_tile_id_and_row_column_are_tied_to_fixed_grid(tmp_path, monkeypatch):
    boundary = box(2500, 2500, 2600, 2600)
    _patch_boundary(monkeypatch, boundary)

    path = build_tiles(tmp_path / "tiles.gpkg", tile_size_m=2000)
    row = read_tiles(path).iloc[0]

    assert row["tile_id"] == "002000_002000_004000_004000"
    assert row["column"] == 1
    assert row["row"] == 1


def test_build_tiles_returns_existing_file_when_overwrite_false(tmp_path, monkeypatch):
    target = tmp_path / "tiles.gpkg"
    target.write_bytes(b"existing")

    def fail_prepare_boundary():
        raise AssertionError("boundary should not be read")

    monkeypatch.setattr(tiles_mod, "_prepare_boundary", fail_prepare_boundary)

    assert build_tiles(target, overwrite=False) == target
    assert target.read_bytes() == b"existing"


def test_overwrite_true_rebuilds_complete_dataset(tmp_path, monkeypatch):
    target = tmp_path / "tiles.gpkg"
    target.write_bytes(b"existing")
    _patch_boundary(monkeypatch, box(100, 100, 1100, 1100))

    path = build_tiles(target, tile_size_m=1000, overwrite=True)
    gdf = read_tiles(path)

    assert path == target
    assert len(gdf) == 4
    assert set(gdf["tile_id"]) == {
        "000000_000000_001000_001000",
        "001000_000000_002000_001000",
        "000000_001000_001000_002000",
        "001000_001000_002000_002000",
    }


@pytest.mark.parametrize("tile_size", [0, -1])
def test_build_tiles_rejects_invalid_tile_size(tmp_path, tile_size):
    with pytest.raises(ValueError, match="tile_size_m"):
        build_tiles(tmp_path / "tiles.gpkg", tile_size_m=tile_size)


def test_build_tiles_rejects_non_integer_inputs(tmp_path):
    with pytest.raises(TypeError, match="tile_size_m"):
        build_tiles(tmp_path / "tiles.gpkg", tile_size_m=2000.0)
    with pytest.raises(TypeError, match="origin_x"):
        build_tiles(tmp_path / "tiles.gpkg", origin_x=0.0)
    with pytest.raises(TypeError, match="origin_y"):
        build_tiles(tmp_path / "tiles.gpkg", origin_y=0.0)


def test_build_tiles_rejects_non_gpkg_target(tmp_path):
    with pytest.raises(ValueError, match=".gpkg"):
        build_tiles(tmp_path / "tiles.geojson")


def test_duplicate_tile_id_validation():
    gdf = gpd.GeoDataFrame(
        {
            "tile_id": ["000000_000000_002000_002000"] * 2,
            "column": [0, 0],
            "row": [0, 0],
            "xmin": [0, 0],
            "ymin": [0, 0],
            "xmax": [2000, 2000],
            "ymax": [2000, 2000],
        },
        geometry=[box(0, 0, 2000, 2000), box(0, 0, 2000, 2000)],
        crs=tiles_mod.EXPECTED_CRS,
    )

    with pytest.raises(ValueError, match="Tile IDs"):
        tiles_mod._validate_tiles(
            gdf,
            box(0, 0, 2000, 2000),
            tile_size_m=2000,
            origin_x=0,
            origin_y=0,
        )


def test_tile_from_row_and_tile_filename():
    row = {
        "tile_id": "080000_440000_082000_442000",
        "column": 40,
        "row": 220,
        "xmin": 80000,
        "ymin": 440000,
        "xmax": 82000,
        "ymax": 442000,
    }

    tile = tile_from_row(row)

    assert isinstance(tile, Tile)
    assert tile.bounds == (80000, 440000, 82000, 442000)
    assert tile.width == 2000
    assert tile.height == 2000
    assert (
        tile_filename("functioneel_landgebruik", tile)
        == "functioneel_landgebruik_080000_440000_082000_442000.tif"
    )


def test_tile_id_format_rejects_unrepresentable_coordinates():
    with pytest.raises(ValueError, match="six digits"):
        tiles_mod._tile_id(-1, 0, 2000, 2000)
    with pytest.raises(ValueError, match="six digits"):
        tiles_mod._tile_id(0, 0, 1000000, 2000)

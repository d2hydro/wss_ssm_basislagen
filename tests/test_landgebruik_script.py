import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_landgebruik_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "landgebruik.py"
    spec = importlib.util.spec_from_file_location("landgebruik_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_landgebruik_script_builds_tiles_vrt_and_cog_in_order(
    tmp_path,
    monkeypatch,
    capsys,
):
    landgebruik = _load_landgebruik_script()
    events = []
    processed_dir = tmp_path / "processed"
    tiles_path = tmp_path / "tiles.gpkg"
    tile_files = [
        processed_dir / "functioneel_landgebruik" / "tiles" / "tile-a.tif",
        processed_dir / "functioneel_landgebruik" / "tiles" / "tile-b.tif",
    ]

    def fake_build_tiles(**kwargs):
        events.append(("build_tiles", kwargs))
        return tiles_path

    def fake_build_landgebruik_tiles(**kwargs):
        events.append(("build_landgebruik_tiles", kwargs))
        return tile_files

    def fake_create_vrt_file(**kwargs):
        events.append(("create_vrt_file", kwargs))
        return kwargs["vrt_file"]

    def fake_create_cog_file(**kwargs):
        events.append(("create_cog_file", kwargs))
        return kwargs["cog_file"]

    monkeypatch.setattr(landgebruik, "datastore", SimpleNamespace(
        processed_data_dir=processed_dir
    ))
    monkeypatch.setattr(landgebruik, "_safe_workers", lambda: 2)
    monkeypatch.setattr(landgebruik, "build_tiles", fake_build_tiles)
    monkeypatch.setattr(
        landgebruik,
        "bouw_functioneel_landgebruik_tiles",
        fake_build_landgebruik_tiles,
    )
    monkeypatch.setattr(landgebruik, "create_vrt_file", fake_create_vrt_file)
    monkeypatch.setattr(landgebruik, "create_cog_file", fake_create_cog_file)

    result = landgebruik.main()

    data_dir = processed_dir / "functioneel_landgebruik"
    tiles_dir = data_dir / "tiles"
    vrt_file = data_dir / "functioneel_landgebruik.vrt"
    cog_file = data_dir / "functioneel_landgebruik.tif"

    assert [event for event, _kwargs in events] == [
        "build_tiles",
        "build_landgebruik_tiles",
        "create_vrt_file",
        "create_cog_file",
    ]
    assert events[0][1] == {"tile_size_m": 5000, "overwrite": False}
    assert events[1][1]["target_dir"] == tiles_dir
    assert events[1][1]["tiles_path"] == tiles_path
    assert events[1][1]["workers"] == 2
    assert events[1][1]["overwrite"] is False
    assert events[2][1] == {"vrt_file": vrt_file, "directory": tiles_dir}
    assert events[3][1] == {
        "vrt_file": vrt_file,
        "cog_file": cog_file,
        "overwrite": False,
    }
    assert result == cog_file

    output = capsys.readouterr().out
    assert "Tiles built: 2" in output
    assert f"VRT built: {vrt_file}" in output
    assert f"National COG built: {cog_file}" in output

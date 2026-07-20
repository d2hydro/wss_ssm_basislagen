import warnings

from waterlagen import _geopandas as wgpd


def test_read_file_suppresses_measured_geometry_warning(monkeypatch):
    def fake_read_file(*args, **kwargs):
        warnings.warn(
            "Measured (M) geometry types are not supported. "
            "Original type 'Measured 3D Polygon' is converted to 'Polygon Z'",
            UserWarning,
            stacklevel=2,
        )
        return "data"

    monkeypatch.setattr(wgpd.gpd, "read_file", fake_read_file)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = wgpd.read_file("source.gpkg")

    assert result == "data"
    assert captured == []


def test_read_file_keeps_other_user_warnings(monkeypatch):
    def fake_read_file(*args, **kwargs):
        warnings.warn("something else", UserWarning, stacklevel=2)
        return "data"

    monkeypatch.setattr(wgpd.gpd, "read_file", fake_read_file)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = wgpd.read_file("source.gpkg")

    assert result == "data"
    assert len(captured) == 1
    assert str(captured[0].message) == "something else"

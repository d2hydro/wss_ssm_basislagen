import warnings

import geopandas as gpd

MEASURED_GEOMETRY_WARNING = (
    r"Measured \(M\) geometry types are not supported\. "
    r"Original type .* is converted to .*"
)


def read_file(*args, **kwargs):
    """Read vector data while suppressing harmless M-geometry conversion warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=MEASURED_GEOMETRY_WARNING,
            category=UserWarning,
        )
        return gpd.read_file(*args, **kwargs)

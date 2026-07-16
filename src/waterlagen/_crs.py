import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pyogrio
from pyproj import CRS


class MissingCRSError(ValueError):
    """Raised when a spatial dataset layer has no source CRS."""


@dataclass(frozen=True)
class LayerCRSInfo:
    layer: str
    geometry_type: str | None
    crs: str | None

    @property
    def is_spatial(self) -> bool:
        return self.geometry_type is not None


def _canonical_crs(value: str | int | CRS) -> CRS:
    return CRS.from_user_input(value)


def _format_crs(value: str | int | CRS) -> str:
    crs = _canonical_crs(value)
    epsg = crs.to_epsg()
    if epsg is not None:
        return f"EPSG:{epsg}"
    return crs.to_string()


def _same_crs(left: str | int | CRS, right: str | int | CRS) -> bool:
    return _canonical_crs(left) == _canonical_crs(right)


def read_layer_crs_info(path: Path) -> list[LayerCRSInfo]:
    """Read CRS metadata for every layer without loading feature data."""
    path = Path(path)
    layers = pyogrio.list_layers(path)
    result: list[LayerCRSInfo] = []
    for layer in layers:
        layer_name = str(layer[0])
        info = pyogrio.read_info(path, layer=layer_name)
        result.append(
            LayerCRSInfo(
                layer=layer_name,
                geometry_type=info.get("geometry_type"),
                crs=info.get("crs"),
            )
        )
    return result


def ensure_dataset_crs(
    path: Path,
    *,
    expected_crs: str | int | CRS,
    logger=None,
) -> None:
    """Ensure all spatial layers in a GeoPackage use the expected CRS."""
    path = Path(path)
    expected_label = _format_crs(expected_crs)
    layer_infos = read_layer_crs_info(path)

    missing = [
        info.layer for info in layer_infos if info.is_spatial and info.crs is None
    ]
    if missing:
        layers = ", ".join(missing)
        raise MissingCRSError(
            f"Dataset layer(s) have no CRS: {layers}. "
            f"Cannot reproject to {expected_label} without a source CRS."
        )

    wrong = [
        info
        for info in layer_infos
        if info.is_spatial
        and info.crs is not None
        and not _same_crs(info.crs, expected_crs)
    ]
    if not wrong:
        return

    if logger is not None:
        wrong_crs_values = sorted({_format_crs(info.crs) for info in wrong if info.crs})
        detected = wrong_crs_values[0] if len(wrong_crs_values) == 1 else "mixed CRS"
        logger.warning(
            f"Dataset CRS is {detected}, expected {expected_label}. "
            "Reprojecting dataset."
        )

    _reproject_geopackage(path, layer_infos=layer_infos, expected_crs=expected_crs)


def _reproject_geopackage(
    path: Path,
    *,
    layer_infos: list[LayerCRSInfo],
    expected_crs: str | int | CRS,
) -> None:
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.reprojected.",
        suffix=".gpkg",
        dir=path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    tmp_path.unlink(missing_ok=True)

    try:
        first_layer = True
        for layer_info in layer_infos:
            if layer_info.is_spatial:
                df = pyogrio.read_dataframe(path, layer=layer_info.layer)
                if df.crs is None:
                    raise MissingCRSError(
                        f"Dataset layer '{layer_info.layer}' has no CRS. "
                        f"Cannot reproject to {_format_crs(expected_crs)} "
                        "without a source CRS."
                    )
                if not _same_crs(df.crs, expected_crs):
                    df = df.to_crs(expected_crs)
            else:
                df = pyogrio.read_dataframe(
                    path,
                    layer=layer_info.layer,
                    read_geometry=False,
                )

            pyogrio.write_dataframe(
                df,
                tmp_path,
                layer=layer_info.layer,
                driver="GPKG",
                append=not first_layer,
            )
            first_layer = False

        _validate_geopackage(tmp_path)
        _validate_spatial_layers_crs(tmp_path, expected_crs=expected_crs)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _validate_geopackage(path: Path) -> None:
    layers = pyogrio.list_layers(path)
    if len(layers) == 0:
        raise ValueError(f"{path} is not a valid GeoPackage: no layers found")


def _validate_spatial_layers_crs(path: Path, *, expected_crs: str | int | CRS) -> None:
    expected_label = _format_crs(expected_crs)
    for layer_info in read_layer_crs_info(path):
        if not layer_info.is_spatial:
            continue
        if layer_info.crs is None:
            raise MissingCRSError(
                f"Converted layer '{layer_info.layer}' has no CRS; expected {expected_label}"
            )
        if not _same_crs(layer_info.crs, expected_crs):
            raise ValueError(
                f"Converted layer '{layer_info.layer}' has CRS "
                f"{_format_crs(layer_info.crs)}; expected {expected_label}"
            )

# %%
import os
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Optional
from xml.etree import ElementTree

import pyogrio
import requests
from osgeo import gdal
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry

from waterlagen import datastore
from waterlagen._crs import (
    MissingCRSError,
    format_crs,
    read_layer_crs_info,
    same_crs,
)
from waterlagen._downloads import (
    DownloadPayloadError,
    stream_download_to_temp,
    validate_geopackage,
)
from waterlagen.logger import get_logger
from waterlagen.settings import settings

logger = get_logger(name=__name__)

ROOT_URL = "https://api.pdok.nl"
DEFAULT_FEATURETYPES = ("waterdeel", "pand")
BGT_PREDEFINED_URL = (
    "https://api.pdok.nl/lv/bgt/download/v1_0/full/predefined/bgt-gmllight-nl-nopbp.zip"
)
DEFAULT_FILENAME = "bgt.gpkg"


@dataclass(frozen=True)
class BgtCustomDownload:
    """Metadata for a custom BGT request."""

    download_dir: Path
    featuretypes: tuple[str, ...]
    request_id: str | None
    download_url: str | None


@dataclass(frozen=True)
class BgtDownload:
    """Metadata for the predefined national BGT download."""

    source_url: str
    target_path: Path
    downloaded_bytes: int
    layer_count: int
    final_crs: str
    total_size_known: bool
    total_bytes: int | None = None
    content_type: str | None = None


def _target_paths(download_dir: Path, featuretypes: Iterable[str]) -> list[Path]:
    return [download_dir / f"bgt_{featuretype}.gpkg" for featuretype in featuretypes]


def request_download(
    featuretypes: Iterable[str], poly_mask: Optional[BaseGeometry]
) -> str:
    """Make a request for a BGT download. Will respond a download request id that can be used for download

    Parameters
    ----------
    featuretypes : Iterable[str]
        BGT feature-types, e.g. ["waterdeel", "pand"]
    poly_mask : shape | None, optional
        Optional polygon-mask to use as geofilter. If not Polygon, shape-bounding box will be used.
        By default None

    Returns
    -------
    str
        BGT downloadRequestId
    """
    featuretypes = tuple(featuretypes)
    url = f"{ROOT_URL}/lv/bgt/download/v1_0/full/custom"
    body = {
        "featuretypes": featuretypes,
        "format": "citygml",
    }

    # add poly-mask
    if poly_mask is not None:
        if not isinstance(poly_mask, Polygon):
            poly_mask = box(*poly_mask.bounds)
        body["geofilter"] = poly_mask.wkt

    # post download request
    response = requests.post(url, json=body)
    response.raise_for_status()

    # return download request-id
    download_request_id = response.json()["downloadRequestId"]
    logger.debug(r"downloadRequestid: {download_request_id}")
    return download_request_id


def poll_downloadstatus(
    download_request_id: str,
    poll_interval_s: int = 5,
) -> str:
    """Poll bgt download status

    Parameters
    ----------
    download_request_id : str
        BGT downloadRequestId as response from`request_bgt_download()`
    poll_interval_s : int, optional
        Interval to poll status (seconds), by default 5

    Returns
    -------
    str
        BGT download URL
    """
    status_url = (
        f"{ROOT_URL}/lv/bgt/download/v1_0/full/custom/{download_request_id}/status"
    )

    waiting = True

    while waiting:
        response = requests.get(status_url, timeout=60)
        response.raise_for_status()

        data = response.json()
        status = data["status"]
        if status == "COMPLETED":
            waiting = False
            download_url = f"{ROOT_URL}{data['_links']['download']['href']}"
            logger.debug(f"status: {status}. download_url: {download_url}")
        else:
            logger.debug(f"status: {status}. progress: {data['progress']}")
            time.sleep(poll_interval_s)

    return download_url


def _temp_gpkg_path(target_path: Path) -> Path:
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.",
        suffix=".gpkg",
        dir=target_path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    tmp_path.unlink(missing_ok=True)
    return tmp_path


def download_to_geopackage(
    download_url: str,
    download_dir: Path,
    crs: int | str = settings.crs,
    *,
    chunk_size: int = 1024 * 1024,
    timeout: int = 30,
) -> Path:
    """Download BGT and safe as GPKG files

    Parameters
    ----------
    download_url : str
        BGT download url
    download_dir : Path
        Download dir
    crs : int | str, optional
        Expected CRS for downloaded BGT layers, by default settings.crs
    chunk_size : int, optional
        Download chunk size in bytes, by default 1 MiB
    timeout : int, optional
        Request timeout in seconds, by default 30

    Returns
    -------
    Path
        download_dir
    """
    # 0) make sure download-dir exists
    download_dir = Path(download_dir)
    download_dir.mkdir(exist_ok=True, parents=True)

    downloaded_file = None
    try:
        downloaded_file = stream_download_to_temp(
            url=download_url,
            target_path=download_dir / "bgt_custom.zip",
            suffix=".zip",
            chunk_size=chunk_size,
            timeout=timeout,
            logger=logger,
            progress=False,
        )
        _convert_bgt_zip_to_separate_geopackages(
            downloaded_file.target_path,
            download_dir,
            expected_crs=crs,
        )
    finally:
        if downloaded_file is not None:
            downloaded_file.target_path.unlink(missing_ok=True)

    return download_dir


def _validate_zip(zip_path: Path) -> list[str]:
    try:
        with zipfile.ZipFile(zip_path) as zf:
            gml_names = [n for n in zf.namelist() if n.lower().endswith(".gml")]
    except zipfile.BadZipFile as exc:
        raise DownloadPayloadError(f"{zip_path} is not a valid ZIP archive") from exc

    if not gml_names:
        raise DownloadPayloadError("Downloaded BGT ZIP contains no GML files")
    return gml_names


def _vsi_zip_member_path(zip_path: Path, member_name: str) -> str:
    return f"/vsizip/{zip_path.resolve().as_posix()}/{member_name}"


def _first_feature_member_sample(
    zf: zipfile.ZipFile,
    member_name: str,
    *,
    max_bytes: int = 64 * 1024 * 1024,
    chunk_size: int = 1024 * 1024,
) -> bytes | None:
    feature_start_tag = b"<gml:featureMember"
    feature_end_tag = b"</gml:featureMember>"
    with zf.open(member_name) as member:
        data = b""
        while len(data) < max_bytes:
            chunk = member.read(chunk_size)
            if not chunk:
                break
            data += chunk
            feature_start = data.find(feature_start_tag)
            feature_end = data.find(feature_end_tag, feature_start)
            if feature_start >= 0 and feature_end >= 0:
                feature_end += len(feature_end_tag)
                return (
                    data[:feature_start]
                    + data[feature_start:feature_end]
                    + b"</gml:FeatureCollection>"
                )
    return None


def _relax_gfs_template(gfs_text: str) -> str:
    root = ElementTree.fromstring(gfs_text)
    for feature_class in root.findall("GMLFeatureClass"):
        dataset_info = feature_class.find("DatasetSpecificInfo")
        if dataset_info is not None:
            feature_class.remove(dataset_info)
        for width in feature_class.findall("./PropertyDefn/Width"):
            width.text = "0"
    return ElementTree.tostring(root, encoding="unicode")


def _write_gfs_template_from_zip_member(
    zf: zipfile.ZipFile,
    member_name: str,
    gml_path: Path,
    *,
    source_crs: str | int,
) -> None:
    sample = _first_feature_member_sample(zf, member_name)
    if sample is None:
        return

    sample_gml = gml_path.with_name(f".{gml_path.stem}.sample.gml")
    sample_gpkg = gml_path.with_name(f".{gml_path.stem}.sample.gpkg")
    sample_gml.write_bytes(sample)
    try:
        options = gdal.VectorTranslateOptions(
            format="GPKG",
            srcSRS=format_crs(source_crs),
            dstSRS=format_crs(source_crs),
            geometryType="CONVERT_TO_LINEAR",
        )
        with gdal.ExceptionMgr(useExceptions=True):
            dataset = gdal.VectorTranslate(
                str(sample_gpkg),
                str(sample_gml),
                options=options,
            )
        if dataset is None:
            return
        dataset = None
        sample_gfs = sample_gml.with_suffix(".gfs")
        if sample_gfs.exists():
            gml_path.with_suffix(".gfs").write_text(
                _relax_gfs_template(sample_gfs.read_text(encoding="utf-8")),
                encoding="utf-8",
            )
    finally:
        sample_gml.unlink(missing_ok=True)
        sample_gml.with_suffix(".gfs").unlink(missing_ok=True)
        sample_gpkg.unlink(missing_ok=True)


def _gml_layer_source_crs(
    gml_path: Path,
    *,
    layer_name: str,
    expected_crs: str | int,
    assign_missing_crs: bool = True,
) -> str:
    info = pyogrio.read_info(gml_path)
    source_crs = info.get("crs")
    expected_label = format_crs(expected_crs)

    if source_crs is None:
        if not assign_missing_crs:
            raise MissingCRSError(
                f"Dataset layer(s) have no CRS: {layer_name}. "
                f"Cannot reproject to {expected_label} without a source CRS."
            )
        logger.warning(f"Layer {layer_name} has no CRS. Assigning {expected_label}.")
        return expected_label

    if not same_crs(source_crs, expected_crs):
        logger.warning(
            f"Layer {layer_name} CRS is {format_crs(source_crs)}, "
            f"expected {expected_label}. Reprojecting layer."
        )

    return source_crs


def _translate_gml_layer_to_geopackage(
    gml_path: Path | str,
    target_path: Path,
    *,
    layer_name: str,
    expected_crs: str | int,
    append: bool,
    assign_missing_crs: bool = True,
    source_crs: str | int | None = None,
) -> None:
    if source_crs is None:
        source_crs = _gml_layer_source_crs(
            Path(gml_path),
            layer_name=layer_name,
            expected_crs=expected_crs,
            assign_missing_crs=assign_missing_crs,
        )
    options = gdal.VectorTranslateOptions(
        format="GPKG",
        accessMode="append" if append else None,
        srcSRS=format_crs(source_crs),
        dstSRS=format_crs(expected_crs),
        layerName=layer_name,
        geometryType="CONVERT_TO_LINEAR",
    )

    with gdal.ExceptionMgr(useExceptions=True):
        dataset = gdal.VectorTranslate(str(target_path), str(gml_path), options=options)
    if dataset is None:
        raise DownloadPayloadError(f"Could not convert {gml_path} to {target_path}")
    dataset = None


def _validate_geopackage_spatial_crs(path: Path, *, expected_crs: str | int) -> None:
    validate_geopackage(path)
    expected_label = format_crs(expected_crs)
    for layer_info in read_layer_crs_info(path):
        if not layer_info.is_spatial:
            continue
        if layer_info.crs is None:
            raise MissingCRSError(
                f"Converted layer '{layer_info.layer}' has no CRS; "
                f"expected {expected_label}"
            )
        if not same_crs(layer_info.crs, expected_crs):
            raise ValueError(
                f"Converted layer '{layer_info.layer}' has CRS "
                f"{format_crs(layer_info.crs)}; expected {expected_label}"
            )


def _convert_bgt_zip_to_separate_geopackages(
    zip_path: Path,
    download_dir: Path,
    *,
    expected_crs: str | int,
) -> int:
    gml_names = _validate_zip(zip_path)
    layer_count = 0

    with tempfile.TemporaryDirectory(dir=download_dir) as extract_dir:
        extract_root = Path(extract_dir)
        with zipfile.ZipFile(zip_path) as zf:
            for gml_name in gml_names:
                layer_name = Path(gml_name).stem
                gml_path = Path(zf.extract(gml_name, path=extract_root))
                gpkg_out = download_dir / f"{layer_name}.gpkg"
                tmp_gpkg = _temp_gpkg_path(gpkg_out)
                try:
                    logger.debug(f"writing {gpkg_out}")
                    _translate_gml_layer_to_geopackage(
                        gml_path,
                        tmp_gpkg,
                        layer_name=layer_name,
                        expected_crs=expected_crs,
                        append=False,
                        assign_missing_crs=True,
                    )
                    _validate_geopackage_spatial_crs(
                        tmp_gpkg,
                        expected_crs=expected_crs,
                    )
                    tmp_gpkg.replace(gpkg_out)
                    layer_count += 1
                except Exception:
                    tmp_gpkg.unlink(missing_ok=True)
                    raise

    return layer_count


def _convert_bgt_zip_to_geopackage(
    zip_path: Path,
    target_path: Path,
    feature_types: Iterable[str],
    *,
    expected_crs: str | int,
    use_vsi_zip: bool = False,
) -> int:
    gml_names = _validate_zip(zip_path)
    feature_types = tuple(
        dict.fromkeys(str(feature_type) for feature_type in feature_types)
    )
    gml_names_by_feature_type = {}
    for gml_name in gml_names:
        gml_stem = PurePosixPath(gml_name).stem
        if gml_stem.lower().startswith("bgt_"):
            gml_names_by_feature_type.setdefault(gml_stem[4:].lower(), gml_name)
    missing_feature_types = [
        feature_type
        for feature_type in feature_types
        if feature_type.lower() not in gml_names_by_feature_type
    ]
    if missing_feature_types:
        missing_labels = ", ".join(
            f"{feature_type} (expected bgt_{feature_type}.gml)"
            for feature_type in missing_feature_types
        )
        logger.warning(
            f"BGT ZIP is missing requested feature types: {missing_labels}"
        )

    selected_gml_names = [
        gml_names_by_feature_type[feature_type.lower()]
        for feature_type in feature_types
        if feature_type.lower() in gml_names_by_feature_type
    ]
    if not selected_gml_names:
        raise DownloadPayloadError(
            "Downloaded BGT ZIP contains none of the requested feature types"
        )

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.",
        suffix=".gpkg",
        dir=target_path.parent,
    )
    os.close(fd)
    tmp_gpkg = Path(tmp_name)
    tmp_gpkg.unlink(missing_ok=True)

    try:
        with tempfile.TemporaryDirectory(dir=target_path.parent) as extract_dir:
            extract_root = Path(extract_dir)
            with zipfile.ZipFile(zip_path) as zf:
                first_layer = True
                for gml_name in selected_gml_names:
                    layer_name = PurePosixPath(gml_name).stem
                    zf.getinfo(gml_name)
                    logger.info(
                        f"Converting {gml_name} to layer {layer_name} "
                        f"in {target_path}"
                    )
                    if use_vsi_zip:
                        gml_path = _vsi_zip_member_path(zip_path, gml_name)
                        logger.debug(f"Reading {gml_name} from ZIP as {gml_path}")
                    else:
                        gml_path = Path(zf.extract(gml_name, path=extract_root))
                        logger.debug(f"Extracted {gml_name} to {gml_path}")
                        _write_gfs_template_from_zip_member(
                            zf,
                            gml_name,
                            gml_path,
                            source_crs=expected_crs,
                        )
                    _translate_gml_layer_to_geopackage(
                        gml_path,
                        layer_name=layer_name,
                        expected_crs=expected_crs,
                        target_path=tmp_gpkg,
                        append=not first_layer,
                        source_crs=expected_crs,
                    )
                    first_layer = False

        _validate_geopackage_spatial_crs(tmp_gpkg, expected_crs=expected_crs)
        layer_count = len(read_layer_crs_info(tmp_gpkg))
        tmp_gpkg.replace(target_path)
        return layer_count
    except Exception:
        tmp_gpkg.unlink(missing_ok=True)
        raise


def _geopackage_layer_count(path: Path) -> int:
    validate_geopackage(path)
    return len(read_layer_crs_info(path))


def download_bgt(
    download_dir: Path = datastore.bgt_dir,
    *,
    featuretypes: Iterable[str] = DEFAULT_FEATURETYPES,
    target_path: Path | None = None,
    overwrite: bool = True,
    url: str = BGT_PREDEFINED_URL,
    progress: bool = True,
    chunk_size: int = 1024 * 1024,
    timeout: int = 30,
) -> BgtDownload:
    """Download the predefined national BGT GML Light ZIP as one GeoPackage."""
    if target_path is None:
        target_path = Path(download_dir) / DEFAULT_FILENAME
    else:
        target_path = Path(target_path)
    target_path.parent.mkdir(exist_ok=True, parents=True)

    if target_path.exists() and not overwrite:
        layer_count = _geopackage_layer_count(target_path)
        return BgtDownload(
            source_url=url,
            target_path=target_path,
            downloaded_bytes=0,
            layer_count=layer_count,
            final_crs=format_crs(settings.crs),
            total_size_known=False,
        )

    downloaded_file = None
    try:
        downloaded_file = stream_download_to_temp(
            url=url,
            target_path=target_path,
            suffix=".zip",
            chunk_size=chunk_size,
            timeout=timeout,
            logger=logger,
            progress=progress,
        )
        layer_count = _convert_bgt_zip_to_geopackage(
            downloaded_file.target_path,
            target_path,
            feature_types=featuretypes,
            expected_crs=settings.crs,
        )
        return BgtDownload(
            source_url=url,
            target_path=target_path,
            downloaded_bytes=downloaded_file.downloaded_bytes,
            layer_count=layer_count,
            final_crs=format_crs(settings.crs),
            total_size_known=downloaded_file.total_size_known,
            total_bytes=downloaded_file.total_bytes,
            content_type=downloaded_file.content_type,
        )
    finally:
        if downloaded_file is not None:
            downloaded_file.target_path.unlink(missing_ok=True)


def bgt_custom_download(
    download_dir: Path = datastore.bgt_dir,
    *,
    featuretypes: Iterable[str] = DEFAULT_FEATURETYPES,
    poly_mask: Optional[BaseGeometry] = None,
    overwrite: bool = True,
    poll_interval_s: int = 5,
) -> BgtCustomDownload:
    """Download BGT features as GeoPackages.

    Parameters
    ----------
    download_dir : Path, optional
        Download dir to store GeoPackages. By default datastore.bgt_dir.
    featuretypes : Iterable[str], optional
        BGT feature-types, by default ("waterdeel", "pand").
    poly_mask : shape | None, optional
        Optional polygon-mask to use as geofilter. If not Polygon, shape-bounding box will be used.
        By default None
    overwrite : bool, optional
        If not True BGT will only be downloaded if all expected output files already exist.
        Default is True.
    poll_interval_s : int, optional
        Interval to poll the BGT download status in seconds, by default 5.

    Returns
    -------
    BgtCustomDownload
        Metadata for the BGT download.
    """
    featuretypes = tuple(featuretypes)
    download_dir = Path(download_dir)
    download_dir.mkdir(exist_ok=True, parents=True)

    if not overwrite and all(
        path.exists() for path in _target_paths(download_dir, featuretypes)
    ):
        return BgtCustomDownload(
            download_dir=download_dir,
            featuretypes=featuretypes,
            request_id=None,
            download_url=None,
        )

    logger.info("Requesting a BGT download")
    download_request_id = request_download(
        featuretypes=featuretypes, poly_mask=poly_mask
    )

    logger.info("Polling status of BGT download")
    download_url = poll_downloadstatus(
        download_request_id=download_request_id,
        poll_interval_s=poll_interval_s,
    )

    logger.info("Downloading result to GeoPackage")
    download_to_geopackage(download_url=download_url, download_dir=download_dir)

    return BgtCustomDownload(
        download_dir=download_dir,
        featuretypes=featuretypes,
        request_id=download_request_id,
        download_url=download_url,
    )


def get_bgt_features(
    featuretypes: Iterable[str],
    poly_mask: Optional[BaseGeometry],
    download_dir: Path,
) -> Path:
    """Download BGT features in GeoPackages.

    Use :func:`bgt_custom_download` for new custom BGT downloads.
    """
    return bgt_custom_download(
        featuretypes=featuretypes,
        poly_mask=poly_mask,
        download_dir=download_dir,
    ).download_dir


def bgt_download(*args, **kwargs) -> BgtDownload:
    """Compatibility alias for :func:`download_bgt`."""
    return download_bgt(*args, **kwargs)

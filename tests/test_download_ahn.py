# %%
import rasterio

from waterlagen.ahn import create_download_dir, download_ahn

SAMPLE_COORDS = [
    (111176.5, 516525.5),
    (113197.0, 512656.5),
    (111375.0, 513895.0),
    (113682.5, 516729.5),
    (112109.5, 512686.0),
]

EXPECTED_SAMPLES_AHN4 = [99.0, -381.0, -29.0, -99.0, -239.0]
EXPECTED_SAMPLES_AHN5 = [96.0, -388.0, -33.0, -102.0, 89.0]


def _sample_raster(vrt_file, coords):
    with rasterio.open(vrt_file) as src:
        return [float(v[0]) for v in src.sample(coords)]


def test_download_ahn4_pdok(ahn_dir):
    """test if PDOK AHN4 downloader works"""
    model = "dtm"
    cell_size = "05"
    ahn_version = 4
    select_indices = ["M_19BZ1"]
    service = "ahn_pdok"

    # make download_dir and assert existing
    download_dir = create_download_dir(
        root_dir=ahn_dir,
        model=model,
        cell_size=cell_size,
        ahn_version=ahn_version,
    )
    assert download_dir.exists()

    # download rasters and assert existing
    vrt_file = download_ahn(
        ahn_dir=download_dir,
        service=service,
        select_indices=select_indices,
        model=model,
        cell_size=cell_size,
        ahn_version=4,
        save_tiles_index=True,
        missing_only=False,
    )

    assert vrt_file.exists()
    assert vrt_file.with_name("M_19BZ1.tif").exists()
    assert vrt_file.with_suffix(".gpkg").exists()
    assert _sample_raster(vrt_file, SAMPLE_COORDS) == EXPECTED_SAMPLES_AHN4


def test_download_ahn5_ahn_nl(ahn_dir):
    """test if AHN.nl AHN5 downloader works"""
    model = "dtm"
    cell_size = "05"
    ahn_version = 5
    select_indices = ["M_19BZ1"]
    service = "ahn_nl"

    # make download_dir and assert existing
    download_dir = create_download_dir(
        root_dir=ahn_dir,
        model=model,
        cell_size=cell_size,
        ahn_version=ahn_version,
    )

    # download rasters and assert existing
    vrt_file = download_ahn(
        ahn_dir=download_dir,
        service=service,
        select_indices=select_indices,
        model=model,
        cell_size=cell_size,
        ahn_version=ahn_version,
        save_tiles_index=True,
        missing_only=False,
    )
    assert vrt_file.exists()
    assert vrt_file.with_name("M_19BZ1.tif").exists()
    assert vrt_file.with_suffix(".gpkg").exists()
    assert _sample_raster(vrt_file, SAMPLE_COORDS) == EXPECTED_SAMPLES_AHN5


def test_download_ahn6_ahn_nl(ahn_dir):
    """test if AHN.nl AHN6 downloader works"""
    model = "dtm"
    cell_size = "05"
    ahn_version = 6
    select_indices = ["249000_466000"]
    service = "ahn_nl"

    # make download_dir and assert existing
    download_dir = create_download_dir(
        root_dir=ahn_dir,
        model=model,
        cell_size=cell_size,
        ahn_version=ahn_version,
    )

    # download rasters and assert existing
    vrt_file = download_ahn(
        ahn_dir=download_dir,
        service=service,
        select_indices=select_indices,
        model=model,
        cell_size=cell_size,
        ahn_version=ahn_version,
        save_tiles_index=True,
    )
    assert vrt_file.exists()
    assert vrt_file.with_name("249000_466000.tif").exists()
    assert vrt_file.with_suffix(".gpkg").exists()

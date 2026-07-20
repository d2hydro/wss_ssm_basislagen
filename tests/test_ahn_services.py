# %%
from waterlagen.ahn.api_config import AHNService

SAMPLE_INDICES = ["M_19BZ1", "M_25BN2", "M_14EN1", "M_14EZ1"]


def test_ahn_pdok():
    """Test if AHN PDOK is online, and tile-specs haven't changed"""
    ahn_service = AHNService(service="ahn_pdok")
    gdf = ahn_service.get_tiles()

    # check if download url is still in fields
    assert ahn_service.download_url_field() in gdf.columns

    # length is still as expected
    assert len(gdf) == 1373

    # samples are in index
    assert len(gdf.loc[SAMPLE_INDICES]) == 4

    # bounds are still as expected
    assert [int(i) for i in gdf.total_bounds] == [10000, 306250, 280000, 618750]


def test_ahn5_ahn_nl():
    """Test if AHN.nl is online, and AHN < 6 tile-specs haven't changed"""
    ahn_service = AHNService(service="ahn_nl")
    gdf = ahn_service.get_tiles(ahn_version=5)

    # length is still as expected
    assert len(gdf) == 1407

    # samples are in index
    assert len(gdf.loc[SAMPLE_INDICES]) == 4

    # bounds are still as expected
    assert [int(i) for i in gdf.total_bounds] == [10000, 306250, 280000, 625000]


def test_ahn6_ahn_nl():
    """Test if AHN.nl is online, and AHN 6 tile-specs haven't changed"""
    ahn_service = AHNService(service="ahn_nl")
    gdf = ahn_service.get_tiles(ahn_version=6)

    # length is still as expected
    assert len(gdf) == 40249

    # bounds are still as expected
    assert [int(i) for i in gdf.total_bounds] == [12103, 304527, 286716, 620132]



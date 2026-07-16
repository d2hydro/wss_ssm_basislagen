# %%
from waterlagen import datastore
from waterlagen.ahn import AHNService, get_tiles_features
from waterlagen.bag import download_bag_light, get_bag_features
from waterlagen.logger import init_logger

logger = init_logger(name="bag_download", log_file=datastore.data_dir / "get_bag.log")
logger.info(f"datastore at dir {datastore}")
# %%
# download BAG in DataStore
bag_gpkg = download_bag_light(download_dir=datastore.bag_dir, overwrite=False)

# # or download bag for specific extent
ahn_indices = ["M_19BZ1"]
ahn_service = AHNService(service="ahn_pdok")
tiles_gdf = get_tiles_features(ahn_service=ahn_service, select_indices=ahn_indices)
bag_pand_gdf = get_bag_features(
    bbox=tiles_gdf.total_bounds, source="bag-light", layer="pand"
)

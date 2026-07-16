# %%
from waterlagen import datastore
from waterlagen.logger import init_logger
from waterlagen.top10nl import download_top10nl

logger = init_logger(
    name="top10NL_download", log_file=datastore.data_dir / "get_top10NL.log"
)
logger.info(f"datastore at dir {datastore}")
# %%
# download BAG in DataStore
bag_gpkg = download_top10nl(download_dir=datastore.top10nl_dir, overwrite=False)

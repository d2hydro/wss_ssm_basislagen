# %%
from waterlagen import datastore
from waterlagen.dijkringen import download_dijkringen
from waterlagen.logger import init_logger

logger = init_logger(
    name="top10NL_download", log_file=datastore.data_dir / "get_dijkringen.log"
)
logger.info(f"datastore at dir {datastore}")
# %%
# download BAG in DataStore
bag_gpkg = download_dijkringen(download_dir=datastore.dijkringen_dir, overwrite=False)

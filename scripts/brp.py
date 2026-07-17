# %%
from waterlagen import datastore
from waterlagen.brp import download_brp
from waterlagen.logger import init_logger

logger = init_logger(name="BRP_download", log_file=datastore.data_dir / "get_brp.log")
logger.info(f"datastore at dir {datastore}")
# %%
# download BRP in DataStore
brp_gpkg = download_brp(download_dir=datastore.brp_dir, overwrite=False)

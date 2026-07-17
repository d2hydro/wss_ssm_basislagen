# %%
from waterlagen import datastore
from waterlagen.bgt import download_bgt
from waterlagen.logger import init_logger

logger = init_logger(name="BGT_download", log_file=datastore.data_dir / "get_bgt.log")
logger.info(f"datastore at dir {datastore}")
# %%
# download BGT in DataStore
bgt_gpkg = download_bgt(
    featuretypes=["wegdeel", "waterdeel", "pand"],
    download_dir=datastore.bgt_dir,
    overwrite=False,
)

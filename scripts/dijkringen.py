# %%
from waterlagen import datastore
from waterlagen.dijkringen import download_dijkringen
from waterlagen.logger import init_logger

logger = init_logger(
    name="download_dijkringen", log_file=datastore.data_dir / "get_dijkringen.log"
)
logger.info(f"datastore at dir {datastore}")
# %%
# download dijkringen in DataStore
dijkringen_gpkg = download_dijkringen(
    download_dir=datastore.dijkringen_dir, overwrite=False
)

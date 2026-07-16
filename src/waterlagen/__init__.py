from importlib.metadata import PackageNotFoundError, version

from waterlagen.datastore import datastore
from waterlagen.settings import settings

__all__ = ["settings", "datastore", "__version__"]

try:
    __version__ = version("waterlagen")
except PackageNotFoundError:
    __version__ = "0+unknown"

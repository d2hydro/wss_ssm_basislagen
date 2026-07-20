import io
import warnings
from typing import Literal

import geopandas as gpd
import requests
from pydantic import BaseModel


class AHNService(BaseModel):
    service: Literal["ahn_pdok", "ahn_nl"] = "ahn_nl"

    def _validate_inputs(
        self, cell_size: Literal["05", "5"], ahn_version: Literal[3, 4, 5, 6]
    ) -> None:
        # validate cell_size (only configurable for ahn.nl)
        if cell_size not in ["05", "5"]:
            raise ValueError(f'`cell_size` should be "05" or "5" not {cell_size}')
        if cell_size == "5" and self.service == "ahn_pdok":
            raise ValueError(
                'Only `cell_size="05"` is valid input with `service`="ahn_pdok"'
            )
        if ahn_version not in [3, 4, 5, 6]:
            raise ValueError(
                f"`ahn_version={ahn_version}` is not valid. Only versions 3-6 are implemented(!)"
            )
        if ahn_version != 4 and self.service == "ahn_pdok":
            raise ValueError(
                'Only `ahn_version=4` is valid input with `service`="ahn_pdok"'
            )

    def get_tiles_url(
        self,
        model: Literal["dtm", "dsm"] = "dtm",
        cell_size: Literal["05", "5"] = "05",
        ahn_version: Literal[3, 4, 5, 6] = 4,
    ):
        """Get tiles url of AHN download service"""
        self._validate_inputs(cell_size, ahn_version)
        if self.service == "ahn_pdok":
            ahn_type = f"{model}_{cell_size}m"
            return f"https://service.pdok.nl/rws/ahn/atom/downloads/{ahn_type}/kaartbladindex.json"
        elif self.service == "ahn_nl":
            if ahn_version < 6:
                return "https://basisdata.nl/hwh-ahn/AUX/bladwijzer.gpkg"
            else:
                return "https://basisdata.nl/hwh-ahn/AUX/bladwijzer_AHN6.gpkg"
        else:
            raise ValueError(
                "Provide valid values for `model`, `cell_size` and `ahn_version`"
            )

    def download_url_field(
        self,
        model: Literal["dtm", "dsm"] = "dtm",
        cell_size: Literal["05", "5"] = "05",
        ahn_version: Literal[3, 4, 5, 6] = 4,
    ):
        self._validate_inputs(cell_size, ahn_version)
        if self.service == "ahn_pdok":
            return "url"
        if self.service == "ahn_nl":
            if ahn_version < 6:
                if model == "dtm":
                    postfix = "M"
                elif model == "dsm":
                    postfix = "R"
                else:
                    raise ValueError(
                        f"{model} invalid value for `model` (choose dtm or dsm)"
                    )
                return f"AHN{ahn_version}_{cell_size}M_{postfix}"
            else:
                return f"{model}_{cell_size}"

    def get_tiles(
        self,
        model: Literal["dtm", "dsm"] = "dtm",
        cell_size: Literal["05", "5"] = "05",
        ahn_version: Literal[3, 4, 5, 6] = 4,
    ) -> gpd.GeoDataFrame:
        """Get AHN tiles in a GeoDataFrame"""
        self._validate_inputs(cell_size, ahn_version)
        # download data
        url = self.get_tiles_url(
            model=model, cell_size=cell_size, ahn_version=ahn_version
        )
        response = requests.get(url=url)
        response.raise_for_status()

        # process into GeoDataframe
        if self.service == "ahn_pdok":
            gdf = gpd.GeoDataFrame.from_features(response.json(), crs=28992)
            gdf.set_index("kaartbladNr", inplace=True)
        elif self.service == "ahn_nl":
            if ahn_version < 6:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        ".*non conformant file extension.*",
                        RuntimeWarning,
                    )
                    gdf = gpd.read_file(io.BytesIO(response.content))
                gdf.index = "M_" + gdf.AHN
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        ".*non conformant file extension.*",
                        RuntimeWarning,
                    )
                    gdf = gpd.read_file(
                        io.BytesIO(response.content), layer="bladindeling_aoi"
                    )
                root_url = "https://fsn1.your-objectstorage.com/hwh-ahn/AHN6/"
                gdf["dtm_05"] = (
                    root_url
                    + "02a_DTM_50cm/AHN"
                    + str(ahn_version)
                    + "_2025_M_"
                    + gdf["bladnaam"]
                    + ".TIF"
                )
                gdf["dtm_5"] = (
                    root_url
                    + "02b_DTM_5m/AHN"
                    + str(ahn_version)
                    + "_2025_M5_"
                    + gdf["bladnaam"]
                    + ".TIF"
                )
                gdf["dsm_05"] = (
                    root_url
                    + "03a_DSM_50cm/AHN"
                    + str(ahn_version)
                    + "_2025_R_"
                    + gdf["bladnaam"]
                    + ".TIF"
                )
                gdf["dsm_5"] = (
                    root_url
                    + "03b_DSM_5m/AHN"
                    + str(ahn_version)
                    + "_2025_R5_"
                    + gdf["bladnaam"]
                    + ".TIF"
                )
                gdf.set_index("bladnaam", inplace=True)

        gdf.index.name = "kaart_index"
        return gdf

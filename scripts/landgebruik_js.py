# %%
"""
Oorspronkelijk script uit Bachelor Eindwerk Jesper Stam (5840929)
Titel: Samengestelde Landgebruikskaart voor het berekenen van schade bij overstromingen
Datum: 19 juni 2026

Aanpassingen:
- feature: automatisch downloaden bag, bgt, brp, en dijkringen naar datastore
- feature: verwijzingen naar datastore i.p.v. lokale bestanden
- fix:  classify_weg wegen["functie"] vervangen voor wegen["bgt-functie"] (kolom functie bestaat niet)
"""

from dataclasses import dataclass
from functools import partial
from math import ceil
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio import features

from waterlagen.bag import download_bag_light
from waterlagen.bgt import download_bgt
from waterlagen.brp import download_brp
from waterlagen.dijkringen import download_dijkringen
from waterlagen.settings import settings

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DATA_DIR = SCRIPT_DIR / "data" / "source_data"
PROCESSED_DATA_DIR = SCRIPT_DIR / "data" / "processed_data"


FUNCTIONEEL_GEBIED_CODES = {
    "sportparken": 91,
    "volkstuinen": 92,
    "begraafplaatsen": 93,
    "parken": 94,
    "bedrijventerreinen": 95,
    "overige functionelegebieden": 96,
    "sportparken buitendijks": 218,
    "volkstuinen buitendijks": 219,
    "begraafplaatsen buitendijks": 220,
    "parken buitendijks": 221,
    "bedrijventerreinen buitendijks": 222,
    "overige functionelegebieden buitendijks": 223,
}

BRP_CODES = {
    "gras": 61,
    "agrarisch gras": 61,
    "mais snij": 62,
    "fruit": 63,
    "boomkwekerijgewassen": 64,
    "granen": 65,
    "akkerbouwgewassen": 66,
    "aardappelen": 67,
    "bloemkwekerijgewassen": 68,
    "overige gewassen": 69,
    "gras buitendijks": 188,
    "agrarisch gras buitendijks": 188,
    "mais snij buitendijks": 189,
    "fruit buitendijks": 190,
    "boomkwekerijgewassen buitendijks": 191,
    "granen buitendijks": 192,
    "akkerbouwgewassen buitendijks": 193,
    "aardappelen buitendijks": 194,
    "bloemkwekerijgewassen buitendijks": 195,
    "overige gewassen buitendijks": 196,
}

WEG_CODES = {
    "spoorwegen": 31,
    "primaire wegen": 32,
    "secundaire wegen": 33,
    "tertiaire wegen": 34,
    "overige wegen": 35,
    "spoorwegen buitendijks": 158,
    "primaire wegen buitendijks": 159,
    "secundaire wegen buitendijks": 160,
    "tertiaire wegen buitendijks": 161,
    "overige wegen buitendijks": 162,
}

BAG_CODES = {
    "woonfunctie 1 verdieping": 2,
    "woonfunctie 2 verdiepingen": 3,
    "woonfunctie 3 verdiepingen": 4,
    "appartementencomplex": 5,
    "winkelfunctie": 6,
    "kantoorfunctie": 7,
    "industriefunctie": 8,
    "onderwijsfunctie": 9,
    "gezondheidszorgfunctie": 10,
    "sportfunctie": 11,
    "bijeenkomstfunctie": 12,
    "overige gebruiksfunctie": 13,
    "logiesfunctie": 14,
    "woonfunctie 1 verdieping buitendijks": 129,
    "woonfunctie 2 verdiepingen buitendijks": 130,
    "woonfunctie 3 verdiepingen buitendijks": 131,
    "appartementencomplex buitendijks": 132,
    "winkelfunctie buitendijks": 133,
    "kantoorfunctie buitendijks": 134,
    "industriefunctie buitendijks": 135,
    "onderwijsfunctie buitendijks": 136,
    "gezondheidszorgfunctie buitendijks": 137,
    "sportfunctie buitendijks": 138,
    "bijeenkomstfunctie buitendijks": 139,
    "overige gebruiksfunctie buitendijks": 140,
    "logiesfunctie buitendijks": 141,
}

COLORMAP = {
    1: (0, 130, 255),
    2: (255, 255, 0),
    3: (255, 175, 0),
    4: (255, 125, 0),
    5: (220, 200, 40),
    6: (255, 255, 140),
    7: (230, 210, 20),
    8: (125, 125, 0),
    9: (180, 180, 0),
    10: (235, 220, 80),
    11: (200, 180, 60),
    12: (240, 220, 40),
    13: (200, 200, 100),
    14: (140, 130, 30),
    31: (150, 0, 0),
    32: (200, 0, 0),
    33: (255, 0, 0),
    34: (255, 60, 60),
    35: (255, 120, 120),
    61: (0, 50, 0),
    62: (0, 100, 0),
    63: (0, 150, 0),
    64: (0, 200, 0),
    65: (0, 250, 0),
    66: (60, 255, 60),
    67: (100, 255, 100),
    68: (140, 255, 140),
    69: (200, 255, 200),
    91: (255, 0, 255),
    92: (180, 100, 180),
    93: (255, 90, 255),
    94: (180, 30, 180),
    95: (100, 0, 100),
    96: (240, 160, 240),
    129: (255, 255, 0),
    130: (255, 175, 0),
    131: (255, 125, 0),
    132: (220, 200, 40),
    133: (255, 255, 140),
    134: (230, 210, 20),
    135: (125, 125, 0),
    136: (180, 180, 0),
    137: (235, 220, 80),
    138: (204, 180, 60),
    139: (240, 220, 40),
    140: (200, 200, 100),
    141: (140, 130, 30),
    158: (150, 0, 0),
    159: (200, 0, 0),
    160: (255, 0, 0),
    161: (255, 60, 60),
    162: (255, 120, 120),
    188: (0, 50, 0),
    189: (0, 100, 0),
    190: (0, 150, 0),
    191: (0, 200, 0),
    192: (0, 250, 0),
    193: (60, 255, 60),
    194: (100, 255, 100),
    195: (140, 255, 140),
    196: (180, 255, 180),
    218: (255, 0, 255),
    219: (180, 100, 180),
    220: (255, 90, 255),
    221: (180, 30, 180),
    222: (100, 0, 100),
    223: (240, 160, 240),
}


@dataclass(frozen=True)
class LandgebruikConfig:
    bgt_gpkg: Path
    bag_gpkg: Path
    brp_gpkg: Path
    top10nl_gpkg: Path
    dijkringen_gpkg: Path
    output_tif: Path
    bounds: tuple[float, float, float, float]
    resolution_m: float
    crs: str
    bgt_water_layer: str = "bgt_waterdeel"
    bgt_wegdeel_layer: str = "bgt_wegdeel"
    bag_pand_layer: str = "pand"
    bag_verblijfsobject_layer: str = "verblijfsobject"
    brp_layer: str = "brp_gewas"
    top10nl_functioneel_gebied_layer: str = "top10nl_functioneel_gebied_vlak"
    dijkringen_layer: str = "dijkring_v_2012"


DEFAULT_CONFIG = LandgebruikConfig(
    bgt_gpkg=SOURCE_DATA_DIR / "bgt" / "bgt.gpkg",
    bag_gpkg=SOURCE_DATA_DIR / "bag" / "bag-light.gpkg",
    brp_gpkg=SOURCE_DATA_DIR / "brp" / "brpgewaspercelen_definitief_2025.gpkg",
    top10nl_gpkg=SOURCE_DATA_DIR / "top10nl" / "top10nl_Compleet.gpkg",
    dijkringen_gpkg=SOURCE_DATA_DIR / "dijkringen" / "dijkringen_historie_2012.gpkg",
    output_tif=PROCESSED_DATA_DIR / "20260528_TUDelft-BSc.tiff",
    bounds=(159261.0, 437220.0, 176475.0, 452279.0),
    resolution_m=0.5,
    crs=settings.crs,
)


def download_sources(config: LandgebruikConfig) -> None:
    """Download source datasets needed for the land-use raster if missing."""
    if not config.bgt_gpkg.exists():
        download_bgt(
            download_dir=config.bgt_gpkg.parent,
            target_path=config.bgt_gpkg,
            featuretypes=[
                config.bgt_water_layer.removeprefix("bgt_"),
                config.bgt_wegdeel_layer.removeprefix("bgt_"),
            ],
            overwrite=False,
        )

    if not config.bag_gpkg.exists():
        download_bag_light(download_dir=config.bag_gpkg.parent, overwrite=False)

    if not config.brp_gpkg.exists():
        download_brp(
            download_dir=config.brp_gpkg.parent,
            filename=config.brp_gpkg.name,
            overwrite=False,
        )

    if not config.dijkringen_gpkg.exists():
        download_dijkringen(
            download_dir=config.dijkringen_gpkg.parent,
            target_path=config.dijkringen_gpkg,
            overwrite=False,
        )


def classify_functionelegebieden(type_fb: object) -> str:
    fb = str(type_fb).lower()
    if any(
        x in fb for x in ["tennispark", "sportcomplex", "circuit", "ijsbaan", "sport"]
    ):
        return "sportparken"
    if any(x in fb for x in ["volkstuinen", "botanische tuin", "heemtuin", "tuin"]):
        return "volkstuinen"
    if any(x in fb for x in ["begraafplaats", "erebegraafplaats"]):
        return "begraafplaatsen"
    if any(x in fb for x in ["park", "landgoed"]):
        return "parken"
    if "bedrijventerrein" in fb:
        return "bedrijventerreinen"
    return "overige functionelegebieden"


def classify_weg(bgt_functie: object) -> str | None:
    weg = str(bgt_functie).lower()
    if "spoorbaan" in weg:
        return "spoorwegen"
    if any(x in weg for x in ["autosnelweg", "autoweg"]):
        return "primaire wegen"
    if any(x in weg for x in ["regionale", "ov"]):
        return "secundaire wegen"
    if any(x in weg for x in ["inrit", "rijbaan lokale weg", "overweg", "woonerf"]):
        return "tertiaire wegen"
    if any(
        x in weg
        for x in [
            "fietspad",
            "voetgangersgebied",
            "voetpad",
            "parkeervlak",
            "ruiterpad",
        ]
    ):
        return "overige wegen"
    return None


def classify_brp_gewas(gewas: object, category: object | None = None) -> str:
    gewas = str(gewas).lower()
    if "gras" in gewas:
        return "agrarisch gras"
    if "aardappel" in gewas:
        return "Aardappelen"
    if "mais" in gewas:
        return "mais snij"
    if any(
        x in gewas
        for x in ["granen", "tarwe", "gerst", "rogge", "haver", "zaad", "triticale"]
    ):
        return "granen"
    if any(
        x in gewas
        for x in [
            "bloem",
            "rozen",
            "chrysant",
            "tulp",
            "lelie",
            "hyacint",
            "narcis",
            "roos",
        ]
    ):
        return "bloemkwekerijgewassen"
    if any(
        x in gewas
        for x in ["chester", "boom", "bomen", "vaste plant", "buxus", "conifeer"]
    ):
        return "boomkwekerijgewassen"
    if any(
        x in gewas
        for x in [
            "appel",
            "peer",
            "kers",
            "fruit",
            "bramen",
            "frambozen",
            "pruimen",
            "vrucht",
            "bessen",
            "noot",
        ]
    ):
        return "fruit"
    if any(
        x in gewas
        for x in [
            "sla",
            "gewas",
            "kool",
            "wortel",
            "groente",
            "erwten",
            "aardbeien",
            "asperges",
            "biet",
            "ui",
            "pompoen",
            "prei",
        ]
    ):
        return "akkerbouwgewassen"
    return "overige gewassen"


def eerste_gebruiksdoel(gebruiksdoel: object) -> str:
    if pd.isna(gebruiksdoel):
        return "overige gebruiksfunctie"

    gebruiksdoel = str(gebruiksdoel).strip()
    if gebruiksdoel == "" or gebruiksdoel.lower() == "nan":
        return "overige gebruiksfunctie"

    return gebruiksdoel.split(",")[0].strip()


def koppeling_hoofdfunctie_aan_panden(
    panden: gpd.GeoDataFrame,
    verblijfsobjecten: gpd.GeoDataFrame,
    pand_id_col: str = "identificatie",
    vbo_pand_id_col: str = "pand_identificatie",
    gebruiksdoel_col: str = "gebruiksdoel",
    oppervlakte_col: str = "oppervlakte",
    status_col: str = "status",
) -> gpd.GeoDataFrame:
    panden = panden.copy()
    vbo = verblijfsobjecten.copy()

    if status_col in panden.columns:
        status = panden[status_col].fillna("").str.lower()
        panden = panden[
            ~status.str.contains("bouwvergunning|sloopvergunning", na=False)
        ].copy()

    if vbo.empty:
        panden["hoofdfunctie"] = "overige gebruiksfunctie"
        return panden

    vbo["vbo_hoofdfunctie"] = vbo[gebruiksdoel_col].apply(eerste_gebruiksdoel)
    vbo[vbo_pand_id_col] = vbo[vbo_pand_id_col].fillna("").astype(str).str.split(",")
    vbo = vbo.explode(vbo_pand_id_col)
    vbo[vbo_pand_id_col] = vbo[vbo_pand_id_col].str.strip()
    vbo = vbo[vbo[vbo_pand_id_col] != ""].copy()
    vbo[oppervlakte_col] = pd.to_numeric(vbo[oppervlakte_col], errors="coerce").fillna(
        0
    )

    functie_per_pand = vbo.groupby(
        [vbo_pand_id_col, "vbo_hoofdfunctie"],
        as_index=False,
    )[oppervlakte_col].sum()
    idx = functie_per_pand.groupby(vbo_pand_id_col)[oppervlakte_col].idxmax()
    hoofdfunctie_per_pand = functie_per_pand.loc[idx].copy()
    hoofdfunctie_per_pand = hoofdfunctie_per_pand.rename(
        columns={
            vbo_pand_id_col: pand_id_col,
            "vbo_hoofdfunctie": "hoofdfunctie",
            oppervlakte_col: "hoofdfunctie_oppervlakte",
        }
    )

    panden = panden.merge(
        hoofdfunctie_per_pand[
            [pand_id_col, "hoofdfunctie", "hoofdfunctie_oppervlakte"]
        ],
        on=pand_id_col,
        how="left",
    )
    panden["hoofdfunctie"] = panden["hoofdfunctie"].fillna("overige gebruiksfunctie")
    return panden


def verfijn_woonfunctie_panden(
    panden: gpd.GeoDataFrame,
    verblijfsobjecten: gpd.GeoDataFrame,
    pand_id_col: str = "identificatie",
    vbo_pand_id_col: str = "pand_identificatie",
    gebruiksdoel_col: str = "gebruiksdoel",
    oppervlakte_col: str = "oppervlakte",
) -> gpd.GeoDataFrame:
    panden = panden.copy()
    vbo = verblijfsobjecten.copy()
    vbo["vbo_hoofdfunctie"] = vbo[gebruiksdoel_col].apply(eerste_gebruiksdoel)

    vbo_woon = vbo[vbo["vbo_hoofdfunctie"] == "woonfunctie"].copy()
    if vbo_woon.empty:
        panden["woon_oppervlakte"] = 0
        panden["aantal_woon_vbo"] = 0
        panden["footprint_m2"] = panden.geometry.area
        panden["aantal_verdiepingen"] = np.nan
        return panden

    vbo_woon[vbo_pand_id_col] = (
        vbo_woon[vbo_pand_id_col].fillna("").astype(str).str.split(",")
    )
    vbo_woon = vbo_woon.explode(vbo_pand_id_col)
    vbo_woon[vbo_pand_id_col] = vbo_woon[vbo_pand_id_col].str.strip()
    vbo_woon[oppervlakte_col] = pd.to_numeric(
        vbo_woon[oppervlakte_col],
        errors="coerce",
    ).fillna(0)

    woon_info = (
        vbo_woon.groupby(vbo_pand_id_col)
        .agg(
            woon_oppervlakte=(oppervlakte_col, "sum"),
            aantal_woon_vbo=(oppervlakte_col, "count"),
        )
        .reset_index()
        .rename(columns={vbo_pand_id_col: pand_id_col})
    )

    panden = panden.merge(woon_info, on=pand_id_col, how="left")
    panden["woon_oppervlakte"] = panden["woon_oppervlakte"].fillna(0)
    panden["aantal_woon_vbo"] = panden["aantal_woon_vbo"].fillna(0)
    panden["footprint_m2"] = panden.geometry.area
    panden["aantal_verdiepingen"] = np.ceil(
        panden["woon_oppervlakte"] / panden["footprint_m2"]
    )
    panden["aantal_verdiepingen"] = panden["aantal_verdiepingen"].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    mask_woon = panden["hoofdfunctie"] == "woonfunctie"
    panden.loc[
        mask_woon & (panden["aantal_woon_vbo"] >= 5),
        "hoofdfunctie",
    ] = "appartementencomplex"
    panden.loc[
        mask_woon
        & (panden["aantal_woon_vbo"] < 5)
        & (panden["aantal_verdiepingen"] <= 1),
        "hoofdfunctie",
    ] = "woonfunctie 1 verdieping"
    panden.loc[
        mask_woon
        & (panden["aantal_woon_vbo"] < 5)
        & (panden["aantal_verdiepingen"] == 2),
        "hoofdfunctie",
    ] = "woonfunctie 2 verdiepingen"
    panden.loc[
        mask_woon
        & (panden["aantal_woon_vbo"] < 5)
        & (panden["aantal_verdiepingen"] >= 3),
        "hoofdfunctie",
    ] = "woonfunctie 3 verdiepingen"
    return panden


def voeg_dijkligging_toe(
    objecten: gpd.GeoDataFrame,
    dijkringen: gpd.GeoDataFrame,
    functiekolom: str,
) -> gpd.GeoDataFrame:
    objecten = objecten.copy()
    dijkringen = dijkringen.copy()

    if objecten.empty:
        objecten["binnendijks"] = pd.Series(dtype=bool)
        return objecten

    if objecten.crs != dijkringen.crs:
        dijkringen = dijkringen.to_crs(objecten.crs)

    dijkring_geom = dijkringen.geometry.make_valid().union_all()
    checkpunten = objecten.geometry.representative_point()
    objecten["binnendijks"] = checkpunten.apply(
        lambda punt: punt.covered_by(dijkring_geom)
    )
    mask_buiten = ~objecten["binnendijks"]
    objecten.loc[mask_buiten, functiekolom] = (
        objecten.loc[mask_buiten, functiekolom].astype(str) + " buitendijks"
    )
    return objecten


def create_profile(config: LandgebruikConfig) -> dict[str, Any]:
    minx, miny, maxx, maxy = config.bounds
    resolution = config.resolution_m
    width = ceil((maxx - minx) / resolution)
    height = ceil((maxy - miny) / resolution)
    transform = rio.transform.from_bounds(minx, miny, maxx, maxy, width, height)
    return rio.profiles.DefaultGTiffProfile(
        count=1,
        dtype="uint8",
        nodata=0,
        compress="PACKBITS",
        width=width,
        height=height,
        blockxsize=4000,
        blockysize=4000,
        transform=transform,
        crs=config.crs,
    )


def rasterize_codes(rasterize: Any, data: gpd.GeoDataFrame) -> None:
    shapes = data[["geometry", "code"]].dropna().values.tolist()
    if shapes:
        rasterize(shapes)


def add_functionele_gebieden(
    rasterize: Any,
    paths: LandgebruikConfig,
    dijkringen: gpd.GeoDataFrame,
    window_bounds: tuple[float, float, float, float],
    config: LandgebruikConfig,
) -> None:
    fgebied = gpd.read_file(
        paths.top10nl_gpkg,
        layer=config.top10nl_functioneel_gebied_layer,
        bbox=window_bounds,
        columns=["typefunctioneelgebied", "geometry"],
    )
    fgebied["cat_fb"] = fgebied["typefunctioneelgebied"].apply(
        classify_functionelegebieden
    )
    fgebied = voeg_dijkligging_toe(fgebied, dijkringen, "cat_fb")
    fgebied["code"] = fgebied["cat_fb"].map(FUNCTIONEEL_GEBIED_CODES)
    rasterize_codes(rasterize, fgebied)


def add_brp(
    rasterize: Any,
    paths: LandgebruikConfig,
    dijkringen: gpd.GeoDataFrame,
    window_bounds: tuple[float, float, float, float],
    config: LandgebruikConfig,
) -> None:
    brp = gpd.read_file(
        paths.brp_gpkg,
        layer=config.brp_layer,
        bbox=window_bounds,
        columns=["gewas", "category", "geometry"],
    )
    brp = brp[brp["category"].str.lower() != "landschapselement"].copy()
    brp["brp_cat"] = brp.apply(
        lambda row: classify_brp_gewas(row["gewas"], row["category"]),
        axis=1,
    )
    brp = voeg_dijkligging_toe(brp, dijkringen, "brp_cat")
    brp["code"] = brp["brp_cat"].map(BRP_CODES)
    rasterize_codes(rasterize, brp)


def add_water(
    rasterize: Any,
    paths: LandgebruikConfig,
    window_bounds: tuple[float, float, float, float],
    config: LandgebruikConfig,
) -> None:
    water = gpd.read_file(
        paths.bgt_gpkg,
        layer=config.bgt_water_layer,
        bbox=window_bounds,
        columns=["bgt-status", "geometry"],
    )
    water["code"] = water["bgt-status"].map({"bestaand": 1})
    rasterize_codes(rasterize, water)


def add_wegen(
    rasterize: Any,
    paths: LandgebruikConfig,
    dijkringen: gpd.GeoDataFrame,
    window_bounds: tuple[float, float, float, float],
    config: LandgebruikConfig,
) -> None:
    wegen = gpd.read_file(
        paths.bgt_gpkg,
        layer=config.bgt_wegdeel_layer,
        bbox=window_bounds,
        columns=["bgt-functie", "geometry"],
    )
    wegen["cat_weg"] = wegen["bgt-functie"].apply(classify_weg)
    wegen = voeg_dijkligging_toe(wegen, dijkringen, "cat_weg")
    wegen["code"] = wegen["cat_weg"].map(WEG_CODES)
    rasterize_codes(rasterize, wegen)


def add_bag(
    rasterize: Any,
    paths: LandgebruikConfig,
    dijkringen: gpd.GeoDataFrame,
    window_bounds: tuple[float, float, float, float],
    config: LandgebruikConfig,
) -> None:
    bag_panden = gpd.read_file(
        paths.bag_gpkg,
        layer=config.bag_pand_layer,
        bbox=window_bounds,
        columns=["identificatie", "status", "geometry"],
    )
    bag_vbo = gpd.read_file(
        paths.bag_gpkg,
        layer=config.bag_verblijfsobject_layer,
        bbox=window_bounds,
        columns=["pand_identificatie", "gebruiksdoel", "oppervlakte", "geometry"],
    )
    bag = koppeling_hoofdfunctie_aan_panden(bag_panden, bag_vbo)
    bag = verfijn_woonfunctie_panden(bag, bag_vbo)
    bag = voeg_dijkligging_toe(bag, dijkringen, "hoofdfunctie")
    bag["hoofdfunctie"] = bag["hoofdfunctie"].astype(str).str.strip().str.lower()
    bag["code"] = bag["hoofdfunctie"].map(BAG_CODES)
    rasterize_codes(rasterize, bag)


def write_landgebruik_raster(config: LandgebruikConfig) -> Path:
    paths = config
    profile = create_profile(config)
    approx_blocks = ceil(profile["width"] / profile["blockxsize"]) * ceil(
        profile["height"] / profile["blockysize"]
    )
    paths.output_tif.parent.mkdir(parents=True, exist_ok=True)
    dijkringen = gpd.read_file(
        paths.dijkringen_gpkg,
        layer=config.dijkringen_layer,
    )

    with rio.open(paths.output_tif, "w", **profile) as r_out:
        for block_number, (window_index, window) in enumerate(
            r_out.block_windows(),
            start=1,
        ):
            window_bounds = rio.windows.bounds(window, r_out.transform)
            print(
                f"{block_number:4d}/{approx_blocks}",
                window_index,
                window_bounds,
                sep="\t",
            )

            out = np.full((window.height, window.width), r_out.nodata)
            rasterize = partial(
                features.rasterize,
                out=out,
                transform=rio.windows.transform(window, r_out.transform),
                all_touched=True,
                merge_alg=rio.enums.MergeAlg.replace,
            )

            add_functionele_gebieden(
                rasterize,
                paths,
                dijkringen,
                window_bounds,
                config,
            )
            add_brp(rasterize, paths, dijkringen, window_bounds, config)
            add_water(rasterize, paths, window_bounds, config)
            add_wegen(rasterize, paths, dijkringen, window_bounds, config)
            add_bag(rasterize, paths, dijkringen, window_bounds, config)

            r_out.write_band(1, out, window=window)

        r_out.set_band_description(1, "Landgebruik")
        r_out.colorinterp = (rio.enums.ColorInterp.palette,)
        r_out.write_colormap(1, COLORMAP)

    return paths.output_tif


download_sources(DEFAULT_CONFIG)
write_landgebruik_raster(DEFAULT_CONFIG)

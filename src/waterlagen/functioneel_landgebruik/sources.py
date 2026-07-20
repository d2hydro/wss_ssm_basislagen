from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry.base import BaseGeometry

from waterlagen import _geopandas as wgpd
from waterlagen.functioneel_landgebruik.classifiers import (
    classify_brp_gewas,
    classify_functioneel_gebied,
    classify_weg,
    eerste_gebruiksdoel,
)
from waterlagen.functioneel_landgebruik.legend import (
    BAG_CODES,
    BRP_CODES,
    FUNCTIONEEL_GEBIED_CODES,
    WEG_CODES,
)


def _with_code(
    gdf: gpd.GeoDataFrame,
    *,
    category_column: str,
    codes: dict[str, int],
) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gpd.GeoDataFrame({"code": pd.Series(dtype="uint8")}, geometry=[], crs=gdf.crs)

    result = gdf.copy()
    result[category_column] = result[category_column].astype(str).str.strip()
    result["code"] = result[category_column].map(codes)
    result = result.dropna(subset=["code", "geometry"]).copy()
    result["code"] = result["code"].astype("uint8")
    return result[["geometry", "code"]]


def classify_inside_dikes(
    gdf: gpd.GeoDataFrame,
    dike_area: BaseGeometry,
    *,
    category_column: str,
) -> gpd.GeoDataFrame:
    """Append ' buitendijks' to categories outside the prepared dike area."""
    result = gdf.copy()
    if result.empty:
        result["binnendijks"] = pd.Series(dtype=bool)
        return result

    check_points = result.geometry.representative_point()
    result["binnendijks"] = check_points.apply(lambda point: point.covered_by(dike_area))
    outside = ~result["binnendijks"]
    result.loc[outside, category_column] = (
        result.loc[outside, category_column].astype(str) + " buitendijks"
    )
    return result


def prepare_functionele_gebieden(
    top10nl_gpkg: Path,
    *,
    layer: str,
    bounds: tuple[float, float, float, float],
    dike_area: BaseGeometry,
) -> gpd.GeoDataFrame:
    gdf = wgpd.read_file(
        top10nl_gpkg,
        layer=layer,
        bbox=bounds,
        columns=["typefunctioneelgebied", "geometry"],
    )
    gdf["categorie"] = gdf["typefunctioneelgebied"].apply(classify_functioneel_gebied)
    gdf = classify_inside_dikes(gdf, dike_area, category_column="categorie")
    return _with_code(
        gdf,
        category_column="categorie",
        codes=FUNCTIONEEL_GEBIED_CODES,
    )


def prepare_brp(
    brp_gpkg: Path,
    *,
    layer: str,
    bounds: tuple[float, float, float, float],
    dike_area: BaseGeometry,
) -> gpd.GeoDataFrame:
    gdf = wgpd.read_file(
        brp_gpkg,
        layer=layer,
        bbox=bounds,
        columns=["gewas", "category", "geometry"],
    )
    if gdf.empty:
        return gpd.GeoDataFrame({"code": pd.Series(dtype="uint8")}, geometry=[], crs=gdf.crs)

    category = gdf["category"].fillna("").astype(str).str.lower()
    gdf = gdf[category != "landschapselement"].copy()
    gdf["categorie"] = gdf.apply(
        lambda row: classify_brp_gewas(row["gewas"], row["category"]),
        axis=1,
    )
    gdf = classify_inside_dikes(gdf, dike_area, category_column="categorie")
    return _with_code(gdf, category_column="categorie", codes=BRP_CODES)


def prepare_water(
    bgt_gpkg: Path,
    *,
    layer: str,
    bounds: tuple[float, float, float, float],
) -> gpd.GeoDataFrame:
    gdf = wgpd.read_file(
        bgt_gpkg,
        layer=layer,
        bbox=bounds,
        columns=["bgt-status", "geometry"],
    )
    if gdf.empty:
        return gpd.GeoDataFrame({"code": pd.Series(dtype="uint8")}, geometry=[], crs=gdf.crs)

    gdf["code"] = gdf["bgt-status"].map({"bestaand": 1})
    gdf = gdf.dropna(subset=["code", "geometry"]).copy()
    gdf["code"] = gdf["code"].astype("uint8")
    return gdf[["geometry", "code"]]


def prepare_wegen(
    bgt_gpkg: Path,
    *,
    layer: str,
    bounds: tuple[float, float, float, float],
    dike_area: BaseGeometry,
) -> gpd.GeoDataFrame:
    gdf = wgpd.read_file(
        bgt_gpkg,
        layer=layer,
        bbox=bounds,
        columns=["bgt-functie", "geometry"],
    )
    gdf["categorie"] = gdf["bgt-functie"].apply(classify_weg)
    gdf = gdf.dropna(subset=["categorie"]).copy()
    gdf = classify_inside_dikes(gdf, dike_area, category_column="categorie")
    return _with_code(gdf, category_column="categorie", codes=WEG_CODES)


def _koppel_hoofdfunctie_aan_panden(
    panden: gpd.GeoDataFrame,
    verblijfsobjecten: gpd.GeoDataFrame,
    *,
    pand_id_col: str = "identificatie",
    vbo_pand_id_col: str = "pand_identificatie",
    gebruiksdoel_col: str = "gebruiksdoel",
    oppervlakte_col: str = "oppervlakte",
    status_col: str = "status",
) -> gpd.GeoDataFrame:
    panden = panden.copy()
    vbo = verblijfsobjecten.copy()

    if status_col in panden.columns:
        status = panden[status_col].fillna("").astype(str).str.lower()
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


def _verfijn_woonfunctie_panden(
    panden: gpd.GeoDataFrame,
    verblijfsobjecten: gpd.GeoDataFrame,
    *,
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


def prepare_bag(
    bag_gpkg: Path,
    *,
    pand_layer: str,
    verblijfsobject_layer: str,
    bounds: tuple[float, float, float, float],
    dike_area: BaseGeometry,
) -> gpd.GeoDataFrame:
    panden = wgpd.read_file(
        bag_gpkg,
        layer=pand_layer,
        bbox=bounds,
        columns=["identificatie", "status", "geometry"],
    )
    vbo = wgpd.read_file(
        bag_gpkg,
        layer=verblijfsobject_layer,
        bbox=bounds,
        columns=["pand_identificatie", "gebruiksdoel", "oppervlakte", "geometry"],
    )
    if panden.empty:
        return gpd.GeoDataFrame({"code": pd.Series(dtype="uint8")}, geometry=[], crs=panden.crs)

    bag = _koppel_hoofdfunctie_aan_panden(panden, vbo)
    bag = _verfijn_woonfunctie_panden(bag, vbo)
    bag = classify_inside_dikes(bag, dike_area, category_column="hoofdfunctie")
    bag["hoofdfunctie"] = bag["hoofdfunctie"].astype(str).str.strip().str.lower()
    return _with_code(bag, category_column="hoofdfunctie", codes=BAG_CODES)

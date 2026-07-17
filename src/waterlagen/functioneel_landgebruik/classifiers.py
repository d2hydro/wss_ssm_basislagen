import pandas as pd


def classify_functioneel_gebied(type_functioneel_gebied: object) -> str:
    value = str(type_functioneel_gebied).lower()
    if any(
        token in value
        for token in ["tennispark", "sportcomplex", "circuit", "ijsbaan", "sport"]
    ):
        return "sportparken"
    if any(token in value for token in ["volkstuinen", "botanische tuin", "heemtuin", "tuin"]):
        return "volkstuinen"
    if any(token in value for token in ["begraafplaats", "erebegraafplaats"]):
        return "begraafplaatsen"
    if any(token in value for token in ["park", "landgoed"]):
        return "parken"
    if "bedrijventerrein" in value:
        return "bedrijventerreinen"
    return "overige functionelegebieden"


def classify_weg(bgt_functie: object) -> str | None:
    value = str(bgt_functie).lower()
    if "spoorbaan" in value:
        return "spoorwegen"
    if any(token in value for token in ["autosnelweg", "autoweg"]):
        return "primaire wegen"
    if any(token in value for token in ["regionale", "ov"]):
        return "secundaire wegen"
    if any(token in value for token in ["inrit", "rijbaan lokale weg", "overweg", "woonerf"]):
        return "tertiaire wegen"
    if any(
        token in value
        for token in [
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
    value = str(gewas).lower()
    if "gras" in value:
        return "agrarisch gras"
    if "aardappel" in value:
        return "Aardappelen"
    if "mais" in value:
        return "mais snij"
    if any(
        token in value
        for token in ["granen", "tarwe", "gerst", "rogge", "haver", "zaad", "triticale"]
    ):
        return "granen"
    if any(
        token in value
        for token in [
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
        token in value
        for token in ["chester", "boom", "bomen", "vaste plant", "buxus", "conifeer"]
    ):
        return "boomkwekerijgewassen"
    if any(
        token in value
        for token in [
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
        token in value
        for token in [
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

    value = str(gebruiksdoel).strip()
    if value == "" or value.lower() == "nan":
        return "overige gebruiksfunctie"
    return value.split(",")[0].strip().lower()

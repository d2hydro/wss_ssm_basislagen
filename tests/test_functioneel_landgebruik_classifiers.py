import pytest

from waterlagen.functioneel_landgebruik.classifiers import (
    classify_brp_gewas,
    classify_functioneel_gebied,
    classify_weg,
    eerste_gebruiksdoel,
)
from waterlagen.functioneel_landgebruik.legend import BRP_CODES


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Sportcomplex", "sportparken"),
        ("Volkstuinen", "volkstuinen"),
        ("Begraafplaats", "begraafplaatsen"),
        ("Stadspark", "parken"),
        ("Bedrijventerrein", "bedrijventerreinen"),
        ("Onbekend", "overige functionelegebieden"),
    ],
)
def test_classify_functioneel_gebied(value, expected):
    assert classify_functioneel_gebied(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("spoorbaan", "spoorwegen"),
        ("autosnelweg", "primaire wegen"),
        ("regionale weg", "secundaire wegen"),
        ("rijbaan lokale weg", "tertiaire wegen"),
        ("fietspad", "overige wegen"),
        ("onbekend", None),
    ],
)
def test_classify_weg(value, expected):
    assert classify_weg(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("blijvend grasland", "agrarisch gras"),
        ("consumptieaardappelen", "Aardappelen"),
        ("mais snijmais", "mais snij"),
        ("wintertarwe", "granen"),
        ("tulpen", "bloemkwekerijgewassen"),
        ("appelboomgaard", "boomkwekerijgewassen"),
        ("peren", "overige gewassen"),
        ("uien", "akkerbouwgewassen"),
        ("onbekend", "overige gewassen"),
    ],
)
def test_classify_brp_gewas(value, expected):
    assert classify_brp_gewas(value) == expected


def test_aardappelen_matches_original_landgebruik_js_behavior():
    assert classify_brp_gewas("aardappelen") == "Aardappelen"
    assert classify_brp_gewas("aardappelen") not in BRP_CODES


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("woonfunctie,kantoorfunctie", "woonfunctie"),
        ("", "overige gebruiksfunctie"),
        (None, "overige gebruiksfunctie"),
        ("Winkelfunctie", "winkelfunctie"),
    ],
)
def test_eerste_gebruiksdoel(value, expected):
    assert eerste_gebruiksdoel(value) == expected

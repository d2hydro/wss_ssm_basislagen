from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from waterlagen._downloads import validate_geopackage
from waterlagen.brp.download import (
    ATOM_FEED_URL,
    BrpFeedError,
    BrpGeoPackageNotFoundError,
    download_brp,
    find_latest_brp_geopackage,
)


class FakeResponse:
    def __init__(self, payload: bytes, *, error: Exception | None = None):
        self.content = payload
        self.payload = payload
        self.error = error
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size: int):
        midpoint = max(1, len(self.payload) // 2)
        yield self.payload[:midpoint]
        if self.error is not None:
            raise self.error
        yield self.payload[midpoint:]


def _atom_feed(entries: str) -> bytes:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:dcterms="http://purl.org/dc/terms/">
  <title>BRP</title>
  {entries}
</feed>
""".encode()


def _entry(
    title: str,
    href: str,
    *,
    published: str | None = None,
    updated: str | None = None,
    issued: str | None = None,
    link_type: str = "application/geopackage+sqlite3",
    extra_link: str = "",
) -> str:
    published_xml = f"<published>{published}</published>" if published else ""
    updated_xml = f"<updated>{updated}</updated>" if updated else ""
    issued_xml = f"<dcterms:issued>{issued}</dcterms:issued>" if issued else ""
    return f"""
  <entry>
    <title>{title}</title>
    {published_xml}
    {updated_xml}
    {issued_xml}
    {extra_link}
    <link rel="enclosure" type="{link_type}" href="{href}" />
  </entry>
"""


def _mock_feed(monkeypatch, payload: bytes):
    def fake_get(url, **kwargs):
        assert url == ATOM_FEED_URL
        assert kwargs == {"timeout": 30}
        return FakeResponse(payload)

    monkeypatch.setattr("waterlagen.brp.download.requests.get", fake_get)


def _valid_gpkg_bytes(path: Path) -> bytes:
    gpkg_path = path / "valid.gpkg"
    gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    )
    gdf.to_file(gpkg_path, driver="GPKG", layer="sample")
    payload = gpkg_path.read_bytes()
    validate_geopackage(gpkg_path)
    return payload


def _mock_feed_and_download(
    monkeypatch,
    *,
    feed: bytes,
    payload: bytes,
    payload_error: Exception | None = None,
):
    def fake_get(url, **kwargs):
        if url == ATOM_FEED_URL:
            return FakeResponse(feed)
        return FakeResponse(payload, error=payload_error)

    monkeypatch.setattr("waterlagen.brp.download.requests.get", fake_get)


def _temp_downloads_for(target: Path) -> list[Path]:
    return list(target.parent.glob(f".{target.name}.*.gpkg"))


def test_find_latest_brp_geopackage_selects_latest_publication(monkeypatch):
    feed = _atom_feed(
        _entry(
            "BRP Gewaspercelen 2023 definitief",
            "https://example.com/brp_2023.gpkg",
            published="2024-01-01T00:00:00Z",
        )
        + _entry(
            "BRP Gewaspercelen 2024 definitief",
            "https://example.com/brp_2024.gpkg",
            updated="2025-01-01T00:00:00Z",
        )
    )
    _mock_feed(monkeypatch, feed)

    release = find_latest_brp_geopackage()

    assert release.download_url == "https://example.com/brp_2024.gpkg"
    assert release.filename == "brp_2024.gpkg"
    assert release.version == "2024"
    assert release.publication_or_update_date is not None


def test_find_latest_brp_geopackage_handles_xml_namespaces(monkeypatch):
    feed = _atom_feed(
        _entry(
            "BRP 2024",
            "downloads/brp_2024.gpkg",
            issued="2024-02-01",
        )
    )
    _mock_feed(monkeypatch, feed)

    release = find_latest_brp_geopackage()

    assert release.download_url == (
        "https://service.pdok.nl/rvo/gewaspercelen/atom/downloads/brp_2024.gpkg"
    )
    assert release.publication_or_update_date is not None


def test_find_latest_brp_geopackage_ignores_element_order(monkeypatch):
    feed = _atom_feed(
        _entry(
            "BRP Gewaspercelen 2025",
            "https://example.com/brp_2025.gpkg",
            published="2026-01-01T00:00:00Z",
        )
        + _entry(
            "BRP Gewaspercelen 2024",
            "https://example.com/brp_2024.gpkg",
            published="2025-01-01T00:00:00Z",
        )
    )
    _mock_feed(monkeypatch, feed)

    release = find_latest_brp_geopackage()

    assert release.filename == "brp_2025.gpkg"


def test_find_latest_brp_geopackage_ignores_non_geopackage_links(monkeypatch):
    feed = _atom_feed(
        _entry(
            "BRP Gewaspercelen 2025",
            "https://example.com/brp_2025.xml",
            published="2026-01-01T00:00:00Z",
            link_type="application/xml",
        )
        + _entry(
            "BRP Gewaspercelen 2024",
            "https://example.com/brp_2024.gpkg",
            published="2025-01-01T00:00:00Z",
        )
    )
    _mock_feed(monkeypatch, feed)

    release = find_latest_brp_geopackage()

    assert release.filename == "brp_2024.gpkg"


def test_find_latest_brp_geopackage_ignores_concept_link_in_same_entry(monkeypatch):
    feed = _atom_feed(
        _entry(
            "Basisregistratie Gewaspercelen BRP Geopackage",
            "https://example.com/downloads/gewaspercelen_concept_2026.gpkg",
            updated="2026-06-26T09:32:35Z",
            extra_link=(
                "<link rel=\"section\" type=\"application/geopackage+sqlite3\" "
                "href=\"https://example.com/downloads/brpgewaspercelen_definitief_2025.gpkg\" "
                "title=\"BRP definitief 2025\" />"
            ),
        )
    )
    _mock_feed(monkeypatch, feed)

    release = find_latest_brp_geopackage()

    assert release.filename == "brpgewaspercelen_definitief_2025.gpkg"
    assert release.version == "2025"
    assert release.version != "2026-06"


def test_find_latest_brp_geopackage_uses_version_when_dates_missing(monkeypatch):
    feed = _atom_feed(
        _entry("BRP Gewaspercelen 2023", "https://example.com/brp_2023.gpkg")
        + _entry("BRP Gewaspercelen 2025", "https://example.com/brp_2025.gpkg")
        + _entry("BRP Gewaspercelen 2024", "https://example.com/brp_2024.gpkg")
    )
    _mock_feed(monkeypatch, feed)

    release = find_latest_brp_geopackage()

    assert release.filename == "brp_2025.gpkg"
    assert release.publication_or_update_date is None
    assert release.version == "2025"


def test_find_latest_brp_geopackage_feed_without_geopackage(monkeypatch):
    feed = _atom_feed(
        _entry(
            "BRP Gewaspercelen 2025",
            "https://example.com/brp_2025.xml",
            published="2026-01-01T00:00:00Z",
            link_type="application/xml",
        )
    )
    _mock_feed(monkeypatch, feed)

    with pytest.raises(BrpGeoPackageNotFoundError, match="No BRP GeoPackage"):
        find_latest_brp_geopackage()


def test_find_latest_brp_geopackage_malformed_xml(monkeypatch):
    _mock_feed(monkeypatch, b"<feed><entry>")

    with pytest.raises(BrpFeedError, match="malformed XML"):
        find_latest_brp_geopackage()


def test_find_latest_brp_geopackage_unexpected_xml(monkeypatch):
    _mock_feed(monkeypatch, b"<html><body>error</body></html>")

    with pytest.raises(BrpFeedError, match="unexpected XML content"):
        find_latest_brp_geopackage()


def test_download_brp_success(monkeypatch, tmp_path):
    gpkg_payload = _valid_gpkg_bytes(tmp_path)
    feed = _atom_feed(
        _entry(
            "BRP Gewaspercelen 2025",
            "https://example.com/brp_2025.gpkg",
            published="2026-01-01T00:00:00Z",
        )
    )
    _mock_feed_and_download(monkeypatch, feed=feed, payload=gpkg_payload)

    result = download_brp(download_dir=tmp_path)

    assert result.path == tmp_path / "brp_2025.gpkg"
    assert result.release.filename == "brp_2025.gpkg"
    assert result.release.version == "2025"
    validate_geopackage(result.path)
    assert _temp_downloads_for(result.path) == []


@pytest.mark.parametrize(
    "payload",
    [
        b"not a geopackage",
        b"<html><body>server error</body></html>",
        b"<?xml version='1.0'?><error>server error</error>",
    ],
)
def test_download_brp_rejects_invalid_payloads(monkeypatch, tmp_path, payload):
    existing_payload = _valid_gpkg_bytes(tmp_path)
    target = tmp_path / "custom.gpkg"
    target.write_bytes(existing_payload)
    feed = _atom_feed(
        _entry(
            "BRP Gewaspercelen 2025",
            "https://example.com/brp_2025.gpkg",
            published="2026-01-01T00:00:00Z",
        )
    )
    _mock_feed_and_download(monkeypatch, feed=feed, payload=payload)

    with pytest.raises(ValueError, match="not a valid GeoPackage"):
        download_brp(download_dir=tmp_path, filename=target.name)

    assert target.read_bytes() == existing_payload
    assert _temp_downloads_for(target) == []


def test_download_brp_interrupted_download_preserves_existing_target(
    monkeypatch, tmp_path
):
    existing_payload = _valid_gpkg_bytes(tmp_path)
    target = tmp_path / "brp_2025.gpkg"
    target.write_bytes(existing_payload)
    feed = _atom_feed(
        _entry(
            "BRP Gewaspercelen 2025",
            "https://example.com/brp_2025.gpkg",
            published="2026-01-01T00:00:00Z",
        )
    )
    _mock_feed_and_download(
        monkeypatch,
        feed=feed,
        payload=existing_payload,
        payload_error=ConnectionError("interrupted"),
    )

    with pytest.raises(ConnectionError, match="interrupted"):
        download_brp(download_dir=tmp_path)

    assert target.read_bytes() == existing_payload
    assert _temp_downloads_for(target) == []

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse
from xml.etree import ElementTree

import requests

from waterlagen import datastore
from waterlagen._downloads import download_geopackage
from waterlagen.logger import get_logger
from waterlagen.settings import settings

logger = get_logger(name=__name__)

ATOM_FEED_URL = (
    "https://service.pdok.nl/rvo/gewaspercelen/atom/"
    "basisregistratie_gewaspercelen_brp.xml"
)
NON_DEFINITIVE_TERMS = ("concept", "preview", "proef", "test", "voorlopig")


class BrpFeedError(ValueError):
    """Raised when the BRP Atom feed cannot be parsed as expected."""


class BrpGeoPackageNotFoundError(BrpFeedError):
    """Raised when no GeoPackage download can be found in the BRP Atom feed."""


@dataclass(frozen=True)
class BrpRelease:
    """Metadata for a BRP GeoPackage release discovered in the Atom feed."""

    download_url: str
    filename: str
    publication_or_update_date: datetime | None
    version: str | None


@dataclass(frozen=True)
class BrpDownload:
    """Metadata for a downloaded BRP release."""

    path: Path
    release: BrpRelease


def _local_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]


def _first_child_text(element: ElementTree.Element, names: tuple[str, ...]) -> str | None:
    for name in names:
        for child in element:
            if _local_name(child.tag) == name and child.text:
                return child.text.strip()
    return None


def _descendant_text(element: ElementTree.Element, names: tuple[str, ...]) -> str | None:
    for name in names:
        for child in element.iter():
            if _local_name(child.tag) == name and child.text:
                return child.text.strip()
    return None


def _parse_atom_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        try:
            return datetime.fromisoformat(f"{normalized}T00:00:00")
        except ValueError:
            return None


def _entry_date(entry: ElementTree.Element) -> datetime | None:
    value = _first_child_text(entry, ("published", "updated"))
    if value is None:
        value = _descendant_text(entry, ("issued", "modified", "date"))
    return _parse_atom_datetime(value)


def _entry_text(entry: ElementTree.Element) -> str:
    parts = []
    for element in entry.iter():
        if element.text:
            parts.append(element.text)
    for link in _entry_links(entry):
        parts.extend(
            value for value in (link.get("href"), link.get("title"), link.get("type")) if value
        )
    return " ".join(parts)


def _contains_non_definitive_term(text: str) -> bool:
    text = text.lower()
    for term in NON_DEFINITIVE_TERMS:
        if term == "test":
            if re.search(r"\btest\w*\b", text):
                return True
        elif term in text:
            return True
    return False


def _link_text(link: ElementTree.Element) -> str:
    return " ".join(
        value
        for value in (link.get("href"), link.get("title"), link.get("type"))
        if value
    )


def _entry_links(entry: ElementTree.Element) -> list[ElementTree.Element]:
    return [element for element in entry.iter() if _local_name(element.tag) == "link"]


def _looks_like_geopackage_link(link: ElementTree.Element) -> bool:
    href = link.get("href")
    if not href:
        return False

    values = [
        href,
        link.get("type") or "",
        link.get("title") or "",
    ]
    text = " ".join(values).lower()
    parsed = urlparse(href)
    path = unquote(parsed.path).lower()
    return (
        path.endswith(".gpkg")
        or ".gpkg" in path
        or "geopackage" in text
        or "gpkg" in text
    )


def _filename_from_url(url: str) -> str:
    filename = Path(unquote(urlparse(url).path)).name
    if not filename:
        raise BrpFeedError(f"BRP GeoPackage URL has no filename: {url}")
    return filename


def _version_from_text(text: str) -> str | None:
    match = re.search(r"(?<!\d)(?:v)?(20\d{2}(?:[._-]?\d{1,2}){0,3})(?!\d)", text)
    if match is None:
        return None
    return match.group(1)


def _release_version(
    filename: str,
    link_text: str,
    entry: ElementTree.Element,
) -> str | None:
    title = _first_child_text(entry, ("title",)) or ""
    content = _first_child_text(entry, ("content", "summary")) or ""
    return (
        _version_from_text(filename)
        or _version_from_text(link_text)
        or _version_from_text(f"{title} {content}")
    )


def _version_key(version: str | None) -> tuple[int, ...]:
    if version is None:
        return ()
    return tuple(int(value) for value in re.findall(r"\d+", version))


def _candidate_sort_key(release: BrpRelease) -> tuple[int, datetime, tuple[int, ...]]:
    if release.publication_or_update_date is not None:
        return (1, release.publication_or_update_date, _version_key(release.version))
    return (0, datetime.min, _version_key(release.version))


def _parse_brp_releases(xml_content: bytes | str, feed_url: str) -> list[BrpRelease]:
    try:
        root = ElementTree.fromstring(xml_content)
    except ElementTree.ParseError as exc:
        raise BrpFeedError("BRP Atom feed is malformed XML") from exc

    if _local_name(root.tag) != "feed":
        raise BrpFeedError("BRP Atom feed has unexpected XML content")

    releases: list[BrpRelease] = []
    for entry in root.iter():
        if _local_name(entry.tag) != "entry":
            continue

        publication_or_update_date = _entry_date(entry)
        for link in _entry_links(entry):
            link_text = _link_text(link)
            if (
                not _looks_like_geopackage_link(link)
                or _contains_non_definitive_term(link_text)
            ):
                continue

            download_url = urljoin(feed_url, link.get("href", ""))
            filename = _filename_from_url(download_url)
            version = _release_version(filename, link_text, entry)
            releases.append(
                BrpRelease(
                    download_url=download_url,
                    filename=filename,
                    publication_or_update_date=publication_or_update_date,
                    version=version,
                )
            )

    return releases


def find_latest_brp_geopackage(
    feed_url: str = ATOM_FEED_URL,
    *,
    timeout: int = 30,
) -> BrpRelease:
    """Find the latest definitive BRP GeoPackage download in the Atom feed."""
    response = requests.get(feed_url, timeout=timeout)
    response.raise_for_status()

    releases = _parse_brp_releases(response.content, feed_url=feed_url)
    if not releases:
        raise BrpGeoPackageNotFoundError(
            f"No BRP GeoPackage download found in Atom feed: {feed_url}"
        )

    return max(releases, key=_candidate_sort_key)


def download_brp(
    download_dir: Path = datastore.brp_dir,
    *,
    filename: str | None = None,
    overwrite: bool = True,
    feed_url: str = ATOM_FEED_URL,
) -> BrpDownload:
    """Download the latest definitive BRP GeoPackage from the Atom feed."""
    release = find_latest_brp_geopackage(feed_url=feed_url)

    download_dir = Path(download_dir)
    target_filename = filename or release.filename
    target_path = download_dir / target_filename
    path = download_geopackage(
        url=release.download_url,
        target_path=target_path,
        overwrite=overwrite,
        logger=logger,
        expected_crs=settings.crs,
    )
    return BrpDownload(path=path, release=release)

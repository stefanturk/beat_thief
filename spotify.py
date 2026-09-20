#!/usr/bin/env python3
"""What a Spotify link says the song is.

Spotify never hands over a file - what it streams is Widevine-encrypted and
only ever decrypted inside an authorized client, so no downloader can take
audio from it and this one doesn't try. What a Spotify link can do is name the
song precisely: title, artist, and exact duration. That's enough to go and
find the same recording on YouTube, which is where the audio actually comes
from (see youtube_match.py). Spotify is the shopping list.

The metadata comes from the public embed page - the little player anyone can
drop into a blog post. It needs no login, no API key and no registration from
whoever is running this, which is the whole reason it was chosen over the
official Web API: nothing to configure. The cost is that the JSON it carries
is Spotify's own internal shape, undocumented and free to change. Everything
that touches that shape is confined to _embed_entity, and it fails by saying
so in plain English rather than by raising a KeyError into someone's face.

If it ever does break, the official Web API returns the same three fields and
would replace track() and playlist() without anything above them noticing."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

EMBED_URL = "https://open.spotify.com/embed/{kind}/{spotify_id}"

# Spotify serves a different (and much less useful) page to clients that
# don't look like browsers.
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

_TIMEOUT_SEC = 15

# How many rows the embed page has been observed to list for a playlist. The
# entity carries no total to check that against, so a longer playlist would
# come back truncated with nothing to say it had been. Callers that hit this
# number exactly are told to treat the list as possibly incomplete - see
# pipeline._resolve_spotify.
EMBED_ROW_LIMIT = 50

# open.spotify.com/track/<id>, the /intl-de/ localized variants, and the
# spotify:track:<id> URIs that the desktop app's "Copy Spotify URI" produces.
# The ?si= tracking parameter falls off with the rest of the query string.
_LINK = re.compile(
    r"open\.spotify\.com/(?:intl-[a-z-]+/)?(track|playlist|album)/([A-Za-z0-9]+)",
    re.IGNORECASE,
)
_URI = re.compile(r"spotify:(track|playlist|album):([A-Za-z0-9]+)", re.IGNORECASE)

_NEXT_DATA = re.compile(
    r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.DOTALL
)


class SpotifyUnavailable(Exception):
    """Spotify's page couldn't be read - offline, a bad link, or the page
    changed shape. Carries a message meant to be shown to a person."""


def spotify_id(url: str) -> tuple[str, str] | None:
    """("track", "4uLU6hMCjMI75M1A2tKUQC") for a Spotify link, or None for
    anything else. How callers decide whether this module is involved at all -
    a YouTube link returns None and takes the ordinary path."""
    if not url:
        return None
    for pattern in (_LINK, _URI):
        match = pattern.search(url)
        if match:
            return match.group(1).lower(), match.group(2)
    return None


def _embed_entity(kind: str, spotify_id_: str) -> dict:
    """The entity blob behind the public embed player.

    The one place that knows Spotify's internal page shape. Every failure -
    network, HTTP, JSON, a moved key - comes out of here as a
    SpotifyUnavailable a person can read."""
    url = EMBED_URL.format(kind=kind, spotify_id=spotify_id_)
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SEC) as response:
            html = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise SpotifyUnavailable(
                "Spotify doesn't have anything at that link - check it's the "
                "right one, and that it isn't a private playlist."
            ) from e
        raise SpotifyUnavailable(f"Spotify wouldn't answer (HTTP {e.code}).") from e
    except Exception as e:
        raise SpotifyUnavailable(f"Couldn't reach Spotify: {e}") from e

    return _entity_from_html(html)


def _entity_from_html(html: str) -> dict:
    """Split out from _embed_entity so the tests can feed it a saved page
    instead of going to the network."""
    match = _NEXT_DATA.search(html)
    if not match:
        raise SpotifyUnavailable(
            "Spotify's page isn't in the shape this app knows how to read - "
            "they've probably changed it."
        )
    try:
        data = json.loads(match.group(1))
        entity = data["props"]["pageProps"]["state"]["data"]["entity"]
    except Exception as e:
        raise SpotifyUnavailable(
            "Spotify's page isn't in the shape this app knows how to read - "
            "they've probably changed it."
        ) from e
    if not isinstance(entity, dict) or not entity:
        raise SpotifyUnavailable(
            "Spotify's page didn't include the song details this app needs."
        )
    return entity


def _artist_names(entity: dict) -> str:
    """"Rick Astley", or "Rick Astley, Elton John" for a collaboration."""
    artists = entity.get("artists") or []
    names = [a.get("name", "") for a in artists if isinstance(a, dict)]
    names = [n for n in names if n]
    if names:
        return ", ".join(names)
    # Playlist rows carry the artist as a plain "subtitle" instead.
    return entity.get("subtitle", "") or ""


def _tidy(text: str) -> str:
    """Spotify's own strings, fit to be typed into a search box.

    Playlist rows separate collaborators with a non-breaking space -
    "Stealth,\xa0The Dap-Kings" - which travels all the way into the YouTube
    query as a character YouTube has no reason to match on."""
    return re.sub(r"\s+", " ", (text or "").replace("\xa0", " ")).strip()


def _song(title: str, artist: str, duration_ms) -> dict:
    title, artist = _tidy(title), _tidy(artist)
    try:
        duration_sec = float(duration_ms) / 1000.0
    except (TypeError, ValueError):
        duration_sec = None
    return {
        "title": title,
        "artist": artist,
        "duration_sec": duration_sec,
        # What gets typed into YouTube's search box. Artist first because
        # that's what pins down which recording of a common title it is.
        "query": f"{artist} {title}".strip(),
    }


def track(url: str) -> dict:
    """{"title", "artist", "duration_sec", "query"} for one Spotify track."""
    found = spotify_id(url)
    if not found:
        raise SpotifyUnavailable("That isn't a Spotify link.")
    kind, spotify_id_ = found
    if kind != "track":
        raise SpotifyUnavailable("That Spotify link isn't a single track.")
    entity = _embed_entity(kind, spotify_id_)
    song = _song(
        entity.get("title") or entity.get("name") or "",
        _artist_names(entity),
        entity.get("duration"),
    )
    if not song["title"]:
        raise SpotifyUnavailable("Spotify's page didn't name that track.")
    return song


def collection(url: str) -> dict:
    """{"name", "songs"} for any Spotify link - a playlist, an album, or a
    single track read as a list of one.

    The name is what the playlist or album is called, which is what a whole
    queue of songs gets filed under. A track link has no such name: one song
    doesn't need a folder built for it."""
    found = spotify_id(url)
    if not found:
        raise SpotifyUnavailable("That isn't a Spotify link.")
    kind, spotify_id_ = found
    if kind == "track":
        return {"name": "", "songs": [track(url)]}
    entity = _embed_entity(kind, spotify_id_)
    rows = entity.get("trackList") or []
    songs = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        title = row.get("title") or ""
        if not title:
            continue
        songs.append(_song(title, row.get("subtitle", "") or "", row.get("duration")))
    if not songs:
        raise SpotifyUnavailable(
            "Spotify's page didn't list any tracks for that link - if it's a "
            "private playlist, this app can't see it."
        )
    return {"name": _tidy(entity.get("title") or entity.get("name") or ""),
            "songs": songs}


def playlist(url: str) -> list[dict]:
    """Every track on a Spotify playlist or album, same shape as track().

    An album is a track list too, and its embed carries the same trackList,
    so it takes this path rather than needing a concept of its own. A track
    link returns a list of one, which is what lets callers treat every
    Spotify link the same way.

    Beware EMBED_ROW_LIMIT: exactly that many rows may mean the page stopped
    listing rather than the playlist ending."""
    return collection(url)["songs"]

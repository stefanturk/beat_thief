#!/usr/bin/env python3
"""Remembers which link each downloaded song came from.

Isolating a stem is slow and it's normal to come back to a song later wanting
another one - drums today, bass tomorrow. Without this you'd have to go and
find the YouTube link again, which is the annoying part, since the mp3 on disk
says nothing about where it came from.

Kept in ~/Library/Application Support, not next to the music: it's bookkeeping
rather than something to look at, and burying it in the music folder would
just be clutter (and would be lost the moment a folder is moved or renamed).

Only the link and the song's path are stored. Which files exist for a song is
never recorded - that's read from disk when asked, so deleting a stem in
Finder is immediately true here too."""

from __future__ import annotations

import json
import os
import time

HISTORY_PATH = os.path.join(
    os.path.expanduser("~"), "Library", "Application Support", "Beat Thief", "history.json"
)

# Enough to cover "songs you were working on lately" without the file growing
# without bound. Oldest entries fall off the end.
MAX_ENTRIES = 200


def _resolve(path: str | None) -> str:
    """Read HISTORY_PATH at call time rather than binding it as a default
    argument, so it can be pointed elsewhere (tests do) and actually take
    effect."""
    return HISTORY_PATH if path is None else path


def _load(path: str | None = None) -> list[dict]:
    path = _resolve(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except (OSError, ValueError):
        return []
    return entries if isinstance(entries, list) else []


def _save(entries: list[dict], path: str | None = None) -> None:
    path = _resolve(path)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entries[:MAX_ENTRIES], f, indent=2)
    except OSError:
        # Remembering is a convenience - never let it take down a run that
        # has already produced the files someone asked for.
        pass


def remember(url: str, song_paths, path: str | None = None) -> None:
    """Record that these songs came from this link, newest first.

    Re-downloading a song moves it back to the top and updates its link
    rather than adding a second entry for it."""
    if not url:
        return
    song_paths = [p for p in song_paths if p]
    if not song_paths:
        return

    entries = _load(path)
    fresh = {os.path.abspath(p) for p in song_paths}
    entries = [e for e in entries if e.get("song") not in fresh]

    now = time.time()
    new = [{"song": os.path.abspath(p), "url": url, "time": now} for p in song_paths]
    _save(new + entries, path)


def entries(path: str | None = None) -> list[dict]:
    """Every remembered song, newest first, skipping any whose mp3 is gone."""
    return [e for e in _load(path) if e.get("song") and os.path.exists(e["song"])]


def url_for(song_path: str, path: str | None = None) -> str:
    """The link a song came from, or "" if it isn't remembered - e.g. a song
    downloaded before this existed, or dropped into the folder by hand."""
    target = os.path.abspath(song_path)
    for entry in _load(path):
        if entry.get("song") == target:
            return entry.get("url", "")
    return ""

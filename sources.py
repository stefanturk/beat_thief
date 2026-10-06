#!/usr/bin/env python3
"""Where every song in a folder came from, and how far a run got.

Two files per folder of downloads - a playlist's folder, or the library
itself for songs that came one at a time:

  .beat_thief_sources.json  the record a run picks up from. Keyed by the
                            song's own id at its source (a Spotify track
                            id, or a YouTube video id), so pasting the same
                            playlist again skips straight past every song
                            already finished, without asking the network
                            about any of them.
  Sources.csv               the same thing for a person: which file is
                            which song, and which YouTube upload it was
                            taken from. Opens in Numbers or Excel.

A 5,000-song playlist is days of downloading, and it will be stopped -
by a closed lid, a lost connection, or somebody wanting their laptop back.
What makes that cheap is this file being written after every song.

Kept to json and csv on purpose: pipeline imports this, and pipeline is
the part of the app that has to start without torch."""

from __future__ import annotations

import csv
import json
import os
import re

MANIFEST_FILENAME = ".beat_thief_sources.json"
CSV_FILENAME = "Sources.csv"

CSV_COLUMNS = ["#", "Song", "Artist", "BPM", "File", "Matched YouTube title",
               "Channel", "YouTube link", "Spotify link", "Note"]

_YOUTUBE_ID = re.compile(r"(?:[?&]v=|youtu\.be/|/shorts/)([A-Za-z0-9_-]{6,})")
_BPM_IN_NAME = re.compile(r"\((\d+(?:\.\d+)?) BPM\)")


def youtube_key(url: str) -> str:
    """"youtube:dQw4w9WgXcQ" for a YouTube link, or the link itself for
    anything that doesn't carry a video id."""
    match = _YOUTUBE_ID.search(url or "")
    return f"youtube:{match.group(1)}" if match else (url or "")


def spotify_key(track_id: str) -> str:
    return f"spotify:{track_id}"


class Sources:
    """The record for one folder. Cheap to open: a folder nobody has
    downloaded into yet just has nothing in it."""

    def __init__(self, folder: str):
        self.folder = folder
        self.songs: dict[str, dict] = {}
        try:
            with open(os.path.join(folder, MANIFEST_FILENAME), encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict) and isinstance(loaded.get("songs"), dict):
                self.songs = loaded["songs"]
        except (OSError, ValueError):
            pass

    def get(self, key: str) -> dict:
        return dict(self.songs.get(key) or {})

    def finished(self, key: str) -> str:
        """Where this song's mp3 is, if a run already finished it and it's
        still there - "" otherwise, which means: do it (again)."""
        record = self.songs.get(key) or {}
        if record.get("status") != "done" or not record.get("file"):
            return ""
        path = os.path.join(self.folder, record["file"])
        return path if os.path.isfile(path) else ""

    def record(self, key: str, **fields) -> None:
        """Update one song and write both files straight away. A path in
        "file" is stored relative to the folder, so the folder can be moved
        or renamed without the record going stale."""
        if not key:
            return
        if fields.get("file") and os.path.isabs(fields["file"]):
            fields["file"] = os.path.relpath(fields["file"], self.folder)
        entry = self.songs.setdefault(key, {})
        entry.update({k: v for k, v in fields.items() if v is not None})
        self.save()

    def save(self) -> None:
        try:
            os.makedirs(self.folder, exist_ok=True)
            manifest = os.path.join(self.folder, MANIFEST_FILENAME)
            tmp = manifest + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"songs": self.songs}, f, indent=1, ensure_ascii=False)
            os.replace(tmp, manifest)
            self._write_csv()
        except OSError:
            # Losing the record costs a re-check next time, not a song.
            pass

    def _write_csv(self) -> None:
        rows = sorted(self.songs.items(),
                      key=lambda kv: (kv[1].get("number") or 10**9, kv[1].get("title") or ""))
        path = os.path.join(self.folder, CSV_FILENAME)
        tmp = path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)
            for key, song in rows:
                file = song.get("file") or ""
                bpm = _BPM_IN_NAME.search(os.path.basename(file))
                note = song.get("note") or ""
                if song.get("check") and not note:
                    note = "Check this match - it was a guess"
                writer.writerow([
                    song.get("number") or "",
                    song.get("title") or "",
                    song.get("artist") or "",
                    bpm.group(1) if bpm else "",
                    file,
                    song.get("youtube_title") or "",
                    song.get("channel") or "",
                    song.get("youtube_url") or "",
                    song.get("spotify_url") or "",
                    note,
                ])
        os.replace(tmp, path)


def folder_holding(output_dir: str, keys: list[str], at_least: float = 0.5) -> str | None:
    """The folder under output_dir whose record already has most of these
    songs - where a pasted list that has no name of its own carries on.

    Measured against whichever is smaller, the list or the folder: the
    first 100 songs of a 1,200-song playlist, downloaded from its link, are
    all in the pasted list, and that folder is where the rest belong.

    Only folders with a record are opened, one level down, so this stays
    quick in a library of thousands of songs."""
    wanted = set(k for k in keys if k)
    if not wanted:
        return None
    best, best_share = None, 0.0
    try:
        names = sorted(os.listdir(output_dir))
    except OSError:
        return None
    for name in names:
        folder = os.path.join(output_dir, name)
        if not os.path.isfile(os.path.join(folder, MANIFEST_FILENAME)):
            continue
        have = set(Sources(folder).songs)
        if not have:
            continue
        share = len(wanted & have) / min(len(wanted), len(have))
        if share > best_share:
            best, best_share = folder, share
    return best if best_share >= at_least else None

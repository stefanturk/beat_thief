#!/usr/bin/env python3
"""Clean up downloaded MP3s: trim dead air, fix volume, tidy titles/tags, remove duplicates."""

from __future__ import annotations

import difflib
import json
import os
import re

from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3NoHeaderError

SANITIZED_ARCHIVE_NAME = ".sanitized_archive.txt"
FLAGGED_FILE_NAME = ".sanitizer_flagged.json"

_JUNK_PATTERNS = [
    r"\(\s*official\s+video\s*\)",
    r"\[\s*official\s+video\s*\]",
    r"\(\s*official\s+audio\s*\)",
    r"\[\s*official\s+audio\s*\]",
    r"\(\s*official\s+music\s+video\s*\)",
    r"\[\s*official\s+music\s+video\s*\]",
    r"\(\s*lyrics?\s*\)",
    r"\[\s*lyrics?\s*\]",
    r"\(\s*lyric\s+video\s*\)",
    r"\[\s*lyric\s+video\s*\]",
    r"\(\s*audio\s*\)",
    r"\[\s*audio\s*\]",
    r"\(\s*visualizer\s*\)",
    r"\[\s*visualizer\s*\]",
    r"\bhd\b",
    r"\b4k\b",
]
_JUNK_RE = re.compile("|".join(_JUNK_PATTERNS), re.IGNORECASE)

DEDUP_THRESHOLD = 0.9


# --- State tracking ---------------------------------------------------------

def load_sanitized_archive(output_dir: str) -> set[str]:
    path = os.path.join(output_dir, SANITIZED_ARCHIVE_NAME)
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def mark_sanitized(output_dir: str, filename: str) -> None:
    path = os.path.join(output_dir, SANITIZED_ARCHIVE_NAME)
    with open(path, "a") as f:
        f.write(filename + "\n")


def load_flagged(output_dir: str) -> list[dict]:
    path = os.path.join(output_dir, FLAGGED_FILE_NAME)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def save_flagged(output_dir: str, flagged: list[dict]) -> None:
    path = os.path.join(output_dir, FLAGGED_FILE_NAME)
    with open(path, "w") as f:
        json.dump(flagged, f, indent=2)


# --- Title cleanup and ID3 tags ---------------------------------------------

def clean_title(stem: str) -> str:
    cleaned = _JUNK_RE.sub("", stem)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s*-\s*$", "", cleaned).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned


def split_title_artist(stem: str) -> tuple[str, str]:
    if " - " in stem:
        title, artist = stem.rsplit(" - ", 1)
        return title, artist
    return stem, ""


def write_id3_tags(path: str, title: str, artist: str) -> None:
    try:
        id3_tags = EasyID3(path)
    except ID3NoHeaderError:
        id3_tags = EasyID3()
        id3_tags.save(path)
        id3_tags = EasyID3(path)
    id3_tags["title"] = title
    id3_tags["artist"] = artist
    id3_tags.save()


# --- Duplicate detection -----------------------------------------------------

def normalize_for_compare(stem: str) -> str:
    lowered = stem.lower()
    stripped = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", stripped).strip()


def find_duplicate_pairs(filenames: list[str], threshold: float = DEDUP_THRESHOLD) -> list[tuple[str, str]]:
    normalized = [(f, normalize_for_compare(os.path.splitext(f)[0])) for f in filenames]
    pairs = []
    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            f1, n1 = normalized[i]
            f2, n2 = normalized[j]
            ratio = difflib.SequenceMatcher(None, n1, n2).ratio()
            if ratio >= threshold:
                pairs.append((f1, f2))
    return pairs

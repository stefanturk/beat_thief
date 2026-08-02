#!/usr/bin/env python3
"""Clean up downloaded MP3s: trim dead air, fix volume, tidy titles/tags, remove duplicates."""

from __future__ import annotations

import difflib
import json
import os
import re
import subprocess
import tempfile

from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3NoHeaderError

from pydub import AudioSegment

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


# --- Audio analysis and manipulation ----------------------------------------

SILENCE_DBFS = -50.0
AMBIGUOUS_DB_BELOW_AVERAGE = 25.0
NORMALIZE_TARGET_DBFS = -1.0
NORMALIZE_TRIGGER_HEADROOM_DB = 3.0
SCAN_CHUNK_MS = 500
SNIPPET_WINDOW_MS = 5000


def load_audio(path: str) -> AudioSegment:
    return AudioSegment.from_mp3(path)


def export_audio(audio: AudioSegment, path: str) -> None:
    audio.export(path, format="mp3", bitrate="320k")


def _region_dbfs(audio: AudioSegment, start_ms: int, end_ms: int) -> float:
    region = audio[start_ms:end_ms]
    return region.dBFS if region.dBFS != float("-inf") else -120.0


def _find_cut_from_start(audio: AudioSegment, avg_dbfs: float) -> int:
    duration_ms = len(audio)
    pos = 0
    while pos < duration_ms:
        level = _region_dbfs(audio, pos, pos + SCAN_CHUNK_MS)
        if level < SILENCE_DBFS or level < avg_dbfs - AMBIGUOUS_DB_BELOW_AVERAGE:
            pos += SCAN_CHUNK_MS
            continue
        break
    return min(pos, duration_ms)


def _find_cut_from_end(audio: AudioSegment, avg_dbfs: float) -> int:
    duration_ms = len(audio)
    pos = duration_ms
    while pos > 0:
        level = _region_dbfs(audio, max(0, pos - SCAN_CHUNK_MS), pos)
        if level < SILENCE_DBFS or level < avg_dbfs - AMBIGUOUS_DB_BELOW_AVERAGE:
            pos -= SCAN_CHUNK_MS
            continue
        break
    return max(pos, 0)


def analyze_cut_candidates(audio: AudioSegment) -> dict:
    avg_dbfs = audio.dBFS
    result: dict = {}

    start_cut_ms = _find_cut_from_start(audio, avg_dbfs)
    if start_cut_ms > 0:
        region_level = _region_dbfs(audio, 0, start_cut_ms)
        classification = "silent" if region_level < SILENCE_DBFS else "ambiguous"
        result["start"] = {"classification": classification, "cut_ms": start_cut_ms}

    end_cut_ms = _find_cut_from_end(audio, avg_dbfs)
    if end_cut_ms < len(audio):
        region_level = _region_dbfs(audio, end_cut_ms, len(audio))
        classification = "silent" if region_level < SILENCE_DBFS else "ambiguous"
        result["end"] = {"classification": classification, "cut_ms": end_cut_ms}

    return result


def trim(audio: AudioSegment, start_ms: int | None, end_ms: int | None) -> AudioSegment:
    start = start_ms if start_ms is not None else 0
    end = end_ms if end_ms is not None else len(audio)
    return audio[start:end]


def apply_fade(audio: AudioSegment, end: str, cut_ms: int) -> AudioSegment:
    if end == "start":
        return audio.fade_in(cut_ms)
    duration = len(audio) - cut_ms
    return audio.fade_out(duration)


def needs_normalization(audio: AudioSegment) -> bool:
    return audio.max_dBFS < NORMALIZE_TARGET_DBFS - NORMALIZE_TRIGGER_HEADROOM_DB


def peak_normalize(audio: AudioSegment) -> AudioSegment:
    gain = NORMALIZE_TARGET_DBFS - audio.max_dBFS
    return audio.apply_gain(gain)


def extract_snippet(audio: AudioSegment, center_ms: int, window_ms: int = SNIPPET_WINDOW_MS) -> AudioSegment:
    start = max(0, center_ms - window_ms)
    end = min(len(audio), center_ms + window_ms)
    return audio[start:end]


def play_snippet(audio: AudioSegment) -> None:
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        audio.export(tmp_path, format="wav")
        subprocess.run(["afplay", tmp_path], check=False)
    finally:
        os.remove(tmp_path)

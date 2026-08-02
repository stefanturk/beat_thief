#!/usr/bin/env python3
"""Clean up downloaded MP3s: trim dead air, fix volume, tidy titles/tags, remove duplicates."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
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


# --- Interactive review of ambiguous cut points -----------------------------

def _prompt_choice() -> str:
    while True:
        choice = input("  (c)ut / (f)ade / (k)eep / (a)djust? ").strip().lower()
        if choice in ("c", "f", "k", "a"):
            return choice
        print("  Please enter c, f, k, or a.")


def _prompt_adjust_seconds() -> float:
    while True:
        raw = input("  Adjust by how many seconds (negative = earlier, positive = later)? ").strip()
        try:
            return float(raw)
        except ValueError:
            print("  Please enter a number.")


def review_flagged(output_dir: str) -> None:
    remaining = load_flagged(output_dir)
    if not remaining:
        print("No ambiguous cuts to review.")
        return

    while remaining:
        flag = remaining[0]
        filename = flag["filename"]
        path = os.path.join(output_dir, filename)

        if not os.path.exists(path):
            remaining.pop(0)
            save_flagged(output_dir, remaining)
            continue

        audio = load_audio(path)
        cut_ms = flag["cut_ms"]
        end = flag["end"]

        label = "intro" if end == "start" else "outro"
        print(f"\n{filename} — possible {label} at {cut_ms / 1000:.1f}s")
        snippet = extract_snippet(audio, cut_ms)
        play_snippet(snippet)

        choice = _prompt_choice()

        if choice == "a":
            delta_seconds = _prompt_adjust_seconds()
            cut_ms = max(0, min(len(audio), cut_ms + int(delta_seconds * 1000)))
            flag["cut_ms"] = cut_ms
            save_flagged(output_dir, remaining)
            continue

        if choice == "c":
            result_audio = trim(
                audio,
                cut_ms if end == "start" else None,
                cut_ms if end == "end" else None,
            )
            export_audio(result_audio, path)
        elif choice == "f":
            result_audio = apply_fade(audio, end, cut_ms)
            export_audio(result_audio, path)

        remaining.pop(0)
        save_flagged(output_dir, remaining)


# --- Per-file / per-folder orchestration ------------------------------------

DEFAULT_OUTPUT = os.path.join(os.path.expanduser("~"), "Downloads", "Song Downloads")


def backup_original(path: str, output_dir: str) -> None:
    originals_dir = os.path.join(output_dir, ".originals")
    os.makedirs(originals_dir, exist_ok=True)
    dest = os.path.join(originals_dir, os.path.basename(path))
    if not os.path.exists(dest):
        shutil.copy2(path, dest)


def sanitize_file(filename: str, output_dir: str) -> list[dict]:
    path = os.path.join(output_dir, filename)
    backup_original(path, output_dir)

    audio = load_audio(path)
    candidates = analyze_cut_candidates(audio)

    trim_start_ms = None
    trim_end_ms = None
    new_flags = []
    for end_key in ("start", "end"):
        if end_key not in candidates:
            continue
        candidate = candidates[end_key]
        if candidate["classification"] == "silent":
            if end_key == "start":
                trim_start_ms = candidate["cut_ms"]
            else:
                trim_end_ms = candidate["cut_ms"]
        else:
            new_flags.append({"filename": filename, "end": end_key, "cut_ms": candidate["cut_ms"]})

    changed = False
    if trim_start_ms is not None or trim_end_ms is not None:
        audio = trim(audio, trim_start_ms, trim_end_ms)
        changed = True

    if needs_normalization(audio):
        audio = peak_normalize(audio)
        changed = True

    if changed:
        export_audio(audio, path)

    stem, ext = os.path.splitext(filename)
    cleaned_stem = clean_title(stem)
    final_filename = filename
    if cleaned_stem != stem:
        final_filename = cleaned_stem + ext
        new_path = os.path.join(output_dir, final_filename)
        os.rename(path, new_path)
        path = new_path
        for flag in new_flags:
            flag["filename"] = final_filename

    title, artist = split_title_artist(cleaned_stem)
    write_id3_tags(path, title, artist)

    mark_sanitized(output_dir, final_filename)
    return new_flags


def _run_dedup(output_dir: str) -> None:
    current_files = sorted(f for f in os.listdir(output_dir) if f.lower().endswith(".mp3"))
    duplicate_pairs = find_duplicate_pairs(current_files)
    if not duplicate_pairs:
        return

    originals_dir = os.path.join(output_dir, ".originals")
    os.makedirs(originals_dir, exist_ok=True)
    for f1, f2 in duplicate_pairs:
        path1 = os.path.join(output_dir, f1)
        path2 = os.path.join(output_dir, f2)
        if not os.path.exists(path1) or not os.path.exists(path2):
            continue
        loser = f2 if os.path.getsize(path1) >= os.path.getsize(path2) else f1
        loser_path = os.path.join(output_dir, loser)
        shutil.move(loser_path, os.path.join(originals_dir, loser))
        print(f"Removed duplicate: {loser}")


def sanitize_folder(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    archive = load_sanitized_archive(output_dir)
    mp3_files = sorted(f for f in os.listdir(output_dir) if f.lower().endswith(".mp3"))

    all_flags = load_flagged(output_dir)
    for filename in mp3_files:
        if filename in archive:
            continue
        try:
            new_flags = sanitize_file(filename, output_dir)
        except Exception as e:
            print(f"  Could not sanitize {filename}, skipping: {e}")
            continue
        all_flags.extend(new_flags)

    save_flagged(output_dir, all_flags)
    _run_dedup(output_dir)

    flagged = load_flagged(output_dir)
    if flagged:
        print(f"\n{len(flagged)} song section(s) need your input.")
        review_flagged(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean up downloaded MP3s: trim dead air, fix volume, tidy titles/tags, remove duplicates."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help=f"Folder of MP3s to sanitize (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Skip processing and resolve any previously-flagged ambiguous cuts",
    )
    args = parser.parse_args()

    if args.review:
        review_flagged(args.path)
    else:
        sanitize_folder(args.path)


if __name__ == "__main__":
    main()

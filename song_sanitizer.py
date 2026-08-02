#!/usr/bin/env python3
"""Clean up downloaded MP3s: trim dead air, fix volume, tidy titles/tags, remove duplicates."""

from __future__ import annotations

import argparse
import difflib
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import termios
import tty

from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3, ID3NoHeaderError, TXXX

from pydub import AudioSegment

DUPLICATES_DIR_NAME = "Duplicates"

_JUNK_PATTERNS = [
    r"\(\s*official\s+video\s*\)",
    r"\[\s*official\s+video\s*\]",
    r"\bofficial\s+video\b",
    r"\(\s*official\s+audio\s*\)",
    r"\[\s*official\s+audio\s*\]",
    r"\bofficial\s+audio\b",
    r"\(\s*official\s+music\s+video\s*\)",
    r"\[\s*official\s+music\s+video\s*\]",
    r"\bofficial\s+music\s+video\b",
    r"\(\s*lyrics?\s*\)",
    r"\[\s*lyrics?\s*\]",
    r"\blyrics?\b",
    r"\(\s*lyric\s+video\s*\)",
    r"\[\s*lyric\s+video\s*\]",
    r"\blyric\s+video\b",
    r"\(\s*audio\s*\)",
    r"\[\s*audio\s*\]",
    r"\(\s*visualizer\s*\)",
    r"\[\s*visualizer\s*\]",
    r"\bvisualizer\b",
    r"\bhd\b",
    r"\b4k\b",
]
_JUNK_RE = re.compile("|".join(_JUNK_PATTERNS), re.IGNORECASE)

DEDUP_THRESHOLD = 0.9


# --- Title cleanup and ID3 tags ---------------------------------------------

def clean_title(stem: str) -> str:
    cleaned = _JUNK_RE.sub("", stem)
    cleaned = re.sub(r"[\(\[]\s*[\)\]]", "", cleaned)  # leftover empty () or [] shells
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s*-\s*$", "", cleaned).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned


def split_title_artist(stem: str) -> tuple[str, str]:
    if " - " in stem:
        title, artist = stem.rsplit(" - ", 1)
    else:
        title, artist = stem, ""

    # "Artist｜Title" (or "Artist|Title") is a common YouTube channel naming
    # convention baked into the video title itself — when present, it's a
    # more reliable artist than whatever came after " - " (often just the
    # uploader/channel name), so it takes precedence.
    for sep in ("｜", "|"):
        if sep in title:
            left, right = title.split(sep, 1)
            if left.strip() and right.strip():
                artist = left.strip()
                title = right.strip()
            break

    return title, artist


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


_SANITIZED_TAG_DESC = "song_sanitizer"


def _mark_as_sanitized(path: str) -> None:
    """Embed a marker in the file's own ID3 tags identifying it as output
    already produced by this tool — not a separate file/folder, so a run
    never has to guess whether a file it's looking at is a finished result
    or something that still needs processing."""
    try:
        tags = ID3(path)
    except ID3NoHeaderError:
        tags = ID3()
    tags.setall("TXXX:" + _SANITIZED_TAG_DESC, [TXXX(desc=_SANITIZED_TAG_DESC, text=["1"])])
    tags.save(path)


def _is_already_sanitized(path: str) -> bool:
    try:
        tags = ID3(path)
    except Exception:
        return False
    return bool(tags.getall("TXXX:" + _SANITIZED_TAG_DESC))


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
SNIPPET_DURATION_MS = 5000
SUSTAINED_LOUD_CHUNKS = 4  # ~2s of continuous non-quiet audio before we consider the song "started"


def load_audio(path: str) -> AudioSegment:
    return AudioSegment.from_mp3(path)


def export_audio(audio: AudioSegment, path: str) -> None:
    audio.export(path, format="mp3", bitrate="320k")


def _region_dbfs(audio: AudioSegment, start_ms: int, end_ms: int) -> float:
    region = audio[start_ms:end_ms]
    return region.dBFS if region.dBFS != float("-inf") else -120.0


def _is_quiet_chunk(level: float, avg_dbfs: float) -> bool:
    return level < SILENCE_DBFS or level < avg_dbfs - AMBIGUOUS_DB_BELOW_AVERAGE


def _find_cut_from_start(audio: AudioSegment, avg_dbfs: float) -> int:
    # A brief loud blip (a knock, a single kick hit) surrounded by silence
    # shouldn't count as "the song has started" — only a sustained run of
    # non-quiet chunks does. This lets a sparse, mostly-silent intro (e.g. a
    # few isolated drum hits) get flagged as one ambiguous region instead of
    # stopping at the very first blip.
    duration_ms = len(audio)
    pos = 0
    consecutive_loud = 0
    while pos < duration_ms:
        level = _region_dbfs(audio, pos, pos + SCAN_CHUNK_MS)
        if _is_quiet_chunk(level, avg_dbfs):
            consecutive_loud = 0
        else:
            consecutive_loud += 1
            if consecutive_loud >= SUSTAINED_LOUD_CHUNKS:
                cut_ms = pos - (consecutive_loud - 1) * SCAN_CHUNK_MS
                return max(cut_ms, 0)
        pos += SCAN_CHUNK_MS
    return duration_ms


def _find_cut_from_end(audio: AudioSegment, avg_dbfs: float) -> int:
    duration_ms = len(audio)
    pos = duration_ms
    consecutive_loud = 0
    while pos > 0:
        level = _region_dbfs(audio, max(0, pos - SCAN_CHUNK_MS), pos)
        if _is_quiet_chunk(level, avg_dbfs):
            consecutive_loud = 0
        else:
            consecutive_loud += 1
            if consecutive_loud >= SUSTAINED_LOUD_CHUNKS:
                cut_ms = pos + (consecutive_loud - 1) * SCAN_CHUNK_MS
                return min(cut_ms, duration_ms)
        pos -= SCAN_CHUNK_MS
    return 0


def analyze_cut_candidates(audio: AudioSegment) -> dict:
    avg_dbfs = audio.dBFS
    result: dict = {}

    start_cut_ms = _find_cut_from_start(audio, avg_dbfs)
    if 0 < start_cut_ms < len(audio):
        region_level = _region_dbfs(audio, 0, start_cut_ms)
        classification = "silent" if region_level < SILENCE_DBFS else "ambiguous"
        result["start"] = {"classification": classification, "cut_ms": start_cut_ms}

    end_cut_ms = _find_cut_from_end(audio, avg_dbfs)
    if 0 < end_cut_ms < len(audio):
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
        return audio.fade_in(max(0, cut_ms))
    duration = max(0, len(audio) - cut_ms)
    return audio.fade_out(duration)


def needs_normalization(audio: AudioSegment) -> bool:
    max_dbfs = audio.max_dBFS
    if not math.isfinite(max_dbfs):
        # Silence/near-silence has no finite peak to normalize against.
        return False
    return max_dbfs < NORMALIZE_TARGET_DBFS - NORMALIZE_TRIGGER_HEADROOM_DB


def peak_normalize(audio: AudioSegment) -> AudioSegment:
    max_dbfs = audio.max_dBFS
    if not math.isfinite(max_dbfs):
        return audio
    gain = NORMALIZE_TARGET_DBFS - max_dbfs
    return audio.apply_gain(gain)


def extract_snippet(audio: AudioSegment, anchor_ms: int, end: str, duration_ms: int = SNIPPET_DURATION_MS) -> AudioSegment:
    """Extract `duration_ms` of audio anchored exactly where the sanitized
    track would begin or end at this cut point — not centered on it, so what
    plays is exactly what the cut would sound like, not a preview that
    straddles material that's about to be removed."""
    if end == "start":
        start = max(0, anchor_ms)
        stop = min(len(audio), start + duration_ms)
    else:
        stop = min(len(audio), anchor_ms)
        start = max(0, stop - duration_ms)
    return audio[start:stop]


def play_snippet(audio: AudioSegment) -> None:
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        audio.export(tmp_path, format="wav")
        subprocess.run(["afplay", tmp_path], check=False)
    finally:
        os.remove(tmp_path)


def _wait_for_space(prompt: str) -> None:
    if not sys.stdin.isatty():
        # Not a real terminal (piped input, non-interactive run) — raw
        # cbreak mode isn't available, so fall back to a plain Enter-to-continue.
        input(prompt)
        return
    print(prompt, end="", flush=True)
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while sys.stdin.read(1) != " ":
            pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    print()


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


def resolve_flags(flags: list[dict], output_dir: str) -> None:
    """Interactively resolve ambiguous cut flags, one at a time, modifying
    each flagged file in place. Purely in-memory — if interrupted partway
    through, whatever wasn't resolved yet is simply left as-is in the
    already-created copy; there's no separate state file to resume from."""
    remaining = list(flags)

    while remaining:
        flag = remaining[0]
        filename = flag["filename"]
        path = os.path.join(output_dir, filename)

        try:
            if not os.path.exists(path):
                remaining.pop(0)
                continue

            audio = load_audio(path)
            cut_ms = flag["cut_ms"]
            end = flag["end"]

            label = "intro" if end == "start" else "outro"
            verb = "start" if end == "start" else "end"
            print(f"\n{filename} — possible {label} at {cut_ms / 1000:.1f}s")
            _wait_for_space(f"  Ready? Press space to hear where the sanitized song would {verb}... ")
            snippet = extract_snippet(audio, cut_ms, end)
            play_snippet(snippet)

            choice = _prompt_choice()

            if choice == "a":
                delta_seconds = _prompt_adjust_seconds()
                cut_ms = max(0, min(len(audio), cut_ms + int(delta_seconds * 1000)))
                flag["cut_ms"] = cut_ms
                continue

            if choice in ("c", "f"):
                # Re-exporting drops any existing ID3 tags, so preserve
                # title/artist across the re-export.
                try:
                    existing_tags = EasyID3(path)
                    title = existing_tags.get("title", [""])[0]
                    artist = existing_tags.get("artist", [""])[0]
                except Exception:
                    title, artist = "", ""

            if choice == "c":
                result_audio = trim(
                    audio,
                    cut_ms if end == "start" else None,
                    cut_ms if end == "end" else None,
                )
                export_audio(result_audio, path)
                write_id3_tags(path, title, artist)

                # Cutting shortens the audio, which invalidates absolute
                # cut_ms offsets stored by any other pending flag for this
                # same file. A start-cut shifts everything earlier by cut_ms;
                # an end-cut only removes trailing audio, which doesn't
                # affect earlier offsets.
                if end == "start":
                    for other in remaining[1:]:
                        if other["filename"] == filename and other["end"] == "end":
                            other["cut_ms"] = max(0, other["cut_ms"] - cut_ms)
            elif choice == "f":
                result_audio = apply_fade(audio, end, cut_ms)
                export_audio(result_audio, path)
                write_id3_tags(path, title, artist)

            remaining.pop(0)
            if not any(f["filename"] == filename for f in remaining):
                # No more pending flags for this file — mark it finished so
                # a later run never re-flags or duplicates it.
                _mark_as_sanitized(path)
        except Exception as e:
            print(f"  Could not review {filename}, skipping: {e}")
            remaining.pop(0)


# --- Per-file / per-folder orchestration ------------------------------------

DEFAULT_OUTPUT = os.path.join(os.path.expanduser("~"), "Downloads", "Song Downloads")




def _derive_title_artist(filename: str) -> tuple[str, str, str]:
    """Return (title, artist, canonical_stem) derived from filename."""
    stem = os.path.splitext(filename)[0]
    cleaned_stem = clean_title(stem)
    title, artist = split_title_artist(cleaned_stem if cleaned_stem else stem)
    canonical_stem = f"{title} - {artist}" if artist else (title or stem)
    return title, artist, canonical_stem


def _existing_companion(output_dir: str, filename: str) -> str | None:
    """If a sanitized copy of filename already exists in output_dir (found
    purely by looking for the filename it would have been given — no hidden
    tracking file involved), return its name. Otherwise None."""
    ext = os.path.splitext(filename)[1]
    _, _, canonical_stem = _derive_title_artist(filename)
    for candidate_stem in (canonical_stem, f"{canonical_stem} (sanitized)"):
        candidate = candidate_stem + ext
        if candidate != filename and os.path.exists(os.path.join(output_dir, candidate)):
            return candidate
    return None


def sanitize_file(filename: str, output_dir: str, replace: bool = False) -> list[dict]:
    """Produce a cleaned-up copy of filename next to it, if it needs one. The
    original file at filename is never opened for writing — only ever read
    from. If nothing about it needs fixing, no new file is created at all.
    If a sanitized copy already exists next to it, it's left alone (skipped).

    If replace is True, the original is deleted once the cleaned-up copy has
    been written successfully (used for automatic post-download cleanup,
    where there's no reason to keep the raw download around). If False, the
    original is kept alongside the new copy (used for one-off/manual runs)."""
    companion = _existing_companion(output_dir, filename)
    if companion:
        print(f"{filename}: already sanitized as '{companion}', nothing to do.")
        return []

    path = os.path.join(output_dir, filename)

    if _is_already_sanitized(path):
        # This file IS a previously-produced copy (e.g. from an earlier run,
        # or already reviewed and kept as-is) — never re-flag or duplicate it.
        print(f"{filename}: already sanitized, nothing to do.")
        return []

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

    needs_trim = trim_start_ms is not None or trim_end_ms is not None
    if needs_trim:
        audio = trim(audio, trim_start_ms, trim_end_ms)

    needs_normalize = needs_normalization(audio)
    if needs_normalize:
        audio = peak_normalize(audio)

    stem, ext = os.path.splitext(filename)
    title, artist, canonical_stem = _derive_title_artist(filename)
    name_changed = canonical_stem != stem

    if not (needs_trim or needs_normalize or name_changed or new_flags):
        # Nothing about this file needs fixing — leave it exactly as it is.
        print(f"{filename}: already good, no changes needed.")
        return new_flags

    preferred_stem = canonical_stem
    self_collision = preferred_stem + ext == filename
    if self_collision:
        # The cleaned-up name is identical to the original, but we still need
        # a modified copy (trim/normalize/pending review) — never overwrite
        # the original while it still exists, so this one gets a
        # distinguishing suffix.
        preferred_stem = f"{canonical_stem} (sanitized)"

    output_path = os.path.join(output_dir, preferred_stem + ext)
    if os.path.exists(output_path):
        # Some other, unrelated song already has this exact cleaned-up name.
        # Never overwrite it — leave this file as it is.
        print(f"{filename}: '{os.path.basename(output_path)}' already exists, skipping to avoid overwriting it.")
        return []
    final_filename = os.path.basename(output_path)

    export_audio(audio, output_path)
    write_id3_tags(output_path, title, artist)

    if replace:
        os.remove(path)
        if self_collision:
            # The original is gone now, so its name is free — drop the
            # "(sanitized)" suffix and use the clean canonical name.
            canonical_path = os.path.join(output_dir, canonical_stem + ext)
            if not os.path.exists(canonical_path):
                os.rename(output_path, canonical_path)
                output_path = canonical_path
                final_filename = os.path.basename(output_path)

    if not new_flags:
        # Only mark it finished once there's nothing left pending review —
        # a flagged copy still needs resolve_flags() to touch it, and that
        # happens right after sanitize_file() returns.
        _mark_as_sanitized(output_path)

    for flag in new_flags:
        flag["filename"] = final_filename

    if replace:
        if final_filename == filename:
            print(f"{filename}: cleaned up.")
        else:
            print(f"{filename}: cleaned up and renamed to '{final_filename}'.")
    else:
        print(f"{filename}: saved cleaned-up copy as '{final_filename}'.")

    return new_flags


def _run_dedup(output_dir: str) -> None:
    current_files = sorted(f for f in os.listdir(output_dir) if f.lower().endswith(".mp3"))
    duplicate_pairs = find_duplicate_pairs(current_files)
    if not duplicate_pairs:
        return

    duplicates_dir = os.path.join(output_dir, DUPLICATES_DIR_NAME)
    os.makedirs(duplicates_dir, exist_ok=True)
    for f1, f2 in duplicate_pairs:
        path1 = os.path.join(output_dir, f1)
        path2 = os.path.join(output_dir, f2)
        if not os.path.exists(path1) or not os.path.exists(path2):
            continue
        loser = f2 if os.path.getsize(path1) >= os.path.getsize(path2) else f1
        loser_path = os.path.join(output_dir, loser)
        shutil.move(loser_path, os.path.join(duplicates_dir, loser))
        print(f"Moved duplicate to {DUPLICATES_DIR_NAME}/: {loser}")


def sanitize_folder(output_dir: str, replace: bool = False) -> None:
    os.makedirs(output_dir, exist_ok=True)
    mp3_files = sorted(f for f in os.listdir(output_dir) if f.lower().endswith(".mp3"))

    if not mp3_files:
        print("No MP3s found to sanitize.")
        return

    all_flags = []
    for filename in mp3_files:
        try:
            new_flags = sanitize_file(filename, output_dir, replace=replace)
        except Exception as e:
            print(f"  Could not sanitize {filename}, skipping: {e}")
            continue
        all_flags.extend(new_flags)

    _run_dedup(output_dir)

    if all_flags:
        print(f"\n{len(all_flags)} song section(s) need your input.")
        resolve_flags(all_flags, output_dir)


def sanitize_single_file(path: str, replace: bool = False) -> None:
    output_dir = os.path.dirname(os.path.abspath(path)) or "."
    filename = os.path.basename(path)

    try:
        flags = sanitize_file(filename, output_dir, replace=replace)
    except Exception as e:
        print(f"  Could not sanitize {filename}, skipping: {e}")
        return

    if flags:
        print(f"\n{len(flags)} song section(s) need your input.")
        resolve_flags(flags, output_dir)


def sanitize_path(path: str, replace: bool = False) -> None:
    if os.path.isfile(path):
        sanitize_single_file(path, replace=replace)
    else:
        sanitize_folder(path, replace=replace)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean up downloaded MP3s: trim dead air, fix volume, tidy titles/tags, remove duplicates."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help=f"Folder of MP3s (or a single MP3 file) to sanitize (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    try:
        sanitize_path(args.path)
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(130)


if __name__ == "__main__":
    main()

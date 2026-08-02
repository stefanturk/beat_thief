# Song Sanitizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `song_sanitizer.py` and supporting modules that clean up downloaded MP3s — trim dead air (with interactive review for ambiguous cases), fix quiet volume, tidy titles, write ID3 tags, and remove duplicates — runnable standalone or automatically after `song_downloader.py`.

**Architecture:** A set of small, single-responsibility modules (`sanitizer_state.py`, `sanitizer_tags.py`, `sanitizer_dedup.py`, `sanitizer_audio.py`, `sanitizer_review.py`) that `song_sanitizer.py` orchestrates via `sanitize_folder()`/`sanitize_file()`, plus a CLI entrypoint. `song_downloader.py` imports and calls `sanitize_folder()` at the end of a run.

**Tech Stack:** Python 3.9, `pydub` (audio ops via ffmpeg), `mutagen` (ID3 tags), stdlib `difflib`/`json`/`subprocess` (dedup, state, playback via macOS `afplay`), stdlib `unittest` for tests (no pytest — matches this repo, which has no existing test framework).

## Global Constraints

- Python 3.9 compatibility required — use `from __future__ import annotations` in any file with `X | Y` type hints (established pattern from `song_downloader.py`).
- No creative audio edits (EQ, compression) — only trim/fade/gain per the spec.
- Peak normalization target: -1dBFS; only apply if peak is more than 3dB of headroom below that (`docs/superpowers/specs/2026-08-01-song-sanitizer-design.md`, "Peak-normalize volume").
- Silence threshold: -50dBFS; ambiguous threshold: more than 25dB below the track's average `dBFS` (spec, "Detect intro/outro cut candidates").
- State files live in the output directory: `.sanitized_archive.txt`, `.sanitizer_flagged.json`, `.originals/` (spec, "State files").
- Playback for review uses macOS `afplay` only (spec, "Explicitly out of scope").
- Every per-file step must be wrapped so one bad file doesn't stop a batch (spec, "Per-file pipeline").

---

### Task 1: State file module

**Files:**
- Create: `song_downloader/sanitizer_state.py`
- Test: `song_downloader/test_sanitizer_state.py`

**Interfaces:**
- Produces: `SANITIZED_ARCHIVE_NAME: str`, `FLAGGED_FILE_NAME: str`, `load_sanitized_archive(output_dir: str) -> set[str]`, `mark_sanitized(output_dir: str, filename: str) -> None`, `load_flagged(output_dir: str) -> list[dict]`, `save_flagged(output_dir: str, flagged: list[dict]) -> None`. A flag dict has keys `filename: str`, `end: str` (`"start"` or `"end"`), `cut_ms: int`.

- [ ] **Step 1: Write the failing tests**

```python
# song_downloader/test_sanitizer_state.py
import os
import shutil
import tempfile
import unittest

import sanitizer_state as state


class TestSanitizerState(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_load_sanitized_archive_missing_file_returns_empty_set(self):
        self.assertEqual(state.load_sanitized_archive(self.tmp_dir), set())

    def test_mark_sanitized_then_load_returns_filename(self):
        state.mark_sanitized(self.tmp_dir, "Song - Artist.mp3")
        state.mark_sanitized(self.tmp_dir, "Other - Artist.mp3")
        archive = state.load_sanitized_archive(self.tmp_dir)
        self.assertEqual(archive, {"Song - Artist.mp3", "Other - Artist.mp3"})

    def test_load_flagged_missing_file_returns_empty_list(self):
        self.assertEqual(state.load_flagged(self.tmp_dir), [])

    def test_save_then_load_flagged_roundtrips(self):
        flags = [{"filename": "Song - Artist.mp3", "end": "start", "cut_ms": 4200}]
        state.save_flagged(self.tmp_dir, flags)
        self.assertEqual(state.load_flagged(self.tmp_dir), flags)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd song_downloader && python3 -m unittest test_sanitizer_state -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sanitizer_state'`

- [ ] **Step 3: Write minimal implementation**

```python
# song_downloader/sanitizer_state.py
"""Read/write the sanitizer's on-disk state: the sanitized archive and pending flags."""

from __future__ import annotations

import json
import os

SANITIZED_ARCHIVE_NAME = ".sanitized_archive.txt"
FLAGGED_FILE_NAME = ".sanitizer_flagged.json"


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd song_downloader && python3 -m unittest test_sanitizer_state -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
cd song_downloader
git add sanitizer_state.py test_sanitizer_state.py
git commit -m "$(cat <<'EOF'
Add sanitizer state tracking module

User-facing:
- No behavior change yet; groundwork for the upcoming song sanitizer feature

Technical:
- sanitizer_state.py: read/write .sanitized_archive.txt and .sanitizer_flagged.json in an output folder
- test_sanitizer_state.py: unit tests for archive/flag roundtrips
EOF
)"
```

---

### Task 2: Title cleanup and ID3 tag module

**Files:**
- Create: `song_downloader/sanitizer_tags.py`
- Test: `song_downloader/test_sanitizer_tags.py`

**Interfaces:**
- Consumes: none from earlier tasks.
- Produces: `clean_title(stem: str) -> str`, `split_title_artist(stem: str) -> tuple[str, str]`, `write_id3_tags(path: str, title: str, artist: str) -> None`.

- [ ] **Step 1: Write the failing tests**

```python
# song_downloader/test_sanitizer_tags.py
import os
import shutil
import subprocess
import tempfile
import unittest

from mutagen.easyid3 import EasyID3

import sanitizer_tags as tags


class TestCleanTitle(unittest.TestCase):
    def test_strips_official_video_tag(self):
        self.assertEqual(
            tags.clean_title("Song Name (Official Video) - Artist"),
            "Song Name - Artist",
        )

    def test_strips_multiple_junk_tags_case_insensitive(self):
        self.assertEqual(
            tags.clean_title("Song Name [OFFICIAL AUDIO] (Lyrics) HD - Artist"),
            "Song Name - Artist",
        )

    def test_leaves_clean_title_unchanged(self):
        self.assertEqual(tags.clean_title("Song Name - Artist"), "Song Name - Artist")


class TestSplitTitleArtist(unittest.TestCase):
    def test_splits_on_last_dash(self):
        self.assertEqual(
            tags.split_title_artist("Song - Name - Artist"),
            ("Song - Name", "Artist"),
        )

    def test_no_dash_returns_empty_artist(self):
        self.assertEqual(tags.split_title_artist("Song Name"), ("Song Name", ""))


class TestWriteId3Tags(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.mp3_path = os.path.join(self.tmp_dir, "test.mp3")
        # 0.5s silent MP3 generated via ffmpeg, used purely as a real MP3 container for tag tests.
        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                "-t", "0.5", "-q:a", "9", self.mp3_path,
            ],
            check=True, capture_output=True,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_writes_title_and_artist(self):
        tags.write_id3_tags(self.mp3_path, "Song Name", "Artist")
        read_tags = EasyID3(self.mp3_path)
        self.assertEqual(read_tags["title"], ["Song Name"])
        self.assertEqual(read_tags["artist"], ["Artist"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd song_downloader && python3 -m unittest test_sanitizer_tags -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sanitizer_tags'`

- [ ] **Step 3: Write minimal implementation**

```python
# song_downloader/sanitizer_tags.py
"""Title cleanup and ID3 tag writing for sanitized MP3s."""

from __future__ import annotations

import re

from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3NoHeaderError

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd song_downloader && python3 -m unittest test_sanitizer_tags -v`
Expected: PASS (6 tests). Requires `ffmpeg` (already a project dependency) and `mutagen` (installed in Task 6's dependency step — if running this task in isolation first, run `pip install mutagen` before testing).

- [ ] **Step 5: Commit**

```bash
cd song_downloader
git add sanitizer_tags.py test_sanitizer_tags.py
git commit -m "$(cat <<'EOF'
Add title cleanup and ID3 tag writing module

User-facing:
- No behavior change yet; groundwork for the upcoming song sanitizer feature

Technical:
- sanitizer_tags.py: clean_title() strips common YouTube junk tags, split_title_artist() parses "Title - Artist", write_id3_tags() writes ID3 tags via mutagen
- test_sanitizer_tags.py: unit tests covering junk stripping, splitting, and real ID3 writes against an ffmpeg-generated silent MP3
EOF
)"
```

---

### Task 3: Duplicate detection module

**Files:**
- Create: `song_downloader/sanitizer_dedup.py`
- Test: `song_downloader/test_sanitizer_dedup.py`

**Interfaces:**
- Consumes: none from earlier tasks.
- Produces: `normalize_for_compare(stem: str) -> str`, `find_duplicate_pairs(filenames: list[str], threshold: float = 0.9) -> list[tuple[str, str]]`.

- [ ] **Step 1: Write the failing tests**

```python
# song_downloader/test_sanitizer_dedup.py
import unittest

import sanitizer_dedup as dedup


class TestNormalizeForCompare(unittest.TestCase):
    def test_lowercases_and_strips_punctuation(self):
        self.assertEqual(
            dedup.normalize_for_compare("Song Name! - Artist."),
            "song name artist",
        )


class TestFindDuplicatePairs(unittest.TestCase):
    def test_finds_near_identical_titles(self):
        files = ["Song Name - Artist.mp3", "Song Name  - Artist.mp3", "Totally Different - Other.mp3"]
        pairs = dedup.find_duplicate_pairs(files)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(set(pairs[0]), {"Song Name - Artist.mp3", "Song Name  - Artist.mp3"})

    def test_no_duplicates_returns_empty_list(self):
        files = ["Song One - Artist.mp3", "Song Two - Other Artist.mp3"]
        self.assertEqual(dedup.find_duplicate_pairs(files), [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd song_downloader && python3 -m unittest test_sanitizer_dedup -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sanitizer_dedup'`

- [ ] **Step 3: Write minimal implementation**

```python
# song_downloader/sanitizer_dedup.py
"""Fuzzy filename-based duplicate detection."""

from __future__ import annotations

import difflib
import os
import re

DEFAULT_THRESHOLD = 0.9


def normalize_for_compare(stem: str) -> str:
    lowered = stem.lower()
    stripped = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", stripped).strip()


def find_duplicate_pairs(filenames: list[str], threshold: float = DEFAULT_THRESHOLD) -> list[tuple[str, str]]:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd song_downloader && python3 -m unittest test_sanitizer_dedup -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
cd song_downloader
git add sanitizer_dedup.py test_sanitizer_dedup.py
git commit -m "$(cat <<'EOF'
Add fuzzy duplicate detection module

User-facing:
- No behavior change yet; groundwork for the upcoming song sanitizer feature

Technical:
- sanitizer_dedup.py: normalize_for_compare() and find_duplicate_pairs() use difflib to fuzzy-match filenames, no new dependency
- test_sanitizer_dedup.py: unit tests for normalization and pair-finding
EOF
)"
```

---

### Task 4: Audio analysis and manipulation module

**Files:**
- Create: `song_downloader/sanitizer_audio.py`
- Test: `song_downloader/test_sanitizer_audio.py`

**Interfaces:**
- Consumes: none from earlier tasks.
- Produces: `load_audio(path: str) -> AudioSegment`, `export_audio(audio: AudioSegment, path: str) -> None`, `analyze_cut_candidates(audio: AudioSegment) -> dict`, `trim(audio: AudioSegment, start_ms: int | None, end_ms: int | None) -> AudioSegment`, `apply_fade(audio: AudioSegment, end: str, cut_ms: int) -> AudioSegment`, `needs_normalization(audio: AudioSegment) -> bool`, `peak_normalize(audio: AudioSegment) -> AudioSegment`, `extract_snippet(audio: AudioSegment, center_ms: int, window_ms: int = 5000) -> AudioSegment`, `play_snippet(audio: AudioSegment) -> None`. `analyze_cut_candidates` returns a dict with optional keys `"start"`/`"end"`, each `{"classification": "silent" | "ambiguous", "cut_ms": int}` (a key is absent if that end is normal volume with no cut needed).

- [ ] **Step 1: Write the failing tests**

```python
# song_downloader/test_sanitizer_audio.py
import os
import shutil
import tempfile
import unittest
from unittest import mock

from pydub import AudioSegment
from pydub.generators import Sine

import sanitizer_audio as audio_ops


def _tone(duration_ms, dbfs_gain=0.0, freq=440):
    tone = Sine(freq).to_audio_segment(duration=duration_ms)
    return tone.apply_gain(dbfs_gain - tone.max_dBFS)


class TestAnalyzeCutCandidates(unittest.TestCase):
    def test_detects_silent_intro(self):
        silent_intro = AudioSegment.silent(duration=3000)
        loud_body = _tone(5000, dbfs_gain=-3)
        track = silent_intro + loud_body
        result = audio_ops.analyze_cut_candidates(track)
        self.assertIn("start", result)
        self.assertEqual(result["start"]["classification"], "silent")
        self.assertGreater(result["start"]["cut_ms"], 2000)

    def test_detects_ambiguous_quiet_intro(self):
        quiet_intro = _tone(3000, dbfs_gain=-45)
        loud_body = _tone(5000, dbfs_gain=-3)
        track = quiet_intro + loud_body
        result = audio_ops.analyze_cut_candidates(track)
        self.assertIn("start", result)
        self.assertEqual(result["start"]["classification"], "ambiguous")

    def test_no_flag_for_consistent_volume_track(self):
        track = _tone(5000, dbfs_gain=-6)
        result = audio_ops.analyze_cut_candidates(track)
        self.assertNotIn("start", result)
        self.assertNotIn("end", result)


class TestTrim(unittest.TestCase):
    def test_trims_start_and_end(self):
        track = _tone(5000, dbfs_gain=-6)
        trimmed = audio_ops.trim(track, start_ms=1000, end_ms=4000)
        self.assertEqual(len(trimmed), 3000)


class TestApplyFade(unittest.TestCase):
    def test_fade_start_reduces_early_volume(self):
        track = _tone(5000, dbfs_gain=-6)
        faded = audio_ops.apply_fade(track, "start", 2000)
        self.assertLess(faded[0:100].dBFS, track[0:100].dBFS)

    def test_fade_end_reduces_late_volume(self):
        track = _tone(5000, dbfs_gain=-6)
        faded = audio_ops.apply_fade(track, "end", 3000)
        self.assertLess(faded[4900:5000].dBFS, track[4900:5000].dBFS)


class TestNormalize(unittest.TestCase):
    def test_needs_normalization_true_for_quiet_track(self):
        quiet = _tone(2000, dbfs_gain=-20)
        self.assertTrue(audio_ops.needs_normalization(quiet))

    def test_needs_normalization_false_for_loud_track(self):
        loud = _tone(2000, dbfs_gain=-1.5)
        self.assertFalse(audio_ops.needs_normalization(loud))

    def test_peak_normalize_brings_peak_near_target(self):
        quiet = _tone(2000, dbfs_gain=-20)
        normalized = audio_ops.peak_normalize(quiet)
        self.assertAlmostEqual(normalized.max_dBFS, audio_ops.NORMALIZE_TARGET_DBFS, delta=0.5)


class TestExtractSnippet(unittest.TestCase):
    def test_extracts_window_around_center(self):
        track = _tone(10000, dbfs_gain=-6)
        snippet = audio_ops.extract_snippet(track, center_ms=5000, window_ms=1000)
        self.assertEqual(len(snippet), 2000)

    def test_clamps_at_track_boundaries(self):
        track = _tone(3000, dbfs_gain=-6)
        snippet = audio_ops.extract_snippet(track, center_ms=200, window_ms=1000)
        self.assertEqual(len(snippet), 1200)


class TestLoadExportRoundtrip(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_export_then_load_preserves_duration(self):
        track = _tone(1000, dbfs_gain=-6)
        path = os.path.join(self.tmp_dir, "test.mp3")
        audio_ops.export_audio(track, path)
        loaded = audio_ops.load_audio(path)
        self.assertAlmostEqual(len(loaded), len(track), delta=100)


class TestPlaySnippet(unittest.TestCase):
    @mock.patch("sanitizer_audio.subprocess.run")
    def test_calls_afplay_with_temp_file(self, mock_run):
        track = _tone(500, dbfs_gain=-6)
        audio_ops.play_snippet(track)
        self.assertTrue(mock_run.called)
        args = mock_run.call_args[0][0]
        self.assertEqual(args[0], "afplay")
        self.assertFalse(os.path.exists(args[1]))  # temp file cleaned up


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd song_downloader && python3 -m unittest test_sanitizer_audio -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sanitizer_audio'`

- [ ] **Step 3: Write minimal implementation**

```python
# song_downloader/sanitizer_audio.py
"""Audio analysis and manipulation for the sanitizer: silence detection, trim, fade, normalize, playback."""

from __future__ import annotations

import os
import subprocess
import tempfile

from pydub import AudioSegment

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd song_downloader && python3 -m unittest test_sanitizer_audio -v`
Expected: PASS (13 tests). Requires `pydub` (installed in Task 6's dependency step — if running this task in isolation first, run `pip install pydub` before testing) and `ffmpeg` (already a project dependency).

- [ ] **Step 5: Commit**

```bash
cd song_downloader
git add sanitizer_audio.py test_sanitizer_audio.py
git commit -m "$(cat <<'EOF'
Add audio analysis and manipulation module for sanitizer

User-facing:
- No behavior change yet; groundwork for the upcoming song sanitizer feature

Technical:
- sanitizer_audio.py: analyze_cut_candidates() classifies intro/outro regions as silent/ambiguous/normal by scanning dBFS from each end against the track average; trim(), apply_fade(), peak_normalize(), extract_snippet(), play_snippet() (via afplay) support the pipeline and review flow
- test_sanitizer_audio.py: unit tests using synthetic pydub tones/silence, no fixture files needed
EOF
)"
```

---

### Task 5: Interactive review module

**Files:**
- Create: `song_downloader/sanitizer_review.py`
- Test: `song_downloader/test_sanitizer_review.py`

**Interfaces:**
- Consumes: `sanitizer_state.load_flagged`, `sanitizer_state.save_flagged` (Task 1); `sanitizer_audio.load_audio`, `sanitizer_audio.extract_snippet`, `sanitizer_audio.play_snippet`, `sanitizer_audio.trim`, `sanitizer_audio.apply_fade`, `sanitizer_audio.export_audio` (Task 4).
- Produces: `review_flagged(output_dir: str) -> None`.

- [ ] **Step 1: Write the failing tests**

```python
# song_downloader/test_sanitizer_review.py
import os
import shutil
import tempfile
import unittest
from unittest import mock

from pydub.generators import Sine

import sanitizer_audio as audio_ops
import sanitizer_review as review
import sanitizer_state as state


def _make_track(path, duration_ms=5000):
    tone = Sine(440).to_audio_segment(duration=duration_ms)
    tone = tone.apply_gain(-6 - tone.max_dBFS)
    audio_ops.export_audio(tone, path)


class TestReviewFlagged(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.filename = "Song - Artist.mp3"
        _make_track(os.path.join(self.tmp_dir, self.filename))

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_no_flags_does_nothing(self):
        review.review_flagged(self.tmp_dir)  # should not raise

    @mock.patch("sanitizer_review.audio_ops.play_snippet")
    @mock.patch("builtins.input", return_value="k")
    def test_keep_choice_leaves_file_and_clears_flag(self, mock_input, mock_play):
        state.save_flagged(self.tmp_dir, [{"filename": self.filename, "end": "start", "cut_ms": 1000}])
        original_size = os.path.getsize(os.path.join(self.tmp_dir, self.filename))

        review.review_flagged(self.tmp_dir)

        self.assertEqual(state.load_flagged(self.tmp_dir), [])
        self.assertEqual(os.path.getsize(os.path.join(self.tmp_dir, self.filename)), original_size)

    @mock.patch("sanitizer_review.audio_ops.play_snippet")
    @mock.patch("builtins.input", return_value="c")
    def test_cut_choice_shortens_file_and_clears_flag(self, mock_input, mock_play):
        state.save_flagged(self.tmp_dir, [{"filename": self.filename, "end": "start", "cut_ms": 1000}])

        review.review_flagged(self.tmp_dir)

        self.assertEqual(state.load_flagged(self.tmp_dir), [])
        result_audio = audio_ops.load_audio(os.path.join(self.tmp_dir, self.filename))
        self.assertLess(len(result_audio), 5000)

    @mock.patch("sanitizer_review.audio_ops.play_snippet")
    @mock.patch("builtins.input", side_effect=["a", "1.0", "k"])
    def test_adjust_then_keep_updates_cut_ms_and_clears_flag(self, mock_input, mock_play):
        state.save_flagged(self.tmp_dir, [{"filename": self.filename, "end": "start", "cut_ms": 1000}])

        review.review_flagged(self.tmp_dir)

        self.assertEqual(state.load_flagged(self.tmp_dir), [])
        self.assertEqual(mock_play.call_count, 2)  # replayed after adjusting

    @mock.patch("sanitizer_review.audio_ops.play_snippet")
    @mock.patch("builtins.input", return_value="k")
    def test_missing_file_skips_flag_without_error(self, mock_input, mock_play):
        state.save_flagged(self.tmp_dir, [{"filename": "Missing.mp3", "end": "start", "cut_ms": 1000}])

        review.review_flagged(self.tmp_dir)

        self.assertEqual(state.load_flagged(self.tmp_dir), [])
        mock_play.assert_not_called()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd song_downloader && python3 -m unittest test_sanitizer_review -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sanitizer_review'`

- [ ] **Step 3: Write minimal implementation**

```python
# song_downloader/sanitizer_review.py
"""Interactive terminal review of ambiguous intro/outro cut points."""

from __future__ import annotations

import os

import sanitizer_audio as audio_ops
import sanitizer_state as state


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
    remaining = state.load_flagged(output_dir)
    if not remaining:
        print("No ambiguous cuts to review.")
        return

    while remaining:
        flag = remaining[0]
        filename = flag["filename"]
        path = os.path.join(output_dir, filename)

        if not os.path.exists(path):
            remaining.pop(0)
            state.save_flagged(output_dir, remaining)
            continue

        audio = audio_ops.load_audio(path)
        cut_ms = flag["cut_ms"]
        end = flag["end"]

        label = "intro" if end == "start" else "outro"
        print(f"\n{filename} — possible {label} at {cut_ms / 1000:.1f}s")
        snippet = audio_ops.extract_snippet(audio, cut_ms)
        audio_ops.play_snippet(snippet)

        choice = _prompt_choice()

        if choice == "a":
            delta_seconds = _prompt_adjust_seconds()
            cut_ms = max(0, min(len(audio), cut_ms + int(delta_seconds * 1000)))
            flag["cut_ms"] = cut_ms
            state.save_flagged(output_dir, remaining)
            continue

        if choice == "c":
            result_audio = audio_ops.trim(
                audio,
                cut_ms if end == "start" else None,
                cut_ms if end == "end" else None,
            )
            audio_ops.export_audio(result_audio, path)
        elif choice == "f":
            result_audio = audio_ops.apply_fade(audio, end, cut_ms)
            audio_ops.export_audio(result_audio, path)

        remaining.pop(0)
        state.save_flagged(output_dir, remaining)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd song_downloader && python3 -m unittest test_sanitizer_review -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
cd song_downloader
git add sanitizer_review.py test_sanitizer_review.py
git commit -m "$(cat <<'EOF'
Add interactive review flow for ambiguous cut points

User-facing:
- No behavior change yet; groundwork for the upcoming song sanitizer feature

Technical:
- sanitizer_review.py: review_flagged() plays a snippet around each flagged cut and applies cut/fade/keep/adjust; saves progress after every decision so Ctrl+C mid-review is resumable
- test_sanitizer_review.py: unit tests mocking input() and playback, covering keep/cut/adjust/missing-file paths
EOF
)"
```

---

### Task 6: Orchestration, CLI entrypoint, and dependency wiring

**Files:**
- Create: `song_downloader/song_sanitizer.py`
- Test: `song_downloader/test_song_sanitizer.py`
- Modify: `song_downloader/requirements.txt`

**Interfaces:**
- Consumes: `sanitizer_state.*` (Task 1), `sanitizer_tags.*` (Task 2), `sanitizer_dedup.find_duplicate_pairs` (Task 3), `sanitizer_audio.*` (Task 4), `sanitizer_review.review_flagged` (Task 5).
- Produces: `DEFAULT_OUTPUT: str`, `backup_original(path: str, output_dir: str) -> None`, `sanitize_file(filename: str, output_dir: str) -> list[dict]` (returns any new flags), `sanitize_folder(output_dir: str) -> None`, `main() -> None`.

- [ ] **Step 1: Add pydub and mutagen to requirements.txt**

```
yt-dlp
pydub
mutagen
```

Run: `pip install pydub mutagen`

- [ ] **Step 2: Write the failing tests**

```python
# song_downloader/test_song_sanitizer.py
import os
import shutil
import tempfile
import unittest
from unittest import mock

from pydub.generators import Sine
from pydub import AudioSegment

import sanitizer_audio as audio_ops
import sanitizer_state as state
import song_sanitizer as sanitizer


def _export(segment, path):
    audio_ops.export_audio(segment, path)


def _loud_tone(duration_ms=5000, dbfs_gain=-3):
    tone = Sine(440).to_audio_segment(duration=duration_ms)
    return tone.apply_gain(dbfs_gain - tone.max_dBFS)


class TestBackupOriginal(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp_dir, "Song - Artist.mp3")
        _export(_loud_tone(1000), self.path)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_copies_file_into_originals(self):
        sanitizer.backup_original(self.path, self.tmp_dir)
        backup_path = os.path.join(self.tmp_dir, ".originals", "Song - Artist.mp3")
        self.assertTrue(os.path.exists(backup_path))

    def test_does_not_overwrite_existing_backup(self):
        sanitizer.backup_original(self.path, self.tmp_dir)
        backup_path = os.path.join(self.tmp_dir, ".originals", "Song - Artist.mp3")
        first_mtime = os.path.getmtime(backup_path)
        sanitizer.backup_original(self.path, self.tmp_dir)
        self.assertEqual(os.path.getmtime(backup_path), first_mtime)


class TestSanitizeFile(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_renames_junky_title_and_writes_tags(self):
        filename = "Song Name (Official Video) - Artist.mp3"
        _export(_loud_tone(3000), os.path.join(self.tmp_dir, filename))

        new_flags = sanitizer.sanitize_file(filename, self.tmp_dir)

        self.assertEqual(new_flags, [])
        self.assertFalse(os.path.exists(os.path.join(self.tmp_dir, filename)))
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, "Song Name - Artist.mp3")))
        archive = state.load_sanitized_archive(self.tmp_dir)
        self.assertIn("Song Name - Artist.mp3", archive)

    def test_flags_ambiguous_intro_without_modifying_audio(self):
        filename = "Song - Artist.mp3"
        quiet_intro = Sine(440).to_audio_segment(duration=3000).apply_gain(-45)
        loud_body = _loud_tone(5000)
        track = quiet_intro + loud_body
        original_len = len(track)
        _export(track, os.path.join(self.tmp_dir, filename))

        new_flags = sanitizer.sanitize_file(filename, self.tmp_dir)

        self.assertEqual(len(new_flags), 1)
        self.assertEqual(new_flags[0]["end"], "start")
        result_audio = audio_ops.load_audio(os.path.join(self.tmp_dir, filename))
        self.assertAlmostEqual(len(result_audio), original_len, delta=200)

    def test_auto_trims_silent_intro(self):
        filename = "Song - Artist.mp3"
        silent_intro = AudioSegment.silent(duration=3000)
        loud_body = _loud_tone(5000)
        track = silent_intro + loud_body
        _export(track, os.path.join(self.tmp_dir, filename))

        new_flags = sanitizer.sanitize_file(filename, self.tmp_dir)

        self.assertEqual(new_flags, [])
        result_audio = audio_ops.load_audio(os.path.join(self.tmp_dir, filename))
        self.assertLess(len(result_audio), 8000)


class TestSanitizeFolder(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_skips_files_already_in_archive(self):
        filename = "Song - Artist.mp3"
        _export(_loud_tone(1000), os.path.join(self.tmp_dir, filename))
        state.mark_sanitized(self.tmp_dir, filename)

        with mock.patch("song_sanitizer.review.review_flagged") as mock_review:
            sanitizer.sanitize_folder(self.tmp_dir)

        mock_review.assert_not_called()
        # No .originals backup means sanitize_file was never called on it.
        self.assertFalse(os.path.exists(os.path.join(self.tmp_dir, ".originals", filename)))

    def test_removes_duplicate_and_calls_review_when_flags_pending(self):
        f1 = "Song Name - Artist.mp3"
        f2 = "Song Name  - Artist.mp3"
        _export(_loud_tone(1000), os.path.join(self.tmp_dir, f1))
        _export(_loud_tone(1000), os.path.join(self.tmp_dir, f2))

        with mock.patch("song_sanitizer.review.review_flagged") as mock_review:
            sanitizer.sanitize_folder(self.tmp_dir)

        remaining = [f for f in os.listdir(self.tmp_dir) if f.endswith(".mp3")]
        self.assertEqual(len(remaining), 1)
        mock_review.assert_not_called()  # no ambiguous flags in this scenario


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd song_downloader && python3 -m unittest test_song_sanitizer -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'song_sanitizer'`

- [ ] **Step 4: Write minimal implementation**

```python
# song_downloader/song_sanitizer.py
#!/usr/bin/env python3
"""Clean up downloaded MP3s: trim dead air, fix volume, tidy titles/tags, remove duplicates."""

from __future__ import annotations

import argparse
import os
import shutil

import sanitizer_audio as audio_ops
import sanitizer_dedup as dedup
import sanitizer_review as review
import sanitizer_state as state
import sanitizer_tags as tags

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

    audio = audio_ops.load_audio(path)
    candidates = audio_ops.analyze_cut_candidates(audio)

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
        audio = audio_ops.trim(audio, trim_start_ms, trim_end_ms)
        changed = True

    if audio_ops.needs_normalization(audio):
        audio = audio_ops.peak_normalize(audio)
        changed = True

    if changed:
        audio_ops.export_audio(audio, path)

    stem, ext = os.path.splitext(filename)
    cleaned_stem = tags.clean_title(stem)
    final_filename = filename
    if cleaned_stem != stem:
        final_filename = cleaned_stem + ext
        new_path = os.path.join(output_dir, final_filename)
        os.rename(path, new_path)
        path = new_path
        for flag in new_flags:
            flag["filename"] = final_filename

    title, artist = tags.split_title_artist(cleaned_stem)
    tags.write_id3_tags(path, title, artist)

    state.mark_sanitized(output_dir, final_filename)
    return new_flags


def _run_dedup(output_dir: str) -> None:
    current_files = sorted(f for f in os.listdir(output_dir) if f.lower().endswith(".mp3"))
    duplicate_pairs = dedup.find_duplicate_pairs(current_files)
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
    archive = state.load_sanitized_archive(output_dir)
    mp3_files = sorted(f for f in os.listdir(output_dir) if f.lower().endswith(".mp3"))

    all_flags = state.load_flagged(output_dir)
    for filename in mp3_files:
        if filename in archive:
            continue
        try:
            new_flags = sanitize_file(filename, output_dir)
        except Exception as e:
            print(f"  Could not sanitize {filename}, skipping: {e}")
            continue
        all_flags.extend(new_flags)

    state.save_flagged(output_dir, all_flags)
    _run_dedup(output_dir)

    flagged = state.load_flagged(output_dir)
    if flagged:
        print(f"\n{len(flagged)} song section(s) need your input.")
        review.review_flagged(output_dir)


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
        review.review_flagged(args.path)
    else:
        sanitize_folder(args.path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd song_downloader && python3 -m unittest test_song_sanitizer -v`
Expected: PASS (7 tests)

- [ ] **Step 6: Commit**

```bash
cd song_downloader
git add song_sanitizer.py test_song_sanitizer.py requirements.txt
git commit -m "$(cat <<'EOF'
Add song sanitizer CLI: trim, normalize, clean titles, tag, dedup

User-facing:
- New standalone command: python3 song_sanitizer.py ["folder"] cleans up downloaded MP3s — trims dead air, boosts quiet volume, tidies junky YouTube titles, writes proper Title/Artist tags, and removes duplicate downloads
- Ambiguous intro/outro cuts (quiet but not silent) are held for your review at the end of the run: it plays the section and asks whether to cut, fade, keep, or adjust the cut point
- python3 song_sanitizer.py --review resumes any review you didn't finish
- Originals are backed up to a hidden .originals/ folder before any change

Technical:
- song_sanitizer.py: sanitize_file() runs the per-file pipeline (backup, trim/normalize, title cleanup + rename, ID3 tags, mark sanitized); sanitize_folder() orchestrates the batch, runs dedup, and triggers review for any pending flags
- requirements.txt: added pydub and mutagen
- test_song_sanitizer.py: unit tests for backup idempotency, per-file pipeline outcomes (rename+tag, ambiguous flag, silent auto-trim), and folder-level skip/dedup/review-trigger behavior
EOF
)"
```

---

### Task 7: Integrate into song_downloader.py

**Files:**
- Modify: `song_downloader/song_downloader.py:183-193` (the summary/exit block at the end of `main()`)

**Interfaces:**
- Consumes: `song_sanitizer.sanitize_folder(output_dir: str) -> None` (Task 6).

- [ ] **Step 1: Add the sanitize call before the final exit**

In `song_downloader.py`, add the import near the top alongside the other imports:

```python
import song_sanitizer
```

Then modify the end of `main()` (currently `song_downloader.py:183-193`):

```python
    print()
    skipped = max((_total_songs or 0) - _downloaded_count - _failed_count, 0)
    parts = [f"downloaded {_downloaded_count} new song{'s' if _downloaded_count != 1 else ''}"]
    if skipped:
        parts.append(f"skipped {skipped} already downloaded")
    if _failed_count:
        parts.append(f"{_failed_count} failed")
    print(f"All done! {', '.join(parts)}.")
    print(f"Your music is in: {args.output}")

    try:
        song_sanitizer.sanitize_folder(args.output)
    except Exception as e:
        print(f"Sanitizing hit a snag, but your downloads are safe: {e}")

    _exit(0 if result == 0 else 1)
```

- [ ] **Step 2: Manually verify the integration**

Run: `cd song_downloader && python3 song_downloader.py "https://music.youtube.com/watch?v=jNQXAC9IVRw" -o /tmp/sd_sanitizer_integration_test`

Expected: download completes, prints the "All done!" summary, then the sanitizer runs against `/tmp/sd_sanitizer_integration_test` (prints nothing extra if the single test song needs no cleanup, or shows review prompts if it does), and the process exits cleanly back to the shell prompt without needing Ctrl+C. Confirm with `ls /tmp/sd_sanitizer_integration_test` that a `.sanitized_archive.txt` now exists alongside the MP3, then `rm -rf /tmp/sd_sanitizer_integration_test`.

- [ ] **Step 3: Commit**

```bash
cd song_downloader
git add song_downloader.py
git commit -m "$(cat <<'EOF'
Automatically sanitize downloads after each run

User-facing:
- Every download run now automatically cleans up the songs it downloaded (trim, normalize, title/tags, dedup) right after the "All done!" summary, using the new song_sanitizer.py
- If sanitizing hits a problem, your downloaded songs are unaffected and a plain message explains it

Technical:
- song_downloader.py: imports song_sanitizer and calls sanitize_folder(args.output) after the summary print, wrapped in try/except so a sanitizer failure can't take down an otherwise-successful download run
EOF
)"
```

---

### Task 8: Update README

**Files:**
- Modify: `song_downloader/README.md`

**Interfaces:**
- Consumes: none (documentation only).

- [ ] **Step 1: Add a "Cleaning up your library" section**

Append to `README.md` after the existing "Re-running the script..." paragraph:

```markdown

## Cleaning up your library

Every download run automatically cleans up the songs it just downloaded:
trimming dead air from the start/end, boosting volume on quiet tracks,
tidying up junky YouTube titles (removing things like "(Official Video)"),
writing proper Title/Artist tags, and removing duplicate downloads.

If a song's intro or outro is quiet but not silent (e.g. a lone instrument or
ambient sound), it's held for your review at the end of the run — you'll hear
the section played and can choose to cut it, fade it instead, keep it as-is,
or adjust exactly where the cut happens.

You can also run the cleanup on its own, any time, against any folder:

```
python3 song_sanitizer.py ["path/to/folder"]
```

If you don't finish reviewing ambiguous cuts (e.g. you Ctrl+C partway
through), resume later with:

```
python3 song_sanitizer.py --review
```

Original files are backed up untouched to a hidden `.originals/` folder
before anything is changed.
```

- [ ] **Step 2: Commit**

```bash
cd song_downloader
git add README.md
git commit -m "$(cat <<'EOF'
Document the song sanitizer in the README

User-facing:
- README now explains automatic post-download cleanup, the standalone song_sanitizer.py command, --review, and the .originals/ backup folder

Technical:
- README.md: new "Cleaning up your library" section
EOF
)"
```

---

## Self-Review Notes

- **Spec coverage:** backup (Task 6), silent auto-trim + ambiguous flagging (Task 4/6), peak normalize (Task 4/6), title cleanup (Task 2/6), ID3 tags (Task 2/6), mark-sanitized semantics matching spec step 6 (Task 6), dedup (Task 3/6), interactive review with cut/fade/keep/adjust and afplay playback (Task 4/5), Ctrl+C resilience via incremental `save_flagged` (Task 5), `--review` flag (Task 6), auto-run integration (Task 7), README (Task 8). All spec sections have a task.
- **Type consistency:** flag dict shape (`filename`/`end`/`cut_ms`) is identical across `sanitizer_state.py`, `sanitizer_audio.analyze_cut_candidates`, `song_sanitizer.sanitize_file`, and `sanitizer_review.review_flagged`. `trim(audio, start_ms, end_ms)` signature and `apply_fade(audio, end, cut_ms)` signature match between Task 4's definition and Task 5/6's call sites.
- **No placeholders:** all steps contain runnable code; no "TBD"/"similar to Task N" shortcuts.

#!/usr/bin/env python3
"""Shared building blocks for isolating a single instrument's part from a
song and transcribing it to MIDI (see drum_isolator.py, bass_isolator.py).
Nothing in here is specific to any one instrument."""

from __future__ import annotations

import json
import os
import pty
import re
import subprocess
import sys
import tempfile

import librosa
import numpy as np
from pydub import AudioSegment

import song_sanitizer

_BAR_WIDTH = 30

# demucs prints its own progress via tqdm as e.g. "  45%|####      | ...",
# plus a couple of one-off informational lines ("Separating track <path>",
# "Selected model is a bag of N models...") that are only ever noise here -
# we already print our own "isolating drums/bass..." message before this
# runs. Rather than try to relay demucs' own raw terminal output (whose
# in-place-updating behavior depends on it correctly detecting a real
# terminal, which isn't reliable across every way this can be run),
# _PERCENT_RE pulls just the percentage back out of it so a single bar of
# our own can be drawn - guaranteed one line, no boilerplate, regardless of
# how demucs itself chose to render.
_PERCENT_RE = re.compile(r"(\d{1,3})%")


def run_demucs(input_path: str, out_dir: str, model_name: str, two_stems: str | None = None) -> str:
    """Run demucs and return the directory containing the separated stems,
    printing our own single-line progress bar (see _PERCENT_RE) rather than
    relaying demucs' own terminal output."""
    cmd = [sys.executable, "-m", "demucs", "-n", model_name, "-o", out_dir]
    if two_stems:
        cmd += ["--two-stems", two_stems]
    cmd.append(input_path)

    # A pty, not a plain pipe, because demucs disables its own progress bar
    # entirely if it doesn't see a real terminal on the other end - we
    # still need it emitting *something* for _PERCENT_RE to read, even
    # though we throw away everything except the percentage itself.
    master_fd, slave_fd = pty.openpty()
    proc = subprocess.Popen(cmd, stdout=slave_fd, stderr=slave_fd, close_fds=True)
    os.close(slave_fd)

    buf = ""
    last_percent = None
    try:
        while True:
            try:
                chunk = os.read(master_fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk.decode(errors="ignore")
            buf = buf[-64:]  # only ever need enough tail to catch the latest "NN%"
            matches = _PERCENT_RE.findall(buf)
            if matches:
                percent = min(int(matches[-1]), 100)
                if percent != last_percent:
                    last_percent = percent
                    filled = int(_BAR_WIDTH * percent / 100)
                    bar = "#" * filled + "-" * (_BAR_WIDTH - filled)
                    sys.stdout.write(f"\r  [{bar}] {percent:3d}%")
                    sys.stdout.flush()
    finally:
        os.close(master_fd)
    proc.wait()
    if last_percent is not None:
        sys.stdout.write("\n")
        sys.stdout.flush()

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    track_name = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(out_dir, model_name, track_name)


ISOLATED_SUFFIX = " (Isolated)"


def song_output_dir(mp3_path: str) -> str:
    """The shared folder a song's isolated instruments (drums, bass, ...)
    are all written into, next to the source mp3 - one folder per song
    rather than a separate one per instrument, so everything for a song
    lives together."""
    title = os.path.splitext(os.path.basename(mp3_path))[0]
    parent = os.path.dirname(os.path.abspath(mp3_path))
    return os.path.join(parent, title + ISOLATED_SUFFIX)


def has_existing_outputs(song_dir: str, label: str, require_midi: bool) -> bool:
    """Whether song_dir already has a wav (and, if require_midi, a matching
    .mid) for the given instrument label (e.g. "Isolated Drums") - matched
    by the label appearing in the filename, so drums and bass sharing one
    song_dir don't see each other's files."""
    if not os.path.isdir(song_dir):
        return False
    entries = os.listdir(song_dir)
    marker = f"({label} at"
    has_wav = any(marker in f and f.endswith(".wav") for f in entries)
    if not require_midi:
        return has_wav
    has_mid = any(marker in f and f.endswith(".mid") for f in entries)
    return has_wav and has_mid


def clear_stale_outputs(song_dir: str, label: str) -> None:
    """Remove any previously-produced .wav/.mid files for this instrument
    label - the filename embeds the tempo, so re-processing with a
    different tempo would otherwise leave old and new copies side by side.
    Only touches files matching this label, leaving any other instrument's
    files in the same shared song_dir alone."""
    if not os.path.isdir(song_dir):
        return
    marker = f"({label} at"
    for f in os.listdir(song_dir):
        if marker in f and (f.endswith(".wav") or f.endswith(".mid")):
            os.remove(os.path.join(song_dir, f))


def write_source_marker(song_dir: str, mp3_path: str, marker_filename: str) -> None:
    stat = os.stat(mp3_path)
    marker = {"path": os.path.abspath(mp3_path), "size": stat.st_size, "mtime": stat.st_mtime}
    with open(os.path.join(song_dir, marker_filename), "w") as f:
        json.dump(marker, f)


def source_marker_matches(song_dir: str, mp3_path: str, marker_filename: str) -> bool:
    """Whether song_dir's existing outputs for this instrument were
    produced from this exact mp3 (matched on size + mtime), not just from a
    folder that happens to be named after this song's title."""
    try:
        with open(os.path.join(song_dir, marker_filename)) as f:
            marker = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    stat = os.stat(mp3_path)
    return marker.get("size") == stat.st_size and marker.get("mtime") == stat.st_mtime


_DEFAULT_TEMPO = 120.0
_MIN_ONSETS_FOR_REFINEMENT = 16
_REFINEMENT_SUBDIVISION = 4  # fit against a 16th-note grid (beat / 4)
_REFINEMENT_BOOTSTRAP_SPAN = 4  # short first pass, to sharpen the estimate before trusting longer spans
_REFINEMENT_MAX_SPAN = 64  # how many onsets ahead to pair each onset with in the second pass
_REFINEMENT_TOLERANCE = 0.15  # fraction of a grid unit a gap may be off by and still count

_TEMPO_WINDOW_SEC = 30.0
_MIN_ONSETS_FOR_WINDOW_TEMPO = 8  # lower bar than a full-song refinement - a single window has far fewer onsets to work with
_TEMPO_DRIFT_THRESHOLD_BPM = 0.3  # any two windows differing by at least this much means the song doesn't have one constant tempo


def _rough_tempo(y: np.ndarray, sr: int) -> float:
    """Array-based rough tempo estimate, only accurate to within a few BPM -
    good enough as a starting point for refine_tempo(). Split out from
    detect_tempo() so a window of already-loaded audio can be estimated
    without round-tripping through disk (see _windowed_tempos)."""
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    return tempo if tempo > 0 else _DEFAULT_TEMPO


def detect_tempo(wav_path: str) -> float:
    """Rough tempo estimate from a wav file - see _rough_tempo."""
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    return _rough_tempo(y, sr)


def _grid_unit_estimate(onset_times: np.ndarray, unit: float, max_span: int) -> float | None:
    """One pass of the gap-ratio averaging described in refine_tempo, for
    onset pairs up to max_span apart. Returns None if nothing usable was
    found (so the caller can fall back to the estimate it already has)."""
    estimates = []
    weights = []
    for offset in range(1, max_span + 1):
        gaps = onset_times[offset:] - onset_times[:-offset]
        multiples = np.round(gaps / unit)
        valid = multiples > 0
        gaps, multiples = gaps[valid], multiples[valid]
        if gaps.size == 0:
            continue
        per_unit = gaps / multiples
        close_enough = np.abs(per_unit - unit) / unit < _REFINEMENT_TOLERANCE
        if not np.any(close_enough):
            continue
        estimates.append(per_unit[close_enough])
        weights.append(multiples[close_enough])  # longer spans -> more precise estimate

    if not estimates:
        return None
    return float(np.average(np.concatenate(estimates), weights=np.concatenate(weights)))


def refine_tempo(onset_times: np.ndarray, initial_tempo: float) -> float:
    """Turn a rough tempo estimate into a precise one using every detected
    onset across a whole song. A single windowed estimate (detect_tempo) is
    only accurate to within a few BPM.

    A naive single global fit (assign each onset an integer grid index from
    the start of the song, then least-squares through all of them) turns
    out to be unstable here: even a couple of BPM of initial error
    accumulates across hundreds of onsets until the "nearest grid index"
    rounding starts picking the wrong integer, and the fit locks onto that
    wrong assignment instead of converging.

    Instead, this looks at gaps between nearby pairs of onsets, works out
    how many grid units each gap likely spans, and derives a small, direct,
    low-drift period estimate from each one - then weight-averages all of
    them. But that alone has the same failure mode at a smaller scale: a
    pair far apart only needs a tiny relative error in the starting guess
    to have its gap rounded to the wrong integer multiple entirely, and
    those (wrong, but heavily-weighted) far-apart pairs would dominate the
    average. So this runs in two passes - a short-span pass first to
    sharpen the estimate enough that long-span pairs can then be trusted,
    then a full-span pass using that sharpened value. Averaging hundreds to
    thousands of these independent estimates in the second pass cancels out
    individual onset-detection jitter and gets this within a small fraction
    of a BPM. Fits against a 16th-note grid rather than the beat itself,
    since hits land on kick quarters, snare backbeats, and hi-hat
    8ths/16ths alike, which uses far more of the song's timing data than
    only looking at one instrument's onsets would.

    This does not fix octave errors (reading half or double the true
    tempo) - that's a separate failure mode of the initial rough estimate;
    refinement just makes whichever octave it picked very precise."""
    onset_times = np.sort(np.asarray(onset_times, dtype=np.float64))
    if initial_tempo <= 0 or onset_times.size < _MIN_ONSETS_FOR_REFINEMENT:
        return initial_tempo

    unit = 60.0 / initial_tempo / _REFINEMENT_SUBDIVISION

    short_span = min(onset_times.size - 1, _REFINEMENT_BOOTSTRAP_SPAN)
    sharpened = _grid_unit_estimate(onset_times, unit, short_span)
    if sharpened is not None:
        unit = sharpened

    full_span = min(onset_times.size - 1, _REFINEMENT_MAX_SPAN)
    if full_span > short_span:
        refined = _grid_unit_estimate(onset_times, unit, full_span)
        if refined is not None:
            unit = refined

    return 60.0 / unit / _REFINEMENT_SUBDIVISION


# A short (~30s) window's own beat tracker can lock onto a different but
# musically-related pulse level than the song's real tempo - e.g. a
# sparser or differently-textured section (a bridge, an instrumental
# break) making a different subdivision of the beat more prominent than
# the main beat itself. That reads as a tempo change but isn't one - it's
# the same tempo measured against a different metric level. Covers
# double/half time (the existing octave-error case refine_tempo's
# docstring already calls out) and 4-against-3 (a beat split into 4 vs
# into 3, e.g. straight 16ths misread against a dotted-8th/triplet feel).
# Deliberately excludes 3/2 (and 2/3): unlike the ratios above, that's also
# a common *genuine* tempo relationship in real tempo changes (e.g. a
# DJ-style 100->150 BPM transition), so auto-correcting it would hide real
# drift this feature exists to catch - reconciling a window's tempo
# against the whole song's before comparing for drift should only remove
# the false positives, not the true ones.
_METRIC_RATIO_CANDIDATES = (1.0, 2.0, 0.5, 4.0 / 3.0, 3.0 / 4.0)
_METRIC_RATIO_TOLERANCE = 0.04  # fraction of the reference tempo a candidate ratio must land within to count as a match


def _reconcile_with_reference(window_tempo: float, reference_tempo: float) -> float:
    """If window_tempo sits at a simple ratio of reference_tempo (close
    enough to plausibly be the same underlying tempo caught at a different
    metric level - see the constants above), rescale it back onto
    reference_tempo's own level and return that instead. Otherwise returns
    window_tempo unchanged, since a genuinely different tempo shouldn't be
    forced to match."""
    if reference_tempo <= 0 or window_tempo <= 0:
        return window_tempo
    best_tempo = window_tempo
    best_error = abs(window_tempo - reference_tempo) / reference_tempo
    for ratio in _METRIC_RATIO_CANDIDATES:
        candidate = window_tempo / ratio
        error = abs(candidate - reference_tempo) / reference_tempo
        if error < _METRIC_RATIO_TOLERANCE and error < best_error:
            best_tempo, best_error = candidate, error
    return best_tempo


def _windowed_tempos(
    y: np.ndarray,
    sr: int,
    onset_times: np.ndarray,
    song_duration_sec: float,
    window_sec: float = _TEMPO_WINDOW_SEC,
    reference_tempo: float | None = None,
) -> list[tuple[float, float, float]]:
    """Independently estimate and refine a tempo within each fixed-length
    window of the song, rather than pooling every onset into one global
    average (which assumes - and hides - a constant tempo throughout).

    Each window gets its own rough estimate (_rough_tempo on just that
    window's audio) rather than reusing the whole song's rough estimate -
    reusing a single shared starting guess would make refine_tempo reject
    a window whose real tempo is too far from it (its grid-fit tolerance
    is only ~15%), which is exactly the failure mode this exists to catch.

    If reference_tempo is given (the whole song's own tempo), each
    window's result is then reconciled against it (see
    _reconcile_with_reference) before being returned, so a window that
    locked onto a different subdivision of the same tempo doesn't read as
    real drift.

    Returns a list of (window_start_sec, window_end_sec, tempo)."""
    onset_times = np.asarray(onset_times, dtype=np.float64)
    windows = []
    start = 0.0
    while start < song_duration_sec:
        end = min(start + window_sec, song_duration_sec)
        y_slice = y[int(start * sr):int(end * sr)]
        window_rough = _rough_tempo(y_slice, sr) if y_slice.size else _DEFAULT_TEMPO
        window_onsets = onset_times[(onset_times >= start) & (onset_times < end)]
        if window_onsets.size >= _MIN_ONSETS_FOR_WINDOW_TEMPO:
            tempo = refine_tempo(window_onsets, window_rough)
        else:
            tempo = window_rough
        if reference_tempo is not None:
            tempo = _reconcile_with_reference(tempo, reference_tempo)
        windows.append((start, end, tempo))
        start = end
    return windows


def _tempo_drift_detected(windows: list[tuple[float, float, float]], threshold_bpm: float = _TEMPO_DRIFT_THRESHOLD_BPM) -> bool:
    tempos = [tempo for _, _, tempo in windows]
    return len(tempos) > 1 and (max(tempos) - min(tempos)) >= threshold_bpm


def _prompt_tempo_choice(windows: list[tuple[float, float, float]]) -> float:
    """Ask which window's tempo to use for the whole song. Defaults to the
    first (beginning of the song) on a bare Enter, since that's usually
    what you'd want when looping the export in a DAW."""
    print("This song's tempo isn't constant throughout - pick which part's tempo to use for the whole export:")
    for i, (start, end, tempo) in enumerate(windows, start=1):
        print(f"  {i}. {start:6.1f}s - {end:6.1f}s: {tempo:.3f} BPM")
    prompt = f"  Which tempo? [1-{len(windows)}, Enter for 1 (the beginning)] "
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return windows[0][2]
        try:
            choice = int(raw)
        except ValueError:
            print("  Please enter a number.")
            continue
        if 1 <= choice <= len(windows):
            return windows[choice - 1][2]
        print(f"  Please enter a number between 1 and {len(windows)}.")


_alignment_cache: dict[tuple[str, int, float], tuple[int, float]] = {}


def song_alignment(mp3_path: str) -> tuple[int, float]:
    """Compute the beat-1 trim point and tempo once, from the full song mix,
    so every instrument isolated from this song can share the exact same
    time origin and tempo grid instead of each independently (and possibly
    inconsistently) re-deriving its own.

    Cached per source mp3 (by path + size + mtime) for the life of the
    process: isolating both drums and bass for the same song calls this
    twice, and without caching a tempo-drift prompt (see
    _prompt_tempo_choice) would ask the same question about the same song
    twice in one run.

    The trim point reuses song_sanitizer's own intro-cut detection against
    the source mp3 - the same detector that already trims the sanitized
    song itself. In the normal pipeline the mp3 passed in has already been
    through song_sanitizer, so this is typically a no-op (~0ms); for
    standalone use against an un-sanitized file it still does real work.

    The tempo is detected and refined against the full (trimmed) mix's own
    onsets rather than any single isolated stem - that way it's available
    regardless of which instruments end up being isolated, and it draws on
    more onset information than any one instrument's onsets alone would."""
    try:
        stat = os.stat(mp3_path)
        cache_key = (os.path.abspath(mp3_path), stat.st_size, stat.st_mtime)
    except OSError:
        cache_key = None
    if cache_key is not None and cache_key in _alignment_cache:
        return _alignment_cache[cache_key]

    audio = AudioSegment.from_file(mp3_path)
    cut_ms = song_sanitizer._find_cut_from_start(audio, audio.dBFS)
    if not (0 < cut_ms < len(audio)):
        cut_ms = 0
    trimmed = audio[cut_ms:] if cut_ms else audio

    tmp_fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    try:
        trimmed.export(tmp_wav, format="wav")
        rough_tempo = detect_tempo(tmp_wav)

        y, sr = librosa.load(tmp_wav, sr=None, mono=True)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        if onset_env.size and np.isfinite(onset_env).any() and onset_env.max() > 0:
            onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=True)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        else:
            onset_times = np.array([], dtype=np.float64)

        tempo = refine_tempo(onset_times, rough_tempo)
    finally:
        os.remove(tmp_wav)

    song_duration_sec = len(trimmed) / 1000.0
    windows = _windowed_tempos(y, sr, onset_times, song_duration_sec, reference_tempo=tempo)
    if _tempo_drift_detected(windows):
        drift = max(t for _, _, t in windows) - min(t for _, _, t in windows)
        print(f"Heads up: this song's tempo drifts by up to {drift:.3f} BPM across its length.")
        if sys.stdin.isatty():
            tempo = _prompt_tempo_choice(windows)
        else:
            # Not a real terminal (piped input, non-interactive run) - don't
            # hang waiting for a choice, just use the beginning of the song.
            tempo = windows[0][2]

    if cache_key is not None:
        _alignment_cache[cache_key] = (cut_ms, tempo)
    return cut_ms, tempo


def trim_and_export(wav_path: str, trim_ms: int, out_path: str) -> None:
    """Slice trim_ms off the front of wav_path (from the shared song_
    alignment(), not a fresh per-file silence scan) and write the result to
    out_path, so every instrument isolated from the same song shares the
    exact same time origin."""
    audio = AudioSegment.from_wav(wav_path)
    if trim_ms:
        audio = audio[trim_ms:]
    audio.export(out_path, format="wav")


def velocities_from_amplitudes(amplitudes: list[float]) -> list[int]:
    """Scale a list of peak hit/note amplitudes into MIDI velocities
    (1-127), normalized against the loudest one in the list - not against
    the raw waveform's global peak sample, which can fall outside any
    detected hit/note window and would then mean nothing ever reaches full
    velocity. Guarantees the loudest hit/note lands at 127 and everything
    else is scaled relative to it."""
    peak = max(amplitudes, default=0.0)
    if peak <= 0:
        return [1 for _ in amplitudes]
    return [int(np.clip(amplitude / peak * 127, 1, 127)) for amplitude in amplitudes]

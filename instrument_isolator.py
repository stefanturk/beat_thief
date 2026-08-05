#!/usr/bin/env python3
"""Shared building blocks for isolating a single instrument's part from a
song and transcribing it to MIDI (see drum_isolator.py, bass_isolator.py).
Nothing in here is specific to any one instrument."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import librosa
import numpy as np
from pydub import AudioSegment

import song_sanitizer


def run_demucs(input_path: str, out_dir: str, model_name: str, two_stems: str | None = None) -> str:
    """Run demucs and return the directory containing the separated stems."""
    cmd = [sys.executable, "-m", "demucs", "-n", model_name, "-o", out_dir]
    if two_stems:
        cmd += ["--two-stems", two_stems]
    cmd.append(input_path)
    # Demucs prints its own progress bar as it separates - leave stdout/
    # stderr connected so the user sees it instead of a long silent wait.
    subprocess.run(cmd, check=True)
    track_name = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(out_dir, model_name, track_name)


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


def _windowed_tempos(y: np.ndarray, sr: int, onset_times: np.ndarray, song_duration_sec: float, window_sec: float = _TEMPO_WINDOW_SEC) -> list[tuple[float, float, float]]:
    """Independently estimate and refine a tempo within each fixed-length
    window of the song, rather than pooling every onset into one global
    average (which assumes - and hides - a constant tempo throughout).

    Each window gets its own rough estimate (_rough_tempo on just that
    window's audio) rather than reusing the whole song's rough estimate -
    reusing a single shared starting guess would make refine_tempo reject
    a window whose real tempo is too far from it (its grid-fit tolerance
    is only ~15%), which is exactly the failure mode this exists to catch.

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


def song_alignment(mp3_path: str) -> tuple[int, float]:
    """Compute the beat-1 trim point and tempo once, from the full song mix,
    so every instrument isolated from this song can share the exact same
    time origin and tempo grid instead of each independently (and possibly
    inconsistently) re-deriving its own.

    The trim point reuses song_sanitizer's own intro-cut detection against
    the source mp3 - the same detector that already trims the sanitized
    song itself. In the normal pipeline the mp3 passed in has already been
    through song_sanitizer, so this is typically a no-op (~0ms); for
    standalone use against an un-sanitized file it still does real work.

    The tempo is detected and refined against the full (trimmed) mix's own
    onsets rather than any single isolated stem - that way it's available
    regardless of which instruments end up being isolated, and it draws on
    more onset information than any one instrument's onsets alone would."""
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
    windows = _windowed_tempos(y, sr, onset_times, song_duration_sec)
    if _tempo_drift_detected(windows):
        drift = max(t for _, _, t in windows) - min(t for _, _, t in windows)
        print(f"Heads up: this song's tempo drifts by up to {drift:.3f} BPM across its length.")
        if sys.stdin.isatty():
            tempo = _prompt_tempo_choice(windows)
        else:
            # Not a real terminal (piped input, non-interactive run) - don't
            # hang waiting for a choice, just use the beginning of the song.
            tempo = windows[0][2]

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

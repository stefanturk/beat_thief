#!/usr/bin/env python3
"""Prepare an isolated stem for listening to inside the app window.

Picking a section by ear needs the audio in the page, and the page can't
have it the obvious way: WebKit refuses to load a `file://` media source
from a `file://` document, so an <audio src> pointing at the wav fails with
a bare "not supported". Measured directly in a real app window, along with
the two things that do work - a local HTTP server, and a data: URI.

This module builds the data: URI, because it keeps the app with no server
and no port (see gui.py). Three things come back:

  - a small mono mp3 of the whole stem, base64'd. It's a listening copy
    only. Whatever gets transcribed is read from the real wav at full
    quality, so this can be as small as it likes.
  - a peak envelope, so the page can draw a waveform without doing any
    audio analysis in JavaScript.
  - every kick in the stem, measured off the real wav (see pulse.py). The
    page snaps markings onto these, draws them, and lines its listening copy
    up against them - the mp3 it plays is not sample-aligned with the wav
    that gets cut, and these are what tell it by how much.

Measured on a 5.8 minute stem: 1.5s to transcode, 3.7MB of base64 across
the bridge in 124ms, and seeks that land exactly."""

from __future__ import annotations

import base64
import os
import subprocess
import tempfile

import numpy as np

import pulse

# What the listening copy is encoded at. Mono and small: it exists to be
# recognised, not to be transcribed, and every byte here crosses the bridge
# as base64 (a third bigger again) before the page can play a note of it.
PREVIEW_BITRATE = "64k"
PREVIEW_SAMPLE_RATE = 22050

# How many buckets the waveform is reduced to. Roughly one per pixel across
# the window - more would be drawing detail nobody can see, and it all has
# to survive being JSON'd across the bridge.
PEAK_BUCKETS = 900

# The peak envelope is measured off a heavily downsampled copy. A waveform
# only has to show where the loud parts are, and reading a six minute stem
# at full rate to draw 900 bars would be most of a second of nothing.
PEAK_SAMPLE_RATE = 4000

_cache: dict[tuple[str, float], dict] = {}


def _run_ffmpeg(args: list[str], stdin: bytes | None = None) -> bytes:
    """ffmpeg writing to stdout, with its own chatter kept out of the way.

    Errors carry ffmpeg's message rather than just a return code - a
    failure here is almost always a codec or a path problem, and the
    message is the whole diagnosis."""
    result = subprocess.run(
        ["ffmpeg", "-loglevel", "error", *args],
        capture_output=True,
        input=stdin,
    )
    if result.returncode != 0:
        detail = result.stderr.decode(errors="ignore").strip() or f"exit {result.returncode}"
        raise RuntimeError(f"ffmpeg failed: {detail}")
    return result.stdout


def _preview_mp3(wav_path: str) -> bytes:
    """The stem as a small mono mp3.

    Written to a temp file rather than taken off a pipe: an mp3 on stdout
    gets no Xing/LAME header, because ffmpeg can't seek back to the start
    to write one. Without it a player has to guess duration and seek by
    extrapolation, which is exactly what a picker does all day."""
    handle, tmp_path = tempfile.mkstemp(suffix=".mp3")
    os.close(handle)
    try:
        _run_ffmpeg([
            "-y", "-i", wav_path,
            "-ac", "1",
            "-ar", str(PREVIEW_SAMPLE_RATE),
            "-b:a", PREVIEW_BITRATE,
            tmp_path,
        ])
        with open(tmp_path, "rb") as preview:
            return preview.read()
    finally:
        os.remove(tmp_path)


def _mono_samples(wav_path: str, rate: int = PEAK_SAMPLE_RATE) -> np.ndarray:
    """The stem as raw mono 16-bit PCM, straight from ffmpeg.

    One decode, no audio library. Split out from _peaks because the kick
    detection reads the same samples, and decoding a six-minute stem twice
    to answer two questions about it would be silly."""
    raw = _run_ffmpeg([
        "-i", wav_path,
        "-ac", "1",
        "-ar", str(rate),
        "-f", "s16le",
        "-",
    ])
    return np.frombuffer(raw, dtype="<i2")


def _peaks(samples: np.ndarray, buckets: int = PEAK_BUCKETS) -> list[float]:
    """A 0-1 loudness bar per bucket, for drawing the waveform.

    Scaled against the stem's own loudest moment rather than full scale: a
    quietly mastered stem would otherwise draw as a flat line, and what the
    waveform is for is telling one part of this song from another."""
    if samples.size == 0:
        return [0.0] * buckets

    # Trim to a whole number of buckets so the reshape is exact; the
    # remainder is at most a bucket's worth at the very end.
    per_bucket = max(1, samples.size // buckets)
    usable = per_bucket * min(buckets, samples.size // per_bucket)
    envelope = np.abs(samples[:usable].astype(np.float32)).reshape(-1, per_bucket).max(axis=1)

    loudest = float(envelope.max())
    if loudest <= 0:
        return [0.0] * len(envelope)
    return [round(float(value) / loudest, 4) for value in envelope]


# How far either way to look for the listening copy's offset, and how sure
# the answer has to be before it is used at all.
_LEAD_SEARCH_SEC = 0.15
_LEAD_MEASURE_SEC = 60.0
_LEAD_FRAME_SEC = 0.001
_LEAD_MIN_RATIO = 1.5


def _rise(samples: np.ndarray, rate: int) -> np.ndarray:
    """Where this audio gets louder, a millisecond at a time."""
    frame = max(1, int(round(_LEAD_FRAME_SEC * rate)))
    frames = samples.size // frame
    if frames < 2:
        return np.zeros(0)
    energy = np.abs(samples[:frames * frame].astype(np.float64)).reshape(frames, frame).mean(axis=1)
    return np.maximum(np.diff(energy, prepend=energy[0]), 0.0)


def _preview_lead(samples: np.ndarray, preview_mp3: bytes, rate: int = PEAK_SAMPLE_RATE) -> float:
    """How far into the listening copy the stem's zero actually sits.

    An mp3 does not come back from a decoder sample-aligned with what went
    in - the encoder puts its own delay on the front, and whether a given
    decoder strips that again is not something to take on trust. The page
    plays the listening copy and cuts the real wav, so any difference lands
    directly on every marking somebody makes by ear.

    So it is measured rather than assumed, and measured by putting the two
    through *identical* processing and asking which shift lines them up. An
    earlier version compared the mp3's loudness against kick times from
    pulse.py and found an 11ms offset that was not an offset at all - it was
    the difference between two ways of deciding when a hit begins. Comparing
    like with like has no room for that.

    Returns 0.0 when nothing lines up convincingly, which is the honest
    answer and also the harmless one."""
    decoded = np.frombuffer(_run_ffmpeg([
        "-i", "pipe:0", "-ac", "1", "-ar", str(rate), "-f", "s16le", "-",
    ], stdin=preview_mp3), dtype="<i2")

    span = int(_LEAD_MEASURE_SEC * rate)
    stem = _rise(samples[:span], rate)
    copy = _rise(decoded[:span], rate)
    reach = int(round(_LEAD_SEARCH_SEC / _LEAD_FRAME_SEC))
    usable = min(stem.size, copy.size) - 2 * reach
    if usable < reach * 4:
        return 0.0

    stem = stem[reach:reach + usable]
    scores = np.array([float(stem @ copy[reach + lag:reach + lag + usable])
                       for lag in range(-reach, reach + 1)])
    average = float(scores.mean())
    if not (average > 0) or float(scores.max()) / average < _LEAD_MIN_RATIO:
        return 0.0
    return float(scores.argmax() - reach) * _LEAD_FRAME_SEC


def duration_sec(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def preview(wav_path: str) -> dict:
    """Everything the page needs to play and draw wav_path.

    Cached per file (by path and mtime) for the life of the process,
    because transcoding is over a second and picking a section means coming
    back to the same stem repeatedly."""
    if not os.path.exists(wav_path):
        raise FileNotFoundError(wav_path)

    key = (os.path.abspath(wav_path), os.path.getmtime(wav_path))
    cached = _cache.get(key)
    if cached is not None:
        return cached

    samples = _mono_samples(wav_path)
    listening_copy = _preview_mp3(wav_path)
    prepared = {
        "audio": "data:audio/mpeg;base64," + base64.b64encode(listening_copy).decode("ascii"),
        # How far the listening copy sits from the stem, measured rather than
        # assumed - see _preview_lead. The page adds this to every position
        # it plays from, so that what you hear is the audio that gets cut.
        "lead": round(_preview_lead(samples, listening_copy), 4),
        "peaks": _peaks(samples),
        # Every kick in the stem, so the page can snap a marking onto one and
        # draw where they are. Measured off the real wav, not off the mp3 the
        # page plays, which is what makes it usable as ground truth for
        # lining the two up (see the page's calibration).
        "kicks": [round(at, 4) for at in pulse.kicks(samples.astype(np.float32), PEAK_SAMPLE_RATE)],
        "duration": duration_sec(wav_path),
        "path": wav_path,
    }
    _cache.clear()  # only ever one stem in hand, and each is megabytes
    _cache[key] = prepared
    return prepared

#!/usr/bin/env python3
"""Does a marked loop come back round on the 1?

Asked of the drumming itself rather than of a tempo. A beat that loops is
one the drummer was already playing on repeat, so the bars that follow the
loop in the stem should sound like the loop. If the loop is a beat long,
what follows is the loop shifted by a beat, and it sounds like it.

So: take the marked start, try a range of loop lengths around the one that
was marked, and for each one compare the drums inside it with the drums
straight after it. The length that matches best is how long the drummer's
phrase really is. If that isn't the length that was marked, the loop would
come back on the wrong beat - and how far out it is says what tempo the grid
should have been at.

Why this rather than measuring a tempo: a tempo measured off a few bars of
kicks has too little to go on (on Dry Rub the kicks agreed on nothing, and
the build's own re-measure was the bug that made its loop a beat long), and
one measured off a loop's own audio has called a loop bad that sounded fine.
This measures the one thing that matters - does it repeat at this length -
on all the drums at once.

Compared as onset envelopes in three bands (kick, snare/body, cymbals), not
as raw audio: a live drummer is never sample-identical twice, but where the
hits fall and which drum they are is. A small slack lets the second pass sit
a few milliseconds off the first, the way a real drummer's does.

What it can't do: say anything when the drummer doesn't repeat - a fill or
a change of section straight after the loop matches no length well. Then
the answer is "no opinion" (None), not a guess.

numpy and scipy only, like pulse.py.
"""

from __future__ import annotations

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfilt

# One envelope frame. A sixteenth at 180 BPM is 83ms, so 5ms resolves a
# length to well under a sixteenth.
HOP_SEC = 0.005

# Kick, the snare's body, cymbals. Separate, so a loop a beat out doesn't
# match on the hats alone - shifted by a beat, the hats still line up and the
# kick and snare don't.
BANDS = ((None, 150.0), (150.0, 2000.0), (5000.0, None))

# How far either side of the marked length is tried, in beats, and how
# finely. A loop a whole beat out is the commonest failure; two covers a
# grid at 2:3 of a short loop's tempo, near enough.
SEARCH_BEATS = 2.5
SEARCH_STEP_BEATS = 1 / 32

# How far the following pass may sit from the first and still count as
# the same hit - the drummer's timing, not the loop's length. Both halves of
# it: the whole pass may sit off by up to SLACK_SEC, and each hit is blurred
# by SMOOTH_SEC so one a few milliseconds early still lands on its twin.
SLACK_SEC = 0.02
SMOOTH_SEC = 0.012

# Below this, what follows the loop doesn't sound like it at any length - a
# fill, a new section, the song ending. No opinion.
MIN_SCORE = 0.3

# Lengths that score within this of the best are treated as ties, and the
# one nearest what was marked wins. A backbeat shifted by half a bar is
# often the same backbeat, and that isn't evidence the marking was wrong.
TIE = 0.04

# How much better than the marked length another one has to match before
# it counts as evidence. Live drumming rarely repeats cleanly at any length,
# so near-equal scores everywhere means "no clear answer", not "a beat out".
CLEAR = 0.15

# A loop out by less than this is a drummer's push and pull, not the wrong
# beat - and a tempo a hair off, which Ableton's metronome won't hear.
OFF_BEATS = 0.4

# At least this much of the stem has to follow the loop to compare with.
MIN_FOLLOWING_BEATS = 4


def _mono(samples: np.ndarray) -> np.ndarray:
    data = samples.astype(np.float64)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(samples.dtype, np.integer):
        data /= float(np.iinfo(samples.dtype).max)
    return data


def envelopes(samples: np.ndarray, rate: float) -> np.ndarray:
    """(bands, frames): how hard something new is hitting, per band, per
    HOP_SEC. Log energy, rectified rise, each band scaled to unit spread so
    a loud kick doesn't drown the hats."""
    hop = max(1, int(round(rate * HOP_SEC)))
    frames = len(samples) // hop
    out = np.zeros((len(BANDS), max(frames, 0)))
    if frames < 2:
        return out
    for row, (low, high) in enumerate(BANDS):
        if low is None:
            sos = butter(4, high, btype="lowpass", fs=rate, output="sos")
        elif high is None:
            sos = butter(4, low, btype="highpass", fs=rate, output="sos")
        else:
            sos = butter(4, (low, high), btype="bandpass", fs=rate, output="sos")
        band = sosfilt(sos, samples)[: frames * hop].reshape(frames, hop)
        energy = np.log1p(1000.0 * np.sqrt((band ** 2).mean(axis=1)))
        rise = np.maximum(0.0, np.diff(energy, prepend=energy[0]))
        rise = gaussian_filter1d(rise, SMOOTH_SEC / HOP_SEC)
        spread = rise.std()
        out[row] = (rise - rise.mean()) / spread if spread > 0 else 0.0
    return out


def _match(env: np.ndarray, length: int, slack: int) -> float | None:
    """How alike env[:, 0:length] and what follows it are, allowing the
    following pass to sit up to `slack` frames off. A correlation, -1..1."""
    first = env[:, :length]
    best = None
    for lag in range(-slack, slack + 1):
        begin = length + lag
        second = env[:, begin:begin + length]
        if begin < 0 or second.shape[1] < length:
            continue
        a, b = first.ravel(), second.ravel()
        a, b = a - a.mean(), b - b.mean()
        norm = np.sqrt((a * a).sum() * (b * b).sum())
        if norm <= 0:
            continue
        score = float((a * b).sum() / norm)
        best = score if best is None else max(best, score)
    return best


def repeat_length(samples: np.ndarray, rate: float, beat_sec: float, beats: int) -> dict | None:
    """The loop length, in beats of `beat_sec`, the drumming in `samples`
    repeats at, searched around `beats`. samples start at the loop's start
    and run on past it.

    Returns {"beats": the best length, "score": how well it matched,
    "marked_score": how well `beats` itself did, "off": whether that is
    clear evidence the loop comes back on the wrong beat (see CLEAR,
    OFF_BEATS)}, or None when nothing repeats well enough to say (see
    MIN_SCORE)."""
    env = envelopes(_mono(samples), rate)
    frames_per_beat = beat_sec / HOP_SEC
    slack = int(round(SLACK_SEC / HOP_SEC))
    # Only lengths that leave a whole second pass to compare against. Short
    # of the stem at the end of a song, the loop is compared with what there
    # is - but never with less than a few beats.
    available = env.shape[1]
    scores = {}
    steps = int(round(SEARCH_BEATS / SEARCH_STEP_BEATS))
    for k in range(-steps, steps + 1):
        length_beats = beats + k * SEARCH_STEP_BEATS
        if length_beats <= 0:
            continue
        length = int(round(length_beats * frames_per_beat))
        if 2 * length + slack > available:
            # Not enough following audio for a full second pass: compare the
            # loop's tail with the same amount of what follows.
            room = available - length - slack
            if room < MIN_FOLLOWING_BEATS * frames_per_beat:
                continue
            trimmed = np.concatenate([env[:, length - room:length], env[:, length:]], axis=1)
            score = _match(trimmed, room, slack)
        else:
            score = _match(env, length, slack)
        if score is not None:
            scores[length_beats] = score
    if not scores:
        return None
    best = max(scores.values())
    if best < MIN_SCORE:
        return None
    close = [length for length, score in scores.items() if score >= best - TIE]
    pick = min(close, key=lambda length: abs(length - beats))
    marked = scores.get(float(beats), -1.0)
    return {
        "beats": float(pick),
        "score": round(scores[pick], 3),
        "marked_score": round(marked, 3),
        # The verdict: the drums clearly repeat at a different length, far
        # enough from the marked one to come back on the wrong beat.
        "off": abs(pick - beats) >= OFF_BEATS and scores[pick] - marked >= CLEAR,
    }


def check(wav_path: str, start_sec: float, beat_sec: float, beats: int) -> dict | None:
    """repeat_length, read straight out of a stem on disk: the loop from
    start_sec, and as much again after it plus the search margin."""
    try:
        rate, data = wavfile.read(wav_path, mmap=True)
    except ValueError:
        # 24-bit can't be mapped; read it whole.
        rate, data = wavfile.read(wav_path)
    longest = (beats + SEARCH_BEATS) * beat_sec
    first = max(0, int(start_sec * rate))
    last = min(len(data), int((start_sec + 2 * longest + SLACK_SEC * 2) * rate))
    if last <= first:
        return None
    return repeat_length(_mono(np.asarray(data[first:last])), rate, beat_sec, beats)

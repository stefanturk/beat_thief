#!/usr/bin/env python3
"""Where the kicks are, and where the bar line sits.

Three small pieces, all deliberately free-standing:

  - KickDetector: audio samples in, kick times out. Fully causal and
    stateful, so it works the same whether it is handed a whole song or a
    stream of 128-sample blocks off a live input.
  - GridPhase: hit times in, grid-line phase out, updated per hit and able
    to forget - the incremental way to track a grid that is moving.
  - fit_grid / grid_track: the same question answered in one pass over a
    section that is already in hand, which is what this app needs.

This module imports nothing but numpy and scipy - no ffmpeg, no file paths,
no Beat Thief. That is on purpose: the same detection is wanted later for a
real-time lighting rig reading a live waveform, and a module that can be
copied out whole is worth more than one that has to be untangled first.
test_pulse.py enforces both the import rule and the block-invariance, so
neither is a claim that can quietly stop being true.

Two things it deliberately does NOT do:

  - Find the tempo from nothing. It refines a tempo it is given (fit_grid
    searches a few percent either side, which matters more than it sounds -
    see there), but it needs a starting guess. Here that comes from the
    stem's filename. A live rig would need a beat tracker in front of this.
  - Tell one bar from another properly. Which sixteenth the grid sits on is
    solid; which step of the bar is the "1" is a best guess off the kick
    pattern alone, and a four-on-the-floor gives it nothing to work with.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from scipy.signal import butter, lfilter, sosfilt, sosfilt_zi

# The band a kick lives in. Chosen to match what the transcriber already
# uses for kick velocity (drum_transcriber's _VELOCITY_BAND), but written
# out as a plain number rather than imported - the whole point of this
# module is that it carries no dependency on the rest of the app.
KICK_BAND = (None, 120.0)

# How long the energy envelope is smoothed over. Long enough to ignore a
# single sample's worth of noise, short enough that two hits a tenth of a
# second apart stay separate. Backtracking (see _emit) takes the lag this
# introduces back off again.
SMOOTH_SEC = 0.02

# Two hits closer together than this are one hit. A kick drum physically
# can't repeat faster, and a drummer's double is one event to a light.
MIN_GAP_SEC = 0.1

# How far back the running level average reaches. A second is long enough to
# ride over a bar of playing and short enough to follow a song getting
# louder.
LEVEL_SEC = 1.0

# How many times its own running level the energy has to stand at to count
# as a hit. Measured on four real isolated drum stems (a hip-hop, two funk,
# one jazz-fusion) rather than picked: this is where the grid the hits agree
# on is sharpest. Higher and a busy stem loses most of its kicks; lower and
# the gaps between them fill up with the tails of the ones already found.
SENSITIVITY = 4.0

# No warmup at all. There used to be one, to ride out the running level
# climbing from zero, and it silently threw away any hit in the opening
# moments - which is exactly where the first hit of a cut loop is. The level
# is bias-corrected instead (see feed), so at the very first sample it reads
# the signal's own value and the ratio is one: a stream that opens mid-kick
# reports that kick, and a stream that opens on noise reports nothing.
WARMUP_SEC = 0.0


def _design(rate: float, band) -> np.ndarray:
    """The band-pass, as second-order sections."""
    low, high = band
    nyquist = rate / 2.0
    if low is not None and high is not None:
        return butter(4, [low / nyquist, min(high / nyquist, 0.99)], btype="band", output="sos")
    if high is not None:
        return butter(4, min(high / nyquist, 0.99), btype="low", output="sos")
    if low is not None:
        return butter(4, low / nyquist, btype="high", output="sos")
    raise ValueError("a band needs at least one edge")


def _dc_group_delay(sos: np.ndarray) -> float:
    """How many samples the filter holds the signal back by, at the low
    frequencies a kick actually occupies.

    This is the first moment of the impulse response, which is exactly the
    group delay at DC. It matters because the detector reports where the
    *filtered* energy starts rising, and that is the true onset plus this.
    Reporting a kick 8ms late every time is the kind of quiet, consistent
    error that makes every cut land slightly wrong.
    """
    impulse = np.zeros(4096)
    impulse[0] = 1.0
    response = sosfilt(sos, impulse)
    weight = response.sum()
    if abs(weight) < 1e-12:
        return 0.0
    return float((np.arange(response.size) * response).sum() / weight)


class KickDetector:
    """Kick onsets out of a stream of samples.

    Feed it audio - all at once, or block by block, in any block sizes you
    like, including sizes smaller than the smoothing window. The answer is
    identical either way: every piece of state that spans a block boundary
    (the filter's memory, the tail of the envelope, the running level, the
    time of the last hit) is carried across explicitly.

    Latency is one smoothing window, about 20ms, which is the price of
    knowing a rise is a rise.
    """

    def __init__(self, rate: float, band=KICK_BAND, min_gap_sec: float = MIN_GAP_SEC,
                 smooth_sec: float = SMOOTH_SEC, sensitivity: float = SENSITIVITY,
                 level_sec: float = LEVEL_SEC, warmup_sec: float = WARMUP_SEC):
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate = float(rate)
        self.sensitivity = float(sensitivity)

        self._sos = _design(self.rate, band)
        self._zi_unit = sosfilt_zi(self._sos)
        self._latency = _dc_group_delay(self._sos)

        self._smooth_n = max(1, int(round(smooth_sec * self.rate)))
        self._gap_n = max(1, int(round(min_gap_sec * self.rate)))
        self._warmup_n = int(round(warmup_sec * self.rate))
        # One-pole average of the rise signal, as a filter so a whole block
        # goes through at once instead of a Python loop per sample.
        self._level_a = 1.0 / max(1.0, level_sec * self.rate)

        self.reset()

    # --- state -----------------------------------------------------------

    def reset(self) -> None:
        """Forget everything, as if no audio had been fed yet."""
        self._zi = self._zi_unit * 0.0
        self._mag_tail = np.zeros(self._smooth_n - 1)
        self._env_tail = np.zeros(self._gap_n)   # for backtracking across a boundary
        self._env_prev = 0.0
        self._level_zi = np.zeros(1)
        self._level_prev = 0.0
        self._was_above = False
        self._n = 0                              # samples consumed, ever
        self._last_hit = -(10 ** 9)              # absolute sample index

    # --- the work --------------------------------------------------------

    def feed(self, block) -> list[float]:
        """Kick times found in this block, in seconds since the stream began.

        Times can point slightly *before* this block - a rise that starts at
        the end of one block is only recognisable in the next, and the
        answer is where it started, not where it was noticed.
        """
        block = np.asarray(block, dtype=np.float64).ravel()
        if block.size == 0:
            return []

        # Start the filter as if the signal had been sitting at its first
        # value forever. From a zero state it instead climbs to the signal
        # over its own settling time, and that climb is a rise like any
        # other - enough to report a kick at time zero in audio that has no
        # kick anywhere in it.
        if self._n == 0:
            self._zi = self._zi_unit * float(block[0])
        filtered, self._zi = sosfilt(self._sos, block, zi=self._zi)

        # Energy envelope: a plain moving average of magnitude, carried over
        # the boundary by keeping the previous block's tail.
        magnitude = np.abs(filtered)
        padded = np.concatenate((self._mag_tail, magnitude))
        if self._smooth_n > 1:
            window = np.full(self._smooth_n, 1.0 / self._smooth_n)
            envelope = np.convolve(padded, window, mode="valid")
            self._mag_tail = padded[-(self._smooth_n - 1):]
            # At the very start the window is averaging over zeros that are
            # not silence, they are the absence of history - so the envelope
            # ramps up for its first window's worth and every stream looks
            # like it opens with a rise. Dividing by how many samples the
            # average has actually seen fixes that, and does it from the
            # absolute sample index so the correction survives being fed in
            # blocks. Without it, audio cut to start exactly on a kick
            # reports a phantom onset at time zero.
            fill = self._smooth_n - 1 - self._n
            if fill > 0:
                seen = np.arange(self._n, self._n + min(fill, envelope.size)) + 1.0
                envelope[:seen.size] *= self._smooth_n / seen
        else:
            envelope = magnitude

        # A hit is energy standing far above what this recording has been
        # sitting at - a ratio, not a fixed line, so nothing needs calibrating
        # per song and a quiet mix reads the same as a loud one. One sample
        # barely moves an average that reaches back a second, so measuring
        # against the level including the current sample costs nothing and
        # makes the very first sample behave: there the two are equal, the
        # ratio is one, and nothing fires on the strength of having no past.
        base = self._n
        self._n += block.size

        level, self._level_zi = lfilter([self._level_a], [1.0, self._level_a - 1.0],
                                        envelope, zi=self._level_zi)
        # The average starts at zero and climbs towards the truth, so for the
        # first few seconds it reads far too low and everything towers over
        # it. Dividing by how much of the average has actually accumulated by
        # each sample corrects that exactly, and does it from the absolute
        # sample index - so it is the same correction however the blocks were
        # cut. The alternative, ignoring the opening seconds, would throw away
        # the start of every song.
        settled = 1.0 - (1.0 - self._level_a) ** (np.arange(base + 1, self._n + 1))
        level = level / np.maximum(settled, 1e-12)
        threshold = level * self.sensitivity
        self._level_prev = float(level[-1])
        self._env_prev = float(envelope[-1])

        # Only the moment it crosses, not every sample it stays above. Without
        # that, one long loud hit reports again every _gap_n samples for as
        # long as it rings.
        above = envelope > threshold
        crossings = np.flatnonzero(above & ~np.concatenate(([self._was_above], above[:-1])))
        self._was_above = bool(above[-1])

        found = []
        for local in crossings:
            at = base + int(local)
            if at < self._warmup_n or at - self._last_hit < self._gap_n:
                continue
            self._last_hit = at
            found.append(self._emit(at, base, envelope))

        # Keep enough envelope to backtrack into from the next block.
        history = np.concatenate((self._env_tail, envelope))
        self._env_tail = history[-self._gap_n:]
        return found

    def _emit(self, at: int, base: int, envelope: np.ndarray) -> float:
        """Where the rise that was noticed at sample `at` actually began.

        The threshold is crossed partway up the slope, which would report
        every kick a little late. Walking back down the slope to its foot
        finds where the rise began instead, and taking the filter's own delay
        off that lands on the transient itself.

        Back down the slope, not back to the quietest sample in the window:
        before a kick in a sparse mix everything is equally quiet, and
        "quietest" would happily walk all the way to the far end of it.
        """
        history = np.concatenate((self._env_tail, envelope))
        # Index of `at` inside `history`: env starts at `base`, and env_tail
        # holds the _gap_n samples before that.
        here = (at - base) + self._env_tail.size
        floor = max(1, here - self._gap_n)
        foot = here
        while foot > floor and history[foot - 1] < history[foot]:
            foot -= 1
        onset = at - (here - foot)
        return max(0.0, (onset - self._latency) / self.rate)


class GridPhase:
    """Where a repeating grid line sits, given how long the repeat is.

    Fed one hit time at a time, it keeps a circular mean of where those hits
    fall within one period. Circular, not plain: hits a hair either side of
    the line have to average to the line, and a plain mean would put them
    half a period away from it. (beat_loop's _grid_origin does the same
    thing for the same reason.)

    `forget` is how much of the past survives each new hit - 1.0 remembers
    everything equally, which is what an offline pass wants; below 1.0 lets
    a section that moved win, which is what a live rig wants.
    """

    def __init__(self, period_sec: float, forget: float = 0.9):
        self.period_sec = float(period_sec) if period_sec > 0 else 0.0
        self.forget = float(forget)
        self.reset()

    def reset(self) -> None:
        self._vector = 0j
        self._weight = 0.0

    def feed(self, at: float) -> None:
        if self.period_sec <= 0:
            return
        angle = 2.0 * math.pi * ((at % self.period_sec) / self.period_sec)
        self._vector = self._vector * self.forget + complex(math.cos(angle), math.sin(angle))
        self._weight = self._weight * self.forget + 1.0

    @property
    def phase(self) -> float:
        """Seconds from zero to the first grid line, in [0, period_sec)."""
        if self.period_sec <= 0 or self._weight <= 0:
            return 0.0
        turns = math.atan2(self._vector.imag, self._vector.real) / (2.0 * math.pi)
        return (turns % 1.0) * self.period_sec

    @property
    def confidence(self) -> float:
        """0 to 1: how much the hits agree. Reported rather than folded into
        the answer, because "0.4 seconds, and nothing agrees" and "0.4
        seconds, dead certain" are different facts and a caller should be
        able to tell them apart."""
        if self.period_sec <= 0 or self._weight <= 0:
            return 0.0
        return abs(self._vector) / self._weight

    def next_line(self, after: float) -> float:
        """The first grid line strictly after `after`. What a lighting cue
        asks for."""
        if self.period_sec <= 0:
            return 0.0
        phase = self.phase
        return phase + (math.floor((after - phase) / self.period_sec) + 1) * self.period_sec


# --- reading a grid off the hits ------------------------------------------

def kicks(samples, rate: float, **options) -> list[float]:
    """Every kick in `samples`, in seconds. One call over the streaming
    detector above, for a caller that has the whole recording in hand."""
    return KickDetector(rate, **options).feed(samples)


# What the grid is fitted against. A sixteenth, not a beat: plenty of real
# kick patterns put a hit on an "and" or an "e", and folding those onto a
# beat drags the answer off the line and drops the agreement below anything
# worth trusting. Every one of them lands on a sixteenth.
SUBDIVISION = 4

# How far from the tempo it is given the fit is allowed to look. The tempo
# reaching this module is a rough whole-song estimate and is routinely off by
# a few tenths of a percent, which is enough to matter (see fit_grid).
TOLERANCE = 0.04
TOLERANCE_STEPS = 401

# Below this many hits there is nothing to fit a grid to, and below this much
# agreement the fit is a coin flip dressed up as a number. Either way the
# honest answer is "no grid", and callers are told so rather than handed a
# number that looks like the others.
MIN_ONSETS = 8
MIN_CONFIDENCE = 0.3

# How much of a song one fit covers, and how far apart consecutive fits sit.
# Sixteen bars is long enough to average out a drummer and short enough that
# the tempo has not moved across it.
WINDOW_BARS = 16
BEATS_PER_BAR = 4


class Grid(NamedTuple):
    """A grid that holds over `start`..`end`.

    `step` is one sixteenth. `downbeat` is an absolute time a bar line falls
    on - the best guess at the "1", which is a weaker claim than the rest of
    this (see fit_grid). `tempo` is what the hits actually say, which is not
    always what the filename says.
    """

    start: float
    end: float
    tempo: float
    step: float
    downbeat: float
    confidence: float

    def line_at(self, index: int) -> float:
        return self.downbeat + index * self.step

    def index_at(self, when: float) -> int:
        """Which step line `when` is nearest to."""
        return int(round((when - self.downbeat) / self.step)) if self.step > 0 else 0

    def snap(self, when: float) -> float:
        return self.line_at(self.index_at(when))


def fit_grid(onsets, tempo: float, tolerance: float = TOLERANCE,
             subdivision: int = SUBDIVISION, beats_per_bar: int = BEATS_PER_BAR):
    """The sixteenth grid `onsets` agree on, searched around `tempo`.

    Returns (step_sec, phase_sec, confidence, downbeat_sec), or a zero
    confidence when there is nothing to go on.

    Why the period is searched rather than taken: the tempo handed in is a
    whole-song estimate, and on real stems it is routinely off by a few
    tenths of a percent. That sounds like nothing and is not - measured on a
    three-minute stem, 0.35% puts the end of the song four and a half
    sixteenths away from where the grid says, so a fit across the whole
    recording agrees on nothing at all (confidence 0.02) while the same hits
    fit over sixteen bars agree strongly (0.6). Hence both halves of this:
    search the period, and fit locally.

    The bar line is the weak part and is deliberately kept separate. Which
    sixteenth the grid sits on is nailed down by the hits; *which* of the
    sixteen steps of the bar is the "1" is settled only by which one carries
    the most hits, and a four-on-the-floor says nothing about it whatsoever.
    Ties go to the earliest. A grid with the lines in the right places and
    the accent on the wrong one is still useful; lines in the wrong places
    are not.
    """
    onsets = [float(at) for at in onsets]
    if tempo <= 0 or len(onsets) < MIN_ONSETS:
        return 0.0, 0.0, 0.0, 0.0

    times = np.asarray(onsets)
    guess = 60.0 / tempo / subdivision
    candidates = guess * np.linspace(1.0 - tolerance, 1.0 + tolerance, TOLERANCE_STEPS)
    vectors = np.array([np.exp(2j * np.pi * (times % step) / step).mean()
                        for step in candidates])
    best = int(np.abs(vectors).argmax())

    step = float(candidates[best])
    confidence = float(abs(vectors[best]))
    phase = (math.atan2(vectors[best].imag, vectors[best].real) / (2.0 * math.pi) % 1.0) * step
    if confidence < MIN_CONFIDENCE:
        return step, phase, confidence, 0.0

    steps_per_bar = beats_per_bar * subdivision
    weight = np.bincount(np.round((times - phase) / step).astype(int) % steps_per_bar,
                         minlength=steps_per_bar)
    return step, phase, confidence, phase + int(weight.argmax()) * step


def downbeat(onsets, tempo: float, **options) -> float:
    """Where the bar line sits across `onsets`, or 0.0 if there is no honest
    answer. Feed it a section, not a whole song - see fit_grid."""
    return fit_grid(onsets, tempo, **options)[3]


def grid_track(onsets, tempo: float, window_bars: int = WINDOW_BARS,
               subdivision: int = SUBDIVISION, beats_per_bar: int = BEATS_PER_BAR,
               tolerance: float = TOLERANCE) -> list[Grid]:
    """The grid across a whole recording, as a run of local fits.

    One fit per half-window, each covering the half it starts on, because no
    single grid holds across a whole song - see fit_grid. Windows with too
    little in them are simply absent from the list, so a caller can tell "no
    grid here" from "a grid here that happens to be poor".
    """
    onsets = sorted(float(at) for at in onsets)
    if tempo <= 0 or len(onsets) < MIN_ONSETS:
        return []

    times = np.asarray(onsets)
    span = window_bars * beats_per_bar * 60.0 / tempo
    hop = span / 2.0
    track = []
    at = times[0]
    while at < times[-1]:
        inside = times[(times >= at) & (times < at + span)]
        step, _, confidence, bar = fit_grid(inside, tempo, tolerance, subdivision, beats_per_bar)
        if confidence >= MIN_CONFIDENCE:
            track.append(Grid(
                start=float(at), end=float(at + hop),
                tempo=60.0 / (step * subdivision), step=step,
                downbeat=bar, confidence=confidence,
            ))
        at += hop
    return track


def grid_at(track, when: float):
    """The Grid covering `when`, or the nearest one, or None. What a page
    redrawing itself as the playhead moves asks for."""
    if not track:
        return None
    for grid in track:
        if grid.start <= when < grid.end:
            return grid
    return min(track, key=lambda grid: min(abs(grid.start - when), abs(grid.end - when)))

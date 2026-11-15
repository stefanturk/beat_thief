#!/usr/bin/env python3
"""Break a song into its distinct parts, without naming them.

A song is a handful of patterns played in turn: this beat for eight bars,
that one for eight, the first again. Given one feature vector per bar, this
finds which bars play the same thing and hands back each distinct pattern
once, with everywhere it plays and a good few bars of it to loop.

It knows nothing about what the features are. For drums they are where in the
bar each drum hits (drum_features below); for a bass or harmony stem they
would be the bar's pitch content. Only the extractor changes, so the same
finder serves every stem.

  1. Bars with next to nothing in them are rests (an intro, a break, the end)
     and are never part of a part.
  2. Every bar is compared with every other, by correlation.
  3. Bars that sound alike are grouped (average-linkage clustering, cut at
     GROUP_DISTANCE).
  4. Consecutive bars of one group make a run. A run shorter than MIN_RUN
     bars is a fill or a transition, not a beat. A group with at least
     MIN_PART_BARS bars left in its runs is a part.
  5. Each part gets one loop: its longest stretch of whole 4-bar phrases from
     a bar line (up to MAX_LOOP_SEC), so the loop runs through the passes and
     keeps the fill that ends each one.

What it can't do: say what a part is (verse, chorus), or tell two parts
apart that differ only in loudness or in what isn't in the features.

numpy and scipy only (plus loop_check, which is too), like pulse.py.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

import loop_check

# A bar with less than this fraction of a typical bar's activity is a rest.
REST_FRACTION = 0.15
# Bars closer than this (1 - correlation, average linkage) are the same beat.
GROUP_DISTANCE = 0.5
# Fewer consecutive bars than this is a fill, not a beat.
MIN_RUN = 2
# A pattern that never plays two bars running (a live drummer who varies it
# every bar) still counts as a beat if it comes back this many times.
MIN_SCATTERED_BARS = 6
# At most this many parts, biggest first: past it they are slight variations.
MAX_PARTS = 8
# A fill of up to this many bars between two runs of one beat doesn't end it.
MERGE_GAP = 2
# A part has at least this many bars across all its runs.
MIN_PART_BARS = 4
# A loop is a whole stretch of a beat's bars, fills and all: a multiple of
# LOOP_PHRASE bars (never fewer than that), as long as the stretch lasts but
# no longer than MAX_LOOP_SEC. A fill of up to MERGE_GAP bars after a stretch
# is part of it, so the loop ends on its fill.
LOOP_PHRASE = 4
MAX_LOOP_SEC = 60.0
# The bar position where the pattern changes most must beat the typical one by
# this much to be called a phrase's one.
PHASE_CONTRAST = 1.3

# Drum features: this many positions per bar (sixteenths).
STEPS = 16


class Part(NamedTuple):
    id: int
    bars: int                           # how many bars it plays, in all
    runs: list                          # [(start_sec, end_sec)]
    loop: tuple                         # (start_sec, bars): a loop worth taking
    confidence: float                   # 0..1: how clearly it is its own beat
    loop_sec: float = 0.0               # how long that loop is

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "bars": self.bars,
            "runs": [[round(a, 3), round(b, 3)] for a, b in self.runs],
            "loop_start": round(self.loop[0], 3),
            "loop_bars": self.loop[1],
            "loop_sec": round(self.loop_sec, 3),
            "confidence": round(self.confidence, 3),
        }


def drum_features(samples: np.ndarray, rate: float, bar_starts) -> np.ndarray:
    """(bars, 3 * STEPS): per bar, how hard the kick, the snare/body and the
    cymbals are hit at each sixteenth. Where in the bar each drum falls is
    what tells a backbeat from a half-time beat from four on the floor."""
    bar_starts = np.asarray(bar_starts, dtype=float)
    if len(bar_starts) < 2:
        return np.zeros((len(bar_starts), 3 * STEPS))
    mono = samples.astype(np.float64)
    if mono.ndim > 1:
        mono = mono.mean(axis=1)
    # Rise above average only: silence has no rise, rather than a constant.
    env = np.maximum(loop_check.envelopes(mono, rate), 0.0)
    bar_len = float(np.median(np.diff(bar_starts)))
    # The envelope's real frame length: its hop is a whole number of samples,
    # so it is not quite loop_check.HOP_SEC, and the difference adds up.
    hop_sec = max(1, int(round(rate * loop_check.HOP_SEC))) / rate
    out = np.zeros((len(bar_starts), env.shape[0], STEPS))
    for i, start in enumerate(bar_starts):
        end = bar_starts[i + 1] if i + 1 < len(bar_starts) else start + bar_len
        # Windows centred on the sixteenths, not starting at them: a hit on a
        # sixteenth is a few ms early or late, and must not flip windows.
        half = (end - start) / STEPS / 2
        edges = np.linspace(start - half, end - half, STEPS + 1) / hop_sec
        for step in range(STEPS):
            a, b = int(round(edges[step])), max(int(round(edges[step + 1])), int(round(edges[step])) + 1)
            a = max(a, 0)
            if a < env.shape[1]:
                out[i, :, step] = env[:, a:b].mean(axis=1)
    return _per_drum(out).reshape(len(bar_starts), -1)


def _per_drum(out: np.ndarray) -> np.ndarray:
    """(bars, bands, steps), each bar's pattern in each band scaled to the same
    size, so the kick's placement counts as much as the hats' - the hats are
    busy in nearly every beat and would drown it. A band that is next to
    silent in a bar stays zero rather than being blown up into noise."""
    level = out.sum(axis=2)                                   # (bars, bands)
    typical = np.array([np.median(col[col > 0]) if (col > 0).any() else 0.0 for col in level.T])
    heard = level >= REST_FRACTION * np.maximum(typical, 1e-12)
    shape = out - out.mean(axis=2, keepdims=True)
    size = np.maximum(np.linalg.norm(shape, axis=2, keepdims=True), 1e-12)
    return np.where(heard[:, :, None] & (level[:, :, None] > 0), shape / size, 0.0)


def _correlation_distance(rows: np.ndarray) -> np.ndarray:
    return pdist(rows, metric="correlation")


def _runs(labels: np.ndarray, active: np.ndarray) -> list[tuple[int, int, int]]:
    """(first bar, one past the last, label): consecutive active bars of one
    group."""
    runs, start = [], None
    for i in range(len(labels) + 1):
        same = (i < len(labels) and active[i] and start is not None
                and labels[i] == labels[start])
        if start is not None and not same:
            runs.append((start, i, int(labels[start])))
            start = None
        if start is None and i < len(labels) and active[i]:
            start = i
    return runs


def _joined(runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Runs of one beat with a short fill between them, as one stretch."""
    out = []
    for a, b in sorted(runs):
        if out and a - out[-1][1] <= MERGE_GAP:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def _gap(runs: list[tuple[int, int]], bar: int) -> int:
    """How many bars `bar` is past the end of the last run before it."""
    ends = [b for _, b in runs if b <= bar]
    return bar - max(ends) + 1 if ends else 10 ** 6


def _phrase_phase(rows: np.ndarray, active: np.ndarray) -> int:
    """Which bar number (mod LOOP_PHRASE) is a phrase's "one". Beats change
    on the one, and the bar before it is the fill: so of every 8 bars, the
    position where the pattern changes most is where a phrase starts. A song
    that changes nowhere in particular (a drummer who varies every bar) says
    nothing, and the grid's own first bar is taken as the one (0)."""
    n = len(rows)
    centred = rows - rows.mean(axis=1, keepdims=True)
    unit = centred / np.maximum(np.linalg.norm(centred, axis=1, keepdims=True), 1e-12)
    change = np.full(n, np.nan)
    both = active[1:] & active[:-1]
    change[1:] = np.where(both, 1.0 - (unit[1:] * unit[:-1]).sum(axis=1), np.nan)
    means = np.full(2 * LOOP_PHRASE, np.nan)
    for k in range(2 * LOOP_PHRASE):
        seen = change[k::2 * LOOP_PHRASE]
        seen = seen[~np.isnan(seen)]
        if len(seen) >= 2:
            means[k] = seen.mean()
    if np.isnan(means).all():
        return 0
    best = int(np.nanargmax(means))
    if means[best] < PHASE_CONTRAST * np.nanmedian(means):
        return 0
    # A fill changes the pattern going in and coming out: the bar after a fill
    # that is nearly as changed as the fill is the one.
    after = means[(best + 1) % (2 * LOOP_PHRASE)]
    if not np.isnan(after) and after >= 0.8 * means[best]:
        best = (best + 1) % (2 * LOOP_PHRASE)
    return best % LOOP_PHRASE


def _loop(stretches: list[tuple[int, int]], similarity: np.ndarray, member: np.ndarray,
          cap_bars: int, phase: int = 0) -> tuple[int, int]:
    """(first bar, bars). A stretch is bars [a, e) of one beat with its
    fills. The loop is the longest run of whole phrases that fits one (and
    the cap), starting a whole number of phrases in; among equal lengths the
    window with the most variety - a fill is unlike the groove, so it counts
    for a window rather than against it. With no stretch of a phrase (a
    drummer who never plays it twice running), the phrase-long window with
    the most of the beat's bars in it."""
    best, best_key = None, None
    for aligned in (True, False):      # a stretch too short to start on a one still gets a loop
        for a, e in stretches:
            if aligned:
                a += (phase - a) % LOOP_PHRASE        # start on a phrase's one
            bars = min((e - a) // LOOP_PHRASE * LOOP_PHRASE, cap_bars)
            if bars < LOOP_PHRASE:
                continue
            for first in range(a, e - bars + 1, LOOP_PHRASE):
                variety = float((1.0 - similarity[first:first + bars]).mean())
                key = (bars, variety, -first)
                if best_key is None or key > best_key:
                    best, best_key = (first, bars), key
        if best is not None:
            break
    if best is not None:
        return best
    n = len(member)
    first = max(range(phase, max(phase + 1, n - LOOP_PHRASE + 1), LOOP_PHRASE),
                key=lambda i: (int(member[i:i + LOOP_PHRASE].sum()), -i))
    return first, LOOP_PHRASE


def find_parts(features, bar_starts) -> list[Part]:
    """The distinct beats in a song. features: (bars, d), one row per bar.
    bar_starts: each bar's start in seconds. Biggest part first."""
    rows = np.asarray(features, dtype=float)
    bar_starts = np.asarray(bar_starts, dtype=float)
    n = len(rows)
    if n < MIN_PART_BARS or n != len(bar_starts) or rows.ndim != 2:
        return []
    bar_len = float(np.median(np.diff(bar_starts))) if n > 1 else 0.0
    if bar_len <= 0:
        return []

    activity = np.linalg.norm(rows, axis=1)
    loud = activity[activity > 0]
    if not len(loud):
        return []
    active = activity >= REST_FRACTION * np.median(loud)
    # A bar with no shape (the same value all through) can't be compared.
    active &= rows.std(axis=1) > 0
    index = np.flatnonzero(active)
    if len(index) < MIN_PART_BARS:
        return []

    labels = np.zeros(n, dtype=int)
    if len(index) == 1:
        labels[index] = 1
    else:
        found = fcluster(linkage(_correlation_distance(rows[index]), "average"),
                         GROUP_DISTANCE, "distance")
        labels[index] = found

    every: dict[int, list[tuple[int, int]]] = {}
    for a, b, label in _runs(labels, active):
        every.setdefault(label, []).append((a, b))
    kept = {}
    for g, r in every.items():
        long_runs = [x for x in r if x[1] - x[0] >= MIN_RUN]
        if sum(b - a for a, b in long_runs) >= MIN_PART_BARS:
            kept[g] = long_runs                        # a fill or two is dropped
        elif sum(b - a for a, b in r) >= MIN_SCATTERED_BARS:
            kept[g] = r                                # comes back, never twice running
    if not kept:
        return []

    centred = rows - rows.mean(axis=1, keepdims=True)
    unit = centred / np.maximum(np.linalg.norm(centred, axis=1, keepdims=True), 1e-12)
    centroids, within = {}, {}
    for g, r in kept.items():
        bars = [i for a, b in r for i in range(a, b)]
        centre = unit[bars].mean(axis=0)
        centre /= max(np.linalg.norm(centre), 1e-12)
        centroids[g] = centre
        within[g] = float((unit[bars] @ centre).mean())

    owner = np.full(n, -1)
    for g, r in kept.items():
        for a, b in r:
            owner[a:b] = g
    cap_bars = max(LOOP_PHRASE, int(MAX_LOOP_SEC / bar_len) // LOOP_PHRASE * LOOP_PHRASE)

    phase = _phrase_phase(rows, active)
    parts = []
    for g, r in kept.items():
        similarity = unit @ centroids[g]
        stretches = []
        for a, b in _joined(r):
            # The fill that ends a pass belongs to it.
            while b < n and active[b] and owner[b] == -1 and _gap(r, b) <= MERGE_GAP:
                b += 1
            stretches.append((a, b))
        first, bars = _loop(stretches, similarity, owner == g, cap_bars, phase)
        others = [float(centroids[g] @ centroids[h]) for h in kept if h != g]
        gap = within[g] - (max(others) if others else 0.0)
        end = lambda i: float(bar_starts[i]) if i < n else float(bar_starts[-1] + bar_len)
        parts.append((
            sum(b - a for a, b in r),
            min(a for a, _ in r),
            [(float(bar_starts[a]), end(b)) for a, b in _joined(r)],
            (float(bar_starts[first]), bars),
            float(np.clip(gap, 0.0, 1.0)),
            end(first + bars) - float(bar_starts[first]),
        ))
    parts.sort(key=lambda p: (-p[0], p[1]))
    parts = parts[:MAX_PARTS]
    return [Part(i, total, runs_sec, loop, conf, sec)
            for i, (total, _, runs_sec, loop, conf, sec) in enumerate(parts)]

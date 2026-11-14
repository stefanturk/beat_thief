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
  5. Each part gets one loop: 1, 2 or 4 bars starting on a bar line, from the
     bars most like the rest of the part, starting where its phrase starts.

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
# A fill of up to this many bars between two runs of one beat doesn't end it.
MERGE_GAP = 2
# A part has at least this many bars across all its runs.
MIN_PART_BARS = 4
# Loops are this long, longest first; the longest that fits a run is used.
LOOP_BARS = (4, 2, 1)

# Drum features: this many positions per bar (sixteenths).
STEPS = 16


class Part(NamedTuple):
    id: int
    bars: int                           # how many bars it plays, in all
    runs: list                          # [(start_sec, end_sec)]
    loop: tuple                         # (start_sec, bars): a loop worth taking
    confidence: float                   # 0..1: how clearly it is its own beat

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "bars": self.bars,
            "runs": [[round(a, 3), round(b, 3)] for a, b in self.runs],
            "loop_start": round(self.loop[0], 3),
            "loop_bars": self.loop[1],
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


def _loop(runs: list[tuple[int, int]], similarity: np.ndarray) -> tuple[int, int]:
    """(first bar, bars): the longest of LOOP_BARS that fits a run, starting
    where a phrase of that length would (the run's start plus whole loops),
    from the window most like the rest of the part."""
    longest = max(b - a for a, b in runs)
    bars = next((n for n in LOOP_BARS if n <= longest), 1)
    best, best_score = None, -np.inf
    for a, b in runs:
        for first in range(a, b - bars + 1, bars):
            score = float(similarity[first:first + bars].mean())
            # Earlier wins a tie, and a longer run is the steadier one.
            score += 1e-6 * (b - a) - 1e-9 * first
            if score > best_score:
                best, best_score = first, score
    return best, bars


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

    runs = [r for r in _runs(labels, active) if r[1] - r[0] >= MIN_RUN]
    by_group: dict[int, list[tuple[int, int]]] = {}
    for a, b, label in runs:
        by_group.setdefault(label, []).append((a, b))
    kept = {g: r for g, r in by_group.items() if sum(b - a for a, b in r) >= MIN_PART_BARS}
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

    parts = []
    for g, r in kept.items():
        similarity = unit @ centroids[g]
        first, bars = _loop(r, similarity)
        others = [float(centroids[g] @ centroids[h]) for h in kept if h != g]
        gap = within[g] - (max(others) if others else 0.0)
        end = lambda i: float(bar_starts[i]) if i < n else float(bar_starts[-1] + bar_len)
        parts.append((
            sum(b - a for a, b in r),
            min(a for a, _ in r),
            [(float(bar_starts[a]), end(b)) for a, b in _joined(r)],
            (float(bar_starts[first]), bars),
            float(np.clip(gap, 0.0, 1.0)),
        ))
    parts.sort(key=lambda p: (-p[0], p[1]))
    return [Part(i, total, runs_sec, loop, conf)
            for i, (total, _, runs_sec, loop, conf) in enumerate(parts)]

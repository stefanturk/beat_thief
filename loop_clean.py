#!/usr/bin/env python3
"""Repair a stolen loop by comparing it with the drummer's other passes.

A drummer plays the same pattern over and over, but a stem splitter damages
every pass differently: a snare comes out hollow here, the cymbals go phasey
there, a bit of ad lib or rhythm guitar leaks through somewhere else. So the
other passes say what each moment of the loop should sound like.

  - Sound the marked pass has and the others don't (bleed): turned down.
  - Sound the others have and the marked pass lacks (hollow, phasey): filled
    in from the pass that sounds most like the typical one - real audio from
    the same drummer, not synthesised.
  - A hit that happens once (a crash, a ghost note): protected. It is
    broadband and nothing else has it, which is what a drum looks like and a
    guitar doesn't.

Known as REPET-SIM when used to pull a repeating part out of a mix; here it
is turned round to repair the repeating part itself.

What it can't do: fix what is the same in every pass. A guitar riff played
identically each bar, or a hit that is hollow in the same way each time,
agrees with itself and survives. And when nothing repeats nearby the loop is
returned untouched, and the report says so.

numpy and scipy only (plus loop_check, which is too), like pulse.py.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, maximum_filter, minimum_filter
from scipy.signal import butter, istft, sosfilt, stft

import loop_check

# The repeat unit is tried at these many bars, shortest first.
UNIT_BARS = (1, 2, 4)
# Passes this many bars either side of the loop are candidates, and no more
# than MAX_REFERENCES (the best matching) are used: a median needs a handful,
# not sixty.
SEARCH_BARS = 32
MAX_REFERENCES = 12
MIN_REFERENCES = 2
# A candidate has to sound this much like the loop (envelope correlation).
MIN_MATCH = 0.5

# How far the drummer may have pushed or pulled a whole pass, and the
# resolution the alignment is found at.
ALIGN_SEC = 0.03
ALIGN_HOP_SEC = 0.001
# A drummer is also a few ms off hit by hit within a pass, so each beat of a
# reference is lined up on its own (this far either way) and the beats are
# crossfaded back together.
LOCAL_ALIGN_SEC = 0.012
CROSSFADE_SEC = 0.004

# A bin this far over / under the consensus is bleed / hollow.
THRESHOLD_DB = 6.0
# And no correction goes beyond this.
MAX_DB = 12.0

# Magnitudes are compared after blurring, because the bins of a noisy drum
# hit are wildly different from pass to pass even when the drum is the same.
BLUR_FRAMES = 1.0
BLUR_BINS = 2.0
# Masks are smoothed, or the repair is a warble of its own.
MASK_FRAMES = 1.0
MASK_BINS = 1.5

# One-off hit protection: a jump of this much over the previous frame, in
# sound the consensus doesn't have, over at least BREADTH of BAND, is a drum.
# It stays protected until it has decayed back to the consensus, but no
# longer than PROTECT_SEC.
ONSET_DB = 6.0
BREADTH = 0.5
BAND = (150.0, 10000.0)
PROTECT_SEC = 1.0

EPS = 1e-9


def _mono(samples: np.ndarray) -> np.ndarray:
    return samples.astype(np.float64).mean(axis=1) if samples.ndim > 1 else samples.astype(np.float64)


def _lag_envelope(x: np.ndarray, rate: float) -> np.ndarray:
    """Where the sound hits, at ALIGN_HOP_SEC: rectified, smoothed."""
    hop = max(1, int(round(rate * ALIGN_HOP_SEC)))
    frames = len(x) // hop
    if frames < 2:
        return np.zeros(0)
    sos = butter(2, (100.0, min(8000.0, rate * 0.45)), btype="bandpass", fs=rate, output="sos")
    band = np.abs(sosfilt(sos, x))[: frames * hop].reshape(frames, hop).mean(axis=1)
    return gaussian_filter1d(band, 2.0)


def _align(loop: np.ndarray, region: np.ndarray, rate: float, start: int, length: int,
           slack_sec: float = ALIGN_SEC) -> int:
    """The lag, in samples, to add to `start` so that region[start+lag:...]
    lines up with `loop`. `region` is the stem; `start` is where the pass
    would begin on a perfectly steady drummer."""
    hop = max(1, int(round(rate * ALIGN_HOP_SEC)))
    slack = int(round(slack_sec / ALIGN_HOP_SEC))
    pad = slack * hop
    lo, hi = start - pad, start + length + pad
    if lo < 0 or hi > len(region):
        return 0
    target = _lag_envelope(loop, rate)
    around = _lag_envelope(region[lo:hi], rate)
    if len(target) < 2 or len(around) < len(target):
        return 0
    target = target - target.mean()
    best, best_lag = -np.inf, 0
    for lag in range(0, 2 * slack + 1):
        part = around[lag:lag + len(target)]
        if len(part) < len(target):
            break
        part = part - part.mean()
        norm = np.sqrt((part * part).sum() * (target * target).sum())
        if norm <= 0:
            continue
        score = float((part * target).sum() / norm)
        if score > best:
            best, best_lag = score, lag
    return (best_lag - slack) * hop


def _references(stem: np.ndarray, rate: float, first: int, length: int, bar_starts) -> list[int]:
    """Start samples of the passes that sound like the loop, best first."""
    bars = np.asarray(sorted(bar_starts), dtype=float)
    if len(bars) < 2 or length < 1:
        return []
    mono = _mono(stem)
    loop = mono[first:first + length]
    if len(loop) < length:
        return []
    marked = int(np.argmin(np.abs(bars - first / rate)))
    # The loop may not start exactly on a bar line; keep the offset.
    shift = int(round(first - bars[marked] * rate))
    loop_env = loop_check.envelopes(loop, rate)
    for unit in UNIT_BARS:
        found = []
        for k in range(-(SEARCH_BARS // unit), SEARCH_BARS // unit + 1):
            index = marked + k * unit
            if k == 0 or not 0 <= index < len(bars):
                continue
            start = int(round(bars[index] * rate)) + shift
            if start < 0 or start + length > len(mono):
                continue
            start += _align(loop, mono, rate, start, length)
            if start < 0 or start + length > len(mono):
                continue
            env = loop_check.envelopes(mono[start:start + length], rate)
            if env.shape != loop_env.shape or env.size == 0:
                continue
            a, b = loop_env.ravel() - loop_env.mean(), env.ravel() - env.mean()
            norm = np.sqrt((a * a).sum() * (b * b).sum())
            score = float((a * b).sum() / norm) if norm > 0 else 0.0
            if score >= MIN_MATCH:
                found.append((score, start))
        if len(found) >= MIN_REFERENCES:
            found.sort(reverse=True)
            return [start for _, start in found[:MAX_REFERENCES]]
    return []


def _beatwise(stem: np.ndarray, mono: np.ndarray, loop: np.ndarray, rate: float,
              start: int, length: int, chunk: int) -> np.ndarray:
    """(channels, length): the pass that begins at `start`, each `chunk` of it
    nudged onto the loop's own hits."""
    fade = max(1, int(round(CROSSFADE_SEC * rate)))
    out = np.zeros((stem.shape[1], length))
    for begin in range(0, length, chunk):
        end = min(length, begin + chunk)
        lag = _align(loop[begin:end], mono, rate, start + begin, end - begin, LOCAL_ALIGN_SEC)
        a = start + begin + lag
        if a < 0 or a + (end - begin) > len(stem):
            a = start + begin
        reach = min(end - begin + fade, len(stem) - a, length - begin)
        if reach <= 0:
            continue
        piece = stem[a:a + reach].T.astype(np.float64)
        weight = np.ones(reach)
        if begin > 0:
            weight[:min(fade, reach)] *= np.linspace(0.0, 1.0, min(fade, reach))
        if reach > end - begin:
            weight[end - begin:] *= np.linspace(1.0, 0.0, reach - (end - begin) + 2)[1:-1][: reach - (end - begin)]
        out[:, begin:begin + reach] += piece * weight
    return out


def _blur(power: np.ndarray) -> np.ndarray:
    return gaussian_filter(power, (BLUR_BINS, BLUR_FRAMES))


def _protected(marked_power, median_power, diff_db, freqs, hop_sec) -> np.ndarray:
    """Frames holding a one-off drum hit and its decay: a sudden jump in
    sound the other passes don't have, across most of the band (a drum), not
    in a few narrow lines (a tone, a chord, a voice)."""
    band = slice(int(np.searchsorted(freqs, BAND[0])), int(np.searchsorted(freqs, BAND[1])))
    mine_db = 10 * np.log10(marked_power[band].sum(axis=0) + EPS)
    theirs_db = 10 * np.log10(median_power[band].sum(axis=0) + EPS)
    jump = np.diff(mine_db, prepend=mine_db[0])
    wide = (diff_db[band] >= ONSET_DB).mean(axis=0)
    onsets = (jump >= ONSET_DB) & (mine_db - theirs_db >= ONSET_DB) & (wide >= BREADTH)
    keep = np.zeros(len(mine_db), dtype=bool)
    longest = max(1, int(PROTECT_SEC / hop_sec))
    for frame in np.flatnonzero(onsets):
        end = frame
        while end < len(mine_db) and end - frame < longest and mine_db[end] - theirs_db[end] >= 3.0:
            end += 1
        keep[max(0, frame - 1):end + 1] = True
    return keep


def _correct(channels: np.ndarray, others: list[np.ndarray], rate: float):
    """channels: (ch, n) the marked loop. others: (ch, n) per reference.
    Returns the repaired (ch, n) and the dB correction per bin."""
    nper = int(2 ** round(np.log2(1024 * rate / 44100.0)))
    nper = max(nper, 256)
    hop = nper // 4

    def spec(x):
        return stft(x, rate, window="hann", nperseg=nper, noverlap=nper - hop, boundary="zeros", padded=True)[2]

    mine = [spec(c) for c in channels]
    theirs = [[spec(c) for c in ref] for ref in others]
    freqs = np.fft.rfftfreq(nper, 1.0 / rate)

    mid = sum(mine) / len(mine)
    mid_refs = [sum(chans) / len(chans) for chans in theirs]
    my_power = _blur(np.abs(mid) ** 2)
    ref_power = np.stack([_blur(np.abs(r) ** 2) for r in mid_refs])
    median = np.median(ref_power, axis=0)
    diff_db = 10 * np.log10((my_power + EPS) / (median + EPS))

    # Bleed: over the consensus. Hollow: under it.
    bump = diff_db > THRESHOLD_DB
    dip = diff_db < -THRESHOLD_DB
    attenuate = -np.clip(np.where(bump, diff_db, 0.0), 0, MAX_DB)
    # Keep the depth of a cut over its neighbours before smoothing it.
    attenuate = minimum_filter(attenuate, size=(3, 3))
    attenuate = gaussian_filter(attenuate, (MASK_BINS, MASK_FRAMES))
    protect = _protected(my_power, median, diff_db, freqs, hop / rate)
    attenuate[:, protect[: attenuate.shape[1]]] = 0.0
    gain = 10 ** (attenuate / 20.0)

    # Fill from the reference whose level at that bin is closest to the median.
    pick = np.abs(np.log(ref_power + EPS) - np.log(median + EPS)[None]).argmin(axis=0)
    weight = np.clip((-diff_db - THRESHOLD_DB) / (THRESHOLD_DB / 2), 0, 1) * dip
    weight = gaussian_filter(maximum_filter(weight, size=(3, 3)), (MASK_BINS, MASK_FRAMES))
    weight = np.clip(weight * 1.5, 0, 1)
    frames = np.arange(my_power.shape[1])[None, :]
    bins = np.arange(my_power.shape[0])[:, None]

    repaired = []
    for c, own in enumerate(mine):
        donor = np.stack([theirs[i][c] for i in range(len(theirs))])[pick, bins, frames]
        # Never fill louder than the consensus itself.
        donor_mag = np.abs(donor) + EPS
        limit = 10 ** (MAX_DB / 20.0) * np.sqrt(median)
        donor = donor * np.minimum(1.0, limit / donor_mag)
        mixed = own * gain * (1.0 - weight) + donor * weight
        repaired.append(istft(mixed, rate, window="hann", nperseg=nper, noverlap=nper - hop,
                              boundary=True)[1])
    out = np.stack([np.resize(r, channels.shape[1]) if len(r) < channels.shape[1] else r[:channels.shape[1]]
                    for r in repaired])
    db = np.where(bump, attenuate, 0.0) - np.where(dip, weight * THRESHOLD_DB, 0.0)
    return out, db


def clean(samples: np.ndarray, rate: float, loop_start_sec: float, loop_len_sec: float,
          bar_starts) -> tuple[np.ndarray, dict]:
    """The loop out of `samples` (a whole drum stem, float, (n,) or (n, ch)),
    repaired against the passes on either side of it. bar_starts are the
    stem's bar lines, in seconds.

    Returns (cleaned, report). cleaned has the loop's exact length and the
    stem's channels. report is {"references": 0} when there was nothing to
    compare with - and then cleaned is the raw cut, untouched - or
    {"references": n, "changed_db": the average size of the corrections,
    "cleaned_fraction": the share of the loop they touched}."""
    first = int(round(loop_start_sec * rate))
    length = int(round(loop_len_sec * rate))
    stem = samples if samples.ndim > 1 else samples[:, None]
    width = stem.shape[1]
    raw = np.zeros((length, width), dtype=np.float32)
    have = max(0, min(length, len(stem) - max(first, 0)))
    if have:
        raw[:have] = stem[max(first, 0):max(first, 0) + have]
    shaped = raw if samples.ndim > 1 else raw[:, 0]

    starts = _references(stem, rate, first, length, bar_starts) if first >= 0 else []
    if len(starts) < MIN_REFERENCES:
        return shaped, {"references": 0}

    channels = raw.T.astype(np.float64)
    bars = np.asarray(sorted(bar_starts), dtype=float)
    chunk = max(1, int(round(float(np.median(np.diff(bars))) * rate / 4)))
    mono = _mono(stem)
    loop_mono = mono[first:first + length]
    others = [_beatwise(stem, mono, loop_mono, rate, s, length, chunk) for s in starts]
    out, db = _correct(channels, others, rate)
    out = np.clip(out, -1.0, 1.0).astype(np.float32).T

    touched = np.abs(db) >= 1.0
    report = {
        "references": len(starts),
        "changed_db": round(float(np.abs(db[touched]).mean()), 2) if touched.any() else 0.0,
        "cleaned_fraction": round(float(touched.mean()), 3),
    }
    return (out if samples.ndim > 1 else out[:, 0]), report

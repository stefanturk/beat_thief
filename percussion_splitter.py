#!/usr/bin/env python3
"""Ask of one drum hit: was that a drum, or was it something being shaken?

The transcription model has five classes and none of them is a tambourine, so
percussion arrives labelled as a snare. This is the audio half of putting it
back where it belongs. It is deliberately a small, weak measurement, and the
weakness is measured rather than assumed - groove_reader takes the votes of a
whole voice and decides. A wrong answer here costs one vote.

What was tried, on the drum stem of Officially Missing You (Brasstracks), and
why it isn't what this does:

  - **Band level in a window.** In a dense mix the loudest thing in the
    150-500Hz window near any onset is the kick's decay or the last snare's
    ring, not this hit. One unimodal blob, no separation anywhere in it.
  - **Velocity.** On-quarter median 102, off-quarter 103.
  - **Attack time and ring length.** Both came back measuring how busy the
    surrounding mix was: ring read 13ms in a sparse intro and 55ms in a dense
    outro, which is the opposite of the truth about those sounds. This is why
    nothing here tries to tell a tambourine from a shaker - that distinction
    is made by groove_reader from what the voice plays, because no measurement
    of the sound supported it.

What works is the *rise*: how much a band jumps at the onset compared with the
moment just before it. That cancels whatever was already ringing, which is the
whole problem with measuring a level. And it has to be read as a difference
between two bands rather than on its own, because every band rises more in a
sparse arrangement - that's a fact about the mix, not the instrument.

So: how much brighter did it get, minus how much more body it got. Medians
over hits that nothing else landed within 70ms of, by section:

    intro   (a voice on the quarters)        +0.6 dB
    outro   (a voice on the sixteenths)      -1.4 dB
    chorus                                   -9.9 dB
    late                                    -11.1 dB
    verse                                   -14.8 dB

The gap is between -1.4 and -9.9, so the line is drawn at -5."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

# The shell of a drum. Deliberately above the kick, which owns everything
# below about 150Hz and would otherwise decide every answer: on real material
# the hits whose low end jumped fifty decibels turned out to be exactly the
# steps the kick class had already flagged.
_BODY_HZ = (180.0, 400.0)

# Jingles, beads, wires. A tambourine and a shaker live here and a drum shell
# doesn't.
_JINGLE_HZ = (7000.0, 14000.0)

# The attack, and the moment just before it that the attack is measured
# against. The gap before zero keeps the onset's own leading edge out of the
# "before" window - onsets are detected on a 10ms frame, so the true attack
# can sit a few milliseconds either side of where it was reported.
_BEFORE_SEC = (-0.035, -0.008)
_AFTER_SEC = (0.0, 0.025)

# Brighter than it got heavier, by this much, and it isn't a drum. See the
# table in the module docstring - the two populations sit at about +0 and
# about -11, and this is the middle of the gap.
_PERCUSSION_SCORE_DB = -5.0

# Under this the hit is treated as unmeasurable and left a snare, which is
# the conservative answer throughout Beat Thief (see
# drum_transcriber._split_hihats). Silence has a rise of zero in every band,
# which would otherwise read as a perfect percussion score.
_AUDIBLE = 1e-4

# What a hit voted "not a drum" is called on the way out. It's provisional:
# groove_reader either confirms the whole voice - possibly renaming it to a
# shaker - or puts it back on the snare. The name is a real drum rack pad
# rather than a placeholder so that a caller who skips groove_reader still
# gets something playable.
PERCUSSION = "tambourine"
SNARE = "snare"


def _band(audio: np.ndarray, sample_rate: int, low: float, high: float) -> np.ndarray:
    nyquist = sample_rate / 2.0
    sos = butter(4, [low / nyquist, min(high, nyquist * 0.99) / nyquist],
                 btype="bandpass", output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def _peak(band: np.ndarray, time_sec: float, span: tuple[float, float], sample_rate: int) -> float:
    start = int((time_sec + span[0]) * sample_rate)
    end = int((time_sec + span[1]) * sample_rate)
    start, end = max(0, start), min(band.size, end)
    return float(np.abs(band[start:end]).max()) if end > start else 0.0


def _db(amplitude: float, floor: float = 1e-7) -> float:
    return 20.0 * float(np.log10(max(amplitude, floor)))


def score(audio: np.ndarray, times: list[float], sample_rate: int) -> list[float]:
    """How much more each hit brightened than it thickened, in decibels.

    Positive is percussion, negative is a drum. Returned separately from the
    verdict so the threshold can be looked at against real material without
    re-deriving it."""
    if not times:
        return []
    body = _band(audio, sample_rate, *_BODY_HZ)
    jingle = _band(audio, sample_rate, *_JINGLE_HZ)

    scores = []
    for time_sec in times:
        loudest = max(_peak(body, time_sec, _AFTER_SEC, sample_rate),
                      _peak(jingle, time_sec, _AFTER_SEC, sample_rate))
        if loudest < _AUDIBLE:
            scores.append(float("-inf"))  # nothing there to measure
            continue
        rise = []
        for band in (jingle, body):
            after = _peak(band, time_sec, _AFTER_SEC, sample_rate)
            before = _peak(band, time_sec, _BEFORE_SEC, sample_rate)
            rise.append(_db(after) - _db(before))
        scores.append(rise[0] - rise[1])
    return scores


def split(audio: np.ndarray, times: list[float], sample_rate: int,
          threshold: float = _PERCUSSION_SCORE_DB) -> list[str]:
    """For each hit the model called a snare: SNARE, or PERCUSSION.

    A vote, not a verdict. groove_reader gets the last word, and is built on
    the assumption that a good fraction of these are wrong.

    `threshold` defaults to the line calibrated on Officially Missing You,
    but a caller with a better idea of what "real" sounds like on this
    specific recording - see drum_transcriber.calibrate_hat_threshold - can
    supply its own."""
    return [PERCUSSION if value > threshold else SNARE
            for value in score(audio, times, sample_rate)]

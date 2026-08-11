#!/usr/bin/env python3
"""Turn a section of an isolated drum stem into a loop.

This is the part that stops trying to transcribe a performance and starts
rebuilding a beat. A whole-song transcription was faithful and unusable; two
bars on a grid are neither, and the difference is that here the output is
allowed to be *tidier* than what was played.

Given a stem, a tempo, and the piece of it somebody marked by ear:

  1. transcribe just that section (drum_transcriber, with padding either
     side so the model has context at the edges)
  2. measure the tempo again from those hits, starting from the song's
     rough estimate (instrument_isolator.refine_tempo)
  3. work out how many bars the section is, at that tempo
  4. find where the grid sits within it, from the hits themselves
  5. quantize onto it, and hand the result to beat_writer

Steps 2 and 4 are the ones that matter, and they're the same idea twice: the
beat determines the grid, not the other way around. A whole-song tempo is an
average of something that drifts across the length of a song, so it's only a
starting point; what the loop is measured against is the drumming inside the
section somebody picked.

Note what step 2 is *not*: dividing the marked span by the bar count. That's
the tempting shortcut and it hands the grid to whichever edge was cut
sloppily - measured on a real two-bar section marked about 7% long, it put
the grid at 97.5 BPM against a song at 105.4 and left a third of the hits
sitting a hair off the halfway line between two sixteenths, so which one they
landed on was a coin toss. The playing is a better clock than the edges.

What isn't done here is second-guessing where the downbeat is. There used to
be a step that scored every rotation of the loop for how much it looked like
4/4 and turned the beat round to suit. It was wrong often enough to matter -
a kick on the and-of-four would win over the one, and the whole loop came out
a beat late - and it was wrong in a way nobody could correct, since it
silently overrode where you clicked. The picker lets you play from the start
you've marked and walk it with the arrow keys, so the one is something you
hear and place. That's a better instrument than a heuristic, and this file's
job is to take it at its word."""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
from typing import NamedTuple

import numpy as np

import beat_writer
import drum_transcriber
import instrument_isolator

# The model is bidirectional and judges a hit partly on what surrounds it,
# so a section cut out and handed over on its own loses hits at both ends.
# Transcribing with this much extra on each side and then discarding it
# costs a second and makes the edges as good as the middle.
_CONTEXT_SEC = 3.0

# The grid the loop is quantized onto. Sixteenths are what nearly every
# drum part is written in, and finer would be preserving the timing detail
# this feature exists to throw away.
STEPS_PER_BAR = 16

# Longest loop worth calling a loop. A marked section far longer than this
# is somebody having missed the end of the phrase, and quantizing four
# minutes onto sixteenths is the thing we just stopped doing.
MAX_BARS = 16

# What drum_transcriber's notes mean, by name, so a loop can be handed to
# beat_writer in its vocabulary rather than in raw note numbers.
_PIECE_FOR_NOTE = {
    36: "kick",
    38: "snare",
    42: "closed hat",
    46: "open hat",
    47: "low-mid tom",
    49: "crash",
}


class Loop(NamedTuple):
    beat: beat_writer.Beat
    bars: int
    origin_sec: float   # where the loop's first step sits in the stem: the
                        # marked start, nudged onto the drumming's own grid
    hits_used: int
    hits_dropped: int
    tempo: float        # the loop's own tempo, measured off the marked section
    song_tempo: float   # what the whole song was estimated at, for comparison


def _section_wav(wav_path: str, start_sec: float, end_sec: float, out_path: str) -> float:
    """Cut [start, end] out of wav_path with _CONTEXT_SEC either side, and
    return how much padding actually made it onto the front (less than
    asked for, when the section starts near the beginning of the file)."""
    lead = min(_CONTEXT_SEC, start_sec)
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y",
            "-ss", f"{start_sec - lead:.6f}",
            "-t", f"{(end_sec - start_sec) + lead + _CONTEXT_SEC:.6f}",
            "-i", wav_path,
            out_path,
        ],
        check=True,
        capture_output=True,
    )
    return lead


def _grid_origin(times: list[float], step_sec: float) -> float:
    """How far off the grid the drumming sits, in seconds. Signed, and
    within half a step either way.

    Each hit's position within its own step is an angle, and averaging
    those as unit vectors gives the offset the whole set agrees on. Doing
    it as angles rather than as plain numbers is what makes it correct at
    the wrap: hits at 1% and 99% of a step are 2% apart, and averaging them
    arithmetically would put the grid at 50% - exactly out of phase, the
    worst answer available.

    Signed rather than in [0, step) for the same reason. "A whole step
    late" and "on time" are the same statement, and only one of them reads
    as one - which matters because floating point turns an exactly-on-time
    set of hits into a hair under a full step often enough to notice.

    Returns 0.0 for hits that agree on nothing, which is the honest answer
    for audio with no discernible pulse."""
    if not times:
        return 0.0

    angles = [(time_sec % step_sec) / step_sec * 2 * math.pi for time_sec in times]
    x = sum(math.cos(angle) for angle in angles)
    y = sum(math.sin(angle) for angle in angles)
    if math.hypot(x, y) < 1e-9:
        return 0.0

    # atan2 gives (-pi, pi], so this lands in (-step/2, step/2].
    return math.atan2(y, x) / (2 * math.pi) * step_sec


def _bar_count(span_sec: float, bar_sec: float) -> int:
    """How many whole bars the marked section is, rounded to the nearest.

    Never zero: somebody who marked a section wants a loop out of it, and
    the shortest honest answer is one bar."""
    return max(1, min(MAX_BARS, round(span_sec / bar_sec)))


def build(
    wav_path: str,
    tempo: float,
    start_sec: float,
    end_sec: float,
    name: str = "Stolen Beat",
) -> Loop:
    """Transcribe [start_sec, end_sec] of wav_path and return it as a
    quantized loop.

    `tempo` is the song's estimate, from the stem's filename. It is not the
    loop's tempo: it's the starting point for measuring the section's own,
    off the drumming inside it (see the module docstring). The two are
    reported side by side as Loop.tempo and Loop.song_tempo; they should now
    agree closely, and a real disagreement means the phrase genuinely runs at
    its own speed."""
    if end_sec <= start_sec:
        raise ValueError("the end of the section has to come after its start")

    handle, section_path = tempfile.mkstemp(suffix=".wav")
    os.close(handle)
    try:
        lead = _section_wav(wav_path, start_sec, end_sec, section_path)
        notes = drum_transcriber.transcribe(section_path)
    finally:
        os.remove(section_path)

    span = end_sec - start_sec
    # Back into the section's own timeline, and drop the padding that was
    # only ever there to give the model something to look at.
    inside = [
        (note.start - lead, note.pitch, note.velocity)
        for note in notes
        if 0 <= (note.start - lead) <= span
    ]

    # The song's estimate gets exactly one job: saying how many bars long the
    # The tempo is measured again here, off the drumming inside the section,
    # with the song's whole-length estimate only as a starting point. That's
    # what makes the beat set the grid: the phrase somebody picked has its
    # own tempo, and a song's drifts across its length (see
    # instrument_isolator's drift warning).
    #
    # Measured rather than taken from the span, which is the tempting shortcut
    # and is wrong: span/bars makes one sloppy edge define the whole grid. On
    # a real two-bar section marked about 7% long it put the grid at 97.5 BPM
    # against a song at 105.4, which left a third of the hits sitting within
    # a fifth of a step of the halfway line - a coin toss for which sixteenth
    # they landed on. Refined off the same section's onsets it came out at
    # 105.2 and nothing was ambiguous.
    loop_tempo = instrument_isolator.refine_tempo(
        np.array(sorted({time_sec for time_sec, _, _ in inside})), tempo)

    bar_sec = beat_writer.BEATS_PER_BAR * 60.0 / loop_tempo
    step_sec = bar_sec / STEPS_PER_BAR
    bars = _bar_count(span, bar_sec)

    origin = _grid_origin([time_sec for time_sec, _, _ in inside], step_sec)
    total_steps = bars * STEPS_PER_BAR

    # Quantize, keeping the loudest of any two hits of the same piece that
    # land on the same step - a flam is two hits a few milliseconds apart,
    # and on a sixteenth grid it can only be one note.
    loudest: dict[tuple[int, str], int] = {}
    dropped = 0
    for time_sec, pitch, velocity in inside:
        piece = _PIECE_FOR_NOTE.get(pitch)
        if piece is None:
            dropped += 1
            continue
        step = round((time_sec - origin) / step_sec)
        if not 0 <= step < total_steps:
            # A hit right on the closing downbeat belongs to the next
            # repeat of the loop, not to a bar that doesn't exist.
            dropped += 1
            continue
        key = (step, piece)
        loudest[key] = max(loudest.get(key, 0), velocity)

    hits = tuple(
        beat_writer.Hit(piece, step, velocity)
        for (step, piece), velocity in sorted(loudest.items())
    )
    beat = beat_writer.Beat(
        tempo=loop_tempo,
        hits=hits,
        steps_per_bar=STEPS_PER_BAR,
        bars=bars,
        name=name,
    )
    return Loop(
        beat=beat,
        bars=bars,
        origin_sec=start_sec + origin,
        hits_used=len(hits),
        hits_dropped=dropped,
        tempo=loop_tempo,
        song_tempo=tempo,
    )


def write(loop: Loop, out_dir: str, title: str) -> str:
    """Write loop next to a song's stems, named so the tempo is readable
    without opening it (see beat_writer.write - Ableton won't read the
    tempo out of the file)."""
    os.makedirs(out_dir, exist_ok=True)
    label = f"{title} ({beat_writer.STOLEN_BEAT_LABEL}, {loop.bars} bar{'s' if loop.bars != 1 else ''})"
    path = os.path.join(out_dir, beat_writer.filename_for(loop.beat, label))
    return beat_writer.write(loop.beat, path)

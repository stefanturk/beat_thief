#!/usr/bin/env python3
"""Turn a section of an isolated drum stem into a loop.

This is the part that stops trying to transcribe a performance and starts
rebuilding a beat. A whole-song transcription was faithful and unusable; two
bars on a grid are neither, and the difference is that here the output is
allowed to be *tidier* than what was played.

Given a stem, a tempo, and the piece of it somebody marked by ear:

  1. transcribe just that section (drum_transcriber, with padding either
     side so the model has context at the edges)
  2. work out how many bars it is, from the song's rough tempo
  3. divide the section itself into that many bars - the section sets the
     grid, and its own tempo falls out of doing so
  4. find where the grid sits within it, from the hits themselves
  5. quantize onto it, and hand the result to beat_writer

Steps 3 and 4 are the ones that matter, and they're the same idea twice: the
beat determines the grid, not the other way around. A whole-song tempo is an
average of something that drifts, so it's trusted only to count bars - after
that the marked phrase is measured against itself, which is also what makes
the loop close seamlessly. And somebody picking a section by ear starts it
wherever they clicked, which is nowhere in particular, so the downbeat is
recovered from the drumming rather than assumed to be where the mouse was."""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
from typing import NamedTuple

import beat_writer
import drum_transcriber

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


# How much each sixteenth of a bar counts as a strong position. The
# downbeat most, the half-bar next, then the other two beats, then the
# eighths, then everything else - the ordinary metrical hierarchy of 4/4.
_METRICAL_WEIGHT = (
    6.0, 0.5, 1.5, 0.5,
    3.0, 0.5, 1.5, 0.5,
    4.0, 0.5, 1.5, 0.5,
    3.0, 0.5, 1.5, 0.5,
)

# What each piece says about where the downbeat is. A kick is the strongest
# evidence and a crash is close behind, since a crash almost always lands
# on one. A snare is scored separately (see _rotation_score) because its
# information is the opposite: it says "not here" about the downbeat and
# "here" about the backbeat. Hats say nothing - they're on everything.
_DOWNBEAT_WEIGHT = {"kick": 1.0, "crash": 1.4, "low-mid tom": 0.2}

# Where a backbeat snare belongs, in sixteenths from the top of the bar.
_BACKBEAT_STEPS = (4, 12)
_BACKBEAT_BONUS = 2.5


class Loop(NamedTuple):
    beat: beat_writer.Beat
    bars: int
    origin_sec: float   # where in the stem the loop's downbeat was found
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


def _rotation_score(placed: dict, rotation: int, total_steps: int) -> float:
    """How much a beat looks like 4/4 when step `rotation` is called one.

    Kicks and crashes score by where they'd land in the bar; snares score
    for landing on the backbeat. Both are needed - kicks alone can't tell a
    bar from its own half, and plenty of beats put a kick on three."""
    score = 0.0
    for (step, piece), velocity in placed.items():
        within_bar = (step - rotation) % total_steps % STEPS_PER_BAR
        loudness = velocity / 127.0
        if piece == "snare":
            if within_bar in _BACKBEAT_STEPS:
                score += _BACKBEAT_BONUS * loudness
        else:
            weight = _DOWNBEAT_WEIGHT.get(piece)
            if weight is not None:
                score += weight * _METRICAL_WEIGHT[within_bar] * loudness
    return score


def _downbeat_rotation(placed: dict, total_steps: int) -> int:
    """Which step of the loop is really beat one.

    Somebody marking a section by ear starts it where they clicked, so the
    quantized pattern is a rotation of the beat rather than the beat - the
    snare comes out on one and three instead of two and four, and dropping
    it at bar 1 in Ableton sounds wrong even though every hit is correct.

    A loop is cyclic, so this costs nothing to fix: try calling each step
    the downbeat and keep whichever reading looks most like 4/4. Only whole
    beats are considered - a downbeat is never a sixteenth off, and letting
    it be one lets a busy hat pattern outvote the kick."""
    if not placed:
        return 0
    beats = range(0, total_steps, STEPS_PER_BAR // beat_writer.BEATS_PER_BAR)
    return max(beats, key=lambda rotation: _rotation_score(placed, rotation, total_steps))


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
    loop's tempo: it only says how many bars were marked, and the loop's own
    tempo is then measured off the marked span (see the module docstring).
    The two are reported side by side as Loop.tempo and Loop.song_tempo,
    because a big disagreement means the section was marked long or short
    and the person who marked it is the only one who can say which."""
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
    # marked section is. Rounding to a whole number of bars is all a
    # whole-song average is good enough for.
    bars = _bar_count(span, beat_writer.BEATS_PER_BAR * 60.0 / tempo)

    # From here the section measures itself. A bar is a bar of what was
    # marked, not a bar of whatever the song averaged out to - the phrase
    # somebody picked has its own tempo, and a song's drifts (see
    # instrument_isolator's drift warning). Deriving the grid from the
    # selection is also the only way the loop can be seamless: it ends
    # exactly where it started, by construction.
    bar_sec = span / bars
    step_sec = bar_sec / STEPS_PER_BAR
    loop_tempo = beat_writer.BEATS_PER_BAR * 60.0 / bar_sec

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

    rotation = _downbeat_rotation(loudest, total_steps)
    hits = tuple(
        beat_writer.Hit(piece, (step - rotation) % total_steps, velocity)
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
        origin_sec=start_sec + origin + rotation * step_sec,
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

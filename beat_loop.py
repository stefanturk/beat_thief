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

The one place the marking is corrected is the start, and only in one way: a
loop is moved on to begin on a kick (_kick_downbeat). People click a little
before the phrase, so a loop that begins on silence is the ordinary outcome
of marking by ear, and starting on a kick is what a loop is almost always
for.

Moving the start moves the end with it, so the loop's last stretch comes from
after the mark - which is why a bar past it is transcribed too (_SPARE_BARS).
It used to be wrapped round from the front instead, and since the front was
the silence somebody clicked into, that silence arrived at the end: a loop
whose last beats were empty while the drumming that belonged there had been
transcribed and thrown away.

That's deliberately narrower than what used to be here. An earlier version
scored every rotation for how much it looked like 4/4 and turned the beat
round to suit; on a real section it could not tell its best answer from its
second best - 59.4 against 58.1 - while the two were a whole beat apart, and
it silently overrode the marking either way. Which beat is the one is a
musical judgement, and the picker's shift-arrows are the instrument for it.
This file only does the part that isn't a judgement."""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
from typing import NamedTuple

import numpy as np

import beat_writer
import drum_transcriber
import groove_reader
import instrument_isolator

# The model is bidirectional and judges a hit partly on what surrounds it,
# so a section cut out and handed over on its own loses hits at both ends.
# Transcribing with this much extra on each side and then discarding it
# costs a second and makes the edges as good as the middle.
_CONTEXT_SEC = 3.0

# The grid the loop is quantized onto. Thirty-seconds: sixteenths are what
# nearly every drum part is *written* in, but they're not what percussion is
# played in. A tambourine doubles up between the sixteenths, and on a
# sixteenth grid the second of the pair lands on the same step as the first
# and the louder of the two wins - the doubling disappears, and what's left
# is a note that no longer matches what you can hear.
#
# The cost is that a sloppy hit can now fall a step off where it belongs
# rather than being pulled into line. That's the trade, and it's the right
# way round: a straight part quantizes to thirty-seconds exactly as it
# quantizes to sixteenths, because every sixteenth is a thirty-second too.
# Only the parts that were being flattened change.
STEPS_PER_BAR = 32

# How much drumming past the marked end to keep, in bars. Moving the loop's
# start onto a kick moves its end by the same amount, and the notes for that
# last stretch are past the mark - so they have to be transcribed too or the
# loop ends in silence. A bar is the most the start can ever move, since the
# kick is looked for within one bar.
_SPARE_BARS = 1

# Longest loop worth calling a loop. A marked section far longer than this
# is somebody having missed the end of the phrase, and quantizing four
# minutes onto a step grid is the thing we just stopped doing.
MAX_BARS = 16

# What drum_transcriber's notes mean, by name, so a loop can be handed to
# beat_writer in its vocabulary rather than in raw note numbers.
_PIECE_FOR_NOTE = {
    36: "kick",
    37: "ghost snare",
    38: "snare",
    39: "tambourine",
    41: "hat percussion",  # provisional - see drum_transcriber._HAT_PERCUSSION_NOTE
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
    hits_inferred: int   # hits groove_reader put back that no onset was
                         # detected for - a stage that quietly adds notes to
                         # your beat has to be able to say that it did
    tempo: float        # the loop's own tempo, measured off the marked section
    song_tempo: float   # what the whole song was estimated at, for comparison


def _section_wav(wav_path: str, start_sec: float, end_sec: float, out_path: str) -> float:
    """Cut [start, end] out of wav_path with _CONTEXT_SEC either side, and
    return how much padding actually made it onto the front (less than
    asked for, when the section starts near the beginning of the file).

    Pass an end past the marked one to keep the drumming that follows it -
    see _SPARE_BARS."""
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


# How close a kick has to come to the strongest one to be preferred for
# being earlier. Loops nearly always want the first strong kick rather than
# the loudest one somewhere later, and without this a beat whose kick on
# three is a few velocity steps harder than its kick on one gets turned
# round - which is how the old downbeat guesser used to ruin a good marking.
_KICK_MARGIN = 0.9

# How much kick the marked start has to already have for the marking to be
# left exactly where it is. Deliberately far below _KICK_MARGIN: this isn't
# asking whether the marked kick is the best one in the bar, only whether
# there is a real kick there rather than a whisper. If there is, the person
# marking heard it and meant it, and moving the cut off it - by up to a
# whole bar, which is what this function is otherwise free to do - hands
# back a file that is not the audio they auditioned. A kick at a quarter of
# the loudest one is a soft kick; a transcription artifact is quieter again.
_KICK_ANCHOR = 0.25


def _kick_downbeat(placed: dict, total_steps: int) -> int:
    """Which step of the marked section to call one, so the loop can start
    there instead.

    A stolen loop is nearly always wanted starting with a kick on the one,
    and a section marked by ear starts wherever the mouse went - usually a
    little before the phrase, since people click early. So the loop is moved
    on to begin at a kick, and runs its full length from there.

    Only kicks vote, and they vote by how hard they are and by turning up in
    every bar. Nothing here scores how much the result "looks like 4/4":
    that's what the old guesser did, and on a real section it could not tell
    its best answer from its second best (59.4 against 58.1) while being a
    whole beat apart. Where the one goes is a musical judgement, and the
    picker's shift-arrows are how you make it. This only does the part that
    isn't a judgement - don't start the loop on silence.

    A marking that already lands on a kick is never moved at all, however
    much louder some later kick is. The picker snaps a click onto a real
    kick and plays from exactly there, so moving the cut afterwards would
    hand back a file that isn't the audio that was auditioned - which is a
    worse failure than starting on the second-best kick in the bar.

    Positions are folded across bars and searched within a single bar, since
    moving a repeating loop on by a whole bar changes nothing you can hear.
    Returns 0 when there are no kicks, which leaves the marking alone."""
    within_bar = min(STEPS_PER_BAR, total_steps)
    force = [0.0] * within_bar
    for (step, piece), velocity in placed.items():
        if piece == "kick":
            force[step % within_bar] += velocity

    strongest = max(force)
    if strongest <= 0:
        return 0

    # Asked of step 0 itself, not of the folded votes: a kick one bar later
    # lands on 0 once folded, and that is a different claim entirely from
    # there being a kick where the marking actually is.
    marked = placed.get((0, "kick"), 0.0)
    loudest = max(v for (step, piece), v in placed.items() if piece == "kick")
    if marked >= loudest * _KICK_ANCHOR:
        return 0

    return next(step for step, f in enumerate(force) if f >= strongest * _KICK_MARGIN)


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
    # What the track inside the MIDI file is called, and so what Ableton
    # calls the clip. A placeholder: write() renames it after the file it
    # ends up in, which is the name worth having. It matters only to a
    # caller that builds a loop and writes it some other way.
    name: str = "Stolen Beat",
    # Told what's happening as it happens, so a caller with no percentage
    # to show (there's no per-chunk progress out of a single model pass,
    # unlike demucs) can still say something other than a static "working"
    # for the thirty-odd seconds this takes. None is a no-op, so nothing
    # else calling build() has to care that this exists.
    on_phase=None,
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

    def phase(message):
        if on_phase is not None:
            on_phase(message)

    span = end_sec - start_sec
    # A bar of the drumming past the marked end, kept rather than thrown
    # away, so that moving the loop's start onto a kick can take the bars it
    # needs from what actually follows (see _SPARE_BARS).
    spare = _SPARE_BARS * beat_writer.BEATS_PER_BAR * 60.0 / tempo if tempo > 0 else 0.0

    phase("Cutting the section...")
    # A song's own hi-hat/ride, not Officially Missing You's - see
    # calibrate_hat_threshold. Cached per file, so stealing several loops out
    # of one song only pays for this once.
    hat_threshold = drum_transcriber.calibrate_hat_threshold(wav_path)

    handle, section_path = tempfile.mkstemp(suffix=".wav")
    os.close(handle)
    try:
        lead = _section_wav(wav_path, start_sec, end_sec + spare, section_path)
        phase("Listening for the hits...")
        notes = drum_transcriber.transcribe(section_path, hat_threshold=hat_threshold)
    finally:
        os.remove(section_path)

    phase("Quantizing the pattern...")

    # Back into the section's own timeline, and drop the padding that was
    # only ever there to give the model something to look at.
    heard = [
        (note.start - lead, note.pitch, note.velocity)
        for note in notes
        if 0 <= (note.start - lead) <= span + spare
    ]
    # What was actually marked. The tempo and the grid come from this and
    # only this - the spare bar is material to build with, not evidence
    # about what somebody picked.
    inside = [hit for hit in heard if hit[0] <= span]

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

    def landed(time_sec: float, from_sec: float) -> int | None:
        """Which step of a loop starting at from_sec a hit belongs to, or
        None if it falls outside one - a hit on the closing downbeat belongs
        to the next repeat, not to a bar that doesn't exist."""
        step = round((time_sec - from_sec) / step_sec)
        return step if 0 <= step < total_steps else None

    def quantize(from_sec: float, played: list) -> dict:
        """The hits of `played` on the grid, as {(step, piece): velocity},
        for the loop starting at from_sec.

        The loudest of two hits of the same piece on one step wins: a flam is
        two hits a few milliseconds apart, and even on a thirty-second grid
        it can only be one note."""
        placed: dict[tuple[int, str], int] = {}
        for time_sec, pitch, velocity in played:
            piece = _PIECE_FOR_NOTE.get(pitch)
            step = landed(time_sec, from_sec)
            if piece is None or step is None:
                continue
            key = (step, piece)
            placed[key] = max(placed.get(key, 0), velocity)
        return placed

    # Where the one is, decided on what was marked, and then the loop is
    # taken from there: a full count of bars starting at that kick, its tail
    # coming out of the spare bar rather than being wrapped round from the
    # front. Wrapping was the bug. People click a little early, so the front
    # of a marked section is usually silence, and rotating carried that
    # silence to the end - a loop whose last two beats were empty while the
    # drumming that belonged there sat just past the mark, transcribed and
    # thrown away.
    rotation = _kick_downbeat(quantize(origin, inside), total_steps)
    from_sec = origin + rotation * step_sec
    loudest = quantize(from_sec, heard)

    # Now that there's a grid, the loop can be read as music rather than as a
    # list of onsets: percussion taken off the snare pad on the evidence of
    # what it plays, and the hits a louder drum was covering put back. This
    # runs after the rotation on purpose - the phases it measures have to be
    # phases of the loop that will actually be written.
    loudest, inferred = groove_reader.refine(loudest, total_steps, STEPS_PER_BAR)

    # What was marked and didn't make it: a piece the map doesn't cover, or
    # a hit left behind at the front when the loop moved on to the kick.
    # Counted over the marked section only - the spare bar is there to build
    # with, and the part of it the loop didn't need was never asked for.
    dropped = sum(
        1 for time_sec, pitch, _ in inside
        if _PIECE_FOR_NOTE.get(pitch) is None or landed(time_sec, from_sec) is None
    )

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
        origin_sec=start_sec + from_sec,
        hits_used=len(hits),
        hits_dropped=dropped,
        hits_inferred=inferred,
        tempo=loop_tempo,
        song_tempo=tempo,
    )


def _free_path(path: str) -> str:
    """path, or the next " (2)", " (3)"... that nothing is using.

    A song is worth more than one beat - a verse and a chorus out of the
    same drums are two different loops - and the name only carries the bar
    count and the tempo, so two sections of the same song can easily ask for
    the same filename. Without this the second steal silently replaces the
    first, which is the worst possible way to find out."""
    stem, ext = os.path.splitext(path)
    nth = 2
    while os.path.exists(path):
        path = f"{stem} ({nth}){ext}"
        nth += 1
    return path


def write(loop: Loop, out_dir: str, title: str) -> str:
    """Write loop next to a song's stems, named so the tempo is readable
    without opening it (see beat_writer.write - Ableton won't read the
    tempo out of the file), and never over a beat that's already there."""
    os.makedirs(out_dir, exist_ok=True)
    path = _free_path(os.path.join(out_dir, beat_writer.stolen_beat_filename(loop.beat, title)))
    # The track inside the file is named after the file. Ableton names the
    # clip you drag in after the track, and that name was the constant
    # "Stolen Beat" - so every loop out of every song arrived in Live called
    # the same thing, and telling two of them apart meant opening them.
    #
    # Taken from the final path rather than built alongside it, so that a
    # second beat out of one song keeps the " (2)" that makes it distinct.
    named = loop.beat._replace(name=os.path.splitext(os.path.basename(path))[0])
    return beat_writer.write(named, path)


def write_wav(loop: Loop, wav_path: str, mid_path: str) -> str:
    """Cut the loop's own span - origin_sec to origin_sec + duration_sec,
    the marking after it was nudged onto a kick, not the raw click - out of
    wav_path and save it beside mid_path.

    Named to match the .mid rather than run through _free_path on its own,
    so the pair a second steal makes keeps its own " (2)" together instead
    of the two drifting out of sync should one ever be written without the
    other."""
    out_path = os.path.splitext(mid_path)[0] + ".wav"
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y",
            "-ss", f"{loop.origin_sec:.6f}",
            "-t", f"{loop.beat.duration_sec:.6f}",
            "-i", wav_path,
            out_path,
        ],
        check=True,
        capture_output=True,
    )
    return out_path

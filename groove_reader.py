#!/usr/bin/env python3
"""Read a quantized loop as music, and rewrite it as what was played.

The transcription that arrives here is honest and wrong. It reports one hit
per onset, labelled with whichever of the model's five classes fitted best,
and on real drumming that means a percussion instrument - a tambourine, a
shaker - lands on the snare, because the model has no class for it and the
snare is the nearest thing it knows.

Measured on the drum stem of Officially Missing You (Brasstracks), the snare
class fires 375 times in three minutes. A backbeat is made of hits eight
sixteenths apart; six of those 375 were. A hundred were a sixteenth apart and
a hundred and fifty a quarter apart. Four bars of the intro put the snare
class on all sixteen quarter notes with no gaps at all, and four bars of the
bridge put the snare class and the hi-hat class on identical steps - one voice
firing two classes at once. None of that is a snare part. It's a tambourine
sitting on the snare pad.

No amount of listening to a single hit fixes this. Every band measurement
tried against that stem came back contaminated by whatever else landed on the
same sixteenth: the loudest thing in the 150-500Hz window near a hit is the
kick's decay, and the hits where the low end jumped fifty decibels turned out
to be exactly the steps the kick class had already flagged. Velocity doesn't
separate them either - on-quarter median 102, off-quarter 103.

What does separate them is what they play. A percussion instrument keeps time:
it repeats on a pulse, right through the bar. A snare is structural - it lands
on two and four, and where it doesn't, it's playing a fill. So the pulse is
the evidence, and this module is where the loop gets read that way:

  - find the pulse the snare class is sitting on, if there is one
  - if the audio's vote agrees it's percussion, take the whole voice off the
    snare pad in one move, rather than one hit at a time
  - restore the hits the pulse says were played and the transcription missed,
    because something louder was on top of them

That last part is the one a per-hit measurement can never do. A tambourine
struck at the same instant as the snare is inside the snare's window, and the
honest local answer is "snare" - so the tambourine's hits on two and four are
gone, which on a backbeat is most of them. The pulse knows they were there.

One thing is thrown away rather than moved: a snare vote cast on a step that
a kick is also on. That vote is not evidence - the kick's shell reaches into
the band that decides "this was a drum", and on the real stem the hits whose
low end jumped fifty decibels were exactly the steps the kick class had
already flagged. Where the pulse says percussion and the only thing arguing
otherwise is a reading taken through a kick, the pulse wins.

Everything here is a grid in and a grid out. No audio, no model, no numbers
that came from one. Every test in test_groove_reader.py is a pattern written
by hand and an assertion about a pattern, which is the point: these rules are
musical judgements, and they should be arguable on paper.

Used by beat_loop.build, after quantizing and after the loop has been moved
onto its kick - so the phases measured here are phases of the loop that will
actually be written."""

from __future__ import annotations

import statistics

# The pieces the snare class can arrive as. drum_transcriber votes on each hit
# with the audio (percussion_splitter) and writes that vote as the piece: a
# hit it thinks has no drum body arrives as "tambourine". The vote is weak per
# hit and is meant to be - it gets aggregated here, over a whole voice, which
# is the only scale it was reliable at when measured.
_SNARE_CLASS = ("snare", "tambourine")

# Voices that keep time, and so are allowed to have missed hits restored. A
# kick, a snare or a crash never is: those are structural, their silences are
# the composition, and inventing one changes the groove rather than clarifying
# it.
_TIMEKEEPING = ("closed hat", "open hat", "tambourine", "shaker")

# Pulses looked for, in steps of a sixteenth grid: sixteenths, eighths and
# quarters. Deliberately not eight - a voice on every eighth step is a
# backbeat, and a backbeat is exactly what has to be left alone. Triplets and
# swing can't be expressed on this grid at all (see beat_loop.STEPS_PER_BAR).
_PERIODS = (1, 2, 4)

# What the pulse's period says the instrument is. A voice on the quarters or
# the eighths is marking the beat, which is a tambourine's job; a voice on
# every sixteenth is a fill, which is a shaker's.
#
# This is a guess about the *role*, not about the sound. Nothing measurable in
# the audio told a tambourine from a shaker on real material - attack time and
# ring length both turned out to be measuring how busy the surrounding mix was
# - so the honest thing is to guess from what it plays and to be easy to
# overrule. One line, right here.
_PIECE_FOR_PERIOD = {1: "shaker", 2: "tambourine", 4: "tambourine"}

# How much of a pulse has to be occupied before it counts as a voice keeping
# time rather than a coincidence. Four bars of quarter notes is sixteen
# positions, so this allows four of them to be missing.
#
# It's also what bounds how much can be put back: nothing is ever filled that
# isn't a gap in a pulse, so at most a quarter of a voice can be inferred, and
# no separate cap is needed.
_PULSE_OCCUPANCY = 0.75

# Fewest hits a pulse can be built from. Without this, three hits landing on
# the quarters of a one bar loop would qualify, and three hits agree with
# almost anything.
_PULSE_MIN_HITS = 6

# How many of a voice's hits may sit off its own pulse. This is the test that
# tells a percussion voice from a snare fill: a fill covers a bar of
# sixteenths densely and leaves the rest of the loop empty, so its own hits
# are the strays.
_PULSE_MAX_STRAY = 0.35

# How much of a pulse has to have voted "not a drum" before the voice comes
# off the snare pad. The vote is weak per hit, so it's only being asked which
# way the whole voice leans - and not even asked for a majority, because the
# ways it goes wrong are one-sided. A hit with another drum on the same
# sixteenth reads as a drum whatever else was in there with it, and a snare
# is the loudest thing in most bars, so contamination pushes votes toward
# "snare" and almost never the other way. Measured on the six four-bar loops
# of the Brasstracks stem: 62% on the intro's tambourine, 44%, 36%, 35%, 33%,
# and 16% on the busiest chorus.
_PERCUSSION_MAJORITY = 0.3

# A voice on every single sixteenth is never taken off the snare pad. On a
# pulse that fine, every snare in the loop is on the pulse by definition, so
# there is no way to tell a percussion line with a snare through it from a
# snare playing sixteenths - and the failure is expensive, because confirming
# it doubles the note count of the busiest bar in the song. Checked against
# the outro of the Brasstracks stem, which is a busy snare and which this
# turns away: without it, that loop went from 83 hits to 125.
_MIN_PERCUSSION_PERIOD = 2

# Pieces whose presence on a step makes the audio's vote there worthless.
# This is the single biggest thing measured in Part 0: the snare-class hits
# whose low end jumped fifty decibels were *exactly* the steps the kick class
# had already flagged. A kick's shell reaches well into the band that says
# "this was a drum", so a hit sharing a sixteenth with one reads as a drum
# whatever it was. Rather than believe it, the pulse decides that step.
def _confounds_the_vote(piece: str) -> bool:
    return piece == "kick" or piece.endswith("tom")

# Under this velocity, a hit left on the snare pad is a ghost note and goes to
# its own pad, so the hits that are meant to sing aren't sharing with the ones
# that aren't. Velocity is already decibels below the loudest snare of this
# section (see drum_transcriber._VELOCITY_DYNAMIC_RANGE_DB), so this is about
# 17dB down rather than a raw level - which is what makes it mean the same
# thing in a quiet section as in a loud one.
#
# It happens here, at the very end, rather than in the transcriber, because
# it's a statement about what's on the snare pad and until this module has
# finished moving voices around, that isn't settled.
_GHOST_SNARE_VELOCITY = 55


def _positions(phase: int, period: int, total_steps: int) -> list[int]:
    return list(range(phase, total_steps, period))


def _best_pulse(steps: set[int], total_steps: int) -> tuple[int, int] | None:
    """The (period, phase) this set of steps is keeping time on, or None.

    Keeping time is three things at once and all three are needed: enough of
    the pulse is occupied, there are enough hits for that to mean anything,
    and not too many hits sit off it."""
    best = None
    for period in _PERIODS:
        for phase in range(period):
            positions = _positions(phase, period, total_steps)
            if not positions:
                continue
            on = steps.intersection(positions)
            if len(on) < _PULSE_MIN_HITS:
                continue
            if len(on) / len(positions) < _PULSE_OCCUPANCY:
                continue
            if (len(steps) - len(on)) / len(steps) > _PULSE_MAX_STRAY:
                continue
            # The pulse that accounts for the most of what was played. A
            # sixteenth voice fits a quarter pulse too, but three quarters of
            # it would be strays, so that candidate is already gone.
            if best is None or len(on) > best[0]:
                best = (len(on), period, phase)
    return (best[1], best[2]) if best else None


def _velocity_like(velocities: dict[int, int], step: int, steps_per_beat: int) -> int:
    """A velocity for a hit that has to be put back: what this voice plays at
    the same kind of position.

    On the beat and off it are taken separately, because a hand accents the
    beat - beat_writer's reference groove writes its hats that way by hand,
    and a restored hit that ignored it would be the one flat note in the
    bar."""
    if not velocities:
        return 1
    on_beat = step % steps_per_beat == 0
    alike = [v for s, v in velocities.items() if (s % steps_per_beat == 0) == on_beat]
    return int(round(statistics.median(alike or list(velocities.values()))))


def _put(placed: dict, step: int, piece: str, velocity: int) -> None:
    """Write a hit, keeping the louder of two on one step and piece."""
    placed[(step, piece)] = max(placed.get((step, piece), 0), velocity)


def _restore_covered(
    placed: dict[tuple[int, str], int],
    piece: str,
    positions: list[int],
    velocities: dict[int, int],
    steps_per_beat: int,
) -> int:
    """Put back the hits this voice's pulse says were played, where something
    else landed on top of them. Returns how many.

    Only positions where some other piece is: that's the difference between
    restoring a hit that was covered and inventing one. An empty position with
    nothing on it stays empty - the drummer stopped there, and that's as much
    a part of the groove as the hits are."""
    covered = [
        step for step in positions
        if (step, piece) not in placed
        and any(key[0] == step and key[1] != piece for key in placed)
    ]
    for step in covered:
        placed[(step, piece)] = _velocity_like(velocities, step, steps_per_beat)
    return len(covered)


def refine(
    placed: dict[tuple[int, str], int],
    total_steps: int,
    steps_per_bar: int,
) -> tuple[dict[tuple[int, str], int], int]:
    """The same loop, read as music. Returns the new grid and how many hits
    were put back.

    `placed` is {(step, piece): velocity}, which is what beat_loop quantizes
    to. Nothing that was played is thrown away; hits move between pieces, and
    hits the pulse says were covered are restored."""
    out = dict(placed)
    steps_per_beat = max(1, steps_per_bar // 4)
    inferred = 0

    # --- the snare class: one voice, or two of them interleaved? ---
    #
    # A step with both a snare and a percussion vote on it counts as a snare.
    # That's the conservative reading, and it costs nothing: if the pulse says
    # percussion was there too, it gets added back below.
    pool: dict[int, str] = {}
    for step, piece in placed:
        if piece in _SNARE_CLASS and pool.get(step) != "snare":
            pool[step] = piece

    percussion: str | None = None
    pulse = _best_pulse(set(pool), total_steps) if pool else None
    if pulse and pulse[0] >= _MIN_PERCUSSION_PERIOD:
        period, phase = pulse
        positions = _positions(phase, period, total_steps)
        on_pulse = [step for step in positions if step in pool]

        # A step with a kick on it can't tell us anything, so it doesn't get
        # to vote either way. What's left is the part of the voice that was
        # actually heard on its own.
        covered_by = {step for step in on_pulse
                      if any(key[0] == step and _confounds_the_vote(key[1]) for key in placed)}
        heard_alone = [step for step in on_pulse
                       if pool[step] != "snare" or step not in covered_by]
        voted = sum(1 for step in heard_alone if pool[step] != "snare")

        if voted >= _PERCUSSION_MAJORITY * len(heard_alone or on_pulse):
            percussion = _PIECE_FOR_PERIOD[period]

            velocities: dict[int, int] = {}
            for step in on_pulse:
                if pool[step] == "snare" and step not in covered_by:
                    continue
                velocities[step] = out.pop((step, pool[step]))
                _put(out, step, percussion, velocities[step])

            # What's left is a snare the audio was able to see clearly, on a
            # step the pulse says the percussion was also on. One onset, two
            # instruments: keep the drum and add the hit that was inside it.
            # This is the case no per-hit measurement can reach, because both
            # sounds are in the one window.
            for step in on_pulse:
                if pool[step] == "snare" and step not in covered_by:
                    _put(out, step, percussion,
                         _velocity_like(velocities, step, steps_per_beat))
                    inferred += 1

            inferred += _restore_covered(out, percussion, positions, velocities,
                                         steps_per_beat)

    # Anything still carrying a percussion vote that didn't turn out to be a
    # voice goes back to being a snare. The per-hit vote is advisory only:
    # scattering single tambourine notes across a loop is the failure this
    # stage exists to prevent, and a lone bright snare hit is a snare.
    for step, piece in list(out):
        if piece in _SNARE_CLASS and piece not in ("snare", percussion):
            _put(out, step, "snare", out.pop((step, piece)))

    # --- every other voice that keeps time: restore what was covered ---
    for piece in _TIMEKEEPING:
        if piece == percussion:
            continue  # just done, above
        steps = {step for (step, this) in out if this == piece}
        if not steps:
            continue
        found = _best_pulse(steps, total_steps)
        if not found:
            continue
        period, phase = found
        velocities = {step: out[(step, piece)] for step in steps}
        inferred += _restore_covered(out, piece, _positions(phase, period, total_steps),
                                     velocities, steps_per_beat)

    # --- and last, what's left on the snare pad: which of it is a ghost ---
    for step, piece in list(out):
        if piece in ("snare", "ghost snare"):
            velocity = out.pop((step, piece))
            _put(out, step,
                 "ghost snare" if velocity < _GHOST_SNARE_VELOCITY else "snare",
                 velocity)

    return out, inferred

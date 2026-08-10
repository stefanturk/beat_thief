#!/usr/bin/env python3
"""Write a drum beat to a MIDI file that drops straight into Ableton.

Everything that ends up as a .mid goes through here, so there's one place
that knows what Ableton accepts and one place to fix when it turns out to be
wrong. A beat is a grid: a tempo, how many steps a bar is divided into, how
many bars, and a hit at some step. Nothing here listens to audio - working
out what the hits are is somebody else's job.

Two things this is careful about, both of which are why beats written by
hand before now didn't line up:

  - Every piece is inside 36-51, the sixteen pads of Ableton's default Drum
    Rack. A note outside that range still exists in the clip but has no pad
    to play, so it's silent and invisible - the worst way for a note to be
    wrong.
  - The clip is exactly a whole number of bars long, so looping it doesn't
    add a gap or eat the downbeat.

Run it directly to write the two reference files (see main)."""

from __future__ import annotations

import argparse
import os
from typing import NamedTuple

import pretty_midi

# Name -> MIDI note, in the order the kit is laid out low to high. Every one
# of these is a pad in the stock Drum Rack (36-51) and carries its General
# MIDI meaning, so the file reads correctly in anything else too.
PIECES = {
    "kick": 36,          # Bass Drum 1
    "side stick": 37,    # Side Stick
    "snare": 38,         # Acoustic Snare
    "low floor tom": 41,
    "closed hat": 42,    # Closed Hi-Hat
    "high floor tom": 43,
    "pedal hat": 44,     # Pedal Hi-Hat
    "low tom": 45,
    "open hat": 46,      # Open Hi-Hat
    "low-mid tom": 47,
    "hi-mid tom": 48,
    "crash": 49,         # Crash Cymbal 1
    "high tom": 50,
    "ride": 51,          # Ride Cymbal 1
}

# The pads a Drum Rack shows. Anything outside is silent in a stock rack.
RACK_LOW, RACK_HIGH = 36, 51

# What a stolen loop's filename carries, the way each isolator's outputs
# carry its label (see instrument_isolator.find_existing_basename). Lives
# here rather than in beat_loop so that reading a song's folder to see
# what's in it doesn't mean importing the transcriber and all of torch.
STOLEN_BEAT_LABEL = "Stolen Beat"

# How long each note is written as. A drum rack plays one-shots, so this is
# cosmetic - but a part whose notes are all hairlines is unreadable in the
# MIDI editor, and one whose notes overlap the next hit looks like a mistake.
# A sixteenth at the beat's own tempo, capped, reads right at any speed.
_NOTE_LENGTH_STEPS = 0.9

# Beats are written in 4/4. Everything here assumes four beats to a bar:
# steps_per_bar is divided by it, and the clip length is a multiple of it.
BEATS_PER_BAR = 4

# Ticks per quarter note. A MIDI file stores every time as a whole number of
# ticks, so this is the finest distinction the format can hold - times land
# on a tick whether we like it or not.
#
# pretty_midi defaults to 220, which at 120 BPM is a 2.3ms grid. That's
# exact for anything on a sixteenth, and wrong for everything that isn't: a
# swung eighth written at step 1.66 came back 0.7ms early. 960 is what DAWs
# generally use, costs nothing, and takes that error under a millisecond.
RESOLUTION = 960


class Hit(NamedTuple):
    """One drum hit. step is in steps from the start of the clip and is a
    float on purpose - a swung eighth sits at 1.66, a triplet at 1.33, and
    neither should need a different grid to express."""

    piece: str
    step: float
    velocity: int = 100


class Beat(NamedTuple):
    """A drum pattern on a grid.

    steps_per_bar is how finely the bar is divided: 16 for sixteenths, which
    is what nearly everything wants, 12 for eighth-note triplets. bars is
    how long the clip is, and is what makes it loop cleanly - it's set
    rather than inferred from the hits, so a bar that ends in silence stays
    a full bar instead of being trimmed to the last note."""

    tempo: float
    hits: tuple
    steps_per_bar: int = 16
    bars: int = 2
    name: str = "Drums"

    @property
    def seconds_per_step(self) -> float:
        return (60.0 / self.tempo) * BEATS_PER_BAR / self.steps_per_bar

    @property
    def total_steps(self) -> int:
        return self.steps_per_bar * self.bars

    @property
    def duration_sec(self) -> float:
        return self.total_steps * self.seconds_per_step


def note_for(piece: str) -> int:
    """The MIDI note a named piece lands on. Raises for a name that isn't a
    piece, rather than guessing - a typo that silently wrote note 0 would
    produce a clip that looks fine and plays nothing."""
    try:
        return PIECES[piece]
    except KeyError:
        raise KeyError(f"no drum piece called {piece!r} - have {', '.join(PIECES)}") from None


def write(beat: Beat, path: str) -> str:
    """Write beat to path as a MIDI file and return the path.

    The tempo goes in the header and every note time is computed from the
    step index, so the grid in the file is exact rather than accumulated.

    Note that Ableton ignores the header tempo on import: it drops the notes
    onto the set's grid at whatever the set's tempo is, which is why the BPM
    goes in the filename too. Set Live to that number and it lines up."""
    midi = pretty_midi.PrettyMIDI(initial_tempo=beat.tempo, resolution=RESOLUTION)
    midi.time_signature_changes = [pretty_midi.TimeSignature(BEATS_PER_BAR, 4, 0.0)]

    track = pretty_midi.Instrument(program=0, is_drum=True, name=beat.name)
    length = _NOTE_LENGTH_STEPS * beat.seconds_per_step
    for hit in sorted(beat.hits, key=lambda h: (h.step, note_for(h.piece))):
        start = hit.step * beat.seconds_per_step
        track.notes.append(
            pretty_midi.Note(
                velocity=int(max(1, min(127, hit.velocity))),
                pitch=note_for(hit.piece),
                start=start,
                end=min(start + length, beat.duration_sec),
            )
        )
    # A clip has to be a whole number of bars or it won't loop, and a MIDI
    # file only runs as far as its last event - a beat whose final bar ends
    # in silence would come in short and loop early. A silent controller
    # message at the bar line carries the file out to full length.
    #
    # It goes on the drum track rather than a track of its own: a MIDI file
    # with two tracks imports into Ableton as two tracks, and the second
    # would be empty.
    track.control_changes.append(
        pretty_midi.ControlChange(number=7, value=100, time=beat.duration_sec)
    )
    midi.instruments.append(track)

    midi.write(path)
    return path


def filename_for(beat: Beat, label: str) -> str:
    """The BPM belongs in the filename, because Ableton won't read it out of
    the file (see write) and it's the number you have to type into Live."""
    return f"{label} ({beat.tempo:g} BPM).mid"


# --- the two reference beats -------------------------------------------
#
# These exist to answer "does what we write actually land in Ableton
# correctly", before anything gets built on an assumption about it.

REFERENCE_TEMPO = 90.0


def reference_groove() -> Beat:
    """A real two-bar beat, in sixteenths. This is the "does it sound right"
    test: a straight backbeat with eighth hats, a ghost note before the
    second snare, an open hat on the "and" of four, and a crash on one."""
    hits = [
        Hit("crash", 0, 112),
    ]
    for bar in (0, 16):
        # Hats on every eighth, accented on the beat the way a hand plays.
        for step in range(0, 16, 2):
            hits.append(Hit("closed hat", bar + step, 92 if step % 4 == 0 else 68))
        hits.append(Hit("kick", bar + 0, 118))
        hits.append(Hit("kick", bar + 6, 104))
        hits.append(Hit("snare", bar + 4, 114))
        hits.append(Hit("snare", bar + 12, 114))
        hits.append(Hit("snare", bar + 11, 34))  # ghost note
    # The open hat replaces the closed one it lands on, rather than stacking.
    hits = [h for h in hits if not (h.piece == "closed hat" and h.step == 30)]
    hits.append(Hit("open hat", 30, 96))
    # A pickup kick in the second bar, so the two bars aren't identical.
    # Step 26 is the "and" of three, which nothing else is on - a second
    # kick on a step that already had one would look like one note and play
    # as a doubled trigger.
    hits.append(Hit("kick", 26, 96))
    return Beat(tempo=REFERENCE_TEMPO, hits=tuple(hits), name="Reference Groove")


def reference_all_pads() -> Beat:
    """Every piece in PIECES, one per beat, in note order, cycling through
    four velocities. Not music - a diagnostic. A mapping mistake or a
    velocity that didn't survive the trip is obvious at a glance here and
    invisible inside a groove."""
    velocities = (20, 50, 80, 110)
    steps_per_beat = 4  # sixteenths
    hits = tuple(
        Hit(piece, index * steps_per_beat, velocities[index % len(velocities)])
        for index, piece in enumerate(PIECES)
    )
    bars = -(-len(PIECES) // BEATS_PER_BAR)  # round up to whole bars
    return Beat(tempo=REFERENCE_TEMPO, hits=hits, bars=bars, name="Reference All Pads")


def write_references(out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for beat, label in ((reference_groove(), "Reference Groove"), (reference_all_pads(), "Reference All Pads")):
        written.append(write(beat, os.path.join(out_dir, filename_for(beat, label))))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the two reference drum beats, for checking how MIDI lands in Ableton."
    )
    parser.add_argument(
        "out_dir",
        nargs="?",
        default=os.path.join(os.path.expanduser("~"), "Music", "Beat Thief"),
        help="Where to write them (default: ~/Music/Beat Thief)",
    )
    args = parser.parse_args()

    for path in write_references(args.out_dir):
        print(f"Wrote {path}")
    print(f"\nSet Ableton's tempo to {REFERENCE_TEMPO:g} BPM before dragging these in - Live ignores")
    print("a MIDI file's own tempo and drops the notes onto the set's grid.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Check a stolen loop's MIDI against its own audio, independent of every
classifier that helped produce it.

percussion_splitter's votes and groove_reader's pulse logic can both be
self-consistent and still wrong about what's actually in the recording - a
bug shared by both stages would pass its own check. This measures the
*output* (the quantized grid) against the *source* (the isolated drum wav)
directly: for each piece, does audio energy show up on the steps the MIDI
says it played, and stay away from the steps it says it didn't?

It also asks the other question a single loop can't answer on its own: put
the cut next to itself four times over - does it still line up? (see seam).

Dev tool only. Not imported by gui.py, beat_loop.py, or anything the app
loads. It must not share the blind spots of the stages it is judging, so it
takes nothing from groove_reader and nothing from percussion_splitter beyond
one frequency range. pulse.py it does use, and that is fine: the app uses it
to choose where to cut, never to decide what was played, so it has no stake
in the answer being measured here.

Run the seeded benchmark:

    python3 beat_fidelity.py

or score one section directly:

    python3 beat_fidelity.py <isolated_drum.wav> <tempo> <start_sec> <end_sec>
"""

from __future__ import annotations

import statistics
import sys
from typing import NamedTuple

import numpy as np
from scipy.signal import butter, sosfilt

import adtof
import beat_loop
import beat_writer
import drum_transcriber
import percussion_splitter
import pulse

SAMPLE_RATE = drum_transcriber.SAMPLE_RATE

# Frequency band each piece is measured in, reusing ranges already chosen
# and justified elsewhere rather than inventing new ones. drum_transcriber's
# _VELOCITY_BAND is indexed by class (kick, snare, tom, hi-hat, cymbal); the
# open hat shares the closed hat's band, and the two structural pads that
# split off the snare class (ghost snare) share the snare's. Tambourine and
# shaker both live in percussion_splitter's jingle band - nothing separates
# those two timbrally (see that module's docstring), so the fidelity check
# can't either.
_BAND_FOR_PIECE = {
    "kick": drum_transcriber._VELOCITY_BAND[drum_transcriber._KICK],
    "snare": drum_transcriber._VELOCITY_BAND[drum_transcriber._SNARE],
    "ghost snare": drum_transcriber._VELOCITY_BAND[drum_transcriber._SNARE],
    "low-mid tom": drum_transcriber._VELOCITY_BAND[drum_transcriber._TOM],
    "closed hat": drum_transcriber._VELOCITY_BAND[drum_transcriber._HIHAT],
    "open hat": drum_transcriber._VELOCITY_BAND[drum_transcriber._HIHAT],
    "crash": drum_transcriber._VELOCITY_BAND[drum_transcriber._CYMBAL],
    "tambourine": percussion_splitter._JINGLE_HZ,
    "shaker": percussion_splitter._JINGLE_HZ,
}

# Peak amplitude is taken in a window this wide around each step's audio
# time, matching drum_transcriber's own velocity measurement window.
_WINDOW_SEC = drum_transcriber._VELOCITY_WINDOW_SEC

# A piece needs at least this many "on" and this many "off" steps before its
# separation means anything - a median of one or two points isn't one.
_MIN_STEPS_FOR_VERDICT = 4


class StepEnergy(NamedTuple):
    step: int
    on: bool    # does the MIDI have a note for this piece on this step
    db: float   # this piece's band energy at this step's audio time
    suspect: bool  # on the wrong side of the on/off gap


class VoiceFidelity(NamedTuple):
    piece: str
    hits: int
    on_db: float | None
    off_db: float | None
    separation_db: float | None  # on_db - off_db; None when too few steps for a verdict
    steps: list[StepEnergy]


def _band(audio: np.ndarray, sample_rate: int, low: float | None, high: float | None) -> np.ndarray:
    """Same shape as the _band helper in percussion_splitter.py and
    drum_transcriber._band_filtered - duplicated on purpose, matching how
    each of those already owns its own copy rather than sharing one."""
    nyquist = sample_rate / 2.0
    if low is not None and high is not None:
        sos = butter(4, [low / nyquist, min(high, nyquist * 0.99) / nyquist], btype="bandpass", output="sos")
    elif low is not None:
        sos = butter(4, low / nyquist, btype="highpass", output="sos")
    elif high is not None:
        sos = butter(4, high / nyquist, btype="lowpass", output="sos")
    else:
        return audio
    return sosfilt(sos, audio).astype(np.float32)


def _db(amplitude: float, floor: float = 1e-7) -> float:
    return 20.0 * float(np.log10(max(amplitude, floor)))


def _peak_db(band: np.ndarray, time_sec: float, sample_rate: int) -> float:
    start = int(time_sec * sample_rate)
    end = start + max(1, int(_WINDOW_SEC * sample_rate))
    start, end = max(0, start), min(band.size, end)
    amplitude = float(np.abs(band[start:end]).max()) if end > start else 0.0
    return _db(amplitude)


def check(loop: beat_loop.Loop, audio: np.ndarray, sample_rate: int) -> list[VoiceFidelity]:
    """One VoiceFidelity per piece the loop uses, each measured against the
    audio independently of whatever classifier decided the MIDI."""
    step_sec = loop.beat.seconds_per_step
    total_steps = loop.beat.total_steps

    on_steps: dict[str, set[int]] = {}
    for hit in loop.beat.hits:
        on_steps.setdefault(hit.piece, set()).add(int(hit.step))

    results = []
    for piece in sorted(on_steps):
        band_range = _BAND_FOR_PIECE.get(piece)
        if band_range is None:
            continue
        band = _band(audio, sample_rate, *band_range)

        steps = []
        for step in range(total_steps):
            time_sec = loop.origin_sec + step * step_sec
            steps.append(StepEnergy(step, step in on_steps[piece], _peak_db(band, time_sec, sample_rate), False))

        on = [s.db for s in steps if s.on]
        off = [s.db for s in steps if not s.on]
        if len(on) < _MIN_STEPS_FOR_VERDICT or len(off) < _MIN_STEPS_FOR_VERDICT:
            results.append(VoiceFidelity(piece, len(on), None, None, None, steps))
            continue

        on_db, off_db = statistics.median(on), statistics.median(off)
        midpoint = (on_db + off_db) / 2.0
        steps = [s._replace(suspect=(s.on and s.db < midpoint) or (not s.on and s.db > midpoint)) for s in steps]
        results.append(VoiceFidelity(piece, len(on), on_db, off_db, on_db - off_db, steps))
    return results


def _format_suspects(fidelity: VoiceFidelity) -> str:
    suspects = [s for s in fidelity.steps if s.suspect]
    if not suspects:
        return "suspects: none"
    midpoint = (fidelity.on_db + fidelity.off_db) / 2.0
    parts = [
        f"{'on' if s.on else 'off'}-step {s.step} ({s.db - midpoint:+.1f}dB {'below' if s.on else 'above'} midpoint)"
        for s in suspects
    ]
    return "suspects: " + ", ".join(parts)


def _delta_str(current: float | None, previous: float | None) -> str:
    if current is None or previous is None:
        return ""
    delta = current - previous
    if abs(delta) < 0.05:
        return " (=)"
    return f" ({delta:+.1f})"


def report(fidelities: list[VoiceFidelity], baseline: dict[str, float | None] | None = None) -> str:
    lines = []
    for f in fidelities:
        delta = _delta_str(f.separation_db, baseline.get(f.piece)) if baseline else ""
        if f.separation_db is None:
            lines.append(f"{f.piece:<12} hits={f.hits:<3} (too few steps for a verdict)")
            continue
        lines.append(
            f"{f.piece:<12} hits={f.hits:<3} on={sum(1 for s in f.steps if s.on):<3} "
            f"off={sum(1 for s in f.steps if not s.on):<3} separation={f.separation_db:+.1f}dB{delta}   "
            f"{_format_suspects(f)}"
        )
    return "\n".join(lines)


def _load_audio(wav_path: str) -> np.ndarray:
    return adtof.create_adtof_processor().load_audio(wav_path)


# --- does the loop actually loop -----------------------------------------
#
# check() above asks whether the MIDI matches the audio. That can be perfect
# while the loop is still unusable, because it says nothing about the two
# ways a *cut* goes wrong:
#
#   - the start misses the one, so the file opens mid-phrase
#   - the length isn't quite N bars, so every repeat slides further off
#
# The obvious test - tile the cut and look at the seam - cannot find the
# second of those, and it is worth saying why, because it looks like it
# should. A tiling repeats at exactly the length that was cut, so the cut's
# own claimed grid always agrees with it perfectly. The error is invisible
# from inside.
#
# What does find it is comparing the tiling against the recording it came
# out of: cut A, and let the source play on into what would have been B.
# If A really is N bars, the drummer's next hit after the loop point lands
# exactly where A's first hit lands when it comes round again. If A is 20ms
# short, it lands 20ms early, and by the fourth pass it is 80ms out.
#
# Everything is reported in milliseconds, which is the unit the mistake is
# heard in.

_SEAM_MIN_ONSETS = 4

# How much of the recording either side of the cut to hand the detector as
# context. A cut that opens partway through a ringing kick gives a detector
# started cold nothing to judge that ringing against, and it reads as a hit
# at time zero - which is exactly the reading "did this start on the one"
# must not get wrong. Feeding it the real audio just before the cut and
# subtracting the offset afterwards removes the question entirely.
_SEAM_LEAD_SEC = 0.5
_SEAM_EDGE_SEC = 0.02


def _kicks_from(audio: np.ndarray, first: int, length: int, sample_rate: int) -> np.ndarray:
    """Kick times inside audio[first:first + length], relative to `first`,
    detected with whatever real audio precedes it as context."""
    lead = min(int(_SEAM_LEAD_SEC * sample_rate), first)
    window = audio[first - lead:first + length]
    found = np.asarray(pulse.kicks(window, sample_rate)) - lead / sample_rate
    # A hit sitting exactly on the boundary is reported a few milliseconds
    # early - the detector walks back to the foot of the rise, and the foot
    # of the very first hit is just outside. Dropping anything negative would
    # throw away precisely the hit this is here to find, so a hair either
    # side of the boundary counts as on it.
    return np.maximum(found[found >= -_SEAM_EDGE_SEC], 0.0)


class Seam(NamedTuple):
    head_ms: float | None    # how far past the start of the file the first kick is
    loop_ms: float | None    # how early (-) or late (+) the loop point comes round
    grid_ms: float | None    # how far the playing sits off the loop's own grid

    def __str__(self) -> str:
        if self.head_ms is None:
            return "loop: not enough kicks in the cut to say"
        parts = [f"starts {self.head_ms:+.0f}ms off the one"]
        if self.loop_ms is not None:
            parts.append(f"loop point {self.loop_ms:+.0f}ms")
        if self.grid_ms is not None:
            parts.append(f"{self.grid_ms:.0f}ms off its own grid")
        return "loop: " + ", ".join(parts)


def _wrap(value: float, period: float) -> float:
    """`value` folded into (-period/2, +period/2]."""
    folded = value % period
    return folded - period if folded > period / 2 else folded


def seam(loop: beat_loop.Loop, audio: np.ndarray, sample_rate: int) -> Seam:
    """Whether the cut starts where it should and comes round where it should."""
    span = loop.beat.duration_sec
    first = int(round(loop.origin_sec * sample_rate))
    length = int(round(span * sample_rate))
    piece = audio[first:first + length]
    if length <= 0 or piece.size < length:
        return Seam(None, None, None)

    inside = _kicks_from(audio, first, length, sample_rate)
    if inside.size < _SEAM_MIN_ONSETS:
        return Seam(None, None, None)
    head_ms = float(inside[0] * 1000)

    # How well the playing agrees with the grid the loop claims. A cut whose
    # tempo is wrong shows up here even before anything is repeated.
    sixteenth = 60.0 / loop.tempo / 4 if loop.tempo > 0 else 0.0
    grid_ms = None
    if sixteenth > 0:
        vector = np.exp(2j * np.pi * (inside % sixteenth) / sixteenth).mean()
        phase = (np.angle(vector) / (2 * np.pi) % 1.0) * sixteenth
        grid_ms = float(np.median([abs(_wrap(at - phase, sixteenth)) for at in inside]) * 1000)

    # Where the loop point falls against where the drummer actually went.
    # `inside[0] + span` is when the loop brings its first kick back round;
    # the source's own next kick after the cut ends is where that should be.
    if audio.size < first + length + length // 2:
        return Seam(head_ms, None, grid_ms)
    continued = _kicks_from(audio, first + length, length, sample_rate)
    if continued.size == 0:
        return Seam(head_ms, None, grid_ms)

    expected = float(continued[0])          # relative to the end of the cut
    return Seam(head_ms, (inside[0] - expected) * 1000, grid_ms)


# One accepted scoreboard, checked in. Committed deliberately (--accept) so
# that "did the pipeline get more faithful" is a line in `git log` rather
# than a number that only ever lived in a terminal.
BASELINE_PATH = "fidelity_baseline.txt"


def _load_baseline(path: str = BASELINE_PATH) -> dict[tuple[str, str], float | None]:
    baseline: dict[tuple[str, str], float | None] = {}
    try:
        with open(path) as handle:
            for line in handle:
                case_label, piece, value = line.rstrip("\n").split("|")
                baseline[(case_label, piece)] = None if value == "None" else float(value)
    except FileNotFoundError:
        pass
    return baseline


def _save_baseline(rows: list[tuple[str, str, float | None]], path: str = BASELINE_PATH) -> None:
    with open(path, "w") as handle:
        for case_label, piece, value in rows:
            handle.write(f"{case_label}|{piece}|{'None' if value is None else value}\n")


def main(argv: list[str]) -> int:
    accept = "--accept" in argv
    argv = [a for a in argv if a != "--accept"]

    if len(argv) not in (0, 4):
        print(__doc__)
        return 1

    if not argv:
        import fidelity_cases
        baseline = _load_baseline()
        rows: list[tuple[str, str, float | None]] = []
        exit_code = 0
        for case in fidelity_cases.CASES:
            loop = beat_loop.build(case.wav_path, case.tempo, case.start_sec, case.end_sec)
            audio = _load_audio(case.wav_path)
            fidelities = check(loop, audio, SAMPLE_RATE)
            case_baseline = {piece: value for (label, piece), value in baseline.items() if label == case.label}
            print(f"\n{case.label}")
            print(report(fidelities, case_baseline))
            print(seam(loop, audio, SAMPLE_RATE))
            rows.extend((case.label, f.piece, f.separation_db) for f in fidelities)
            if not fidelity_cases.check_expectations(case, fidelities):
                exit_code = 1
        if accept:
            _save_baseline(rows)
            print(f"\nWrote {BASELINE_PATH}")
        return exit_code

    wav_path, tempo, start_sec, end_sec = argv
    loop = beat_loop.build(wav_path, float(tempo), float(start_sec), float(end_sec))
    audio = _load_audio(wav_path)
    print(report(check(loop, audio, SAMPLE_RATE)))
    print(seam(loop, audio, SAMPLE_RATE))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

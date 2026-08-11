#!/usr/bin/env python3
"""Check a stolen loop's MIDI against its own audio, independent of every
classifier that helped produce it.

percussion_splitter's votes and groove_reader's pulse logic can both be
self-consistent and still wrong about what's actually in the recording - a
bug shared by both stages would pass its own check. This measures the
*output* (the quantized grid) against the *source* (the isolated drum wav)
directly: for each piece, does audio energy show up on the steps the MIDI
says it played, and stay away from the steps it says it didn't?

Dev tool only. Not imported by gui.py, beat_loop.py, or anything the app
loads, and imports nothing from percussion_splitter or groove_reader - it
must not share their blind spots.

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
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

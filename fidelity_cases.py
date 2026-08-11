#!/usr/bin/env python3
"""The benchmark suite beat_fidelity.py runs by default: real sections of
real stems, each with what should be true of it by ear.

Grows over time - adding a stolen song's stem as a case is one line. Seeded
from the only ground truth Beat Thief's percussion split has (see
`officially-missing-you-tambourine-labels` in project memory): Stefan's
by-ear labelling of where the tambourine plays in the drum stem of
"Officially Missing You" (Brasstracks)."""

from __future__ import annotations

from typing import NamedTuple

import beat_fidelity

_OFFICIALLY_MISSING_YOU = (
    "/Users/Stefan/Music/Beat Thief/Officially Missing You (Bonus Track) - Brasstracks/"
    "Officially Missing You (Bonus Track) - Brasstracks (Isolated Drums at 105.373 BPM).wav"
)
_OMY_TEMPO = 105.373


class Case(NamedTuple):
    label: str
    wav_path: str
    tempo: float
    start_sec: float
    end_sec: float
    # piece -> True (must show up with at least one hit) or False (must not).
    # The only ground truth this has - a numeric score with no by-ear check
    # behind it can drift arbitrarily and still "improve".
    expect: dict[str, bool]


CASES = [
    Case(
        "Officially Missing You - solo tambourine (133-143s)",
        _OFFICIALLY_MISSING_YOU, _OMY_TEMPO, 133.0, 143.0,
        {"tambourine": True, "snare": False},
    ),
    Case(
        "Officially Missing You - intro, no tambourine (15-25s)",
        _OFFICIALLY_MISSING_YOU, _OMY_TEMPO, 15.0, 25.0,
        {"tambourine": False},
    ),
    Case(
        "Officially Missing You - verse, tambourine present (40-50s)",
        _OFFICIALLY_MISSING_YOU, _OMY_TEMPO, 40.0, 50.0,
        {"tambourine": True},
    ),
]


def check_expectations(case: Case, fidelities: list[beat_fidelity.VoiceFidelity]) -> bool:
    """Print and return whether every by-ear expectation for `case` held.
    A hard FAIL regardless of what the separation numbers say - that's the
    only ground truth here, and no score is allowed to override it."""
    present = {f.piece for f in fidelities}
    ok = True
    for piece, expected in case.expect.items():
        actual = piece in present
        if actual != expected:
            state = "present" if actual else "absent"
            wanted = "present" if expected else "absent"
            print(f"  expect FAIL: {piece} should be {wanted}, is {state}")
            ok = False
    return ok

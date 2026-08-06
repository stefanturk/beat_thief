"""ADTOF drum transcription model, vendored. See NOTICE.md for where this
came from and what was changed.

This package is only the model and its audio front end. Everything Beat
Thief does with the model's output - peak picking, velocity, the open/closed
hi-hat split, the note map, writing the MIDI - lives in drum_transcriber.py,
which is the only thing that should import this.

Vendored rather than pip-installed for two reasons. The published package
doesn't import on Python 3.9 (its post_processing.py annotates a function
signature with `Sequence[float] | float` and has no `from __future__ import
annotations`, so the annotation is evaluated at import time and raises), and
that broken module is the one part we don't want anyway - it writes a flat
velocity of 100 on every note, a 120 BPM header on every song, and a kick on
note 35, which sits below the first pad of Ableton's default Drum Rack."""

from __future__ import annotations

import os

from .audio import AudioProcessor, create_adtof_processor, process_audio_file
from .model import (
    ADTOFFrameRNN,
    calculate_n_bins,
    create_frame_rnn_model,
    load_audio_for_model,
    load_pytorch_weights,
)

WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.pth")

# The model's five outputs, in the order it emits them, as General MIDI note
# numbers. Beat Thief remaps some of these before writing a file - see
# drum_transcriber._NOTE_FOR_CLASS.
LABELS_5 = (35, 38, 47, 42, 49)

# Per-class peak-picking thresholds published with the model. These are the
# values its 88.5% F-measure was measured at; changing one changes that
# number, so they're recorded here rather than tuned by feel.
THRESHOLDS_5 = (0.22, 0.24, 0.32, 0.22, 0.30)

# Frames per second of the model's output activations.
FPS = 100

__all__ = [
    "ADTOFFrameRNN",
    "AudioProcessor",
    "FPS",
    "LABELS_5",
    "THRESHOLDS_5",
    "WEIGHTS_PATH",
    "calculate_n_bins",
    "create_adtof_processor",
    "create_frame_rnn_model",
    "load_audio_for_model",
    "load_pytorch_weights",
    "process_audio_file",
]

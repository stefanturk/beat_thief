#!/usr/bin/env python3
"""Isolate a song's bass from downloaded MP3s: the bass, pulled out of the
song with Demucs, noise-gated and written as a wav. Trimmed and tempo-
aligned to the same shared song_alignment() as every other isolated
instrument (see instrument_isolator.py), so it lines up exactly with the
drums, harmony and vocals exports from the same song.

There's no MIDI here any more. It was a pitch-tracked guess that was worse
than Ableton's own audio-to-MIDI, and it went out with the whole-song drum
MIDI it was built to line up with."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from pydub import AudioSegment

import instrument_isolator

DEFAULT_OUTPUT = os.path.join(os.path.expanduser("~"), "Downloads", "Song Downloads")

_HTDEMUCS_MODEL = "htdemucs"

# What every bass output filename for a song contains, and what identifies
# bass files (vs. e.g. a sibling drums export) within a song's shared
# "<title> (Isolated)" folder - see instrument_isolator.
_LABEL = "Isolated Bass"

# Written alongside each song's bass output files to identify exactly which
# source mp3 they were produced from - see instrument_isolator.
# source_marker_matches. Doesn't depend on tempo/filename, so it can be
# checked before song_alignment() (and its slow tempo work + possible
# interactive drift prompt) ever runs.
_SOURCE_MARKER_FILENAME = ".bass_source.json"

# Demucs' bass separation isn't perfect - what's left over after the real
# bass content is imperfect stem "bleed"/noise-floor artifacts, which
# confuse pitch tracking if left in. Calibrated against a real isolated
# bass stem: its noise-floor lead-in peaked at ~5.25% of the file's overall
# peak amplitude, matching this threshold almost exactly. Drums don't get
# this treatment - kick/snare/cymbal hits already sit well above the noise
# floor, so there's nothing worth gating out there.
_NOISE_GATE_THRESHOLD_RATIO = 0.05
_NOISE_GATE_WINDOW_MS = 20  # short enough to not gate out quick, quiet notes; long enough for a stable RMS reading
_NOISE_GATE_FADE_MS = 5  # smooths each gate on/off transition so it doesn't click


def _apply_noise_gate(wav_path: str) -> None:
    """Silence any part of wav_path that sits below
    _NOISE_GATE_THRESHOLD_RATIO of the file's own peak amplitude - these are
    near-noise-floor artifacts of imperfect stem separation, not real bass
    content, and left in they're audible hiss under an otherwise clean stem.

    Gates by short RMS window rather than per-sample, with the gate curve
    itself smoothed by a short moving average, so a gate transition fades
    rather than clicking."""
    audio = AudioSegment.from_wav(wav_path)
    channels = audio.channels
    sr = audio.frame_rate
    raw = np.array(audio.get_array_of_samples())
    if raw.size == 0:
        return

    frames = raw.reshape(-1, channels).astype(np.float64)
    peak = np.abs(frames).max()
    if peak <= 0:
        return
    threshold = _NOISE_GATE_THRESHOLD_RATIO * peak

    window_frames = max(1, int(_NOISE_GATE_WINDOW_MS / 1000 * sr))
    n_frames = frames.shape[0]
    gate = np.ones(n_frames, dtype=np.float64)
    for start in range(0, n_frames, window_frames):
        end = min(start + window_frames, n_frames)
        window_rms = np.sqrt(np.mean(frames[start:end] ** 2))
        if window_rms < threshold:
            gate[start:end] = 0.0

    fade_frames = max(1, int(_NOISE_GATE_FADE_MS / 1000 * sr))
    kernel = np.ones(fade_frames) / fade_frames
    gate = np.convolve(gate, kernel, mode="same")

    gated = frames * gate[:, np.newaxis]
    max_val = float(2 ** (8 * audio.sample_width - 1))
    gated = np.clip(gated, -max_val, max_val - 1)
    gated_interleaved = gated.reshape(-1).astype(raw.dtype)

    gated_audio = audio._spawn(gated_interleaved.tobytes())
    gated_audio.export(wav_path, format="wav")


def _output_basename(title: str, tempo: float) -> str:
    return f"{title} ({_LABEL} at {tempo:.3f} BPM)"


def isolate_bass(mp3_path: str, context: instrument_isolator.RunContext | None = None) -> bool:
    """Produce an isolated bass wav for a single song, written into its
    shared "<title> (Isolated)" folder alongside any other instrument
    exported from the same song. Returns False (skipped) if a bass output
    already exists for this exact source mp3 (see
    instrument_isolator.source_marker_matches) - an existing output whose
    marker is missing or doesn't match is treated as stale (e.g. a leftover
    folder from an earlier run or a different file that happened to share
    this title) and reprocessed rather than trusted."""
    context = context or instrument_isolator.DEFAULT_CONTEXT
    title = os.path.splitext(os.path.basename(mp3_path))[0]
    song_dir = instrument_isolator.song_output_dir(mp3_path)
    marker_matches = instrument_isolator.source_marker_matches(song_dir, mp3_path, _SOURCE_MARKER_FILENAME)

    if marker_matches and instrument_isolator.has_existing_outputs(song_dir, _LABEL, require_midi=False):
        print(f"{title}: bass already isolated, nothing to do.")
        return False

    print(f"{title}: isolating bass (this can take a few minutes)...")
    trim_ms, tempo = instrument_isolator.song_alignment(mp3_path, interactive=context.interactive)

    stem_dir = instrument_isolator.separated_stems(mp3_path, _HTDEMUCS_MODEL, context)
    bass_wav = os.path.join(stem_dir, "bass.wav")

    os.makedirs(song_dir, exist_ok=True)
    instrument_isolator.clear_stale_outputs(song_dir, _LABEL)
    basename = _output_basename(title, tempo)
    wav_path = os.path.join(song_dir, basename + ".wav")
    instrument_isolator.trim_and_export(bass_wav, trim_ms, wav_path)
    _apply_noise_gate(wav_path)

    instrument_isolator.write_source_marker(song_dir, mp3_path, _SOURCE_MARKER_FILENAME)

    print(f"{title}: bass isolated ({tempo:.3f} BPM).")
    return True


def isolate_bass_for_folder(output_dir: str, context: instrument_isolator.RunContext | None = None) -> None:
    mp3_files = sorted(f for f in os.listdir(output_dir) if f.lower().endswith(".mp3"))
    if not mp3_files:
        print("No MP3s found to isolate bass from.")
        return

    for filename in mp3_files:
        path = os.path.join(output_dir, filename)
        try:
            isolate_bass(path, context=context)
        except Exception as e:
            print(f"  Could not isolate bass for {filename}, skipping: {e}")


def isolate_bass_for_single_file(path: str, context: instrument_isolator.RunContext | None = None) -> None:
    # Left to propagate - see the matching comment in
    # drum_isolator.isolate_drums_for_single_file.
    isolate_bass(path, context=context)


def isolate_bass_for_path(path: str, context: instrument_isolator.RunContext | None = None) -> None:
    if os.path.isfile(path):
        isolate_bass_for_single_file(path, context=context)
    else:
        isolate_bass_for_folder(path, context=context)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isolate bass to a wav from downloaded MP3s."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help=f"Folder of MP3s (or a single MP3 file) to isolate bass from (default: {DEFAULT_OUTPUT})",
    )

    args = parser.parse_args(sys.argv[1:])

    try:
        isolate_bass_for_path(args.path)
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(130)
    finally:
        instrument_isolator.clear_stem_cache()


if __name__ == "__main__":
    main()

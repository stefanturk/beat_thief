#!/usr/bin/env python3
"""Isolate a song's instrumental from downloaded MP3s - everything but the
singing, as one file. Just audio, no MIDI step, like vocals_isolator.py.

Not a fifth part of the song: it's drums, bass and harmony played together
(see drum_isolator.py, bass_isolator.py, harmony_isolator.py), so it
overlaps them. Here so that "the song without the vocals" is one square
rather than three files to line up in a DAW.

Costs no more than any other stem. demucs separates all four sources in
one pass whatever's asked for - its own --two-stems vocals instrumental is
the other three summed afterwards - and that pass is shared with every
other instrument taken from the song (see instrument_isolator.
separated_stems). Summed from demucs' raw stems rather than the exported
ones, so nothing bass_isolator's noise gate takes out goes missing here."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.io import wavfile

import instrument_isolator

DEFAULT_OUTPUT = os.path.join(os.path.expanduser("~"), "Downloads", "Song Downloads")

_HTDEMUCS_MODEL = "htdemucs"

# What every instrumental output filename for a song contains, and what
# identifies instrumental files within a song's folder - see
# instrument_isolator.
_LABEL = "Isolated Instrumental"

# Written alongside each song's instrumental output file to identify
# exactly which source mp3 it was produced from - see instrument_isolator.
# source_marker_matches.
_SOURCE_MARKER_FILENAME = ".instrumental_source.json"

# The three demucs sources that aren't the vocals.
_SOURCES = ("drums", "bass", "other")


def _output_basename(title: str) -> str:
    # No BPM suffix, like vocals - nothing about it is tempo-dependent.
    return f"{title} ({_LABEL})"


def _summed_wav(stem_dir: str) -> str:
    """Write drums + bass + other into the separation's own folder and
    return its path. Kept there rather than beside the song: it's an
    intermediate, and the stem cache already clears that folder away.

    Added in a wider type and held at full scale, so a loud passage clips
    the way the song itself would rather than wrapping round into noise."""
    path = os.path.join(stem_dir, "instrumental.wav")
    if os.path.exists(path):
        return path
    rate, total = None, None
    for name in _SOURCES:
        rate, samples = wavfile.read(os.path.join(stem_dir, name + ".wav"))
        dtype = samples.dtype
        total = samples.astype(np.float64) if total is None else total + samples
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        total = np.clip(total, info.min, info.max)
    wavfile.write(path, rate, total.astype(dtype))
    return path


def isolate_instrumental(mp3_path: str, context: instrument_isolator.RunContext | None = None) -> bool:
    """Produce an instrumental wav for a single song, written into its
    folder alongside any other instrument exported from the same song.
    Returns False (skipped) if one already exists for this exact source mp3
    (see instrument_isolator.source_marker_matches)."""
    context = context or instrument_isolator.DEFAULT_CONTEXT
    title = instrument_isolator.song_title(mp3_path)
    song_dir = instrument_isolator.song_output_dir(mp3_path)
    marker_matches = instrument_isolator.source_marker_matches(song_dir, mp3_path, _SOURCE_MARKER_FILENAME)

    if marker_matches and instrument_isolator.has_existing_outputs(song_dir, _LABEL, require_midi=False):
        print(f"{title}: instrumental already isolated, nothing to do.")
        return False

    print(f"{title}: isolating the instrumental (this can take a few minutes)...")
    trim_ms, _tempo = instrument_isolator.song_alignment(mp3_path, interactive=context.interactive)

    stem_dir = instrument_isolator.separated_stems(mp3_path, _HTDEMUCS_MODEL, context)
    instrumental_wav = _summed_wav(stem_dir)

    os.makedirs(song_dir, exist_ok=True)
    instrument_isolator.clear_stale_outputs(song_dir, _LABEL)
    wav_path = os.path.join(song_dir, _output_basename(title) + ".wav")
    instrument_isolator.trim_and_export(instrumental_wav, trim_ms, wav_path)

    instrument_isolator.write_source_marker(song_dir, mp3_path, _SOURCE_MARKER_FILENAME)

    print(f"{title}: instrumental isolated.")
    return True


def isolate_instrumental_for_folder(output_dir: str, context: instrument_isolator.RunContext | None = None) -> None:
    mp3_files = sorted(f for f in os.listdir(output_dir) if f.lower().endswith(".mp3"))
    if not mp3_files:
        print("No MP3s found to isolate an instrumental from.")
        return

    for filename in mp3_files:
        path = os.path.join(output_dir, filename)
        try:
            isolate_instrumental(path, context=context)
        except Exception as e:
            print(f"  Could not isolate the instrumental for {filename}, skipping: {e}")


def isolate_instrumental_for_single_file(path: str, context: instrument_isolator.RunContext | None = None) -> None:
    # Left to propagate - see the matching comment in
    # drum_isolator.isolate_drums_for_single_file.
    isolate_instrumental(path, context=context)


def isolate_instrumental_for_path(path: str, context: instrument_isolator.RunContext | None = None) -> None:
    if os.path.isfile(path):
        isolate_instrumental_for_single_file(path, context=context)
    else:
        isolate_instrumental_for_folder(path, context=context)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isolate the instrumental (everything but the vocals) from downloaded MP3s."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help=f"Folder of MP3s (or a single MP3 file) to take the instrumental from (default: {DEFAULT_OUTPUT})",
    )

    args = parser.parse_args(sys.argv[1:])

    try:
        isolate_instrumental_for_path(args.path)
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(130)
    finally:
        instrument_isolator.clear_stem_cache()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Isolate a song's vocals from downloaded MP3s - the lead and any backing
singing, with the band taken out from under it. Just audio, no MIDI step -
meant to drop straight into a DAW alongside the drums/bass/harmony exports
from the same song (see drum_isolator.py, bass_isolator.py,
harmony_isolator.py, instrument_isolator.py).

Those four stems are the whole song between them: nothing in the mix
belongs to two of them and nothing to none of them, so drums, bass, harmony
and vocals played together give the song back."""

from __future__ import annotations

import argparse
import os
import sys

import instrument_isolator

DEFAULT_OUTPUT = os.path.join(os.path.expanduser("~"), "Downloads", "Song Downloads")

_HTDEMUCS_MODEL = "htdemucs"

# What every vocals output filename for a song contains, and what
# identifies vocals files (vs. e.g. a sibling drums/bass export) within a
# song's shared "<title> (Isolated)" folder - see instrument_isolator.
_LABEL = "Isolated Vocals"

# Written alongside each song's vocals output file to identify exactly
# which source mp3 it was produced from - see instrument_isolator.
# source_marker_matches. Doesn't depend on the filename, so it can be
# checked before song_alignment() ever runs.
_SOURCE_MARKER_FILENAME = ".vocals_source.json"


def _output_basename(title: str) -> str:
    # No BPM suffix, unlike drums/bass - there's no MIDI step here for a
    # tempo to matter to, so nothing about the filename is tempo-dependent.
    return f"{title} ({_LABEL})"


def isolate_vocals(mp3_path: str, context: instrument_isolator.RunContext | None = None) -> bool:
    """Produce an isolated vocals wav for a single song, written into its
    shared "<title> (Isolated)" folder alongside any other instrument
    exported from the same song. Returns False (skipped) if a vocals output
    already exists for this exact source mp3 (see
    instrument_isolator.source_marker_matches) - an existing output whose
    marker is missing or doesn't match is treated as stale and reprocessed
    rather than trusted."""
    context = context or instrument_isolator.DEFAULT_CONTEXT
    title = os.path.splitext(os.path.basename(mp3_path))[0]
    song_dir = instrument_isolator.song_output_dir(mp3_path)
    marker_matches = instrument_isolator.source_marker_matches(song_dir, mp3_path, _SOURCE_MARKER_FILENAME)

    if marker_matches and instrument_isolator.has_existing_outputs(song_dir, _LABEL, require_midi=False):
        print(f"{title}: vocals already isolated, nothing to do.")
        return False

    print(f"{title}: isolating vocals (this can take a few minutes)...")
    trim_ms, _tempo = instrument_isolator.song_alignment(mp3_path, interactive=context.interactive)

    stem_dir = instrument_isolator.separated_stems(mp3_path, _HTDEMUCS_MODEL, context)
    vocals_wav = os.path.join(stem_dir, "vocals.wav")

    os.makedirs(song_dir, exist_ok=True)
    instrument_isolator.clear_stale_outputs(song_dir, _LABEL)
    basename = _output_basename(title)
    wav_path = os.path.join(song_dir, basename + ".wav")
    instrument_isolator.trim_and_export(vocals_wav, trim_ms, wav_path)

    instrument_isolator.write_source_marker(song_dir, mp3_path, _SOURCE_MARKER_FILENAME)

    print(f"{title}: vocals isolated.")
    return True


def isolate_vocals_for_folder(output_dir: str, context: instrument_isolator.RunContext | None = None) -> None:
    mp3_files = sorted(f for f in os.listdir(output_dir) if f.lower().endswith(".mp3"))
    if not mp3_files:
        print("No MP3s found to isolate vocals from.")
        return

    for filename in mp3_files:
        path = os.path.join(output_dir, filename)
        try:
            isolate_vocals(path, context=context)
        except Exception as e:
            print(f"  Could not isolate vocals for {filename}, skipping: {e}")


def isolate_vocals_for_single_file(path: str, context: instrument_isolator.RunContext | None = None) -> None:
    # Left to propagate - see the matching comment in
    # drum_isolator.isolate_drums_for_single_file.
    isolate_vocals(path, context=context)


def isolate_vocals_for_path(path: str, context: instrument_isolator.RunContext | None = None) -> None:
    if os.path.isfile(path):
        isolate_vocals_for_single_file(path, context=context)
    else:
        isolate_vocals_for_folder(path, context=context)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isolate vocals from downloaded MP3s."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help=f"Folder of MP3s (or a single MP3 file) to isolate vocals from (default: {DEFAULT_OUTPUT})",
    )

    args = parser.parse_args(sys.argv[1:])

    try:
        isolate_vocals_for_path(args.path)
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(130)
    finally:
        instrument_isolator.clear_stem_cache()


if __name__ == "__main__":
    main()

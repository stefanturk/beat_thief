#!/usr/bin/env python3
"""Isolate a song's harmony from downloaded MP3s: everything left after
drums, bass and vocals are pulled out - guitars, keys, pads, whatever else
is holding the chords up. Just audio, no MIDI step - meant to drop straight
into a DAW alongside the drums/bass/vocals exports from the same song (see
drum_isolator.py, bass_isolator.py, vocals_isolator.py,
instrument_isolator.py).

Together those four stems are the whole song: nothing belongs to two of
them and nothing to none of them, so playing them at once gives the song
back. Harmony used to include the vocals as well, which made that
impossible - there was no instrumental and no way to hear a vocal on its
own."""

from __future__ import annotations

import argparse
import os
import sys

import instrument_isolator

DEFAULT_OUTPUT = os.path.join(os.path.expanduser("~"), "Downloads", "Song Downloads")

_HTDEMUCS_MODEL = "htdemucs"

# What every harmony output filename for a song contains, and what
# identifies harmony files (vs. e.g. a sibling drums/bass export) within a
# song's shared "<title> (Isolated)" folder - see instrument_isolator.
_LABEL = "Isolated Harmony"

# Written alongside each song's harmony output file to identify exactly
# which source mp3 it was produced from - see instrument_isolator.
# source_marker_matches. Doesn't depend on the filename, so it can be
# checked before song_alignment() ever runs.
#
# The "_v2" is deliberate and load-bearing: harmony used to mean "other +
# vocals", and every harmony wav produced before that change still has the
# vocals in it. Those files sit in folders whose old .harmony_source.json
# says they're up to date, so under the old name they'd be trusted forever
# and quietly hand back a stem that isn't what harmony means any more.
# Changing the name makes every one of them read as stale, so it's rebuilt.
_SOURCE_MARKER_FILENAME = ".harmony_source_v2.json"


def _output_basename(title: str) -> str:
    # No BPM suffix, unlike drums/bass - there's no MIDI step here for a
    # tempo to matter to, so nothing about the filename is tempo-dependent.
    return f"{title} ({_LABEL})"


def isolate_harmony(mp3_path: str, context: instrument_isolator.RunContext | None = None) -> bool:
    """Produce an isolated harmony wav (everything but drums, bass and
    vocals) for a single song, written into its shared "<title> (Isolated)" folder
    alongside any other instrument exported from the same song. Returns
    False (skipped) if a harmony output already exists for this exact
    source mp3 (see instrument_isolator.source_marker_matches) - an
    existing output whose marker is missing or doesn't match is treated as
    stale and reprocessed rather than trusted."""
    context = context or instrument_isolator.DEFAULT_CONTEXT
    title = os.path.splitext(os.path.basename(mp3_path))[0]
    song_dir = instrument_isolator.song_output_dir(mp3_path)
    marker_matches = instrument_isolator.source_marker_matches(song_dir, mp3_path, _SOURCE_MARKER_FILENAME)

    if marker_matches and instrument_isolator.has_existing_outputs(song_dir, _LABEL, require_midi=False):
        print(f"{title}: harmony already isolated, nothing to do.")
        return False

    print(f"{title}: isolating harmony (this can take a few minutes)...")
    trim_ms, _tempo = instrument_isolator.song_alignment(mp3_path, interactive=context.interactive)

    stem_dir = instrument_isolator.separated_stems(mp3_path, _HTDEMUCS_MODEL, context)
    other_wav = os.path.join(stem_dir, "other.wav")

    os.makedirs(song_dir, exist_ok=True)
    instrument_isolator.clear_stale_outputs(song_dir, _LABEL)
    basename = _output_basename(title)
    wav_path = os.path.join(song_dir, basename + ".wav")
    instrument_isolator.trim_and_export(other_wav, trim_ms, wav_path)

    instrument_isolator.write_source_marker(song_dir, mp3_path, _SOURCE_MARKER_FILENAME)

    print(f"{title}: harmony isolated.")
    return True


def isolate_harmony_for_folder(output_dir: str, context: instrument_isolator.RunContext | None = None) -> None:
    mp3_files = sorted(f for f in os.listdir(output_dir) if f.lower().endswith(".mp3"))
    if not mp3_files:
        print("No MP3s found to isolate harmony from.")
        return

    for filename in mp3_files:
        path = os.path.join(output_dir, filename)
        try:
            isolate_harmony(path, context=context)
        except Exception as e:
            print(f"  Could not isolate harmony for {filename}, skipping: {e}")


def isolate_harmony_for_single_file(path: str, context: instrument_isolator.RunContext | None = None) -> None:
    try:
        isolate_harmony(path, context=context)
    except Exception as e:
        print(f"  Could not isolate harmony for {os.path.basename(path)}, skipping: {e}")


def isolate_harmony_for_path(path: str, context: instrument_isolator.RunContext | None = None) -> None:
    if os.path.isfile(path):
        isolate_harmony_for_single_file(path, context=context)
    else:
        isolate_harmony_for_folder(path, context=context)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isolate harmony (everything but drums, bass and vocals) from downloaded MP3s."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help=f"Folder of MP3s (or a single MP3 file) to isolate harmony from (default: {DEFAULT_OUTPUT})",
    )

    args = parser.parse_args(sys.argv[1:])

    try:
        isolate_harmony_for_path(args.path)
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(130)
    finally:
        instrument_isolator.clear_stem_cache()


if __name__ == "__main__":
    main()

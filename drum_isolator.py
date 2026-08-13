#!/usr/bin/env python3
"""Isolate a song's drums from downloaded MP3s: the full drum mix, pulled
out of the song with Demucs and written as a wav. Trimmed and tempo-aligned
to the same shared song_alignment() as every other isolated instrument (see
instrument_isolator.py), so it lines up exactly with the bass, harmony and
vocals exports from the same song.

There's no MIDI here any more. Transcribing four minutes of a live drummer
produced a faithful and unusable wall of notes; what's actually wanted is a
couple of bars with a good beat in them, on a grid, that loop. That's a
different feature working on a chosen section, and drum_transcriber.py is
still here for it."""

from __future__ import annotations

import argparse
import os
import sys

import instrument_isolator

DEFAULT_OUTPUT = os.path.join(os.path.expanduser("~"), "Downloads", "Song Downloads")

_HTDEMUCS_MODEL = "htdemucs"

# What every drums output filename for a song contains, and what
# identifies drum files (vs. e.g. a sibling bass export) within a song's
# shared "<title> (Isolated)" folder - see instrument_isolator.
_LABEL = "Isolated Drums"

# Written alongside each song's drum output files to identify exactly which
# source mp3 they were produced from - see instrument_isolator.
# source_marker_matches. Doesn't depend on tempo/filename, so it can be
# checked before song_alignment() (and its slow tempo work + possible
# interactive drift prompt) ever runs.
_SOURCE_MARKER_FILENAME = ".drums_source.json"


def _output_basename(title: str, tempo: float) -> str:
    return f"{title} ({_LABEL} at {tempo:.3f} BPM)"


def isolate_drums(mp3_path: str, context: instrument_isolator.RunContext | None = None) -> bool:
    """Produce an isolated drums wav for a single song, written into its
    shared "<title> (Isolated)" folder alongside any other instrument
    exported from the same song. Returns False (skipped) if a drums output
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
        print(f"{title}: drums already isolated, nothing to do.")
        return False

    print(f"{title}: isolating drums (this can take a few minutes)...")
    trim_ms, tempo = instrument_isolator.song_alignment(mp3_path, interactive=context.interactive)

    stem_dir = instrument_isolator.separated_stems(mp3_path, _HTDEMUCS_MODEL, context)
    drums_wav = os.path.join(stem_dir, "drums.wav")

    os.makedirs(song_dir, exist_ok=True)
    instrument_isolator.clear_stale_outputs(song_dir, _LABEL)
    basename = _output_basename(title, tempo)
    wav_path = os.path.join(song_dir, basename + ".wav")
    instrument_isolator.trim_and_export(drums_wav, trim_ms, wav_path)

    instrument_isolator.write_source_marker(song_dir, mp3_path, _SOURCE_MARKER_FILENAME)

    print(f"{title}: drums isolated ({tempo:.3f} BPM).")
    return True


def isolate_drums_for_folder(output_dir: str, context: instrument_isolator.RunContext | None = None) -> None:
    mp3_files = sorted(f for f in os.listdir(output_dir) if f.lower().endswith(".mp3"))
    if not mp3_files:
        print("No MP3s found to isolate drums from.")
        return

    for filename in mp3_files:
        path = os.path.join(output_dir, filename)
        try:
            isolate_drums(path, context=context)
        except Exception as e:
            print(f"  Could not isolate drums for {filename}, skipping: {e}")


def isolate_drums_for_single_file(path: str, context: instrument_isolator.RunContext | None = None) -> None:
    # Left to propagate, not swallowed like the folder loop's per-file catch:
    # pipeline._isolate_songs calls this directly for one song at a time and
    # depends on a real failure surfacing as a run error (see
    # test_pipeline.TestSeparatedAudioIsCleanedUp), so the GUI can say what
    # went wrong instead of quietly reporting "done" with no drums wav and
    # no explanation - the printed message here goes nowhere in a windowed
    # app with no attached console.
    isolate_drums(path, context=context)


def isolate_drums_for_path(path: str, context: instrument_isolator.RunContext | None = None) -> None:
    if os.path.isfile(path):
        isolate_drums_for_single_file(path, context=context)
    else:
        isolate_drums_for_folder(path, context=context)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isolate drums to a wav from downloaded MP3s."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help=f"Folder of MP3s (or a single MP3 file) to isolate drums from (default: {DEFAULT_OUTPUT})",
    )

    args = parser.parse_args(sys.argv[1:])

    try:
        isolate_drums_for_path(args.path)
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(130)
    finally:
        instrument_isolator.clear_stem_cache()


if __name__ == "__main__":
    main()

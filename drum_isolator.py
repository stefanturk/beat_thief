#!/usr/bin/env python3
"""Isolate a song's drums from downloaded MP3s: an isolated drum wav (the
full drum mix, pulled out of the song with Demucs) and a matching MIDI file
transcribed from it - no manual conversion needed. Both are trimmed and
tempo-aligned to the same shared song_alignment() as every other isolated
instrument (see instrument_isolator.py), so the drums MIDI lines up exactly
with the bass MIDI and any future instrument exports from the same song.

The transcription itself lives in drum_transcriber.py, which runs a trained
model over the drum wav and works out six pieces (kick, snare, toms, closed
and open hi-hat, cymbal) with per-hit velocity. This module is about which
files get written where."""

from __future__ import annotations

import argparse
import os
import sys

import pretty_midi

import drum_transcriber
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

# Written alongside each song's drum MIDI, and versioned separately from
# the wav's marker above, because the two go stale for different reasons.
# The wav is stale when the source mp3 changes; the MIDI is also stale when
# the transcription itself changes, and a .mid written by an older one has
# nothing about it to say so.
#
# Kept apart so that a transcriber change costs seconds rather than
# minutes: a stale .mid is rebuilt onto the wav that's already sitting
# there, instead of invalidating the wav and putting the song through
# demucs again for audio that would come out identical.
#
# Bump this whenever drum_transcriber's output changes. _v2 is the trained
# model replacing the spectral-centroid guesswork that came before it.
_MIDI_MARKER_FILENAME = ".drums_midi_v2.json"


def _write_drum_midi(wav_path: str, midi_path: str, tempo: float) -> None:
    """Transcribe the isolated drum wav (see drum_transcriber) and write
    every hit to a single midi_path track.

    Notes keep the exact times they were detected at - nothing is quantized
    or snapped to a grid, so ghost notes, flams and swing survive. The tempo
    comes from the shared song_alignment() (see isolate_drums) rather than a
    fresh per-file estimate, so this file's grid is the same one every other
    instrument from this song was exported against."""
    all_notes = drum_transcriber.transcribe(wav_path) if os.path.exists(wav_path) else []

    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    drum_track = pretty_midi.Instrument(program=0, is_drum=True, name=f"Drums ({round(tempo)})")
    drum_track.notes = sorted(all_notes, key=lambda note: note.start)
    midi.instruments.append(drum_track)
    midi.write(midi_path)


def _output_basename(title: str, tempo: float) -> str:
    return f"{title} ({_LABEL} at {tempo:.3f} BPM)"


def isolate_drums(mp3_path: str, write_midi: bool = True, context: instrument_isolator.RunContext | None = None) -> bool:
    """Produce an isolated drums wav (and, if write_midi, a matching MIDI)
    for a single song, written into its shared "<title> (Isolated)" folder
    alongside any other instrument exported from the same song. Returns
    False (skipped) if drum outputs already exist for this exact source mp3
    (see instrument_isolator.source_marker_matches) - existing outputs
    whose marker is missing or doesn't match are treated as stale (e.g. a
    leftover folder from an earlier run or a different file that happened
    to share this title) and reprocessed rather than trusted.

    If the wav is already there and the MIDI is missing - or was written by
    an older transcriber (see _MIDI_MARKER_FILENAME) - this transcribes onto
    the existing wav instead of re-running demucs. Demucs is by far the
    slowest part of this, and nothing about the wav changes based on whether
    MIDI is also requested or on how the MIDI is made."""
    context = context or instrument_isolator.DEFAULT_CONTEXT
    title = os.path.splitext(os.path.basename(mp3_path))[0]
    song_dir = instrument_isolator.song_output_dir(mp3_path)
    marker_matches = instrument_isolator.source_marker_matches(song_dir, mp3_path, _SOURCE_MARKER_FILENAME)
    midi_is_current = instrument_isolator.source_marker_matches(song_dir, mp3_path, _MIDI_MARKER_FILENAME)

    if (
        marker_matches
        and instrument_isolator.has_existing_outputs(song_dir, _LABEL, write_midi)
        and (midi_is_current or not write_midi)
    ):
        print(f"{title}: drums already isolated, nothing to do.")
        return False

    if marker_matches and instrument_isolator.has_existing_outputs(song_dir, _LABEL, require_midi=False):
        basename = instrument_isolator.find_existing_basename(song_dir, _LABEL)
        wav_path = os.path.join(song_dir, basename + ".wav")
        midi_path = os.path.join(song_dir, basename + ".mid")
        tempo = instrument_isolator.parse_tempo_from_basename(basename)
        rebuilt = os.path.exists(midi_path)
        _write_drum_midi(wav_path, midi_path, tempo)
        instrument_isolator.write_source_marker(song_dir, mp3_path, _MIDI_MARKER_FILENAME)
        print(f"{title}: drums MIDI {'rebuilt' if rebuilt else 'added'} ({tempo:.3f} BPM).")
        return True

    print(f"{title}: isolating drums (this can take a few minutes)...")
    trim_ms, tempo = instrument_isolator.song_alignment(mp3_path, interactive=context.interactive)

    stem_dir = instrument_isolator.separated_stems(mp3_path, _HTDEMUCS_MODEL, context)
    drums_wav = os.path.join(stem_dir, "drums.wav")

    os.makedirs(song_dir, exist_ok=True)
    instrument_isolator.clear_stale_outputs(song_dir, _LABEL)
    basename = _output_basename(title, tempo)
    wav_path = os.path.join(song_dir, basename + ".wav")
    instrument_isolator.trim_and_export(drums_wav, trim_ms, wav_path)

    if write_midi:
        midi_path = os.path.join(song_dir, basename + ".mid")
        _write_drum_midi(wav_path, midi_path, tempo)
        instrument_isolator.write_source_marker(song_dir, mp3_path, _MIDI_MARKER_FILENAME)

    instrument_isolator.write_source_marker(song_dir, mp3_path, _SOURCE_MARKER_FILENAME)

    print(f"{title}: drums isolated{' + MIDI' if write_midi else ''} ({tempo:.3f} BPM).")
    return True


def isolate_drums_for_folder(output_dir: str, write_midi: bool = True, context: instrument_isolator.RunContext | None = None) -> None:
    mp3_files = sorted(f for f in os.listdir(output_dir) if f.lower().endswith(".mp3"))
    if not mp3_files:
        print("No MP3s found to isolate drums from.")
        return

    for filename in mp3_files:
        path = os.path.join(output_dir, filename)
        try:
            isolate_drums(path, write_midi=write_midi, context=context)
        except Exception as e:
            print(f"  Could not isolate drums for {filename}, skipping: {e}")


def isolate_drums_for_single_file(path: str, write_midi: bool = True, context: instrument_isolator.RunContext | None = None) -> None:
    try:
        isolate_drums(path, write_midi=write_midi, context=context)
    except Exception as e:
        print(f"  Could not isolate drums for {os.path.basename(path)}, skipping: {e}")


def isolate_drums_for_path(path: str, write_midi: bool = True, context: instrument_isolator.RunContext | None = None) -> None:
    if os.path.isfile(path):
        isolate_drums_for_single_file(path, write_midi=write_midi, context=context)
    else:
        isolate_drums_for_folder(path, write_midi=write_midi, context=context)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isolate drums (wav, optionally + MIDI) from downloaded MP3s."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help=f"Folder of MP3s (or a single MP3 file) to isolate drums from (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--midi",
        action="store_true",
        help="Also write a MIDI file, not just the isolated wav: six pieces (kick, snare, toms, closed and open hi-hat, cymbal) with per-hit velocity, laid out for Ableton's default Drum Rack. Can also be given as a bare 'midi' argument.",
    )

    raw_args = sys.argv[1:]
    bare_midi = any(tok.lower() == "midi" for tok in raw_args)
    remaining_args = [tok for tok in raw_args if tok.lower() != "midi"]
    args = parser.parse_args(remaining_args)
    args.midi = args.midi or bare_midi

    try:
        isolate_drums_for_path(args.path, write_midi=args.midi)
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(130)
    finally:
        instrument_isolator.clear_stem_cache()


if __name__ == "__main__":
    main()

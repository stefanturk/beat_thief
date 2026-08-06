#!/usr/bin/env python3
"""Isolate a song's drums from downloaded MP3s: an isolated drum wav (the
full drum mix, pulled out of the song with Demucs) and a matching MIDI file
built by detecting hits directly in it — no manual conversion needed. Both
are trimmed and tempo-aligned to the same shared song_alignment() as every
other isolated instrument (see instrument_isolator.py), so the drums MIDI
lines up exactly with the bass MIDI and any future instrument exports from
the same song."""

from __future__ import annotations

import argparse
import os
import sys

import librosa
import numpy as np
import pretty_midi

import instrument_isolator

DEFAULT_OUTPUT = os.path.join(os.path.expanduser("~"), "Downloads", "Song Downloads")

_HTDEMUCS_MODEL = "htdemucs"

_NOTE_DURATION_SEC = 0.05

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

# Without per-instrument stems there's no ML model telling a kick hit from
# a snare hit apart anymore, but the mixed drums.wav still carries enough
# spectral information to guess: kick hits are almost all low-frequency
# energy, snares and toms sit in the middle, and cymbals/hi-hats are
# dominated by high-frequency energy. A hit's spectral centroid (the
# "center of mass" of its frequency content, in Hz) right after each onset
# is a simple, cheap way to tell those apart without a second model.
# Notes match Ableton's default Drum Rack mapping.
_KICK_NOTE = 36     # Bass Drum 1
_SNARE_NOTE = 38    # Acoustic Snare
_CYMBAL_NOTE = 42   # Closed Hi-Hat

# Fallback absolute thresholds, used only when a song has too few onsets to
# derive its own (see _detect_note_events) - real, in-context drum hits run
# far hotter than these, since a mixed recording's attack transients carry
# broadband energy that a clean isolated tone doesn't.
_KICK_MAX_CENTROID_HZ = 300.0
_SNARE_MAX_CENTROID_HZ = 2000.0

# How many onsets are needed before classifying hits relative to the song's
# own centroid distribution instead of falling back to the absolute
# thresholds above.
_MIN_ONSETS_FOR_RELATIVE_CLASSIFICATION = 12
_KICK_PERCENTILE = 33  # bottom third of a song's onsets -> kick
_SNARE_PERCENTILE = 67  # middle third -> snare, top third -> cymbal


_VELOCITY_WINDOW_SEC = 0.03
_CLASSIFY_WINDOW_SEC = 0.06  # longer than the velocity window, to capture more of a kick's low-frequency sustain past its initial (broadband) attack click


def _hit_centroid(window: np.ndarray, sr: int) -> float | None:
    """Spectral centroid (the "center of mass" of frequency content, in Hz)
    of a short hit window, or None if the window is silent/too small.

    Computed directly with numpy rather than librosa.feature.spectral_
    centroid, which defaults to n_fft=2048 - larger than our short hit
    window, so it silently zero-pads. Zero-padding a signal with sharp
    edges introduces spectral leakage (artificial high-frequency energy
    from the abrupt discontinuity), which was skewing nearly every real
    hit toward "cymbal" regardless of its actual pitch. A Hann window
    tapers those edges instead of leaving them abrupt, and sizing the FFT
    to the window itself avoids the padding altogether."""
    if window.size < 2:
        return None
    windowed = window * np.hanning(window.size)
    spectrum = np.abs(np.fft.rfft(windowed))
    total_energy = spectrum.sum()
    if total_energy <= 0:
        return None
    freqs = np.fft.rfftfreq(window.size, d=1.0 / sr)
    return float(np.sum(freqs * spectrum) / total_energy)


def _note_for_centroid(centroid: float | None, kick_threshold: float, snare_threshold: float) -> int:
    if centroid is None or centroid < kick_threshold:
        return _KICK_NOTE
    if centroid < snare_threshold:
        return _SNARE_NOTE
    return _CYMBAL_NOTE


def _detect_note_events(wav_path: str) -> list[pretty_midi.Note]:
    """Detect hit onsets in the isolated drum mix and turn each one into a
    MIDI note, guessing kick/snare/cymbal per hit from its spectral content
    since there's no separate stem per instrument.

    Classification is relative to the song's own onsets rather than fixed
    Hz cutoffs: a mixed recording's attack transients carry broadband
    energy no matter the instrument, so absolute thresholds tuned on clean
    isolated tones read almost every real hit as "cymbal". Splitting a
    song's own onsets into thirds by centroid (kick = lowest third, snare =
    middle third, cymbal = top third) adapts to each song's own mix/EQ
    instead. Falls back to fixed thresholds when there are too few onsets
    to make that split meaningful.

    Velocity is derived from the waveform's peak amplitude just after each
    onset, normalized against the loudest hit in the file (see
    instrument_isolator.velocities_from_amplitudes)."""
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    if onset_env.size == 0 or not np.isfinite(onset_env).any() or onset_env.max() <= 0:
        return []

    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    velocity_window_samples = max(1, int(_VELOCITY_WINDOW_SEC * sr))
    classify_window_samples = max(1, int(_CLASSIFY_WINDOW_SEC * sr))

    amplitudes = []
    centroids = []
    for start in onset_times:
        start_sample = int(start * sr)
        velocity_window = y[start_sample:start_sample + velocity_window_samples]
        classify_window = y[start_sample:start_sample + classify_window_samples]
        amplitudes.append(np.abs(velocity_window).max() if velocity_window.size else 0.0)
        centroids.append(_hit_centroid(classify_window, sr))

    if not amplitudes:
        return []

    velocities = instrument_isolator.velocities_from_amplitudes(amplitudes)

    valid_centroids = np.array([c for c in centroids if c is not None], dtype=np.float64)
    if valid_centroids.size >= _MIN_ONSETS_FOR_RELATIVE_CLASSIFICATION:
        kick_threshold = float(np.percentile(valid_centroids, _KICK_PERCENTILE))
        snare_threshold = float(np.percentile(valid_centroids, _SNARE_PERCENTILE))
    else:
        kick_threshold = _KICK_MAX_CENTROID_HZ
        snare_threshold = _SNARE_MAX_CENTROID_HZ

    notes = []
    for start, velocity, centroid in zip(onset_times, velocities, centroids):
        pitch = _note_for_centroid(centroid, kick_threshold, snare_threshold)
        notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=float(start), end=float(start) + _NOTE_DURATION_SEC))
    return notes


def _write_drum_midi(wav_path: str, midi_path: str, tempo: float) -> None:
    """Detect hits in the isolated drum wav and write them all to a single
    midi_path track, each classified as kick/snare/cymbal (see
    _detect_note_events). Notes keep their raw onset-detected times (no
    quantizing/snapping) - tempo comes from the shared song_alignment()
    (see isolate_drums), not a fresh per-file estimate, so it matches every
    other instrument exported from the same song."""
    all_notes = _detect_note_events(wav_path) if os.path.exists(wav_path) else []

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

    If the wav is already there (from an earlier run that didn't ask for
    MIDI) and only the MIDI is now missing, this transcribes onto the
    existing wav instead of re-running demucs - demucs is by far the
    slowest part of this, and nothing about the wav changes based on
    whether MIDI is also requested."""
    context = context or instrument_isolator.DEFAULT_CONTEXT
    title = os.path.splitext(os.path.basename(mp3_path))[0]
    song_dir = instrument_isolator.song_output_dir(mp3_path)
    marker_matches = instrument_isolator.source_marker_matches(song_dir, mp3_path, _SOURCE_MARKER_FILENAME)

    if marker_matches and instrument_isolator.has_existing_outputs(song_dir, _LABEL, write_midi):
        print(f"{title}: drums already isolated, nothing to do.")
        return False

    if marker_matches and instrument_isolator.has_existing_outputs(song_dir, _LABEL, require_midi=False):
        basename = instrument_isolator.find_existing_basename(song_dir, _LABEL)
        wav_path = os.path.join(song_dir, basename + ".wav")
        midi_path = os.path.join(song_dir, basename + ".mid")
        tempo = instrument_isolator.parse_tempo_from_basename(basename)
        _write_drum_midi(wav_path, midi_path, tempo)
        print(f"{title}: drums MIDI added ({tempo:.3f} BPM).")
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
        help="(Deprecated, may be removed later - MIDI quality is currently worse than Ableton's own audio-to-MIDI.) Also write a MIDI file, not just the isolated wav. Can also be given as a bare 'midi' argument.",
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

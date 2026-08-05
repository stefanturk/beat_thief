#!/usr/bin/env python3
"""Isolate a song's bass from downloaded MP3s: a bass.wav (the isolated bass
part, pulled out of the song with Demucs) and a bass.mid built by tracking
its pitch directly. Both are trimmed and tempo-aligned to the same shared
song_alignment() as every other isolated instrument (see
instrument_isolator.py), so bass.mid lines up exactly with drums.mid and any
future instrument exports from the same song."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

import librosa
import numpy as np
import pretty_midi

import instrument_isolator

BASS_DIR_NAME = "Bass"
DEFAULT_OUTPUT = os.path.join(os.path.expanduser("~"), "Downloads", "Song Downloads")

_HTDEMUCS_MODEL = "htdemucs"

BASS_WAV_FILENAME = "bass.wav"
MIDI_FILENAME = "bass.mid"

_EXPECTED_OUTPUTS = (BASS_WAV_FILENAME, MIDI_FILENAME)

# Bass is monophonic, so unlike drums (one fixed note per hit, guessed from
# spectral shape) each note's own pitch has to be tracked directly. Covers
# standard 4/5-string bass guitar range plus some headroom for slap/high
# notes and low synth bass.
_PITCH_FMIN_HZ = librosa.note_to_hz("E1")
_PITCH_FMAX_HZ = librosa.note_to_hz("G4")
_HOP_LENGTH = 512
_FRAME_LENGTH = 4096  # long enough to fit several periods of _PITCH_FMIN_HZ, avoiding pyin's low-frequency inaccuracy warning

_VELOCITY_WINDOW_SEC = 0.05
_MIN_NOTE_SEC = 0.05  # discard fragments too short to be a real note


def _detect_bass_notes(wav_path: str) -> list[pretty_midi.Note]:
    """Track pitch through the isolated bass stem and turn it into discrete
    MIDI notes.

    librosa.pyin gives a per-frame pitch estimate (f0, in Hz) plus a voiced
    flag. That's turned into a MIDI note number per frame, then grouped into
    notes: a new note starts whenever the rounded pitch changes, voicing
    drops to silence, or an onset lands on the same pitch (a re-attack -
    otherwise a repeated bassline note would get glued into one long note).

    Velocity is derived from the waveform's peak amplitude right after each
    note's start, normalized against the loudest note in the file (see
    instrument_isolator.velocities_from_amplitudes)."""
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    if y.size == 0:
        return []

    f0, voiced_flag, _ = librosa.pyin(y, fmin=_PITCH_FMIN_HZ, fmax=_PITCH_FMAX_HZ, sr=sr, hop_length=_HOP_LENGTH, frame_length=_FRAME_LENGTH)
    if f0 is None or f0.size == 0:
        return []
    frame_times = librosa.times_like(f0, sr=sr, hop_length=_HOP_LENGTH)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=_HOP_LENGTH)
    if onset_env.size and np.isfinite(onset_env).any() and onset_env.max() > 0:
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=_HOP_LENGTH, backtrack=True)
    else:
        onset_frames = np.array([], dtype=int)
    onset_frame_set = set(onset_frames.tolist())

    voiced = voiced_flag & ~np.isnan(f0)
    rounded_pitches = np.round(librosa.hz_to_midi(np.where(voiced, f0, 1.0)))

    segments = []  # (start_frame, end_frame_exclusive, pitch)
    current_pitch = None
    current_start = None
    for i in range(len(rounded_pitches)):
        if not voiced[i]:
            if current_pitch is not None:
                segments.append((current_start, i, current_pitch))
                current_pitch = None
            continue
        pitch = rounded_pitches[i]
        if current_pitch is None or pitch != current_pitch or i in onset_frame_set:
            if current_pitch is not None:
                segments.append((current_start, i, current_pitch))
            current_pitch = pitch
            current_start = i
    if current_pitch is not None:
        segments.append((current_start, len(rounded_pitches), current_pitch))

    if not segments:
        return []

    song_duration_sec = len(y) / sr
    velocity_window_samples = max(1, int(_VELOCITY_WINDOW_SEC * sr))

    note_infos = []
    amplitudes = []
    for start_frame, end_frame, pitch in segments:
        start_time = float(frame_times[start_frame])
        end_time = float(frame_times[end_frame]) if end_frame < len(frame_times) else song_duration_sec
        if end_time - start_time < _MIN_NOTE_SEC:
            continue
        start_sample = int(start_time * sr)
        window = y[start_sample:start_sample + velocity_window_samples]
        amplitudes.append(np.abs(window).max() if window.size else 0.0)
        note_infos.append((start_time, end_time, int(pitch)))

    if not note_infos:
        return []

    velocities = instrument_isolator.velocities_from_amplitudes(amplitudes)

    notes = []
    for (start_time, end_time, pitch), velocity in zip(note_infos, velocities):
        notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=start_time, end=end_time))
    return notes


def _write_bass_midi(song_dir: str, tempo: float) -> None:
    """Detect notes in bass.wav and write them to a single bass.mid track.
    Notes keep their raw pitch-tracked times (no quantizing/snapping) -
    tempo comes from the shared song_alignment() (see isolate_bass), not a
    fresh per-file estimate, so it matches every other instrument exported
    from the same song."""
    bass_wav = os.path.join(song_dir, BASS_WAV_FILENAME)
    all_notes = _detect_bass_notes(bass_wav) if os.path.exists(bass_wav) else []

    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    bass_track = pretty_midi.Instrument(program=33, is_drum=False, name=f"Bass ({round(tempo)})")  # 33 = Electric Bass (finger)
    bass_track.notes = sorted(all_notes, key=lambda note: note.start)
    midi.instruments.append(bass_track)
    midi.write(os.path.join(song_dir, MIDI_FILENAME))


def isolate_bass(mp3_path: str, bass_root: str) -> bool:
    """Produce bass.wav and a bass.mid for a single song under
    bass_root/<title>/. Returns False (skipped) if both already exist."""
    title = os.path.splitext(os.path.basename(mp3_path))[0]
    song_dir = os.path.join(bass_root, title)

    if os.path.isdir(song_dir) and all(os.path.exists(os.path.join(song_dir, f)) for f in _EXPECTED_OUTPUTS):
        print(f"{title}: bass already isolated, nothing to do.")
        return False

    print(f"{title}: isolating bass (this can take a few minutes)...")
    trim_ms, tempo = instrument_isolator.song_alignment(mp3_path)

    tmp_dir = tempfile.mkdtemp()
    try:
        bass_stem_dir = instrument_isolator.run_demucs(mp3_path, tmp_dir, _HTDEMUCS_MODEL, two_stems="bass")
        bass_wav = os.path.join(bass_stem_dir, "bass.wav")

        os.makedirs(song_dir, exist_ok=True)
        instrument_isolator.trim_and_export(bass_wav, trim_ms, os.path.join(song_dir, BASS_WAV_FILENAME))

        print(f"{title}: transcribing to MIDI...")
        _write_bass_midi(song_dir, tempo)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"{title}: bass.wav and bass.mid saved to {song_dir}")
    return True


def isolate_bass_for_folder(output_dir: str) -> None:
    mp3_files = sorted(f for f in os.listdir(output_dir) if f.lower().endswith(".mp3"))
    if not mp3_files:
        print("No MP3s found to isolate bass from.")
        return

    bass_root = os.path.join(output_dir, BASS_DIR_NAME)
    for filename in mp3_files:
        path = os.path.join(output_dir, filename)
        try:
            isolate_bass(path, bass_root)
        except Exception as e:
            print(f"  Could not isolate bass for {filename}, skipping: {e}")


def isolate_bass_for_single_file(path: str) -> None:
    output_dir = os.path.dirname(os.path.abspath(path)) or "."
    bass_root = os.path.join(output_dir, BASS_DIR_NAME)
    try:
        isolate_bass(path, bass_root)
    except Exception as e:
        print(f"  Could not isolate bass for {os.path.basename(path)}, skipping: {e}")


def isolate_bass_for_path(path: str) -> None:
    if os.path.isfile(path):
        isolate_bass_for_single_file(path)
    else:
        isolate_bass_for_folder(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isolate bass (bass.wav + bass.mid) from downloaded MP3s."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help=f"Folder of MP3s (or a single MP3 file) to isolate bass from (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    try:
        isolate_bass_for_path(args.path)
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(130)


if __name__ == "__main__":
    main()

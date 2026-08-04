#!/usr/bin/env python3
"""Isolate drum stems from downloaded MP3s: a full drums.wav, a first pass
at splitting it into individual kit pieces (kick/snare/toms), with cymbals
and hi-hat still bundled together as one file for now (the free local model
this uses doesn't separate those two yet), and a combined drums.mid built by
detecting hits in each isolated stem — no manual per-stem MIDI conversion
or combining needed. Any dead air or drum-less intro is trimmed off the
front first, so the stems and MIDI both start on beat 1, and the MIDI's own
tempo is detected and its notes snapped to a 16th-note grid at that tempo,
so it actually lands on the click instead of importing at the wrong BPM."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import librosa
import numpy as np
import pretty_midi
from pydub import AudioSegment

import song_sanitizer

DRUMS_DIR_NAME = "Drums"
DEFAULT_OUTPUT = os.path.join(os.path.expanduser("~"), "Downloads", "Song Downloads")

_HTDEMUCS_MODEL = "htdemucs"

# drumsep (github.com/inagoy/drumsep) is just the standard `demucs` CLI
# pointed at a custom Hybrid-Demucs checkpoint. Its install script downloads
# that checkpoint from Google Drive into a local "model" repo directory,
# which we mirror here rather than any separate inference code.
_DRUMSEP_MODEL_NAME = "49469ca8"
_DRUMSEP_GDRIVE_FILE_ID = "1-Dm666ScPkg8Gt2-lK3Ua0xOudWHZBGC"
_DRUMSEP_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drumsep_model")

_STEM_NAMES = ("drums.wav", "kick.wav", "snare.wav", "toms.wav", "cymbals.wav")

MIDI_FILENAME = "drums.mid"
_NOTE_DURATION_SEC = 0.05

# Drum map notes, matching Ableton's own default Drum Rack mapping — a
# MIDI file built from these can be dropped straight onto a stock drum rack.
# cymbals and toms are each a single note for now since those stems aren't
# split any further than one file per instrument group.
_MIDI_NOTE_MAP = {
    "kick": 36,     # Bass Drum 1
    "snare": 38,    # Acoustic Snare
    "cymbals": 40,  # combined cymbals/hi-hat stem
    "toms": 45,     # Low Tom
}

_EXPECTED_OUTPUTS = _STEM_NAMES + (MIDI_FILENAME,)


def _map_drumsep_stem_name(filename: str) -> str | None:
    """The drumsep checkpoint's own stem filenames aren't documented in
    English (its README describes them as Bombo/Redoblante/Platillos/Toms),
    so match loosely on either language instead of assuming one exact name."""
    lowered = filename.lower()
    if "kick" in lowered or "bombo" in lowered:
        return "kick"
    if "snare" in lowered or "redoblante" in lowered or "caja" in lowered:
        return "snare"
    if "tom" in lowered:
        return "toms"
    if "cymbal" in lowered or "platillo" in lowered or "hihat" in lowered or "hi-hat" in lowered or "hh" in lowered:
        return "cymbals"
    return None


def _ensure_drumsep_model() -> None:
    if os.path.isdir(_DRUMSEP_MODEL_DIR) and os.listdir(_DRUMSEP_MODEL_DIR):
        return
    print("  Downloading the kit-piece separation model (one-time)...")
    os.makedirs(_DRUMSEP_MODEL_DIR, exist_ok=True)
    try:
        subprocess.run(
            [sys.executable, "-m", "gdown", _DRUMSEP_GDRIVE_FILE_ID, "-O", _DRUMSEP_MODEL_DIR + os.sep],
            check=True,
        )
    except Exception:
        shutil.rmtree(_DRUMSEP_MODEL_DIR, ignore_errors=True)
        raise


def _run_demucs(input_path: str, out_dir: str, model_name: str, repo: str | None = None, two_stems: str | None = None) -> str:
    """Run demucs and return the directory containing the separated stems."""
    cmd = [sys.executable, "-m", "demucs", "-n", model_name, "-o", out_dir]
    if repo:
        cmd += ["--repo", repo]
    if two_stems:
        cmd += ["--two-stems", two_stems]
    cmd.append(input_path)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    track_name = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(out_dir, model_name, track_name)


_VELOCITY_WINDOW_SEC = 0.03


def _detect_note_events(wav_path: str, midi_note: int) -> list[pretty_midi.Note]:
    """Detect hit onsets in a single isolated drum stem and turn each one
    into a MIDI note. This is a much easier problem than transcribing drums
    from a full mix — the stem already isolates one instrument, so a plain
    onset detector does a solid job on timing.

    Velocity is derived from the waveform's peak amplitude just after each
    onset, normalized against the loudest hit in the stem — not from the
    onset-strength envelope's own peak, which is dominated by rare outlier
    spikes (e.g. a single unusually sharp transient) and made everything
    else round down to the minimum velocity when used directly."""
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    if onset_env.size == 0 or not np.isfinite(onset_env).any() or onset_env.max() <= 0:
        return []

    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    peak_amplitude = np.abs(y).max()
    if peak_amplitude <= 0:
        return []

    window_samples = max(1, int(_VELOCITY_WINDOW_SEC * sr))
    notes = []
    for start in onset_times:
        start_sample = int(start * sr)
        window = y[start_sample:start_sample + window_samples]
        hit_amplitude = np.abs(window).max() if window.size else 0.0
        velocity = int(np.clip(hit_amplitude / peak_amplitude * 127, 1, 127))
        notes.append(pretty_midi.Note(velocity=velocity, pitch=midi_note, start=float(start), end=float(start) + _NOTE_DURATION_SEC))
    return notes


_DEFAULT_TEMPO = 120.0
_GRID_SUBDIVISION = 16  # snap hits to the nearest 16th note


def _detect_tempo(drums_wav_path: str) -> float:
    """Estimate the song's tempo from the full isolated drums stem (a
    clearer, more periodic signal to track than any single kit piece).
    Without this, the MIDI file silently defaults to 120 BPM regardless of
    the song's real tempo, which is exactly why a straight import lands off
    the grid — the notes' absolute-time positions are correct, but nothing
    ties them to the right bars/beats until the tempo is set to match."""
    y, sr = librosa.load(drums_wav_path, sr=None, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    return tempo if tempo > 0 else _DEFAULT_TEMPO


def _quantize_time(time_sec: float, tempo: float) -> float:
    grid_sec = (60.0 / tempo) * (4.0 / _GRID_SUBDIVISION)
    return round(time_sec / grid_sec) * grid_sec


def _write_drum_midi(song_dir: str) -> None:
    """Combine hits detected across all the isolated stems into a single
    drums.mid, so there's one file to drag onto one MIDI track instead of
    converting and merging each stem by hand. The file's own tempo is set
    to the detected BPM and every note is snapped to the nearest 16th note
    at that tempo, so it lands on the grid instead of at its raw (slightly
    human/jittery) onset-detected time."""
    drums_wav = os.path.join(song_dir, "drums.wav")
    tempo = _detect_tempo(drums_wav) if os.path.exists(drums_wav) else _DEFAULT_TEMPO

    all_notes = []
    for stem_name, midi_note in _MIDI_NOTE_MAP.items():
        wav_path = os.path.join(song_dir, stem_name + ".wav")
        if os.path.exists(wav_path):
            all_notes.extend(_detect_note_events(wav_path, midi_note))

    # Snap each hit onto the grid, and drop duplicates that land on the same
    # pitch + grid slot (keeping the louder one) instead of double-triggering.
    quantized: dict[tuple[int, float], pretty_midi.Note] = {}
    for note in all_notes:
        q_start = max(0.0, _quantize_time(note.start, tempo))
        key = (note.pitch, round(q_start, 6))
        if key not in quantized or note.velocity > quantized[key].velocity:
            note.start = q_start
            note.end = q_start + _NOTE_DURATION_SEC
            quantized[key] = note

    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    drum_track = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    drum_track.notes = sorted(quantized.values(), key=lambda note: note.start)
    midi.instruments.append(drum_track)
    midi.write(os.path.join(song_dir, MIDI_FILENAME))


def _trim_leading_silence(drums_wav_path: str) -> str:
    """Cut any dead air (or a quiet, drum-less intro) off the front of the
    isolated drums stem, reusing song_sanitizer's own intro-cut detection so
    this lines up with how the song's own intro gets trimmed — otherwise the
    first real hit lands however many seconds into the file the original
    intro happened to be, instead of on beat 1. Returns the (possibly
    unchanged) path to the trimmed file, written next to the original."""
    audio = AudioSegment.from_wav(drums_wav_path)
    cut_ms = song_sanitizer._find_cut_from_start(audio, audio.dBFS)
    if not (0 < cut_ms < len(audio)):
        return drums_wav_path
    trimmed = song_sanitizer.trim(audio, cut_ms, None)
    trimmed_path = os.path.join(os.path.dirname(drums_wav_path), "drums_trimmed.wav")
    trimmed.export(trimmed_path, format="wav")
    return trimmed_path


def isolate_drums(mp3_path: str, drums_root: str) -> bool:
    """Produce drums.wav, kick.wav, snare.wav, toms.wav, cymbals.wav, and a
    combined drums.mid for a single song under drums_root/<title>/. Returns
    False (skipped) if all of those already exist."""
    title = os.path.splitext(os.path.basename(mp3_path))[0]
    song_dir = os.path.join(drums_root, title)

    if os.path.isdir(song_dir) and all(os.path.exists(os.path.join(song_dir, f)) for f in _EXPECTED_OUTPUTS):
        print(f"{title}: drum stems already exist, nothing to do.")
        return False

    print(f"{title}: isolating drums (this can take a few minutes)...")
    tmp_dir = tempfile.mkdtemp()
    try:
        drums_stem_dir = _run_demucs(mp3_path, tmp_dir, _HTDEMUCS_MODEL, two_stems="drums")
        drums_wav = os.path.join(drums_stem_dir, "drums.wav")
        drums_wav = _trim_leading_silence(drums_wav)

        os.makedirs(song_dir, exist_ok=True)
        shutil.copy(drums_wav, os.path.join(song_dir, "drums.wav"))

        print(f"{title}: splitting into kick/snare/toms/cymbals...")
        _ensure_drumsep_model()
        kit_stem_dir = _run_demucs(drums_wav, tmp_dir, _DRUMSEP_MODEL_NAME, repo=_DRUMSEP_MODEL_DIR)

        for filename in os.listdir(kit_stem_dir):
            our_name = _map_drumsep_stem_name(filename)
            if our_name:
                shutil.copy(os.path.join(kit_stem_dir, filename), os.path.join(song_dir, our_name + ".wav"))

        print(f"{title}: transcribing stems to MIDI...")
        _write_drum_midi(song_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"{title}: drum stems and MIDI saved to {song_dir}")
    return True


def isolate_drums_for_folder(output_dir: str) -> None:
    mp3_files = sorted(f for f in os.listdir(output_dir) if f.lower().endswith(".mp3"))
    if not mp3_files:
        print("No MP3s found to isolate drums from.")
        return

    drums_root = os.path.join(output_dir, DRUMS_DIR_NAME)
    for filename in mp3_files:
        path = os.path.join(output_dir, filename)
        try:
            isolate_drums(path, drums_root)
        except Exception as e:
            print(f"  Could not isolate drums for {filename}, skipping: {e}")


def isolate_drums_for_single_file(path: str) -> None:
    output_dir = os.path.dirname(os.path.abspath(path)) or "."
    drums_root = os.path.join(output_dir, DRUMS_DIR_NAME)
    try:
        isolate_drums(path, drums_root)
    except Exception as e:
        print(f"  Could not isolate drums for {os.path.basename(path)}, skipping: {e}")


def isolate_drums_for_path(path: str) -> None:
    if os.path.isfile(path):
        isolate_drums_for_single_file(path)
    else:
        isolate_drums_for_folder(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isolate drum stems (kick/snare/toms/cymbals) from downloaded MP3s."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help=f"Folder of MP3s (or a single MP3 file) to isolate drums from (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    try:
        isolate_drums_for_path(args.path)
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(130)


if __name__ == "__main__":
    main()

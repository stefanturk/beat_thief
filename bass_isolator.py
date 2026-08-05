#!/usr/bin/env python3
"""Isolate a song's bass from downloaded MP3s: an isolated bass wav (pulled
out of the song with Demucs) and a matching MIDI file built by tracking its
pitch directly. Both are trimmed and tempo-aligned to the same shared
song_alignment() as every other isolated instrument (see
instrument_isolator.py), so the bass MIDI lines up exactly with the drums
MIDI and any future instrument exports from the same song."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

import librosa
import numpy as np
import pretty_midi
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

# Bass is monophonic, so unlike drums (one fixed note per hit, guessed from
# spectral shape) each note's own pitch has to be tracked directly. Covers
# standard 4/5-string bass guitar range plus some headroom for slap/high
# notes and low synth bass.
_PITCH_FMIN_HZ = librosa.note_to_hz("E1")
_PITCH_FMAX_HZ = librosa.note_to_hz("G4")
_HOP_LENGTH = 512
_FRAME_LENGTH = 4096  # long enough to fit several periods of _PITCH_FMIN_HZ, avoiding pyin's low-frequency inaccuracy warning

_VELOCITY_WINDOW_SEC = 0.05
_MIN_NOTE_SEC = 0.12  # discard fragments too short to be a real note

# pyin's frame-to-frame pitch estimate is noisy even on a single sustained
# note (it can wobble by a semitone or two from one 12ms frame to the next),
# and a real pitch slide/bend passes through several intermediate semitones
# on its way to landing on a note. Naively starting a new note on every
# rounded-pitch change turned both of those into a flood of tiny, mostly
# duplicate notes. Smoothing the pitch curve first, then only confirming a
# pitch change once it's held for several consecutive frames, fixes both:
# single-frame jitter never lasts long enough to register, and a slide's
# transient in-between semitones get skipped in favor of wherever it settles.
# Tuned against a real isolated bass stem (Redbone) - weaker smoothing/
# confirmation left obvious jitter-driven duplicate/staircase notes in the
# output; this setting was the point where those disappeared without
# noticeably flattening the song's actual rhythm.
_SMOOTHING_WINDOW_FRAMES = 9
_PITCH_CONFIRM_SEC = 0.1  # how long a new pitch must hold before it counts as a real note change

# A genuine repeated note (the same pitch played twice in a row) needs an
# onset to split it from the note before it, since pitch alone can't tell
# the difference - but treating every onset as a re-attack was splitting
# single held notes on spurious mid-note onsets too. Requiring a minimum
# amount of time to have passed since the current note started filters
# those out without losing real fast repeated notes.
_MIN_TIME_BETWEEN_ONSET_SPLITS_SEC = 0.1
_ONSET_DELTA = 0.3  # higher than librosa's percussive-tuned default, see _detect_bass_notes

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


def _smooth_pitch_curve(midi_pitches: np.ndarray, voiced: np.ndarray, window: int) -> np.ndarray:
    """Median-filter the pitch curve over voiced frames only, so a brief
    unvoiced gap doesn't get bridged/smeared across by the window."""
    smoothed = midi_pitches.copy()
    half = window // 2
    n = len(midi_pitches)
    for i in range(n):
        if not voiced[i]:
            continue
        lo, hi = max(0, i - half), min(n, i + half + 1)
        neighborhood = midi_pitches[lo:hi][voiced[lo:hi]]
        if neighborhood.size:
            smoothed[i] = np.median(neighborhood)
    return smoothed


def _segment_notes(rounded_pitches: np.ndarray, voiced: np.ndarray, onset_frame_set: set, frame_times: np.ndarray) -> list[tuple[int, int, float]]:
    """Group per-frame pitch estimates into discrete notes, requiring a
    pitch change to hold for confirm_frames before it's accepted - see
    _detect_bass_notes for why."""
    confirm_frames = max(1, round(_PITCH_CONFIRM_SEC / (frame_times[1] - frame_times[0]))) if len(frame_times) > 1 else 1

    segments = []  # (start_frame, end_frame_exclusive, pitch)
    confirmed_pitch = None
    confirmed_start = None
    candidate_pitch = None
    candidate_start = None
    candidate_count = 0

    def close_current(end_frame):
        if confirmed_pitch is not None:
            segments.append((confirmed_start, end_frame, confirmed_pitch))

    for i in range(len(rounded_pitches)):
        if not voiced[i]:
            close_current(i)
            confirmed_pitch = None
            candidate_pitch = None
            candidate_count = 0
            continue

        pitch = rounded_pitches[i]
        is_onset = i in onset_frame_set
        re_attack = (
            confirmed_pitch is not None
            and pitch == confirmed_pitch
            and is_onset
            and frame_times[i] - frame_times[confirmed_start] >= _MIN_TIME_BETWEEN_ONSET_SPLITS_SEC
        )

        if confirmed_pitch is None:
            confirmed_pitch, confirmed_start = pitch, i
            candidate_pitch, candidate_count = None, 0
            continue

        if re_attack:
            close_current(i)
            confirmed_pitch, confirmed_start = pitch, i
            candidate_pitch, candidate_count = None, 0
            continue

        if pitch == confirmed_pitch:
            candidate_pitch, candidate_count = None, 0
            continue

        # Pitch differs from the confirmed note - track how long it holds
        # before treating it as a real change rather than transient jitter.
        if pitch == candidate_pitch:
            candidate_count += 1
        else:
            candidate_pitch, candidate_count = pitch, 1

        if candidate_count >= confirm_frames:
            change_at = i - candidate_count + 1
            close_current(change_at)
            confirmed_pitch, confirmed_start = candidate_pitch, change_at
            candidate_pitch, candidate_count = None, 0

    close_current(len(rounded_pitches))
    return segments


def _apply_noise_gate(wav_path: str) -> None:
    """Silence any part of wav_path that sits below
    _NOISE_GATE_THRESHOLD_RATIO of the file's own peak amplitude - these are
    near-noise-floor artifacts of imperfect stem separation, not real bass
    content, and left in they add spurious low-confidence pitch estimates
    for _detect_bass_notes to (mis)transcribe.

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


def _detect_bass_notes(wav_path: str) -> list[pretty_midi.Note]:
    """Track pitch through the isolated bass stem and turn it into discrete
    MIDI notes.

    librosa.pyin gives a per-frame pitch estimate (f0, in Hz) plus a voiced
    flag. That's smoothed (see _smooth_pitch_curve) and turned into a MIDI
    note number per frame, then grouped into notes (see _segment_notes): a
    new note starts whenever a pitch change holds long enough to be real,
    voicing drops to silence, or an onset lands on the same pitch after the
    current note has been playing a little while (a re-attack - otherwise a
    repeated bassline note would get glued into one long note).

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
        # A higher-than-default delta (librosa's default is tuned for
        # percussive onsets) makes the re-attack check in _segment_notes
        # much less prone to firing on noise floor/estimation jitter in a
        # sustained note - on a held tone with light broadband noise mixed
        # in (typical of an imperfectly isolated bass stem), the default
        # delta produced 20+ spurious onsets across 2 seconds of otherwise
        # completely steady pitch.
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=_HOP_LENGTH, backtrack=True, delta=_ONSET_DELTA)
    else:
        onset_frames = np.array([], dtype=int)
    onset_frame_set = set(onset_frames.tolist())

    voiced = voiced_flag & ~np.isnan(f0)
    midi_pitches = librosa.hz_to_midi(np.where(voiced, f0, 1.0))
    smoothed_pitches = _smooth_pitch_curve(midi_pitches, voiced, _SMOOTHING_WINDOW_FRAMES)
    rounded_pitches = np.round(smoothed_pitches)

    segments = _segment_notes(rounded_pitches, voiced, onset_frame_set, frame_times)

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


def _write_bass_midi(wav_path: str, midi_path: str, tempo: float) -> None:
    """Detect notes in the isolated bass wav and write them to a single
    midi_path track. Notes keep their raw pitch-tracked times (no
    quantizing/snapping) - tempo comes from the shared song_alignment()
    (see isolate_bass), not a fresh per-file estimate, so it matches every
    other instrument exported from the same song."""
    all_notes = _detect_bass_notes(wav_path) if os.path.exists(wav_path) else []

    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    bass_track = pretty_midi.Instrument(program=33, is_drum=False, name=f"Bass ({round(tempo)})")  # 33 = Electric Bass (finger)
    bass_track.notes = sorted(all_notes, key=lambda note: note.start)
    midi.instruments.append(bass_track)
    midi.write(midi_path)


def _output_basename(title: str, tempo: float) -> str:
    return f"{title} ({_LABEL} at {tempo:.3f} BPM)"


def isolate_bass(mp3_path: str, write_midi: bool = True) -> bool:
    """Produce an isolated bass wav (and, if write_midi, a matching MIDI)
    for a single song, written into its shared "<title> (Isolated)" folder
    alongside any other instrument exported from the same song. Returns
    False (skipped) if bass outputs already exist for this exact source mp3
    (see instrument_isolator.source_marker_matches) - existing outputs
    whose marker is missing or doesn't match are treated as stale (e.g. a
    leftover folder from an earlier run or a different file that happened
    to share this title) and reprocessed rather than trusted."""
    title = os.path.splitext(os.path.basename(mp3_path))[0]
    song_dir = instrument_isolator.song_output_dir(mp3_path)

    if instrument_isolator.has_existing_outputs(song_dir, _LABEL, write_midi) and instrument_isolator.source_marker_matches(
        song_dir, mp3_path, _SOURCE_MARKER_FILENAME
    ):
        print(f"{title}: bass already isolated, nothing to do.")
        return False

    print(f"{title}: isolating bass (this can take a few minutes)...")
    trim_ms, tempo = instrument_isolator.song_alignment(mp3_path)

    tmp_dir = tempfile.mkdtemp()
    try:
        bass_stem_dir = instrument_isolator.run_demucs(mp3_path, tmp_dir, _HTDEMUCS_MODEL, two_stems="bass")
        bass_wav = os.path.join(bass_stem_dir, "bass.wav")

        os.makedirs(song_dir, exist_ok=True)
        instrument_isolator.clear_stale_outputs(song_dir, _LABEL)
        basename = _output_basename(title, tempo)
        wav_path = os.path.join(song_dir, basename + ".wav")
        instrument_isolator.trim_and_export(bass_wav, trim_ms, wav_path)
        _apply_noise_gate(wav_path)

        saved = os.path.basename(wav_path)
        if write_midi:
            midi_path = os.path.join(song_dir, basename + ".mid")
            print(f"{title}: transcribing to MIDI...")
            _write_bass_midi(wav_path, midi_path, tempo)
            saved += f" and {os.path.basename(midi_path)}"

        instrument_isolator.write_source_marker(song_dir, mp3_path, _SOURCE_MARKER_FILENAME)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"{title}: {saved} saved to {song_dir}")
    return True


def isolate_bass_for_folder(output_dir: str, write_midi: bool = True) -> None:
    mp3_files = sorted(f for f in os.listdir(output_dir) if f.lower().endswith(".mp3"))
    if not mp3_files:
        print("No MP3s found to isolate bass from.")
        return

    for filename in mp3_files:
        path = os.path.join(output_dir, filename)
        try:
            isolate_bass(path, write_midi=write_midi)
        except Exception as e:
            print(f"  Could not isolate bass for {filename}, skipping: {e}")


def isolate_bass_for_single_file(path: str, write_midi: bool = True) -> None:
    try:
        isolate_bass(path, write_midi=write_midi)
    except Exception as e:
        print(f"  Could not isolate bass for {os.path.basename(path)}, skipping: {e}")


def isolate_bass_for_path(path: str, write_midi: bool = True) -> None:
    if os.path.isfile(path):
        isolate_bass_for_single_file(path, write_midi=write_midi)
    else:
        isolate_bass_for_folder(path, write_midi=write_midi)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isolate bass (wav, optionally + MIDI) from downloaded MP3s."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help=f"Folder of MP3s (or a single MP3 file) to isolate bass from (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--midi",
        action="store_true",
        help="Also write a MIDI file, not just the isolated wav. Can also be given as a bare 'midi' argument.",
    )

    raw_args = sys.argv[1:]
    bare_midi = any(tok.lower() == "midi" for tok in raw_args)
    remaining_args = [tok for tok in raw_args if tok.lower() != "midi"]
    args = parser.parse_args(remaining_args)
    args.midi = args.midi or bare_midi

    try:
        isolate_bass_for_path(args.path, write_midi=args.midi)
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(130)


if __name__ == "__main__":
    main()

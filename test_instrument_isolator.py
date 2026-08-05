import os
import shutil
import tempfile
import unittest

import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine

import instrument_isolator


def _click_track(bpm, beats, hit_ms=30, freq=1000):
    """A steady metronome-like click at an exact tempo, for exercising tempo
    detection/refinement against a known-correct answer."""
    interval_ms = 60000.0 / bpm
    gap_ms = interval_ms - hit_ms
    beat_audio = Sine(freq).to_audio_segment(duration=hit_ms).fade_out(hit_ms // 2)
    track = AudioSegment.silent(duration=0)
    for _ in range(beats):
        track += beat_audio + AudioSegment.silent(duration=int(gap_ms))
    return track


class TestDetectTempo(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_detects_tempo_of_a_steady_click_track(self):
        wav_path = os.path.join(self.tmp_dir, "click.wav")
        _click_track(bpm=120, beats=24).export(wav_path, format="wav")

        tempo = instrument_isolator.detect_tempo(wav_path)

        # Octave errors (reading half or double the real tempo) are a known
        # limitation of automatic tempo detection, so accept any of those
        # instead of requiring an exact match to 120.
        ratio = tempo / 120.0
        self.assertTrue(any(abs(ratio - r) < 0.05 for r in (0.5, 1.0, 2.0)), tempo)


class TestRefineTempo(unittest.TestCase):
    def test_recovers_precise_tempo_from_noisy_onsets(self):
        true_bpm = 137.73
        period = 60.0 / true_bpm
        rng = np.random.default_rng(0)
        n_hits = 300
        onset_times = np.arange(n_hits) * period
        onset_times += rng.normal(scale=0.004, size=n_hits)  # ~4ms onset-detection jitter

        # Deliberately start from a rough/wrong initial estimate, matching
        # how detect_tempo's single windowed guess is only approximate.
        refined = instrument_isolator.refine_tempo(onset_times, initial_tempo=140.0)

        self.assertAlmostEqual(refined, true_bpm, delta=0.1)

    def test_too_few_onsets_returns_the_initial_estimate_unchanged(self):
        onset_times = np.array([0.0, 0.5, 1.0])

        refined = instrument_isolator.refine_tempo(onset_times, initial_tempo=95.0)

        self.assertEqual(refined, 95.0)


class TestSongAlignment(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_trims_a_silent_intro_and_detects_tempo(self):
        silent_intro = AudioSegment.silent(duration=3000)
        beat = _click_track(bpm=120, beats=24)
        track = silent_intro + beat
        mp3_path = os.path.join(self.tmp_dir, "song.mp3")
        track.export(mp3_path, format="mp3")

        trim_ms, tempo = instrument_isolator.song_alignment(mp3_path)

        # The ~3s silent intro should be detected and trimmed.
        self.assertGreater(trim_ms, 1500)
        self.assertLess(trim_ms, len(track))
        ratio = tempo / 120.0
        self.assertTrue(any(abs(ratio - r) < 0.05 for r in (0.5, 1.0, 2.0)), tempo)

    def test_no_intro_to_trim_returns_zero(self):
        beat = _click_track(bpm=120, beats=24)
        mp3_path = os.path.join(self.tmp_dir, "song.mp3")
        beat.export(mp3_path, format="mp3")

        trim_ms, _ = instrument_isolator.song_alignment(mp3_path)

        self.assertEqual(trim_ms, 0)


class TestTrimAndExport(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_slices_the_given_amount_off_the_front(self):
        track = Sine(200).to_audio_segment(duration=3000)
        wav_path = os.path.join(self.tmp_dir, "in.wav")
        track.export(wav_path, format="wav")
        out_path = os.path.join(self.tmp_dir, "out.wav")

        instrument_isolator.trim_and_export(wav_path, 1000, out_path)

        trimmed = AudioSegment.from_wav(out_path)
        self.assertAlmostEqual(len(trimmed), 2000, delta=50)

    def test_zero_trim_leaves_length_unchanged(self):
        track = Sine(200).to_audio_segment(duration=3000)
        wav_path = os.path.join(self.tmp_dir, "in.wav")
        track.export(wav_path, format="wav")
        out_path = os.path.join(self.tmp_dir, "out.wav")

        instrument_isolator.trim_and_export(wav_path, 0, out_path)

        untrimmed = AudioSegment.from_wav(out_path)
        self.assertAlmostEqual(len(untrimmed), 3000, delta=50)


class TestVelocitiesFromAmplitudes(unittest.TestCase):
    def test_loudest_amplitude_reaches_127(self):
        velocities = instrument_isolator.velocities_from_amplitudes([0.1, 0.5, 1.0, 0.2])

        self.assertEqual(max(velocities), 127)
        self.assertEqual(velocities[2], 127)

    def test_relative_ordering_is_preserved(self):
        velocities = instrument_isolator.velocities_from_amplitudes([0.1, 0.5, 1.0, 0.2])

        self.assertLess(velocities[0], velocities[3])
        self.assertLess(velocities[3], velocities[1])
        self.assertLess(velocities[1], velocities[2])

    def test_empty_list_returns_empty(self):
        self.assertEqual(instrument_isolator.velocities_from_amplitudes([]), [])

    def test_all_zero_amplitudes_returns_minimum_velocity(self):
        self.assertEqual(instrument_isolator.velocities_from_amplitudes([0.0, 0.0]), [1, 1])


if __name__ == "__main__":
    unittest.main()

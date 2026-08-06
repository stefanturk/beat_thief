import os
import shutil
import tempfile
import unittest
from unittest import mock

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


def _onsets_from_audio(y, sr):
    onset_env = instrument_isolator.librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = instrument_isolator.librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=True)
    return instrument_isolator.librosa.frames_to_time(onset_frames, sr=sr)


class TestWindowedTempos(unittest.TestCase):
    def test_splits_a_tempo_change_into_separate_windows(self):
        track = _click_track(bpm=100, beats=50) + _click_track(bpm=150, beats=70)
        y = np.array(track.set_channels(1).get_array_of_samples()).astype(np.float32) / 32768.0
        sr = track.frame_rate
        onset_times = _onsets_from_audio(y, sr)
        song_duration_sec = len(track) / 1000.0

        windows = instrument_isolator._windowed_tempos(y, sr, onset_times, song_duration_sec, window_sec=30.0)

        self.assertEqual(len(windows), 2)
        self.assertAlmostEqual(windows[0][2], 100.0, delta=2.0)
        self.assertAlmostEqual(windows[1][2], 150.0, delta=2.0)

    def test_a_window_with_too_few_onsets_falls_back_to_a_rough_estimate(self):
        track = _click_track(bpm=95, beats=3)  # too few hits to refine
        y = np.array(track.set_channels(1).get_array_of_samples()).astype(np.float32) / 32768.0
        sr = track.frame_rate
        onset_times = _onsets_from_audio(y, sr)
        song_duration_sec = len(track) / 1000.0

        windows = instrument_isolator._windowed_tempos(y, sr, onset_times, song_duration_sec, window_sec=30.0)

        self.assertEqual(len(windows), 1)
        self.assertGreater(windows[0][2], 0.0)

    def test_reference_tempo_reconciles_a_window_caught_at_a_different_subdivision(self):
        # Regression test: a real, constant-tempo song was reported as
        # "drifting" from ~108 to ~144 BPM (a 4/3 ratio) purely because a
        # later window's own beat tracker locked onto a different
        # subdivision of the same beat, not because the song's tempo
        # actually changed.
        track = _click_track(bpm=108, beats=54) + _click_track(bpm=144, beats=72)
        y = np.array(track.set_channels(1).get_array_of_samples()).astype(np.float32) / 32768.0
        sr = track.frame_rate
        onset_times = _onsets_from_audio(y, sr)
        song_duration_sec = len(track) / 1000.0

        windows = instrument_isolator._windowed_tempos(
            y, sr, onset_times, song_duration_sec, window_sec=30.0, reference_tempo=108.0
        )

        for _, _, tempo in windows:
            self.assertAlmostEqual(tempo, 108.0, delta=3.0)
        self.assertFalse(instrument_isolator._tempo_drift_detected(windows))


class TestSnapTempoToWholeNumberIfClose(unittest.TestCase):
    def test_a_tempo_close_to_a_whole_number_is_rounded(self):
        self.assertEqual(instrument_isolator._snap_tempo_to_whole_number_if_close(108.038), 108.0)

    def test_a_tempo_further_from_a_whole_number_is_left_alone(self):
        self.assertEqual(instrument_isolator._snap_tempo_to_whole_number_if_close(108.138), 108.138)


class TestSnapIfNoisyAroundAWholeNumber(unittest.TestCase):
    def test_windows_scattered_tightly_around_a_whole_number_are_snapped(self):
        # Regression test: a real, essentially-constant ~157 BPM song
        # triggered the tempo-drift prompt purely from ordinary per-window
        # jitter (max-min was under 1 BPM, comfortably above the 0.3 BPM
        # drift threshold but not a real tempo change).
        windows = [
            (0, 30, 157.600), (30, 60, 156.935), (60, 90, 156.875), (90, 120, 156.845),
            (120, 150, 156.741), (150, 180, 157.037), (180, 210, 157.143), (210, 240, 156.862),
            (240, 270, 157.143), (270, 300, 156.955), (300, 319.1, 157.155),
        ]

        self.assertEqual(instrument_isolator._snap_if_noisy_around_a_whole_number(windows), 157.0)

    def test_a_genuine_two_part_tempo_change_is_not_snapped(self):
        windows = [(0.0, 30.0, 100.0), (30.0, 60.0, 150.0)]

        self.assertIsNone(instrument_isolator._snap_if_noisy_around_a_whole_number(windows))

    def test_a_stable_non_whole_tempo_is_not_snapped(self):
        windows = [(0.0, 30.0, 137.6), (30.0, 60.0, 137.7), (60.0, 90.0, 137.65)]

        self.assertIsNone(instrument_isolator._snap_if_noisy_around_a_whole_number(windows))


class TestReconcileWithReference(unittest.TestCase):
    def test_a_4_3_ratio_is_reconciled_onto_the_reference(self):
        self.assertAlmostEqual(instrument_isolator._reconcile_with_reference(144.0, 108.0), 108.0, delta=0.5)

    def test_an_octave_error_is_reconciled_onto_the_reference(self):
        self.assertAlmostEqual(instrument_isolator._reconcile_with_reference(240.0, 120.0), 120.0, delta=0.5)

    def test_a_3_2_ratio_is_left_unchanged_since_it_can_be_a_genuine_tempo_change(self):
        self.assertEqual(instrument_isolator._reconcile_with_reference(150.0, 100.0), 150.0)

    def test_an_unrelated_tempo_is_left_unchanged(self):
        self.assertEqual(instrument_isolator._reconcile_with_reference(95.0, 120.0), 95.0)


class TestTempoDriftDetected(unittest.TestCase):
    def test_flags_windows_that_differ_beyond_the_threshold(self):
        windows = [(0.0, 30.0, 100.0), (30.0, 60.0, 100.5)]
        self.assertTrue(instrument_isolator._tempo_drift_detected(windows))

    def test_does_not_flag_windows_within_the_threshold(self):
        windows = [(0.0, 30.0, 100.0), (30.0, 60.0, 100.1)]
        self.assertFalse(instrument_isolator._tempo_drift_detected(windows))

    def test_a_single_window_is_never_flagged(self):
        windows = [(0.0, 30.0, 100.0)]
        self.assertFalse(instrument_isolator._tempo_drift_detected(windows))


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

    @mock.patch("builtins.input", return_value="2")
    @mock.patch("sys.stdin")
    def test_tempo_drift_prompts_and_uses_the_chosen_window(self, mock_stdin, mock_input):
        mock_stdin.isatty.return_value = True
        track = _click_track(bpm=100, beats=50) + _click_track(bpm=150, beats=70)
        mp3_path = os.path.join(self.tmp_dir, "song.mp3")
        track.export(mp3_path, format="mp3")

        _, tempo = instrument_isolator.song_alignment(mp3_path)

        mock_input.assert_called_once()
        self.assertAlmostEqual(tempo, 150.0, delta=2.0)

    def test_tempo_drift_defaults_to_the_first_window_when_not_interactive(self):
        track = _click_track(bpm=100, beats=50) + _click_track(bpm=150, beats=70)
        mp3_path = os.path.join(self.tmp_dir, "song.mp3")
        track.export(mp3_path, format="mp3")

        with mock.patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            _, tempo = instrument_isolator.song_alignment(mp3_path)

        self.assertAlmostEqual(tempo, 100.0, delta=2.0)

    @mock.patch("builtins.input")
    def test_interactive_false_never_prompts_even_from_a_real_terminal(self, mock_input):
        # The GUI passes interactive=False explicitly rather than relying on
        # isatty, so the answer can't depend on how the .app inherits stdin.
        track = _click_track(bpm=100, beats=50) + _click_track(bpm=150, beats=70)
        mp3_path = os.path.join(self.tmp_dir, "song.mp3")
        track.export(mp3_path, format="mp3")

        with mock.patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            _, tempo = instrument_isolator.song_alignment(mp3_path, interactive=False)

        mock_input.assert_not_called()
        self.assertAlmostEqual(tempo, 100.0, delta=2.0)


class TestRunDemucsProgress(unittest.TestCase):
    """run_demucs parses demucs' own output for a percentage; where that
    percentage goes is what these cover."""

    def _run_with_output(self, raw_output, on_percent=None):
        fake_proc = mock.MagicMock()
        chunks = [raw_output.encode(), b""]
        fake_proc.stdout.read.side_effect = chunks
        fake_proc.returncode = 0
        with mock.patch("subprocess.Popen", return_value=fake_proc):
            with mock.patch("sys.stdout.write") as mock_write:
                stem_dir = instrument_isolator.run_demucs(
                    "/tmp/Some Song.mp3", "/tmp/out", "htdemucs", on_percent=on_percent
                )
        return stem_dir, mock_write

    def test_percentages_go_to_the_callback_and_not_to_stdout(self):
        percents = []
        _, mock_write = self._run_with_output(
            " 12%|##   |\r 58%|#####  |\r100%|######|\r", on_percent=percents.append
        )

        self.assertEqual(percents, [12, 58, 100])
        mock_write.assert_not_called()

    def test_without_a_callback_it_still_draws_the_terminal_bar(self):
        _, mock_write = self._run_with_output(" 58%|#####  |\r")

        written = "".join(call.args[0] for call in mock_write.call_args_list)
        self.assertIn("58%", written)

    def test_returns_the_stem_directory_for_the_track(self):
        stem_dir, _ = self._run_with_output(" 100%|#|\r", on_percent=lambda p: None)

        self.assertEqual(stem_dir, os.path.join("/tmp/out", "htdemucs", "Some Song"))


class TestSeparatedStems(unittest.TestCase):
    """One demucs pass per song, shared by every instrument that wants it -
    demucs computes all four sources whichever one you asked for."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.mp3_path = os.path.join(self.tmp_dir, "Song - Artist.mp3")
        with open(self.mp3_path, "wb") as f:
            f.write(b"fake mp3 bytes")

    def tearDown(self):
        instrument_isolator.clear_stem_cache()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _fake_run_demucs(self, input_path, out_dir, model_name, two_stems=None, on_percent=None, should_cancel=None):
        stem_dir = os.path.join(out_dir, model_name, "Song - Artist")
        os.makedirs(stem_dir, exist_ok=True)
        return stem_dir

    def test_asking_twice_for_the_same_song_separates_it_once(self):
        with mock.patch("instrument_isolator.run_demucs", side_effect=self._fake_run_demucs) as mock_demucs:
            first = instrument_isolator.separated_stems(self.mp3_path, "htdemucs")
            second = instrument_isolator.separated_stems(self.mp3_path, "htdemucs")

        self.assertEqual(first, second)
        mock_demucs.assert_called_once()

    def test_it_asks_for_every_stem_not_just_two(self):
        with mock.patch("instrument_isolator.run_demucs", side_effect=self._fake_run_demucs) as mock_demucs:
            instrument_isolator.separated_stems(self.mp3_path, "htdemucs")

        self.assertNotIn("two_stems", mock_demucs.call_args.kwargs)

    def test_a_changed_song_is_separated_again(self):
        # A re-download or a sanitizer pass rewrites the mp3; the previous
        # separation no longer describes it.
        with mock.patch("instrument_isolator.run_demucs", side_effect=self._fake_run_demucs) as mock_demucs:
            instrument_isolator.separated_stems(self.mp3_path, "htdemucs")
            os.utime(self.mp3_path, (0, 0))
            instrument_isolator.separated_stems(self.mp3_path, "htdemucs")

        self.assertEqual(mock_demucs.call_count, 2)

    def test_a_cached_pass_still_reports_progress(self):
        # An instant stage would otherwise leave a window's bar stuck at
        # whatever the last real percentage was.
        seen = []
        context = instrument_isolator.RunContext(on_percent=seen.append)
        with mock.patch("instrument_isolator.run_demucs", side_effect=self._fake_run_demucs):
            instrument_isolator.separated_stems(self.mp3_path, "htdemucs", context)
            seen.clear()
            instrument_isolator.separated_stems(self.mp3_path, "htdemucs", context)

        self.assertEqual(seen, [100])

    def test_clearing_the_cache_removes_the_separated_audio(self):
        with mock.patch("instrument_isolator.run_demucs", side_effect=self._fake_run_demucs):
            stem_dir = instrument_isolator.separated_stems(self.mp3_path, "htdemucs")
        self.assertTrue(os.path.isdir(stem_dir))

        instrument_isolator.clear_stem_cache()

        self.assertFalse(os.path.exists(stem_dir))

    def test_moving_on_to_the_next_song_frees_the_previous_one(self):
        # A separated song is hundreds of megabytes; a folder run would fill
        # a disk if every song's stayed around.
        other_mp3 = os.path.join(self.tmp_dir, "Other - Artist.mp3")
        with open(other_mp3, "wb") as f:
            f.write(b"another song")

        with mock.patch("instrument_isolator.run_demucs", side_effect=self._fake_run_demucs):
            first = instrument_isolator.separated_stems(self.mp3_path, "htdemucs")
            second = instrument_isolator.separated_stems(other_mp3, "htdemucs")

        self.assertFalse(os.path.exists(first))
        self.assertTrue(os.path.isdir(second))

    def test_clearing_an_empty_cache_is_fine(self):
        instrument_isolator.clear_stem_cache()  # nothing separated yet

    def test_a_cancelled_pass_leaves_nothing_behind_and_nothing_cached(self):
        def cancel_immediately(*args, **kwargs):
            raise instrument_isolator.Cancelled()

        with mock.patch("instrument_isolator.run_demucs", side_effect=cancel_immediately):
            with self.assertRaises(instrument_isolator.Cancelled):
                instrument_isolator.separated_stems(self.mp3_path, "htdemucs")

        self.assertEqual(instrument_isolator._stem_cache, {})
        self.assertEqual(instrument_isolator._stem_temp_dirs, [])


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

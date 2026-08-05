import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine

import drum_isolator


def _hits(count, hit_ms=80, gap_ms=400, freq=200):
    """A short percussive-ish hit repeated `count` times with silence
    between, standing in for a real isolated drum stem."""
    silence = AudioSegment.silent(duration=gap_ms)
    track = AudioSegment.silent(duration=200)
    for _ in range(count):
        hit = Sine(freq).to_audio_segment(duration=hit_ms).fade_out(hit_ms // 2)
        track += hit + silence
    return track


class TestIsolateDrums(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.drums_root = os.path.join(self.tmp_dir, "Drums")
        self.mp3_path = os.path.join(self.tmp_dir, "Song - Artist.mp3")
        with open(self.mp3_path, "wb") as f:
            f.write(b"fake mp3 bytes")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _fake_run_demucs(self, input_path, out_dir, model_name, two_stems=None):
        # Mimic demucs's own output layout well enough for isolate_drums to
        # find what it's looking for, without actually running any model.
        track_name = os.path.splitext(os.path.basename(input_path))[0]
        stem_dir = os.path.join(out_dir, model_name, track_name)
        os.makedirs(stem_dir, exist_ok=True)
        with open(os.path.join(stem_dir, "drums.wav"), "wb") as f:
            f.write(b"drums")
        return stem_dir

    def _fake_trim_and_export(self, wav_path, trim_ms, out_path):
        shutil.copy(wav_path, out_path)

    def _fake_write_drum_midi(self, song_dir, tempo):
        # Real onset detection needs real audio; these tests only care about
        # file orchestration, so stand in with an empty placeholder file.
        with open(os.path.join(song_dir, drum_isolator.MIDI_FILENAME), "wb") as f:
            f.write(b"midi")

    @mock.patch("drum_isolator._write_drum_midi")
    @mock.patch("instrument_isolator.trim_and_export")
    @mock.patch("instrument_isolator.run_demucs")
    @mock.patch("instrument_isolator.song_alignment")
    def test_produces_drums_wav_and_midi(self, mock_alignment, mock_run_demucs, mock_trim, mock_write_midi):
        mock_alignment.return_value = (0, 120.0)
        mock_run_demucs.side_effect = self._fake_run_demucs
        mock_trim.side_effect = self._fake_trim_and_export
        mock_write_midi.side_effect = self._fake_write_drum_midi

        result = drum_isolator.isolate_drums(self.mp3_path, self.drums_root)

        self.assertTrue(result)
        song_dir = os.path.join(self.drums_root, "Song - Artist")
        for name in drum_isolator._EXPECTED_OUTPUTS:
            self.assertTrue(os.path.exists(os.path.join(song_dir, name)), name)
        mock_write_midi.assert_called_once_with(song_dir, 120.0)

    @mock.patch("instrument_isolator.run_demucs")
    def test_skips_when_all_outputs_already_exist(self, mock_run_demucs):
        song_dir = os.path.join(self.drums_root, "Song - Artist")
        os.makedirs(song_dir)
        for name in drum_isolator._EXPECTED_OUTPUTS:
            with open(os.path.join(song_dir, name), "wb") as f:
                f.write(b"x")

        result = drum_isolator.isolate_drums(self.mp3_path, self.drums_root)

        self.assertFalse(result)
        mock_run_demucs.assert_not_called()

    @mock.patch("drum_isolator._write_drum_midi")
    @mock.patch("instrument_isolator.trim_and_export")
    @mock.patch("instrument_isolator.run_demucs")
    @mock.patch("instrument_isolator.song_alignment")
    def test_reruns_when_the_midi_file_is_missing(self, mock_alignment, mock_run_demucs, mock_trim, mock_write_midi):
        song_dir = os.path.join(self.drums_root, "Song - Artist")
        os.makedirs(song_dir)
        with open(os.path.join(song_dir, drum_isolator.DRUMS_WAV_FILENAME), "wb") as f:
            f.write(b"x")
        # drums.mid missing

        mock_alignment.return_value = (0, 120.0)
        mock_run_demucs.side_effect = self._fake_run_demucs
        mock_trim.side_effect = self._fake_trim_and_export
        mock_write_midi.side_effect = self._fake_write_drum_midi

        result = drum_isolator.isolate_drums(self.mp3_path, self.drums_root)

        self.assertTrue(result)
        for name in drum_isolator._EXPECTED_OUTPUTS:
            self.assertTrue(os.path.exists(os.path.join(song_dir, name)), name)


class TestDetectNoteEvents(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_detects_roughly_one_note_per_hit(self):
        wav_path = os.path.join(self.tmp_dir, "drums.wav")
        _hits(4).export(wav_path, format="wav")

        notes = drum_isolator._detect_note_events(wav_path)

        # Onset detection on synthetic tones won't be pixel-perfect, but it
        # should land in the ballpark of the four hits we actually made.
        self.assertGreaterEqual(len(notes), 2)
        self.assertLessEqual(len(notes), 6)
        for note in notes:
            self.assertIn(note.pitch, (36, 38, 42))
            self.assertTrue(1 <= note.velocity <= 127)

    def test_silence_produces_no_notes(self):
        wav_path = os.path.join(self.tmp_dir, "silence.wav")
        AudioSegment.silent(duration=2000).export(wav_path, format="wav")

        notes = drum_isolator._detect_note_events(wav_path)

        self.assertEqual(notes, [])

    def test_loudest_hit_reaches_full_velocity(self):
        wav_path = os.path.join(self.tmp_dir, "drums.wav")
        _hits(4).export(wav_path, format="wav")

        notes = drum_isolator._detect_note_events(wav_path)

        self.assertTrue(any(note.velocity == 127 for note in notes))


class TestHitCentroid(unittest.TestCase):
    def _window(self, freq, sr=44100, duration_sec=0.03):
        t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
        return np.sin(2 * np.pi * freq * t).astype(np.float32), sr

    def test_low_frequency_tone_has_low_centroid(self):
        window, sr = self._window(100)
        low_centroid = drum_isolator._hit_centroid(window, sr)

        window, sr = self._window(6000)
        high_centroid = drum_isolator._hit_centroid(window, sr)

        self.assertLess(low_centroid, high_centroid)

    def test_empty_window_returns_none(self):
        self.assertIsNone(drum_isolator._hit_centroid(np.array([]), 44100))

    def test_silent_window_returns_none(self):
        self.assertIsNone(drum_isolator._hit_centroid(np.zeros(100), 44100))


class TestNoteForCentroid(unittest.TestCase):
    def test_below_kick_threshold_is_kick(self):
        self.assertEqual(drum_isolator._note_for_centroid(100.0, kick_threshold=300.0, snare_threshold=2000.0), drum_isolator._KICK_NOTE)

    def test_between_thresholds_is_snare(self):
        self.assertEqual(drum_isolator._note_for_centroid(1000.0, kick_threshold=300.0, snare_threshold=2000.0), drum_isolator._SNARE_NOTE)

    def test_above_snare_threshold_is_cymbal(self):
        self.assertEqual(drum_isolator._note_for_centroid(5000.0, kick_threshold=300.0, snare_threshold=2000.0), drum_isolator._CYMBAL_NOTE)

    def test_none_defaults_to_kick(self):
        self.assertEqual(drum_isolator._note_for_centroid(None, kick_threshold=300.0, snare_threshold=2000.0), drum_isolator._KICK_NOTE)


class TestWriteDrumMidi(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_writes_notes_for_every_detected_hit(self):
        _hits(5).export(os.path.join(self.tmp_dir, "drums.wav"), format="wav")

        drum_isolator._write_drum_midi(self.tmp_dir, tempo=120.0)

        midi_path = os.path.join(self.tmp_dir, drum_isolator.MIDI_FILENAME)
        self.assertTrue(os.path.exists(midi_path))

        import pretty_midi
        midi = pretty_midi.PrettyMIDI(midi_path)
        self.assertEqual(len(midi.instruments), 1)
        self.assertTrue(midi.instruments[0].is_drum)
        pitches = {note.pitch for note in midi.instruments[0].notes}
        self.assertTrue(pitches.issubset({36, 38, 42}))
        self.assertGreater(len(midi.instruments[0].notes), 0)
        # Notes should be written in start-time order.
        starts = [note.start for note in midi.instruments[0].notes]
        self.assertEqual(starts, sorted(starts))

    def test_missing_drums_wav_produces_an_empty_midi_file(self):
        drum_isolator._write_drum_midi(self.tmp_dir, tempo=120.0)

        import pretty_midi
        # An instrument with zero notes isn't written back out by pretty_midi
        # on round-trip, so the file legitimately has no instruments at all.
        midi = pretty_midi.PrettyMIDI(os.path.join(self.tmp_dir, drum_isolator.MIDI_FILENAME))
        notes = [note for instrument in midi.instruments for note in instrument.notes]
        self.assertEqual(notes, [])

    def test_embeds_the_given_tempo_without_snapping_notes(self):
        _hits(6).export(os.path.join(self.tmp_dir, "drums.wav"), format="wav")

        drum_isolator._write_drum_midi(self.tmp_dir, tempo=123.4)

        import pretty_midi
        midi = pretty_midi.PrettyMIDI(os.path.join(self.tmp_dir, drum_isolator.MIDI_FILENAME))
        _, tempi = midi.get_tempo_changes()
        # write_drum_midi should embed exactly the tempo it's given, not
        # detect/refine its own - that's now song_alignment()'s job, shared
        # across every instrument isolated from the same song.
        self.assertAlmostEqual(tempi[0], 123.4, delta=0.1)

        # Notes should keep their raw onset-detected times untouched - not
        # snapped to any musical grid - matching what _detect_note_events
        # alone would produce on the same file. A small tolerance accounts
        # for MIDI's own tick resolution: writing/reading a .mid file always
        # rounds absolute times to the nearest tick (a couple of ms here),
        # regardless of any snapping logic - that's the file format, not us.
        expected_starts = sorted(n.start for n in drum_isolator._detect_note_events(
            os.path.join(self.tmp_dir, "drums.wav")))
        actual_starts = sorted(n.start for n in midi.instruments[0].notes)
        self.assertEqual(len(actual_starts), len(expected_starts))
        for actual, expected in zip(actual_starts, expected_starts):
            self.assertAlmostEqual(actual, expected, delta=0.01)


class TestIsolateDrumsForFolder(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_no_mp3s_prints_message_and_does_nothing(self):
        with mock.patch("drum_isolator.isolate_drums") as mock_isolate:
            drum_isolator.isolate_drums_for_folder(self.tmp_dir)
        mock_isolate.assert_not_called()

    @mock.patch("drum_isolator.isolate_drums")
    def test_processes_each_mp3_in_folder(self, mock_isolate):
        for name in ("A - Artist.mp3", "B - Artist.mp3"):
            with open(os.path.join(self.tmp_dir, name), "wb") as f:
                f.write(b"x")

        drum_isolator.isolate_drums_for_folder(self.tmp_dir)

        self.assertEqual(mock_isolate.call_count, 2)
        called_paths = {call.args[0] for call in mock_isolate.call_args_list}
        self.assertEqual(
            called_paths,
            {os.path.join(self.tmp_dir, "A - Artist.mp3"), os.path.join(self.tmp_dir, "B - Artist.mp3")},
        )

    @mock.patch("drum_isolator.isolate_drums")
    def test_one_failure_does_not_stop_the_rest(self, mock_isolate):
        for name in ("A - Artist.mp3", "B - Artist.mp3"):
            with open(os.path.join(self.tmp_dir, name), "wb") as f:
                f.write(b"x")
        mock_isolate.side_effect = [RuntimeError("boom"), True]

        drum_isolator.isolate_drums_for_folder(self.tmp_dir)

        self.assertEqual(mock_isolate.call_count, 2)


class TestIsolateDrumsForPath(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @mock.patch("drum_isolator.isolate_drums")
    def test_single_file_dispatches_with_sibling_drums_folder(self, mock_isolate):
        mp3_path = os.path.join(self.tmp_dir, "Song - Artist.mp3")
        with open(mp3_path, "wb") as f:
            f.write(b"x")

        drum_isolator.isolate_drums_for_path(mp3_path)

        mock_isolate.assert_called_once_with(mp3_path, os.path.join(self.tmp_dir, drum_isolator.DRUMS_DIR_NAME))

    @mock.patch("drum_isolator.isolate_drums_for_folder")
    def test_folder_dispatches_to_folder_handler(self, mock_folder):
        drum_isolator.isolate_drums_for_path(self.tmp_dir)
        mock_folder.assert_called_once_with(self.tmp_dir)


if __name__ == "__main__":
    unittest.main()

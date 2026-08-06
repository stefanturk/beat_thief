import os
import shutil
import tempfile
import unittest
from unittest import mock

from pydub import AudioSegment
from pydub.generators import Sine

import harmony_isolator
import instrument_isolator


class TestIsolateHarmony(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.mp3_path = os.path.join(self.tmp_dir, "Song - Artist.mp3")
        with open(self.mp3_path, "wb") as f:
            f.write(b"fake mp3 bytes")
        self.song_dir = os.path.join(self.tmp_dir, "Song - Artist (Isolated)")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _fake_run_demucs(self, input_path, out_dir, model_name, two_stems=None, on_percent=None, should_cancel=None):
        self.assertIsNone(two_stems)  # harmony needs a full separation, not two-stems
        track_name = os.path.splitext(os.path.basename(input_path))[0]
        stem_dir = os.path.join(out_dir, model_name, track_name)
        os.makedirs(stem_dir, exist_ok=True)
        Sine(220).to_audio_segment(duration=200).export(os.path.join(stem_dir, "other.wav"), format="wav")
        Sine(440).to_audio_segment(duration=200).export(os.path.join(stem_dir, "vocals.wav"), format="wav")
        # drums.wav / bass.wav also exist in a real run but harmony ignores them.
        return stem_dir

    def _fake_trim_and_export(self, wav_path, trim_ms, out_path):
        shutil.copy(wav_path, out_path)

    @mock.patch("instrument_isolator.trim_and_export")
    @mock.patch("instrument_isolator.run_demucs")
    @mock.patch("instrument_isolator.song_alignment")
    def test_produces_harmony_wav(self, mock_alignment, mock_run_demucs, mock_trim):
        mock_alignment.return_value = (0, 120.0)
        mock_run_demucs.side_effect = self._fake_run_demucs
        mock_trim.side_effect = self._fake_trim_and_export

        result = harmony_isolator.isolate_harmony(self.mp3_path)

        self.assertTrue(result)
        basename = "Song - Artist (Isolated Harmony)"
        wav_path = os.path.join(self.song_dir, basename + ".wav")
        self.assertTrue(os.path.exists(wav_path))
        self.assertTrue(os.path.exists(os.path.join(self.song_dir, harmony_isolator._SOURCE_MARKER_FILENAME)))

    @mock.patch("instrument_isolator.run_demucs")
    def test_skips_when_outputs_already_exist_and_match_the_source_mp3(self, mock_run_demucs):
        os.makedirs(self.song_dir)
        basename = "Song - Artist (Isolated Harmony)"
        with open(os.path.join(self.song_dir, basename + ".wav"), "wb") as f:
            f.write(b"x")
        instrument_isolator.write_source_marker(self.song_dir, self.mp3_path, harmony_isolator._SOURCE_MARKER_FILENAME)

        result = harmony_isolator.isolate_harmony(self.mp3_path)

        self.assertFalse(result)
        mock_run_demucs.assert_not_called()

    @mock.patch("instrument_isolator.trim_and_export")
    @mock.patch("instrument_isolator.run_demucs")
    @mock.patch("instrument_isolator.song_alignment")
    def test_reprocesses_when_the_marker_is_missing(self, mock_alignment, mock_run_demucs, mock_trim):
        os.makedirs(self.song_dir)
        basename = "Song - Artist (Isolated Harmony)"
        with open(os.path.join(self.song_dir, basename + ".wav"), "wb") as f:
            f.write(b"x")
        # no marker - so this wav can't be trusted as coming from self.mp3_path.

        mock_alignment.return_value = (0, 120.0)
        mock_run_demucs.side_effect = self._fake_run_demucs
        mock_trim.side_effect = self._fake_trim_and_export

        result = harmony_isolator.isolate_harmony(self.mp3_path)

        self.assertTrue(result)
        mock_run_demucs.assert_called_once()

    @mock.patch("instrument_isolator.trim_and_export")
    @mock.patch("instrument_isolator.run_demucs")
    @mock.patch("instrument_isolator.song_alignment")
    def test_reprocesses_when_the_marker_does_not_match(self, mock_alignment, mock_run_demucs, mock_trim):
        os.makedirs(self.song_dir)
        basename = "Song - Artist (Isolated Harmony)"
        with open(os.path.join(self.song_dir, basename + ".wav"), "wb") as f:
            f.write(b"x")
        stale_mp3 = os.path.join(self.tmp_dir, "stale.mp3")
        with open(stale_mp3, "wb") as f:
            f.write(b"different bytes")
        instrument_isolator.write_source_marker(self.song_dir, stale_mp3, harmony_isolator._SOURCE_MARKER_FILENAME)

        mock_alignment.return_value = (0, 120.0)
        mock_run_demucs.side_effect = self._fake_run_demucs
        mock_trim.side_effect = self._fake_trim_and_export

        result = harmony_isolator.isolate_harmony(self.mp3_path)

        self.assertTrue(result)
        mock_run_demucs.assert_called_once()


class TestMixHarmony(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_mixes_both_stems_into_one_wav(self):
        other_wav = os.path.join(self.tmp_dir, "other.wav")
        vocals_wav = os.path.join(self.tmp_dir, "vocals.wav")
        out_path = os.path.join(self.tmp_dir, "harmony.wav")
        Sine(220).to_audio_segment(duration=500).export(other_wav, format="wav")
        Sine(440).to_audio_segment(duration=500).export(vocals_wav, format="wav")

        harmony_isolator._mix_harmony(other_wav, vocals_wav, out_path)

        self.assertTrue(os.path.exists(out_path))
        mixed = AudioSegment.from_wav(out_path)
        self.assertGreater(mixed.rms, 0)
        self.assertAlmostEqual(len(mixed), 500, delta=50)


class TestIsolateHarmonyForFolder(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_no_mp3s_prints_message_and_does_nothing(self):
        with mock.patch("harmony_isolator.isolate_harmony") as mock_isolate:
            harmony_isolator.isolate_harmony_for_folder(self.tmp_dir)
        mock_isolate.assert_not_called()

    @mock.patch("harmony_isolator.isolate_harmony")
    def test_processes_each_mp3_in_folder(self, mock_isolate):
        for name in ("A - Artist.mp3", "B - Artist.mp3"):
            with open(os.path.join(self.tmp_dir, name), "wb") as f:
                f.write(b"x")

        harmony_isolator.isolate_harmony_for_folder(self.tmp_dir)

        self.assertEqual(mock_isolate.call_count, 2)
        called_paths = {call.args[0] for call in mock_isolate.call_args_list}
        self.assertEqual(
            called_paths,
            {os.path.join(self.tmp_dir, "A - Artist.mp3"), os.path.join(self.tmp_dir, "B - Artist.mp3")},
        )

    @mock.patch("harmony_isolator.isolate_harmony")
    def test_one_failure_does_not_stop_the_rest(self, mock_isolate):
        for name in ("A - Artist.mp3", "B - Artist.mp3"):
            with open(os.path.join(self.tmp_dir, name), "wb") as f:
                f.write(b"x")
        mock_isolate.side_effect = [RuntimeError("boom"), True]

        harmony_isolator.isolate_harmony_for_folder(self.tmp_dir)

        self.assertEqual(mock_isolate.call_count, 2)


class TestIsolateHarmonyForPath(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @mock.patch("harmony_isolator.isolate_harmony")
    def test_single_file_dispatches_directly(self, mock_isolate):
        mp3_path = os.path.join(self.tmp_dir, "Song - Artist.mp3")
        with open(mp3_path, "wb") as f:
            f.write(b"x")

        harmony_isolator.isolate_harmony_for_path(mp3_path)

        mock_isolate.assert_called_once_with(mp3_path, context=None)

    @mock.patch("harmony_isolator.isolate_harmony_for_folder")
    def test_folder_dispatches_to_folder_handler(self, mock_folder):
        harmony_isolator.isolate_harmony_for_path(self.tmp_dir)
        mock_folder.assert_called_once_with(self.tmp_dir, context=None)


if __name__ == "__main__":
    unittest.main()

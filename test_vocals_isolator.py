import os
import shutil
import tempfile
import unittest
from unittest import mock

from pydub.generators import Sine

import instrument_isolator
import vocals_isolator


class TestIsolateVocals(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.mp3_path = os.path.join(self.tmp_dir, "Song - Artist.mp3")
        with open(self.mp3_path, "wb") as f:
            f.write(b"fake mp3 bytes")
        # Everything for a song lives in the mp3's own folder now.
        self.song_dir = self.tmp_dir

    def tearDown(self):
        instrument_isolator.clear_stem_cache()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _fake_run_demucs(self, input_path, out_dir, model_name, two_stems=None, on_percent=None, should_cancel=None):
        self.assertIsNone(two_stems)  # one full separation, shared by every instrument
        track_name = os.path.splitext(os.path.basename(input_path))[0]
        stem_dir = os.path.join(out_dir, model_name, track_name)
        os.makedirs(stem_dir, exist_ok=True)
        for name, freq in (("vocals", 440), ("other", 220), ("drums", 110), ("bass", 55)):
            Sine(freq).to_audio_segment(duration=200).export(os.path.join(stem_dir, name + ".wav"), format="wav")
        return stem_dir

    def _fake_trim_and_export(self, wav_path, trim_ms, out_path):
        shutil.copy(wav_path, out_path)

    @mock.patch("instrument_isolator.trim_and_export")
    @mock.patch("instrument_isolator.run_demucs")
    @mock.patch("instrument_isolator.song_alignment")
    def test_produces_a_vocals_wav_in_the_songs_shared_folder(self, mock_alignment, mock_run_demucs, mock_trim):
        mock_alignment.return_value = (0, 120.0)
        mock_run_demucs.side_effect = self._fake_run_demucs
        mock_trim.side_effect = self._fake_trim_and_export

        result = vocals_isolator.isolate_vocals(self.mp3_path)

        self.assertTrue(result)
        wav_path = os.path.join(self.song_dir, "Song - Artist (Isolated Vocals).wav")
        self.assertTrue(os.path.exists(wav_path))
        self.assertTrue(os.path.exists(os.path.join(self.song_dir, vocals_isolator._SOURCE_MARKER_FILENAME)))

    @mock.patch("instrument_isolator.trim_and_export")
    @mock.patch("instrument_isolator.run_demucs")
    @mock.patch("instrument_isolator.song_alignment")
    def test_exports_the_vocals_stem_and_not_another_one(self, mock_alignment, mock_run_demucs, mock_trim):
        mock_alignment.return_value = (0, 120.0)
        mock_run_demucs.side_effect = self._fake_run_demucs
        mock_trim.side_effect = self._fake_trim_and_export

        vocals_isolator.isolate_vocals(self.mp3_path)

        self.assertEqual(os.path.basename(mock_trim.call_args.args[0]), "vocals.wav")

    @mock.patch("instrument_isolator.trim_and_export")
    @mock.patch("instrument_isolator.run_demucs")
    @mock.patch("instrument_isolator.song_alignment")
    def test_no_midi_is_written_alongside_it(self, mock_alignment, mock_run_demucs, mock_trim):
        # There's no single line to transcribe out of a sung phrase, so
        # unlike drums/bass the filename carries no BPM either.
        mock_alignment.return_value = (0, 174.0)
        mock_run_demucs.side_effect = self._fake_run_demucs
        mock_trim.side_effect = self._fake_trim_and_export

        vocals_isolator.isolate_vocals(self.mp3_path)

        produced = os.listdir(self.song_dir)
        self.assertFalse([f for f in produced if f.endswith(".mid")])
        self.assertIn("Song - Artist (Isolated Vocals).wav", produced)

    @mock.patch("instrument_isolator.run_demucs")
    def test_skips_when_outputs_already_exist_and_match_the_source_mp3(self, mock_run_demucs):
        with open(os.path.join(self.song_dir, "Song - Artist (Isolated Vocals).wav"), "wb") as f:
            f.write(b"x")
        instrument_isolator.write_source_marker(self.song_dir, self.mp3_path, vocals_isolator._SOURCE_MARKER_FILENAME)

        result = vocals_isolator.isolate_vocals(self.mp3_path)

        self.assertFalse(result)
        mock_run_demucs.assert_not_called()

    @mock.patch("instrument_isolator.trim_and_export")
    @mock.patch("instrument_isolator.run_demucs")
    @mock.patch("instrument_isolator.song_alignment")
    def test_reprocesses_when_the_marker_is_missing(self, mock_alignment, mock_run_demucs, mock_trim):
        with open(os.path.join(self.song_dir, "Song - Artist (Isolated Vocals).wav"), "wb") as f:
            f.write(b"x")
        # no marker - so this wav can't be trusted as coming from self.mp3_path.

        mock_alignment.return_value = (0, 120.0)
        mock_run_demucs.side_effect = self._fake_run_demucs
        mock_trim.side_effect = self._fake_trim_and_export

        result = vocals_isolator.isolate_vocals(self.mp3_path)

        self.assertTrue(result)
        mock_run_demucs.assert_called_once()

    @mock.patch("instrument_isolator.trim_and_export")
    @mock.patch("instrument_isolator.run_demucs")
    @mock.patch("instrument_isolator.song_alignment")
    def test_reprocesses_when_the_marker_does_not_match(self, mock_alignment, mock_run_demucs, mock_trim):
        with open(os.path.join(self.song_dir, "Song - Artist (Isolated Vocals).wav"), "wb") as f:
            f.write(b"x")
        stale_mp3 = os.path.join(self.tmp_dir, "stale.mp3")
        with open(stale_mp3, "wb") as f:
            f.write(b"different bytes")
        instrument_isolator.write_source_marker(self.song_dir, stale_mp3, vocals_isolator._SOURCE_MARKER_FILENAME)

        mock_alignment.return_value = (0, 120.0)
        mock_run_demucs.side_effect = self._fake_run_demucs
        mock_trim.side_effect = self._fake_trim_and_export

        result = vocals_isolator.isolate_vocals(self.mp3_path)

        self.assertTrue(result)
        mock_run_demucs.assert_called_once()

    @mock.patch("instrument_isolator.trim_and_export")
    @mock.patch("instrument_isolator.run_demucs")
    @mock.patch("instrument_isolator.song_alignment")
    def test_the_run_context_reaches_demucs_and_the_tempo_step(self, mock_alignment, mock_run_demucs, mock_trim):
        mock_alignment.return_value = (0, 120.0)
        mock_run_demucs.side_effect = self._fake_run_demucs
        mock_trim.side_effect = self._fake_trim_and_export

        seen = []

        def note_percent(percent):
            seen.append(percent)

        def never_cancel():
            return False

        context = instrument_isolator.RunContext(
            on_percent=note_percent, interactive=False, should_cancel=never_cancel
        )
        vocals_isolator.isolate_vocals(self.mp3_path, context=context)

        mock_alignment.assert_called_once_with(self.mp3_path, interactive=False)
        self.assertIs(mock_run_demucs.call_args.kwargs["should_cancel"], never_cancel)


class TestIsolateVocalsForFolder(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_no_mp3s_does_nothing(self):
        with mock.patch("vocals_isolator.isolate_vocals") as mock_isolate:
            vocals_isolator.isolate_vocals_for_folder(self.tmp_dir)
        mock_isolate.assert_not_called()

    @mock.patch("vocals_isolator.isolate_vocals")
    def test_processes_each_mp3_in_folder(self, mock_isolate):
        for name in ("A - Artist.mp3", "B - Artist.mp3"):
            with open(os.path.join(self.tmp_dir, name), "wb") as f:
                f.write(b"x")

        vocals_isolator.isolate_vocals_for_folder(self.tmp_dir)

        self.assertEqual(mock_isolate.call_count, 2)

    @mock.patch("vocals_isolator.isolate_vocals")
    def test_one_failure_does_not_stop_the_rest(self, mock_isolate):
        for name in ("A - Artist.mp3", "B - Artist.mp3"):
            with open(os.path.join(self.tmp_dir, name), "wb") as f:
                f.write(b"x")
        mock_isolate.side_effect = [RuntimeError("boom"), True]

        vocals_isolator.isolate_vocals_for_folder(self.tmp_dir)

        self.assertEqual(mock_isolate.call_count, 2)


class TestIsolateVocalsForPath(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @mock.patch("vocals_isolator.isolate_vocals")
    def test_single_file_dispatches_directly(self, mock_isolate):
        mp3_path = os.path.join(self.tmp_dir, "Song - Artist.mp3")
        with open(mp3_path, "wb") as f:
            f.write(b"x")

        vocals_isolator.isolate_vocals_for_path(mp3_path)

        mock_isolate.assert_called_once_with(mp3_path, context=None)

    @mock.patch("vocals_isolator.isolate_vocals_for_folder")
    def test_folder_dispatches_to_folder_handler(self, mock_folder):
        vocals_isolator.isolate_vocals_for_path(self.tmp_dir)
        mock_folder.assert_called_once_with(self.tmp_dir, context=None)


if __name__ == "__main__":
    unittest.main()

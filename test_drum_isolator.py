import os
import shutil
import tempfile
import unittest
from unittest import mock

import drum_isolator


class TestMapDrumsepStemName(unittest.TestCase):
    def test_maps_english_names(self):
        self.assertEqual(drum_isolator._map_drumsep_stem_name("kick.wav"), "kick")
        self.assertEqual(drum_isolator._map_drumsep_stem_name("snare.wav"), "snare")
        self.assertEqual(drum_isolator._map_drumsep_stem_name("toms.wav"), "toms")
        self.assertEqual(drum_isolator._map_drumsep_stem_name("cymbals.wav"), "cymbals_hihat")
        self.assertEqual(drum_isolator._map_drumsep_stem_name("hihat.wav"), "cymbals_hihat")

    def test_maps_spanish_checkpoint_names(self):
        self.assertEqual(drum_isolator._map_drumsep_stem_name("bombo.wav"), "kick")
        self.assertEqual(drum_isolator._map_drumsep_stem_name("redoblante.wav"), "snare")
        self.assertEqual(drum_isolator._map_drumsep_stem_name("caja.wav"), "snare")
        self.assertEqual(drum_isolator._map_drumsep_stem_name("platillos.wav"), "cymbals_hihat")

    def test_case_insensitive(self):
        self.assertEqual(drum_isolator._map_drumsep_stem_name("KICK.WAV"), "kick")

    def test_unrecognized_name_returns_none(self):
        self.assertIsNone(drum_isolator._map_drumsep_stem_name("mystery.wav"))


class TestIsolateDrums(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.drums_root = os.path.join(self.tmp_dir, "Drums")
        self.mp3_path = os.path.join(self.tmp_dir, "Song - Artist.mp3")
        with open(self.mp3_path, "wb") as f:
            f.write(b"fake mp3 bytes")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _fake_run_demucs(self, input_path, out_dir, model_name, repo=None, two_stems=None):
        # Mimic demucs's own output layout well enough for isolate_drums to
        # find what it's looking for, without actually running any model.
        track_name = os.path.splitext(os.path.basename(input_path))[0]
        stem_dir = os.path.join(out_dir, model_name, track_name)
        os.makedirs(stem_dir, exist_ok=True)
        if two_stems == "drums":
            with open(os.path.join(stem_dir, "drums.wav"), "wb") as f:
                f.write(b"drums")
        else:
            for name in ("bombo.wav", "redoblante.wav", "toms.wav", "platillos.wav"):
                with open(os.path.join(stem_dir, name), "wb") as f:
                    f.write(name.encode())
        return stem_dir

    @mock.patch("drum_isolator._ensure_drumsep_model")
    @mock.patch("drum_isolator._run_demucs")
    def test_produces_all_five_stems(self, mock_run_demucs, mock_ensure_model):
        mock_run_demucs.side_effect = self._fake_run_demucs

        result = drum_isolator.isolate_drums(self.mp3_path, self.drums_root)

        self.assertTrue(result)
        song_dir = os.path.join(self.drums_root, "Song - Artist")
        for name in drum_isolator._STEM_NAMES:
            self.assertTrue(os.path.exists(os.path.join(song_dir, name)), name)
        mock_ensure_model.assert_called_once()

    @mock.patch("drum_isolator._ensure_drumsep_model")
    @mock.patch("drum_isolator._run_demucs")
    def test_skips_when_all_stems_already_exist(self, mock_run_demucs, mock_ensure_model):
        song_dir = os.path.join(self.drums_root, "Song - Artist")
        os.makedirs(song_dir)
        for name in drum_isolator._STEM_NAMES:
            with open(os.path.join(song_dir, name), "wb") as f:
                f.write(b"x")

        result = drum_isolator.isolate_drums(self.mp3_path, self.drums_root)

        self.assertFalse(result)
        mock_run_demucs.assert_not_called()
        mock_ensure_model.assert_not_called()

    @mock.patch("drum_isolator._ensure_drumsep_model")
    @mock.patch("drum_isolator._run_demucs")
    def test_reruns_when_a_stem_is_missing(self, mock_run_demucs, mock_ensure_model):
        song_dir = os.path.join(self.drums_root, "Song - Artist")
        os.makedirs(song_dir)
        with open(os.path.join(song_dir, "drums.wav"), "wb") as f:
            f.write(b"x")
        # kick/snare/toms/cymbals_hihat missing

        mock_run_demucs.side_effect = self._fake_run_demucs

        result = drum_isolator.isolate_drums(self.mp3_path, self.drums_root)

        self.assertTrue(result)
        for name in drum_isolator._STEM_NAMES:
            self.assertTrue(os.path.exists(os.path.join(song_dir, name)), name)


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

import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
from scipy.io import wavfile

import instrument_isolator
import instrumental_isolator


class TestIsolateInstrumental(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.mp3_path = os.path.join(self.tmp_dir, "Song - Artist.mp3")
        with open(self.mp3_path, "wb") as f:
            f.write(b"fake mp3 bytes")
        self.song_dir = self.tmp_dir
        self.stems = {
            "drums": np.full((100, 2), 1000, dtype=np.int16),
            "bass": np.full((100, 2), 200, dtype=np.int16),
            "other": np.full((100, 2), 30, dtype=np.int16),
            "vocals": np.full((100, 2), 4, dtype=np.int16),
        }

    def tearDown(self):
        instrument_isolator.clear_stem_cache()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _fake_run_demucs(self, input_path, out_dir, model_name, two_stems=None, on_percent=None, should_cancel=None):
        self.assertIsNone(two_stems)  # one full separation, shared by every instrument
        stem_dir = os.path.join(out_dir, model_name, "Song - Artist")
        os.makedirs(stem_dir, exist_ok=True)
        for name, samples in self.stems.items():
            wavfile.write(os.path.join(stem_dir, name + ".wav"), 44100, samples)
        return stem_dir

    def _fake_trim_and_export(self, wav_path, trim_ms, out_path):
        shutil.copy(wav_path, out_path)

    def _isolate(self):
        with mock.patch("instrument_isolator.song_alignment", return_value=(0, 120.0)), \
             mock.patch("instrument_isolator.run_demucs", side_effect=self._fake_run_demucs) as demucs, \
             mock.patch("instrument_isolator.trim_and_export", side_effect=self._fake_trim_and_export):
            result = instrumental_isolator.isolate_instrumental(self.mp3_path)
        return result, demucs

    def test_the_instrumental_is_drums_bass_and_harmony_added_together(self):
        result, _ = self._isolate()

        self.assertTrue(result)
        _rate, samples = wavfile.read(os.path.join(self.song_dir, "Song - Artist (Isolated Instrumental).wav"))
        self.assertTrue((samples == 1230).all())  # no vocals (4) in it

    def test_only_the_instrumental_lands_in_the_songs_folder(self):
        self._isolate()

        wavs = [f for f in os.listdir(self.song_dir) if f.endswith(".wav")]
        self.assertEqual(wavs, ["Song - Artist (Isolated Instrumental).wav"])

    def test_a_loud_sum_is_held_at_full_scale_rather_than_wrapping_round(self):
        self.stems["drums"] = np.full((100, 2), 30000, dtype=np.int16)
        self.stems["bass"] = np.full((100, 2), 30000, dtype=np.int16)

        self._isolate()

        _rate, samples = wavfile.read(os.path.join(self.song_dir, "Song - Artist (Isolated Instrumental).wav"))
        self.assertTrue((samples == 32767).all())

    def test_one_separation_serves_the_vocals_and_the_instrumental(self):
        import vocals_isolator
        with mock.patch("instrument_isolator.song_alignment", return_value=(0, 120.0)), \
             mock.patch("instrument_isolator.run_demucs", side_effect=self._fake_run_demucs) as demucs, \
             mock.patch("instrument_isolator.trim_and_export", side_effect=self._fake_trim_and_export):
            vocals_isolator.isolate_vocals(self.mp3_path)
            instrumental_isolator.isolate_instrumental(self.mp3_path)

        demucs.assert_called_once()

    def test_skips_when_it_is_already_there_for_this_mp3(self):
        with open(os.path.join(self.song_dir, "Song - Artist (Isolated Instrumental).wav"), "wb") as f:
            f.write(b"x")
        instrument_isolator.write_source_marker(self.song_dir, self.mp3_path, instrumental_isolator._SOURCE_MARKER_FILENAME)

        result, demucs = self._isolate()

        self.assertFalse(result)
        demucs.assert_not_called()


if __name__ == "__main__":
    unittest.main()

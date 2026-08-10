import os
import shutil
import tempfile
import unittest
from unittest import mock

import pretty_midi
from pydub import AudioSegment
from pydub.generators import Sine

import drum_isolator
import drum_transcriber  # noqa: F401 - imported so mock.patch can find it by name
import instrument_isolator


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
        self.mp3_path = os.path.join(self.tmp_dir, "Song - Artist.mp3")
        with open(self.mp3_path, "wb") as f:
            f.write(b"fake mp3 bytes")
        self.song_dir = os.path.join(self.tmp_dir, "Song - Artist (Isolated)")

    def tearDown(self):
        instrument_isolator.clear_stem_cache()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _fake_run_demucs(self, input_path, out_dir, model_name, two_stems=None, on_percent=None, should_cancel=None):
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

    @mock.patch("instrument_isolator.trim_and_export")
    @mock.patch("instrument_isolator.run_demucs")
    @mock.patch("instrument_isolator.song_alignment")
    def test_run_context_reaches_demucs_and_alignment(self, mock_alignment, mock_run_demucs, mock_trim):
        # The GUI's progress callback and non-interactive choice are only
        # useful if they survive the trip down to where the work happens.
        mock_alignment.return_value = (0, 120.0)
        mock_run_demucs.side_effect = self._fake_run_demucs
        mock_trim.side_effect = self._fake_trim_and_export
        sink = []

        def note_percent(percent):
            sink.append(percent)

        context = instrument_isolator.RunContext(on_percent=note_percent, interactive=False)

        drum_isolator.isolate_drums(self.mp3_path, context=context)

        self.assertIs(mock_run_demucs.call_args.kwargs["on_percent"], note_percent)
        self.assertIs(mock_alignment.call_args.kwargs["interactive"], False)

    @mock.patch("instrument_isolator.trim_and_export")
    @mock.patch("instrument_isolator.run_demucs")
    @mock.patch("instrument_isolator.song_alignment")
    def test_produces_a_drums_wav(self, mock_alignment, mock_run_demucs, mock_trim):
        mock_alignment.return_value = (0, 120.0)
        mock_run_demucs.side_effect = self._fake_run_demucs
        mock_trim.side_effect = self._fake_trim_and_export

        result = drum_isolator.isolate_drums(self.mp3_path)

        self.assertTrue(result)
        basename = "Song - Artist (Isolated Drums at 120.000 BPM)"
        self.assertTrue(os.path.exists(os.path.join(self.song_dir, basename + ".wav")))
        self.assertTrue(os.path.exists(os.path.join(self.song_dir, drum_isolator._SOURCE_MARKER_FILENAME)))

    @mock.patch("instrument_isolator.run_demucs")
    def test_skips_when_outputs_already_exist_and_match_the_source_mp3(self, mock_run_demucs):
        os.makedirs(self.song_dir)
        basename = "Song - Artist (Isolated Drums at 120.000 BPM)"
        with open(os.path.join(self.song_dir, basename + ".wav"), "wb") as f:
            f.write(b"x")
        instrument_isolator.write_source_marker(self.song_dir, self.mp3_path, drum_isolator._SOURCE_MARKER_FILENAME)

        result = drum_isolator.isolate_drums(self.mp3_path)

        self.assertFalse(result)
        mock_run_demucs.assert_not_called()

    @mock.patch("instrument_isolator.trim_and_export")
    @mock.patch("instrument_isolator.run_demucs")
    @mock.patch("instrument_isolator.song_alignment")
    def test_reruns_when_a_wav_is_there_but_no_marker_confirms_it(self, mock_alignment, mock_run_demucs, mock_trim):
        os.makedirs(self.song_dir)
        basename = "Song - Artist (Isolated Drums at 120.000 BPM)"
        with open(os.path.join(self.song_dir, basename + ".wav"), "wb") as f:
            f.write(b"x")
        # No marker, so this wav can't be trusted as coming from
        # self.mp3_path and is reprocessed rather than accepted.

        mock_alignment.return_value = (0, 120.0)
        mock_run_demucs.side_effect = self._fake_run_demucs
        mock_trim.side_effect = self._fake_trim_and_export

        result = drum_isolator.isolate_drums(self.mp3_path)

        self.assertTrue(result)
        mock_run_demucs.assert_called_once()
        self.assertTrue(os.path.exists(os.path.join(self.song_dir, basename + ".wav")))

    @mock.patch("instrument_isolator.trim_and_export")
    @mock.patch("instrument_isolator.run_demucs")
    @mock.patch("instrument_isolator.song_alignment")
    def test_reruns_and_replaces_stale_outputs_when_the_source_mp3_marker_does_not_match(
        self, mock_alignment, mock_run_demucs, mock_trim
    ):
        os.makedirs(self.song_dir)
        stale = "Song - Artist (Isolated Drums at 90.000 BPM)"
        with open(os.path.join(self.song_dir, stale + ".wav"), "wb") as f:
            f.write(b"x")
        instrument_isolator.write_source_marker(
            self.song_dir, self.mp3_path, drum_isolator._SOURCE_MARKER_FILENAME
        )
        with open(self.mp3_path, "wb") as f:
            f.write(b"different bytes entirely")

        mock_alignment.return_value = (0, 120.0)
        mock_run_demucs.side_effect = self._fake_run_demucs
        mock_trim.side_effect = self._fake_trim_and_export

        result = drum_isolator.isolate_drums(self.mp3_path)

        self.assertTrue(result)
        self.assertFalse(os.path.exists(os.path.join(self.song_dir, stale + ".wav")))
        fresh = "Song - Artist (Isolated Drums at 120.000 BPM)"
        self.assertTrue(os.path.exists(os.path.join(self.song_dir, fresh + ".wav")))


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
    def test_single_file_dispatches_directly(self, mock_isolate):
        mp3_path = os.path.join(self.tmp_dir, "Song - Artist.mp3")
        with open(mp3_path, "wb") as f:
            f.write(b"x")

        drum_isolator.isolate_drums_for_path(mp3_path)

        mock_isolate.assert_called_once_with(mp3_path, context=None)

    @mock.patch("drum_isolator.isolate_drums_for_folder")
    def test_folder_dispatches_to_folder_handler(self, mock_folder):
        drum_isolator.isolate_drums_for_path(self.tmp_dir)
        mock_folder.assert_called_once_with(self.tmp_dir, context=None)


if __name__ == "__main__":
    unittest.main()

import os
import shutil
import tempfile
import unittest
from unittest import mock

import librosa
from pydub import AudioSegment
from pydub.generators import Sine

import bass_isolator


def _notes(note_names, note_ms=600, gap_ms=200):
    """A short sequence of held, distinctly-pitched tones with silence
    between them, standing in for a real isolated bass stem."""
    track = AudioSegment.silent(duration=100)
    for note_name in note_names:
        freq = librosa.note_to_hz(note_name)
        tone = Sine(freq).to_audio_segment(duration=note_ms).fade_in(10).fade_out(30)
        track += tone + AudioSegment.silent(duration=gap_ms)
    return track


class TestIsolateBass(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.bass_root = os.path.join(self.tmp_dir, "Bass")
        self.mp3_path = os.path.join(self.tmp_dir, "Song - Artist.mp3")
        with open(self.mp3_path, "wb") as f:
            f.write(b"fake mp3 bytes")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _fake_run_demucs(self, input_path, out_dir, model_name, two_stems=None):
        track_name = os.path.splitext(os.path.basename(input_path))[0]
        stem_dir = os.path.join(out_dir, model_name, track_name)
        os.makedirs(stem_dir, exist_ok=True)
        with open(os.path.join(stem_dir, "bass.wav"), "wb") as f:
            f.write(b"bass")
        return stem_dir

    def _fake_trim_and_export(self, wav_path, trim_ms, out_path):
        shutil.copy(wav_path, out_path)

    def _fake_write_bass_midi(self, song_dir, tempo):
        with open(os.path.join(song_dir, bass_isolator.MIDI_FILENAME), "wb") as f:
            f.write(b"midi")

    @mock.patch("bass_isolator._write_bass_midi")
    @mock.patch("instrument_isolator.trim_and_export")
    @mock.patch("instrument_isolator.run_demucs")
    @mock.patch("instrument_isolator.song_alignment")
    def test_produces_bass_wav_and_midi(self, mock_alignment, mock_run_demucs, mock_trim, mock_write_midi):
        mock_alignment.return_value = (0, 120.0)
        mock_run_demucs.side_effect = self._fake_run_demucs
        mock_trim.side_effect = self._fake_trim_and_export
        mock_write_midi.side_effect = self._fake_write_bass_midi

        result = bass_isolator.isolate_bass(self.mp3_path, self.bass_root)

        self.assertTrue(result)
        song_dir = os.path.join(self.bass_root, "Song - Artist")
        for name in bass_isolator._EXPECTED_OUTPUTS:
            self.assertTrue(os.path.exists(os.path.join(song_dir, name)), name)
        mock_write_midi.assert_called_once_with(song_dir, 120.0)

    @mock.patch("instrument_isolator.run_demucs")
    def test_skips_when_all_outputs_already_exist(self, mock_run_demucs):
        song_dir = os.path.join(self.bass_root, "Song - Artist")
        os.makedirs(song_dir)
        for name in bass_isolator._EXPECTED_OUTPUTS:
            with open(os.path.join(song_dir, name), "wb") as f:
                f.write(b"x")

        result = bass_isolator.isolate_bass(self.mp3_path, self.bass_root)

        self.assertFalse(result)
        mock_run_demucs.assert_not_called()

    @mock.patch("bass_isolator._write_bass_midi")
    @mock.patch("instrument_isolator.trim_and_export")
    @mock.patch("instrument_isolator.run_demucs")
    @mock.patch("instrument_isolator.song_alignment")
    def test_reruns_when_the_midi_file_is_missing(self, mock_alignment, mock_run_demucs, mock_trim, mock_write_midi):
        song_dir = os.path.join(self.bass_root, "Song - Artist")
        os.makedirs(song_dir)
        with open(os.path.join(song_dir, bass_isolator.BASS_WAV_FILENAME), "wb") as f:
            f.write(b"x")
        # bass.mid missing

        mock_alignment.return_value = (0, 120.0)
        mock_run_demucs.side_effect = self._fake_run_demucs
        mock_trim.side_effect = self._fake_trim_and_export
        mock_write_midi.side_effect = self._fake_write_bass_midi

        result = bass_isolator.isolate_bass(self.mp3_path, self.bass_root)

        self.assertTrue(result)
        for name in bass_isolator._EXPECTED_OUTPUTS:
            self.assertTrue(os.path.exists(os.path.join(song_dir, name)), name)


class TestDetectBassNotes(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_detects_roughly_one_note_per_played_note(self):
        note_names = ["E1", "A1", "D2", "G2"]
        wav_path = os.path.join(self.tmp_dir, "bass.wav")
        _notes(note_names).export(wav_path, format="wav")

        notes = bass_isolator._detect_bass_notes(wav_path)

        self.assertGreaterEqual(len(notes), 2)
        self.assertLessEqual(len(notes), 8)
        for note in notes:
            self.assertTrue(1 <= note.velocity <= 127)
            self.assertGreater(note.end, note.start)

        # Pitches detected should land close to at least some of the notes
        # we actually played (pitch tracking on synthetic tones won't be
        # pixel-perfect, but should be in the right ballpark).
        expected_pitches = {round(librosa.note_to_midi(n)) for n in note_names}
        detected_pitches = {note.pitch for note in notes}
        self.assertTrue(any(abs(p - e) <= 1 for p in detected_pitches for e in expected_pitches))

    def test_silence_produces_no_notes(self):
        wav_path = os.path.join(self.tmp_dir, "silence.wav")
        AudioSegment.silent(duration=2000).export(wav_path, format="wav")

        notes = bass_isolator._detect_bass_notes(wav_path)

        self.assertEqual(notes, [])

    def test_loudest_note_reaches_full_velocity(self):
        wav_path = os.path.join(self.tmp_dir, "bass.wav")
        _notes(["E1", "A1", "D2"]).export(wav_path, format="wav")

        notes = bass_isolator._detect_bass_notes(wav_path)

        self.assertTrue(any(note.velocity == 127 for note in notes))


class TestWriteBassMidi(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_writes_notes_for_detected_pitches(self):
        _notes(["E1", "A1", "D2"]).export(os.path.join(self.tmp_dir, "bass.wav"), format="wav")

        bass_isolator._write_bass_midi(self.tmp_dir, tempo=120.0)

        midi_path = os.path.join(self.tmp_dir, bass_isolator.MIDI_FILENAME)
        self.assertTrue(os.path.exists(midi_path))

        import pretty_midi
        midi = pretty_midi.PrettyMIDI(midi_path)
        self.assertEqual(len(midi.instruments), 1)
        self.assertFalse(midi.instruments[0].is_drum)
        self.assertGreater(len(midi.instruments[0].notes), 0)
        starts = [note.start for note in midi.instruments[0].notes]
        self.assertEqual(starts, sorted(starts))

    def test_missing_bass_wav_produces_an_empty_midi_file(self):
        bass_isolator._write_bass_midi(self.tmp_dir, tempo=120.0)

        import pretty_midi
        midi = pretty_midi.PrettyMIDI(os.path.join(self.tmp_dir, bass_isolator.MIDI_FILENAME))
        notes = [note for instrument in midi.instruments for note in instrument.notes]
        self.assertEqual(notes, [])

    def test_embeds_the_given_tempo(self):
        _notes(["E1", "A1"]).export(os.path.join(self.tmp_dir, "bass.wav"), format="wav")

        bass_isolator._write_bass_midi(self.tmp_dir, tempo=123.4)

        import pretty_midi
        midi = pretty_midi.PrettyMIDI(os.path.join(self.tmp_dir, bass_isolator.MIDI_FILENAME))
        _, tempi = midi.get_tempo_changes()
        self.assertAlmostEqual(tempi[0], 123.4, delta=0.1)


class TestIsolateBassForFolder(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_no_mp3s_prints_message_and_does_nothing(self):
        with mock.patch("bass_isolator.isolate_bass") as mock_isolate:
            bass_isolator.isolate_bass_for_folder(self.tmp_dir)
        mock_isolate.assert_not_called()

    @mock.patch("bass_isolator.isolate_bass")
    def test_processes_each_mp3_in_folder(self, mock_isolate):
        for name in ("A - Artist.mp3", "B - Artist.mp3"):
            with open(os.path.join(self.tmp_dir, name), "wb") as f:
                f.write(b"x")

        bass_isolator.isolate_bass_for_folder(self.tmp_dir)

        self.assertEqual(mock_isolate.call_count, 2)
        called_paths = {call.args[0] for call in mock_isolate.call_args_list}
        self.assertEqual(
            called_paths,
            {os.path.join(self.tmp_dir, "A - Artist.mp3"), os.path.join(self.tmp_dir, "B - Artist.mp3")},
        )

    @mock.patch("bass_isolator.isolate_bass")
    def test_one_failure_does_not_stop_the_rest(self, mock_isolate):
        for name in ("A - Artist.mp3", "B - Artist.mp3"):
            with open(os.path.join(self.tmp_dir, name), "wb") as f:
                f.write(b"x")
        mock_isolate.side_effect = [RuntimeError("boom"), True]

        bass_isolator.isolate_bass_for_folder(self.tmp_dir)

        self.assertEqual(mock_isolate.call_count, 2)


class TestIsolateBassForPath(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @mock.patch("bass_isolator.isolate_bass")
    def test_single_file_dispatches_with_sibling_bass_folder(self, mock_isolate):
        mp3_path = os.path.join(self.tmp_dir, "Song - Artist.mp3")
        with open(mp3_path, "wb") as f:
            f.write(b"x")

        bass_isolator.isolate_bass_for_path(mp3_path)

        mock_isolate.assert_called_once_with(mp3_path, os.path.join(self.tmp_dir, bass_isolator.BASS_DIR_NAME))

    @mock.patch("bass_isolator.isolate_bass_for_folder")
    def test_folder_dispatches_to_folder_handler(self, mock_folder):
        bass_isolator.isolate_bass_for_path(self.tmp_dir)
        mock_folder.assert_called_once_with(self.tmp_dir)


if __name__ == "__main__":
    unittest.main()

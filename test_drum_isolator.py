import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine

import drum_isolator
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

    def _fake_write_drum_midi(self, wav_path, midi_path, tempo):
        # Real onset detection needs real audio; these tests only care about
        # file orchestration, so stand in with an empty placeholder file.
        with open(midi_path, "wb") as f:
            f.write(b"midi")

    @mock.patch("drum_isolator._write_drum_midi")
    @mock.patch("instrument_isolator.trim_and_export")
    @mock.patch("instrument_isolator.run_demucs")
    @mock.patch("instrument_isolator.song_alignment")
    def test_run_context_reaches_demucs_and_alignment(
        self, mock_alignment, mock_run_demucs, mock_trim, mock_write_midi
    ):
        # The GUI's progress callback and non-interactive choice are only
        # useful if they survive the trip down to where the work happens.
        mock_alignment.return_value = (0, 120.0)
        mock_run_demucs.side_effect = self._fake_run_demucs
        mock_trim.side_effect = self._fake_trim_and_export
        mock_write_midi.side_effect = self._fake_write_drum_midi
        sink = []

        def note_percent(percent):
            sink.append(percent)

        context = instrument_isolator.RunContext(on_percent=note_percent, interactive=False)

        drum_isolator.isolate_drums(self.mp3_path, context=context)

        self.assertIs(mock_run_demucs.call_args.kwargs["on_percent"], note_percent)
        self.assertIs(mock_alignment.call_args.kwargs["interactive"], False)

    @mock.patch("drum_isolator._write_drum_midi")
    @mock.patch("instrument_isolator.trim_and_export")
    @mock.patch("instrument_isolator.run_demucs")
    @mock.patch("instrument_isolator.song_alignment")
    def test_produces_drums_wav_and_midi(self, mock_alignment, mock_run_demucs, mock_trim, mock_write_midi):
        mock_alignment.return_value = (0, 120.0)
        mock_run_demucs.side_effect = self._fake_run_demucs
        mock_trim.side_effect = self._fake_trim_and_export
        mock_write_midi.side_effect = self._fake_write_drum_midi

        result = drum_isolator.isolate_drums(self.mp3_path)

        self.assertTrue(result)
        song_dir = self.song_dir
        basename = "Song - Artist (Isolated Drums at 120.000 BPM)"
        self.assertTrue(os.path.exists(os.path.join(song_dir, basename + ".wav")))
        self.assertTrue(os.path.exists(os.path.join(song_dir, basename + ".mid")))
        mock_write_midi.assert_called_once_with(
            os.path.join(song_dir, basename + ".wav"), os.path.join(song_dir, basename + ".mid"), 120.0
        )
        # A source marker is written too, so a re-run against the same mp3
        # can tell these outputs are already up to date.
        self.assertTrue(os.path.exists(os.path.join(song_dir, drum_isolator._SOURCE_MARKER_FILENAME)))

    @mock.patch("drum_isolator._write_drum_midi")
    @mock.patch("instrument_isolator.trim_and_export")
    @mock.patch("instrument_isolator.run_demucs")
    @mock.patch("instrument_isolator.song_alignment")
    def test_write_midi_false_produces_only_a_wav(self, mock_alignment, mock_run_demucs, mock_trim, mock_write_midi):
        mock_alignment.return_value = (0, 120.0)
        mock_run_demucs.side_effect = self._fake_run_demucs
        mock_trim.side_effect = self._fake_trim_and_export

        result = drum_isolator.isolate_drums(self.mp3_path, write_midi=False)

        self.assertTrue(result)
        basename = "Song - Artist (Isolated Drums at 120.000 BPM)"
        self.assertTrue(os.path.exists(os.path.join(self.song_dir, basename + ".wav")))
        self.assertFalse(os.path.exists(os.path.join(self.song_dir, basename + ".mid")))
        mock_write_midi.assert_not_called()

    @mock.patch("instrument_isolator.run_demucs")
    def test_skips_when_outputs_already_exist_and_match_the_source_mp3(self, mock_run_demucs):
        os.makedirs(self.song_dir)
        basename = "Song - Artist (Isolated Drums at 120.000 BPM)"
        with open(os.path.join(self.song_dir, basename + ".wav"), "wb") as f:
            f.write(b"x")
        with open(os.path.join(self.song_dir, basename + ".mid"), "wb") as f:
            f.write(b"x")
        instrument_isolator.write_source_marker(self.song_dir, self.mp3_path, drum_isolator._SOURCE_MARKER_FILENAME)

        result = drum_isolator.isolate_drums(self.mp3_path)

        self.assertFalse(result)
        mock_run_demucs.assert_not_called()

    @mock.patch("drum_isolator._write_drum_midi")
    @mock.patch("instrument_isolator.trim_and_export")
    @mock.patch("instrument_isolator.run_demucs")
    @mock.patch("instrument_isolator.song_alignment")
    def test_reruns_when_the_midi_file_is_missing_and_no_marker_confirms_the_wav(
        self, mock_alignment, mock_run_demucs, mock_trim, mock_write_midi
    ):
        os.makedirs(self.song_dir)
        basename = "Song - Artist (Isolated Drums at 120.000 BPM)"
        with open(os.path.join(self.song_dir, basename + ".wav"), "wb") as f:
            f.write(b"x")
        # no .mid file present yet, and no marker either - so this wav
        # can't be trusted as coming from self.mp3_path, and the whole
        # thing (not just the MIDI) has to be reprocessed.

        mock_alignment.return_value = (0, 120.0)
        mock_run_demucs.side_effect = self._fake_run_demucs
        mock_trim.side_effect = self._fake_trim_and_export
        mock_write_midi.side_effect = self._fake_write_drum_midi

        result = drum_isolator.isolate_drums(self.mp3_path)

        self.assertTrue(result)
        mock_run_demucs.assert_called_once()
        self.assertTrue(os.path.exists(os.path.join(self.song_dir, basename + ".wav")))
        self.assertTrue(os.path.exists(os.path.join(self.song_dir, basename + ".mid")))

    @mock.patch("drum_isolator._write_drum_midi")
    @mock.patch("instrument_isolator.run_demucs")
    def test_adds_midi_to_an_already_isolated_wav_without_rerunning_demucs(self, mock_run_demucs, mock_write_midi):
        # Regression test: requesting midi against a song already isolated
        # (wav-only, from an earlier midi-less run) used to re-run demucs
        # from scratch just to add the MIDI, even though the wav - by far
        # the slow part - was already there and still valid.
        os.makedirs(self.song_dir)
        basename = "Song - Artist (Isolated Drums at 120.000 BPM)"
        with open(os.path.join(self.song_dir, basename + ".wav"), "wb") as f:
            f.write(b"x")
        instrument_isolator.write_source_marker(self.song_dir, self.mp3_path, drum_isolator._SOURCE_MARKER_FILENAME)
        mock_write_midi.side_effect = self._fake_write_drum_midi

        result = drum_isolator.isolate_drums(self.mp3_path, write_midi=True)

        self.assertTrue(result)
        mock_run_demucs.assert_not_called()
        self.assertTrue(os.path.exists(os.path.join(self.song_dir, basename + ".mid")))
        mock_write_midi.assert_called_once_with(
            os.path.join(self.song_dir, basename + ".wav"), os.path.join(self.song_dir, basename + ".mid"), 120.0
        )

    @mock.patch("drum_isolator._write_drum_midi")
    @mock.patch("instrument_isolator.trim_and_export")
    @mock.patch("instrument_isolator.run_demucs")
    @mock.patch("instrument_isolator.song_alignment")
    def test_reruns_and_replaces_stale_outputs_when_the_source_mp3_marker_does_not_match(
        self, mock_alignment, mock_run_demucs, mock_trim, mock_write_midi
    ):
        os.makedirs(self.song_dir)
        stale_basename = "Song - Artist (Isolated Drums at 99.000 BPM)"
        with open(os.path.join(self.song_dir, stale_basename + ".wav"), "wb") as f:
            f.write(b"stale")
        with open(os.path.join(self.song_dir, stale_basename + ".mid"), "wb") as f:
            f.write(b"stale")
        # No marker at all - e.g. a leftover folder from before this feature
        # existed, or from an unrelated song that happened to share a title.

        mock_alignment.return_value = (0, 130.5)
        mock_run_demucs.side_effect = self._fake_run_demucs
        mock_trim.side_effect = self._fake_trim_and_export
        mock_write_midi.side_effect = self._fake_write_drum_midi

        result = drum_isolator.isolate_drums(self.mp3_path)

        self.assertTrue(result)
        mock_run_demucs.assert_called_once()
        new_basename = "Song - Artist (Isolated Drums at 130.500 BPM)"
        self.assertTrue(os.path.exists(os.path.join(self.song_dir, new_basename + ".wav")))
        self.assertTrue(os.path.exists(os.path.join(self.song_dir, new_basename + ".mid")))
        # The stale, wrongly-named files from the mismatched marker are gone.
        self.assertFalse(os.path.exists(os.path.join(self.song_dir, stale_basename + ".wav")))
        self.assertFalse(os.path.exists(os.path.join(self.song_dir, stale_basename + ".mid")))


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
        wav_path = os.path.join(self.tmp_dir, "drums.wav")
        midi_path = os.path.join(self.tmp_dir, "drums.mid")
        _hits(5).export(wav_path, format="wav")

        drum_isolator._write_drum_midi(wav_path, midi_path, tempo=120.0)

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
        wav_path = os.path.join(self.tmp_dir, "drums.wav")
        midi_path = os.path.join(self.tmp_dir, "drums.mid")
        drum_isolator._write_drum_midi(wav_path, midi_path, tempo=120.0)

        import pretty_midi
        # An instrument with zero notes isn't written back out by pretty_midi
        # on round-trip, so the file legitimately has no instruments at all.
        midi = pretty_midi.PrettyMIDI(midi_path)
        notes = [note for instrument in midi.instruments for note in instrument.notes]
        self.assertEqual(notes, [])

    def test_embeds_the_given_tempo_without_snapping_notes(self):
        wav_path = os.path.join(self.tmp_dir, "drums.wav")
        midi_path = os.path.join(self.tmp_dir, "drums.mid")
        _hits(6).export(wav_path, format="wav")

        drum_isolator._write_drum_midi(wav_path, midi_path, tempo=123.4)

        import pretty_midi
        midi = pretty_midi.PrettyMIDI(midi_path)
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
        for call in mock_isolate.call_args_list:
            self.assertTrue(call.kwargs.get("write_midi", True))

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

        mock_isolate.assert_called_once_with(mp3_path, write_midi=True, context=None)

    @mock.patch("drum_isolator.isolate_drums_for_folder")
    def test_folder_dispatches_to_folder_handler(self, mock_folder):
        drum_isolator.isolate_drums_for_path(self.tmp_dir)
        mock_folder.assert_called_once_with(self.tmp_dir, write_midi=True, context=None)


if __name__ == "__main__":
    unittest.main()

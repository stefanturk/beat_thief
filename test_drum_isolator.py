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
        instrument_isolator.write_source_marker(self.song_dir, self.mp3_path, drum_isolator._MIDI_MARKER_FILENAME)

        result = drum_isolator.isolate_drums(self.mp3_path)

        self.assertFalse(result)
        mock_run_demucs.assert_not_called()

    @mock.patch("drum_isolator._write_drum_midi")
    @mock.patch("instrument_isolator.run_demucs")
    def test_a_midi_from_an_older_transcriber_is_rebuilt_on_the_wav_already_there(
        self, mock_run_demucs, mock_write_midi
    ):
        # A .mid has nothing in it to say which transcriber wrote it, so it
        # carries a marker of its own. When that marker is missing or from
        # an older version the MIDI is redone - but the wav is untouched and
        # demucs doesn't run, because nothing about the audio changed.
        os.makedirs(self.song_dir)
        basename = "Song - Artist (Isolated Drums at 120.000 BPM)"
        for suffix in (".wav", ".mid"):
            with open(os.path.join(self.song_dir, basename + suffix), "wb") as f:
                f.write(b"x")
        instrument_isolator.write_source_marker(self.song_dir, self.mp3_path, drum_isolator._SOURCE_MARKER_FILENAME)
        mock_write_midi.side_effect = self._fake_write_drum_midi

        result = drum_isolator.isolate_drums(self.mp3_path)

        self.assertTrue(result)
        mock_run_demucs.assert_not_called()
        mock_write_midi.assert_called_once()
        # And it says so, so the next run leaves it alone.
        self.assertTrue(instrument_isolator.source_marker_matches(
            self.song_dir, self.mp3_path, drum_isolator._MIDI_MARKER_FILENAME))

    @mock.patch("instrument_isolator.run_demucs")
    def test_an_older_midi_is_left_alone_when_no_midi_was_asked_for(self, mock_run_demucs):
        os.makedirs(self.song_dir)
        basename = "Song - Artist (Isolated Drums at 120.000 BPM)"
        for suffix in (".wav", ".mid"):
            with open(os.path.join(self.song_dir, basename + suffix), "wb") as f:
                f.write(b"x")
        instrument_isolator.write_source_marker(self.song_dir, self.mp3_path, drum_isolator._SOURCE_MARKER_FILENAME)

        result = drum_isolator.isolate_drums(self.mp3_path, write_midi=False)

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


class TestWriteDrumMidi(unittest.TestCase):
    """The MIDI file itself. The transcription behind it is
    drum_transcriber's business and is stubbed out here - what's under test
    is that whatever it returns reaches the file unaltered, at the tempo the
    song was aligned to."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.wav_path = os.path.join(self.tmp_dir, "drums.wav")
        self.midi_path = os.path.join(self.tmp_dir, "drums.mid")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _existing_wav(self):
        _hits(2).export(self.wav_path, format="wav")

    @mock.patch("drum_transcriber.transcribe")
    def test_writes_every_transcribed_hit_to_one_drum_track(self, mock_transcribe):
        self._existing_wav()
        mock_transcribe.return_value = [
            pretty_midi.Note(velocity=100, pitch=38, start=0.5, end=0.55),
            pretty_midi.Note(velocity=90, pitch=36, start=0.0, end=0.05),
            pretty_midi.Note(velocity=70, pitch=46, start=0.25, end=0.4),
        ]

        drum_isolator._write_drum_midi(self.wav_path, self.midi_path, tempo=120.0)

        midi = pretty_midi.PrettyMIDI(self.midi_path)
        self.assertEqual(len(midi.instruments), 1)
        self.assertTrue(midi.instruments[0].is_drum)
        notes = midi.instruments[0].notes
        self.assertEqual([note.pitch for note in notes], [36, 46, 38])
        self.assertEqual([note.velocity for note in notes], [90, 70, 100])
        # Written in start-time order, whatever order they arrived in.
        starts = [note.start for note in notes]
        self.assertEqual(starts, sorted(starts))

    @mock.patch("drum_transcriber.transcribe")
    def test_missing_drums_wav_produces_an_empty_midi_file(self, mock_transcribe):
        drum_isolator._write_drum_midi(self.wav_path, self.midi_path, tempo=120.0)

        mock_transcribe.assert_not_called()
        # An instrument with zero notes isn't written back out by pretty_midi
        # on round-trip, so the file legitimately has no instruments at all.
        midi = pretty_midi.PrettyMIDI(self.midi_path)
        notes = [note for instrument in midi.instruments for note in instrument.notes]
        self.assertEqual(notes, [])

    @mock.patch("drum_transcriber.transcribe")
    def test_embeds_the_given_tempo_without_snapping_notes(self, mock_transcribe):
        self._existing_wav()
        # Deliberately off any grid a 123.4 BPM song could have.
        raw_times = [0.017, 0.493, 1.031, 1.507]
        mock_transcribe.return_value = [
            pretty_midi.Note(velocity=100, pitch=36, start=t, end=t + 0.05) for t in raw_times
        ]

        drum_isolator._write_drum_midi(self.wav_path, self.midi_path, tempo=123.4)

        midi = pretty_midi.PrettyMIDI(self.midi_path)
        _, tempi = midi.get_tempo_changes()
        # Exactly the tempo it's given, not one detected here - that's
        # song_alignment()'s job, shared across every instrument isolated
        # from the same song.
        self.assertAlmostEqual(tempi[0], 123.4, delta=0.1)

        # Hits keep the moment they were detected. Nothing is quantized, so
        # ghost notes, flams and swing survive into the file. The tolerance
        # is MIDI's own tick resolution - writing and reading a .mid always
        # rounds to the nearest tick - not slack for snapping.
        actual = sorted(note.start for note in midi.instruments[0].notes)
        self.assertEqual(len(actual), len(raw_times))
        for got, expected in zip(actual, raw_times):
            self.assertAlmostEqual(got, expected, delta=0.005)


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

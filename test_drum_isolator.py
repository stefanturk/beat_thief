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


class TestMapDrumsepStemName(unittest.TestCase):
    def test_maps_english_names(self):
        self.assertEqual(drum_isolator._map_drumsep_stem_name("kick.wav"), "kick")
        self.assertEqual(drum_isolator._map_drumsep_stem_name("snare.wav"), "snare")
        self.assertEqual(drum_isolator._map_drumsep_stem_name("toms.wav"), "toms")
        self.assertEqual(drum_isolator._map_drumsep_stem_name("cymbals.wav"), "cymbals")
        self.assertEqual(drum_isolator._map_drumsep_stem_name("hihat.wav"), "cymbals")

    def test_maps_spanish_checkpoint_names(self):
        self.assertEqual(drum_isolator._map_drumsep_stem_name("bombo.wav"), "kick")
        self.assertEqual(drum_isolator._map_drumsep_stem_name("redoblante.wav"), "snare")
        self.assertEqual(drum_isolator._map_drumsep_stem_name("caja.wav"), "snare")
        self.assertEqual(drum_isolator._map_drumsep_stem_name("platillos.wav"), "cymbals")

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

    def _fake_write_drum_midi(self, song_dir):
        # Real onset detection needs real audio; these tests only care about
        # file orchestration, so stand in with an empty placeholder file.
        with open(os.path.join(song_dir, drum_isolator.MIDI_FILENAME), "wb") as f:
            f.write(b"midi")

    @mock.patch("drum_isolator._write_drum_midi")
    @mock.patch("drum_isolator._ensure_drumsep_model")
    @mock.patch("drum_isolator._trim_leading_silence", side_effect=lambda path: path)
    @mock.patch("drum_isolator._run_demucs")
    def test_produces_all_stems_and_midi(self, mock_run_demucs, mock_trim, mock_ensure_model, mock_write_midi):
        mock_run_demucs.side_effect = self._fake_run_demucs
        mock_write_midi.side_effect = self._fake_write_drum_midi

        result = drum_isolator.isolate_drums(self.mp3_path, self.drums_root)

        self.assertTrue(result)
        song_dir = os.path.join(self.drums_root, "Song - Artist")
        for name in drum_isolator._EXPECTED_OUTPUTS:
            self.assertTrue(os.path.exists(os.path.join(song_dir, name)), name)
        mock_ensure_model.assert_called_once()
        mock_write_midi.assert_called_once_with(song_dir)

    @mock.patch("drum_isolator._ensure_drumsep_model")
    @mock.patch("drum_isolator._run_demucs")
    def test_skips_when_all_outputs_already_exist(self, mock_run_demucs, mock_ensure_model):
        song_dir = os.path.join(self.drums_root, "Song - Artist")
        os.makedirs(song_dir)
        for name in drum_isolator._EXPECTED_OUTPUTS:
            with open(os.path.join(song_dir, name), "wb") as f:
                f.write(b"x")

        result = drum_isolator.isolate_drums(self.mp3_path, self.drums_root)

        self.assertFalse(result)
        mock_run_demucs.assert_not_called()
        mock_ensure_model.assert_not_called()

    @mock.patch("drum_isolator._write_drum_midi")
    @mock.patch("drum_isolator._ensure_drumsep_model")
    @mock.patch("drum_isolator._trim_leading_silence", side_effect=lambda path: path)
    @mock.patch("drum_isolator._run_demucs")
    def test_reruns_when_the_midi_file_is_missing(self, mock_run_demucs, mock_trim, mock_ensure_model, mock_write_midi):
        song_dir = os.path.join(self.drums_root, "Song - Artist")
        os.makedirs(song_dir)
        for name in drum_isolator._STEM_NAMES:
            with open(os.path.join(song_dir, name), "wb") as f:
                f.write(b"x")
        # drums.mid missing

        mock_run_demucs.side_effect = self._fake_run_demucs
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
        wav_path = os.path.join(self.tmp_dir, "kick.wav")
        _hits(4).export(wav_path, format="wav")

        notes = drum_isolator._detect_note_events(wav_path, midi_note=36)

        # Onset detection on synthetic tones won't be pixel-perfect, but it
        # should land in the ballpark of the four hits we actually made.
        self.assertGreaterEqual(len(notes), 2)
        self.assertLessEqual(len(notes), 6)
        for note in notes:
            self.assertEqual(note.pitch, 36)
            self.assertTrue(1 <= note.velocity <= 127)

    def test_silence_produces_no_notes(self):
        wav_path = os.path.join(self.tmp_dir, "silence.wav")
        AudioSegment.silent(duration=2000).export(wav_path, format="wav")

        notes = drum_isolator._detect_note_events(wav_path, midi_note=36)

        self.assertEqual(notes, [])


class TestDetectTempo(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_detects_tempo_of_a_steady_click_track(self):
        wav_path = os.path.join(self.tmp_dir, "drums.wav")
        _click_track(bpm=120, beats=24).export(wav_path, format="wav")

        tempo = drum_isolator._detect_tempo(wav_path)

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
        # how _detect_tempo's single windowed guess is only approximate.
        refined = drum_isolator._refine_tempo(onset_times, initial_tempo=140.0)

        self.assertAlmostEqual(refined, true_bpm, delta=0.1)

    def test_too_few_onsets_returns_the_initial_estimate_unchanged(self):
        onset_times = np.array([0.0, 0.5, 1.0])

        refined = drum_isolator._refine_tempo(onset_times, initial_tempo=95.0)

        self.assertEqual(refined, 95.0)


class TestWriteDrumMidi(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_combines_all_present_stems_into_one_file(self):
        _hits(3).export(os.path.join(self.tmp_dir, "kick.wav"), format="wav")
        _hits(2, freq=800).export(os.path.join(self.tmp_dir, "snare.wav"), format="wav")
        # toms.wav / cymbals.wav intentionally absent

        drum_isolator._write_drum_midi(self.tmp_dir)

        midi_path = os.path.join(self.tmp_dir, drum_isolator.MIDI_FILENAME)
        self.assertTrue(os.path.exists(midi_path))

        import pretty_midi
        midi = pretty_midi.PrettyMIDI(midi_path)
        self.assertEqual(len(midi.instruments), 1)
        self.assertTrue(midi.instruments[0].is_drum)
        pitches = {note.pitch for note in midi.instruments[0].notes}
        self.assertTrue(pitches.issubset({36, 38}))
        self.assertGreater(len(midi.instruments[0].notes), 0)
        # Notes should be written in start-time order.
        starts = [note.start for note in midi.instruments[0].notes]
        self.assertEqual(starts, sorted(starts))

    def test_embeds_a_precisely_refined_tempo_without_snapping_notes(self):
        true_bpm = 123.4
        click = _click_track(bpm=true_bpm, beats=200)
        click.export(os.path.join(self.tmp_dir, "drums.wav"), format="wav")
        click.export(os.path.join(self.tmp_dir, "kick.wav"), format="wav")

        drum_isolator._write_drum_midi(self.tmp_dir)

        import pretty_midi
        midi = pretty_midi.PrettyMIDI(os.path.join(self.tmp_dir, drum_isolator.MIDI_FILENAME))
        _, tempi = midi.get_tempo_changes()
        tempo = tempi[0]

        # Octave errors are a separate, accepted failure mode - check
        # precision within whichever octave the rough estimate landed on.
        targets = [octave * true_bpm for octave in (0.5, 1.0, 2.0)]
        closest_target = min(targets, key=lambda target: abs(tempo - target))
        self.assertLess(abs(tempo - closest_target), 0.1, tempo)

        # Notes should keep their raw onset-detected times untouched - not
        # snapped to any musical grid - matching what _detect_note_events
        # alone would produce on the same file. A small tolerance accounts
        # for MIDI's own tick resolution: writing/reading a .mid file always
        # rounds absolute times to the nearest tick (a couple of ms here),
        # regardless of any snapping logic - that's the file format, not us.
        expected_starts = sorted(n.start for n in drum_isolator._detect_note_events(
            os.path.join(self.tmp_dir, "kick.wav"), midi_note=36))
        actual_kick_starts = sorted(n.start for n in midi.instruments[0].notes if n.pitch == 36)
        self.assertEqual(len(actual_kick_starts), len(expected_starts))
        for actual, expected in zip(actual_kick_starts, expected_starts):
            self.assertAlmostEqual(actual, expected, delta=0.01)


class TestTrimLeadingSilence(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_trims_a_silent_intro_so_the_beat_starts_near_zero(self):
        silent_intro = AudioSegment.silent(duration=3000)
        beat = Sine(200).to_audio_segment(duration=3000)
        track = silent_intro + beat
        wav_path = os.path.join(self.tmp_dir, "drums.wav")
        track.export(wav_path, format="wav")

        trimmed_path = drum_isolator._trim_leading_silence(wav_path)

        self.assertNotEqual(trimmed_path, wav_path)
        trimmed_audio = AudioSegment.from_wav(trimmed_path)
        # The ~3s silent intro should be gone, leaving mostly just the beat.
        self.assertLess(len(trimmed_audio), len(track) - 1500)

    def test_no_intro_to_trim_leaves_the_file_alone(self):
        beat = Sine(200).to_audio_segment(duration=3000)
        wav_path = os.path.join(self.tmp_dir, "drums.wav")
        beat.export(wav_path, format="wav")

        trimmed_path = drum_isolator._trim_leading_silence(wav_path)

        self.assertEqual(trimmed_path, wav_path)


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

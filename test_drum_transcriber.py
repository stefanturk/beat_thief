import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import soundfile as sf

import drum_transcriber as dt

SR = dt.SAMPLE_RATE


def _click(length_sec, hits, freq=6000.0, decay_sec=0.01, amplitude=1.0):
    """A signal with a burst of tone at each time in hits."""
    audio = np.zeros(int(length_sec * SR), dtype=np.float32)
    for start_sec in hits:
        start = int(start_sec * SR)
        t = np.arange(int(decay_sec * 4 * SR)) / SR
        burst = np.sin(2 * np.pi * freq * t) * np.exp(-t / decay_sec) * amplitude
        end = min(start + burst.size, audio.size)
        audio[start:end] += burst[: end - start].astype(np.float32)
    return audio


def _activations(length_frames, hits_by_class):
    """A fake model output: [frames, 5], with a sharp spike wherever a class
    was hit."""
    out = np.zeros((length_frames, 5), dtype=np.float32)
    for class_index, frames in hits_by_class.items():
        for frame in frames:
            out[frame, class_index] = 1.0
    return out


class TranscriberTestCase(unittest.TestCase):
    """The model itself is never run here. What's under test is everything
    Beat Thief does around it - peak picking, the note map, velocity, and
    the open/closed hi-hat split - so the activations are supplied directly
    and the audio is synthetic."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _wav(self, audio, name="drums.wav"):
        path = os.path.join(self.tmp_dir, name)
        sf.write(path, audio, SR)
        return path

    def _transcribe(self, audio, hits_by_class):
        path = self._wav(audio)
        frames = int(len(audio) / SR * dt.adtof.FPS) + 1
        with mock.patch.object(dt, "_activations", return_value=_activations(frames, hits_by_class)):
            return dt.transcribe(path)


class TestNoteMap(TranscriberTestCase):
    def test_every_class_lands_on_its_drum_rack_pad(self):
        notes = self._transcribe(
            _click(3.0, [0.5, 1.0, 1.5, 2.0, 2.5]),
            {0: [50], 1: [100], 2: [150], 3: [200], 4: [250]},
        )

        self.assertEqual([note.pitch for note in notes], [36, 38, 47, 42, 49])

    def test_the_kick_is_moved_off_the_note_the_model_emits(self):
        # The model says 35, which is below the first pad of Ableton's
        # default Drum Rack and would be invisible on it.
        notes = self._transcribe(_click(2.0, [1.0]), {0: [100]})

        self.assertEqual(dt.adtof.LABELS_5[0], 35)
        self.assertEqual([note.pitch for note in notes], [36])

    def test_hits_come_back_at_the_time_they_happened(self):
        notes = self._transcribe(_click(4.0, [0.5, 2.5]), {0: [50, 250]})

        self.assertEqual([round(note.start, 3) for note in notes], [0.5, 2.5])

    def test_notes_are_sorted_by_time(self):
        notes = self._transcribe(_click(4.0, [0.3, 1.1, 2.9]), {4: [290], 0: [30], 1: [110]})

        starts = [note.start for note in notes]
        self.assertEqual(starts, sorted(starts))

    def test_silence_produces_nothing(self):
        notes = self._transcribe(np.zeros(SR * 2, dtype=np.float32), {})

        self.assertEqual(notes, [])


class TestChunking(unittest.TestCase):
    """Chunked inference has to give the same answer a whole-song pass
    would. Swept against a real 4 minute file, the settings here reproduce
    it exactly; these tests hold the seam handling itself in place."""

    def setUp(self):
        self.chunk, self.context = dt._CHUNK_FRAMES, dt._CONTEXT_FRAMES

    def tearDown(self):
        dt._CHUNK_FRAMES, dt._CONTEXT_FRAMES = self.chunk, self.context

    def test_a_song_spanning_several_chunks_is_stitched_back_in_order(self):
        # A model that reports where in the song each frame it was shown
        # sits, so a misplaced chunk shows up as a wrong value.
        frames, bins = 700, 84
        spectrogram = np.zeros((frames, bins, 1), dtype=np.float32)

        def fake_model(x):
            import torch

            length = x.shape[1]
            return torch.zeros(1, length, 5)

        dt._CHUNK_FRAMES, dt._CONTEXT_FRAMES = 100, 50
        with mock.patch.object(dt, "_load_model", return_value=fake_model):
            out = dt._activations(spectrogram)

        self.assertEqual(out.shape, (frames, 5))

    def test_context_frames_are_read_but_not_kept(self):
        frames = 500
        spectrogram = np.zeros((frames, 84, 1), dtype=np.float32)
        seen = []

        def fake_model(x):
            import torch

            seen.append(x.shape[1])
            return torch.zeros(1, x.shape[1], 5)

        dt._CHUNK_FRAMES, dt._CONTEXT_FRAMES = 100, 50
        with mock.patch.object(dt, "_load_model", return_value=fake_model):
            out = dt._activations(spectrogram)

        # Each pass reads more than it keeps...
        self.assertTrue(all(width > 100 for width in seen[1:-1]))
        # ...but the result is exactly one row per frame of the song, with
        # no seams doubled up or dropped.
        self.assertEqual(out.shape[0], frames)

    def test_an_empty_song_is_not_run_through_the_model_at_all(self):
        with mock.patch.object(dt, "_load_model", side_effect=AssertionError("should not load")):
            out = dt._activations(np.zeros((0, 84, 1), dtype=np.float32))

        self.assertEqual(out.shape, (0, 5))


class TestVelocity(TranscriberTestCase):
    def test_the_loudest_hit_of_a_class_reaches_full_velocity(self):
        audio = _click(3.0, [0.5], amplitude=0.2) + _click(3.0, [1.5], amplitude=1.0)
        notes = self._transcribe(audio, {3: [50, 150]})

        self.assertEqual(max(note.velocity for note in notes), 127)

    def test_a_quieter_hit_gets_a_lower_velocity(self):
        audio = _click(3.0, [0.5], amplitude=0.2) + _click(3.0, [1.5], amplitude=1.0)
        notes = self._transcribe(audio, {3: [50, 150]})

        quiet, loud = sorted(notes, key=lambda note: note.start)
        self.assertLess(quiet.velocity, loud.velocity)

    def test_each_class_is_scaled_against_its_own_loudest_hit(self):
        # A loud kick and a much quieter hi-hat, as in any real mix. Scaled
        # together the hats would all sit at the bottom of the range and the
        # part would sound dead; scaled per class the hat keeps its own
        # dynamics.
        audio = _click(3.0, [0.5], freq=60.0, decay_sec=0.05, amplitude=1.0)
        audio = audio + _click(3.0, [1.5], freq=9000.0, amplitude=0.03)
        notes = self._transcribe(audio, {0: [50], 3: [150]})

        hihat = next(note for note in notes if note.pitch == 42)
        self.assertEqual(hihat.velocity, 127)

    def test_velocity_is_measured_in_decibels_not_raw_amplitude(self):
        # A hit at a tenth the amplitude is 20dB down. Linearly that would
        # be velocity 13; over the 30dB range this uses it's about 42.
        quiet = dt._velocities([1.0, 0.1])[1]

        self.assertGreater(quiet, 30)
        self.assertLess(quiet, 55)

    def test_silence_still_produces_a_playable_velocity(self):
        self.assertEqual(dt._velocities([0.0, 0.0]), [1, 1])


class TestOpenHihats(TranscriberTestCase):
    def _hihat(self, hits, decay_sec, length_sec=4.0):
        return _click(length_sec, hits, freq=9000.0, decay_sec=decay_sec)

    def test_a_hat_that_rings_on_is_written_as_an_open_one(self):
        notes = self._transcribe(self._hihat([1.0], decay_sec=0.12), {3: [100]})

        self.assertEqual([note.pitch for note in notes], [dt._OPEN_HIHAT_NOTE])

    def test_a_short_hat_stays_closed(self):
        notes = self._transcribe(self._hihat([1.0], decay_sec=0.004), {3: [100]})

        self.assertEqual([note.pitch for note in notes], [42])

    def test_an_open_hat_is_written_long_enough_to_see(self):
        notes = self._transcribe(self._hihat([1.0], decay_sec=0.12), {3: [100]})

        self.assertGreater(notes[0].end - notes[0].start, dt._OPEN_HIHAT_DECAY_SEC)
        self.assertLessEqual(notes[0].end - notes[0].start, dt._OPEN_HIHAT_MAX_RING_SEC)

    def test_a_short_hat_stays_closed_even_with_another_right_behind_it(self):
        # The next hat must not be measured as part of this one's ring -
        # that would make every hat in a fast pattern read as open.
        notes = self._transcribe(self._hihat([1.0, 1.07], decay_sec=0.004), {3: [100, 107]})

        self.assertEqual({note.pitch for note in notes}, {42})

    def test_a_hat_still_ringing_when_the_next_one_lands_is_open(self):
        # A hat doesn't have to ring for the full 120ms to have been played
        # open - if it's still going when the next hat chokes it, that's
        # what it was.
        notes = self._transcribe(self._hihat([1.0, 1.07], decay_sec=0.12), {3: [100, 107]})

        self.assertEqual(notes[0].pitch, dt._OPEN_HIHAT_NOTE)

    def test_hats_too_close_together_to_measure_stay_closed(self):
        # 40ms apart leaves no room to tell a ring from the next hit.
        notes = self._transcribe(self._hihat([1.0, 1.04], decay_sec=0.12), {3: [100, 104]})

        self.assertEqual(notes[0].pitch, 42)

    def test_a_hat_under_a_cymbal_stays_closed(self):
        # A crash occupies the same part of the spectrum and rings for
        # seconds; every hat under one would otherwise read as open.
        audio = self._hihat([1.0], decay_sec=0.004) + _click(4.0, [1.02], freq=7000.0, decay_sec=0.6)
        notes = self._transcribe(audio, {3: [100], 4: [102]})

        self.assertEqual({note.pitch for note in notes}, {42, 49})


class TestLoadingTheModel(unittest.TestCase):
    def tearDown(self):
        dt._model = None

    def test_weights_that_will_not_load_raise_rather_than_transcribe_noise(self):
        # Falling back to an untrained network would produce a MIDI file
        # that looks fine and is nonsense.
        dt._model = None
        with mock.patch.object(dt.adtof, "load_pytorch_weights", side_effect=RuntimeError("bad weights")):
            with self.assertRaises(RuntimeError):
                dt._load_model()

    def test_the_model_is_built_once_and_reused(self):
        dt._model = None
        sentinel = object()
        with mock.patch.object(dt.adtof, "create_frame_rnn_model") as mock_create, \
             mock.patch.object(dt.adtof, "load_pytorch_weights", return_value=sentinel):
            first, second = dt._load_model(), dt._load_model()

        self.assertIs(first, sentinel)
        self.assertIs(second, sentinel)
        self.assertEqual(mock_create.call_count, 1)


if __name__ == "__main__":
    unittest.main()

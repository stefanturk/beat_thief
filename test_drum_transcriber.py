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


def _drum(length_sec, hits, amplitude=1.0):
    """A hit with a shell under it: a 220Hz body plus the crack on top. What
    percussion_splitter is looking for is the body, so this is what has to be
    used wherever a test means an actual snare - a bare high burst is a
    tambourine as far as the splitter is concerned, and correctly so."""
    body = _click(length_sec, hits, freq=220.0, decay_sec=0.05, amplitude=amplitude)
    crack = _click(length_sec, hits, freq=3000.0, decay_sec=0.01, amplitude=amplitude * 0.3)
    return body + crack


def _shaken(length_sec, hits, amplitude=1.0):
    """A hit with no shell at all - all jingle, no body."""
    return _click(length_sec, hits, freq=9000.0, decay_sec=0.02, amplitude=amplitude)


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

    def _transcribe(self, audio, hits_by_class, **kwargs):
        path = self._wav(audio)
        frames = int(len(audio) / SR * dt.adtof.FPS) + 1
        with mock.patch.object(dt, "_activations", return_value=_activations(frames, hits_by_class)):
            return dt.transcribe(path, **kwargs)


class TestNoteMap(TranscriberTestCase):
    def test_every_class_lands_on_its_drum_rack_pad(self):
        # Not about percussion classification - a bare click has no body and
        # would otherwise read as a tambourine on both the snare and hi-hat
        # class (see TestTheSnareClassSplitsThreeWays / TestHiHatPercussion),
        # which isn't what this test is checking.
        with mock.patch.object(dt.percussion_splitter, "split",
                                side_effect=lambda audio, times, sr: [dt.percussion_splitter.SNARE] * len(times)):
            notes = self._transcribe(
                _click(3.0, [0.5, 1.5, 2.0, 2.5]) + _drum(3.0, [1.0]),
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


class TestTheSnareClassSplitsThreeWays(TranscriberTestCase):
    """The model has one class for everything that hit like a snare, and a
    real kit has more than one thing that does. These are the votes cast
    here; groove_reader is what decides once the loop is on a grid."""

    def test_a_hit_with_a_shell_under_it_is_a_snare(self):
        notes = self._transcribe(_drum(2.0, [1.0]), {1: [100]})

        self.assertEqual([note.pitch for note in notes], [38])

    def test_a_hit_with_no_shell_at_all_is_percussion(self):
        notes = self._transcribe(_shaken(2.0, [1.0]), {1: [100]})

        self.assertEqual([note.pitch for note in notes], [dt._PERCUSSION_NOTE])

    def test_a_quiet_percussion_hit_is_still_percussion(self):
        # How loud it was is not this stage's business - which of what's left
        # on the snare pad is a ghost note is decided by groove_reader, after
        # the voices have stopped moving between pads.
        audio = _shaken(3.0, [0.5], amplitude=1.0) + _shaken(3.0, [1.5], amplitude=0.1)
        notes = self._transcribe(audio, {1: [50, 150]})

        self.assertEqual({note.pitch for note in notes}, {dt._PERCUSSION_NOTE})

    def test_a_silent_onset_stays_a_snare(self):
        # Nothing to measure is not evidence of percussion, even though
        # silence has the same rise in every band.
        notes = self._transcribe(np.zeros(SR * 2, dtype=np.float32), {1: [100]})

        self.assertEqual([note.pitch for note in notes], [38])

    def test_every_pad_the_snare_class_can_reach_is_in_the_rack(self):
        for note in (dt._PERCUSSION_NOTE, dt._NOTE_FOR_CLASS[dt._SNARE]):
            self.assertTrue(36 <= note <= 51, f"{note} is outside the drum rack")


def _real_hihat(length_sec, hits, amplitude=1.0):
    """A hi-hat with a shell under the jingle, at the ratio that scores as a
    real drum on percussion_splitter (see TestOpenHihats and
    drum_transcriber._HAT_PERCUSSION_NOTE). What percussion_splitter is
    looking for is the body - a bare jingle click with nothing under it is a
    tambourine as far as it's concerned, same as _shaken is for the snare
    class, and correctly so."""
    jingle = _click(length_sec, hits, freq=9000.0, decay_sec=0.02, amplitude=amplitude * 0.25)
    body = _click(length_sec, hits, freq=220.0, decay_sec=0.05, amplitude=amplitude * 0.75)
    return jingle + body


class TestHiHatPercussion(TranscriberTestCase):
    """Same vote as the snare class, taken on the hi-hat class's hits - see
    drum_transcriber._HAT_PERCUSSION_NOTE for why it's a second pool rather
    than folded into the first. Hits here are spaced well apart; a hi-hat
    right behind another is exactly the kind of noisy single-hit case
    groove_reader's whole-voice vote exists to absorb, not something this
    per-hit vote is expected to get right alone."""

    def test_a_hit_with_a_shell_under_it_is_a_real_hihat(self):
        notes = self._transcribe(_real_hihat(2.0, [1.0]), {3: [100]})

        self.assertEqual([note.pitch for note in notes], [dt._NOTE_FOR_CLASS[dt._HIHAT]])

    def test_a_hit_with_no_shell_at_all_is_percussion(self):
        notes = self._transcribe(_shaken(2.0, [1.0]), {3: [100]})

        self.assertEqual([note.pitch for note in notes], [dt._HAT_PERCUSSION_NOTE])

    def test_a_quiet_percussion_hit_is_still_percussion(self):
        audio = _shaken(3.0, [0.5], amplitude=1.0) + _shaken(3.0, [1.5], amplitude=0.1)
        notes = self._transcribe(audio, {3: [50, 150]})

        self.assertEqual({note.pitch for note in notes}, {dt._HAT_PERCUSSION_NOTE})

    def test_a_silent_onset_stays_a_real_hihat(self):
        notes = self._transcribe(np.zeros(SR * 2, dtype=np.float32), {3: [100]})

        self.assertEqual([note.pitch for note in notes], [dt._NOTE_FOR_CLASS[dt._HIHAT]])

    def test_the_percussion_pad_is_in_the_rack(self):
        # Provisional and never written by name - beat_loop._PIECE_FOR_NOTE
        # maps it, and groove_reader always resolves it away before a beat
        # reaches beat_writer - but if it ever did leak through, it has to
        # land somewhere a stock Drum Rack can play.
        self.assertTrue(36 <= dt._HAT_PERCUSSION_NOTE <= 51)

    def test_a_caller_can_supply_a_different_line(self):
        # A real hi-hat that reads as percussion against the default line
        # (see calibrate_hat_threshold) shouldn't, against one fitted to a
        # brighter-sounding kit.
        audio = _real_hihat(2.0, [1.0])
        default_notes = self._transcribe(audio, {3: [100]})
        self.assertEqual([n.pitch for n in default_notes], [dt._NOTE_FOR_CLASS[dt._HIHAT]])

        notes = self._transcribe(audio, {3: [100]}, hat_threshold=-100.0)
        self.assertEqual([n.pitch for n in notes], [dt._HAT_PERCUSSION_NOTE])


class TestCalibratingTheHatThreshold(TranscriberTestCase):
    """calibrate_hat_threshold reads a whole file's hi-hat class and sets the
    percussion line relative to what's typical there, rather than trusting
    the fixed line calibrated on one other song (see the module docstring
    and beat-thief-jazz-genre-limits in project memory)."""

    def setUp(self):
        super().setUp()
        dt.calibrate_hat_threshold.cache_clear()

    def _calibrate(self, audio, hits_by_class):
        path = self._wav(audio)
        frames = int(len(audio) / SR * dt.adtof.FPS) + 1
        with mock.patch.object(dt, "_activations", return_value=_activations(frames, hits_by_class)):
            return dt.calibrate_hat_threshold(path)

    def test_too_few_hits_falls_back_to_the_fixed_line(self):
        hits = [0.5 + i for i in range(dt._MIN_HITS_FOR_CALIBRATION - 1)]
        audio = sum((_real_hihat(len(hits) + 2.0, [t]) for t in hits), start=np.zeros(
            int((len(hits) + 2.0) * SR), dtype=np.float32))
        threshold = self._calibrate(audio, {3: [int(t * dt.adtof.FPS) for t in hits]})

        self.assertEqual(threshold, dt.percussion_splitter._PERCUSSION_SCORE_DB)

    def test_the_line_sits_above_a_mostly_real_kit_own_median(self):
        length = dt._MIN_HITS_FOR_CALIBRATION + 2.0
        hits = [1.0 + i for i in range(dt._MIN_HITS_FOR_CALIBRATION)]
        audio = sum((_real_hihat(length, [t]) for t in hits),
                    start=np.zeros(int(length * SR), dtype=np.float32))
        threshold = self._calibrate(audio, {3: [int(t * dt.adtof.FPS) for t in hits]})

        # A real hit's own score should land comfortably under the line
        # calibrated from a population of hits just like it.
        score = dt.percussion_splitter.score(audio, [hits[0]], SR)[0]
        self.assertLess(score, threshold)

    def test_is_cached_per_file(self):
        hits = [1.0 + i for i in range(dt._MIN_HITS_FOR_CALIBRATION)]
        length = dt._MIN_HITS_FOR_CALIBRATION + 2.0
        audio = sum((_real_hihat(length, [t]) for t in hits),
                    start=np.zeros(int(length * SR), dtype=np.float32))
        path = self._wav(audio)
        frames = int(len(audio) / SR * dt.adtof.FPS) + 1

        with mock.patch.object(dt, "_activations",
                                return_value=_activations(frames, {3: [int(t * dt.adtof.FPS) for t in hits]})) as activations:
            dt.calibrate_hat_threshold(path)
            dt.calibrate_hat_threshold(path)

        self.assertEqual(activations.call_count, 1)


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
        # Not about percussion classification - see test_every_class_lands_on_its_drum_rack_pad.
        with mock.patch.object(dt.percussion_splitter, "split",
                                side_effect=lambda audio, times, sr: [dt.percussion_splitter.SNARE] * len(times)):
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
    """Open-vs-closed ring detection, not percussion classification - see
    TestHiHatPercussion for that. percussion_splitter is mocked off here so a
    bare test click (no body, which genuinely would read as a tambourine -
    see _shaken) doesn't interfere with what these tests are about."""

    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(
            dt.percussion_splitter, "split",
            side_effect=lambda audio, times, sr: [dt.percussion_splitter.SNARE] * len(times))
        self.addCleanup(patcher.stop)
        patcher.start()

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

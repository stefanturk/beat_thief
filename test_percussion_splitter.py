import unittest

import numpy as np

import percussion_splitter as ps

SR = 44100


def _burst(length_sec, at_sec, freq, decay_sec=0.02, amplitude=1.0):
    audio = np.zeros(int(length_sec * SR), dtype=np.float32)
    t = np.arange(int(decay_sec * 6 * SR)) / SR
    shape = (np.sin(2 * np.pi * freq * t) * np.exp(-t / decay_sec) * amplitude).astype(np.float32)
    start = int(at_sec * SR)
    end = min(start + shape.size, audio.size)
    audio[start:end] += shape[: end - start]
    return audio


def _drum(length_sec, at_sec, amplitude=1.0):
    """A shell and the crack on top of it."""
    return (_burst(length_sec, at_sec, 220.0, 0.05, amplitude)
            + _burst(length_sec, at_sec, 3000.0, 0.01, amplitude * 0.3))


def _shaken(length_sec, at_sec, amplitude=1.0):
    """All jingle, no shell."""
    return _burst(length_sec, at_sec, 9000.0, 0.02, amplitude)


def _drone(length_sec, freq, amplitude):
    t = np.arange(int(length_sec * SR)) / SR
    return (np.sin(2 * np.pi * freq * t) * amplitude).astype(np.float32)


class TestTheVote(unittest.TestCase):
    def test_a_hit_with_a_shell_is_a_drum(self):
        self.assertEqual(ps.split(_drum(2.0, 1.0), [1.0], SR), [ps.SNARE])

    def test_a_hit_with_no_shell_is_percussion(self):
        self.assertEqual(ps.split(_shaken(2.0, 1.0), [1.0], SR), [ps.PERCUSSION])

    def test_several_hits_come_back_in_order(self):
        audio = _drum(3.0, 0.5) + _shaken(3.0, 1.5) + _drum(3.0, 2.5)

        self.assertEqual(ps.split(audio, [0.5, 1.5, 2.5], SR),
                         [ps.SNARE, ps.PERCUSSION, ps.SNARE])

    def test_nothing_to_measure_is_not_evidence_of_percussion(self):
        # Silence has the same rise in every band, which scores as a perfect
        # percussion hit unless it's guarded. It has to come back a snare -
        # the conservative answer, the way an unmeasurable hi-hat stays
        # closed.
        self.assertEqual(ps.split(np.zeros(SR * 2, dtype=np.float32), [1.0], SR), [ps.SNARE])

    def test_no_hits_is_not_an_error(self):
        self.assertEqual(ps.split(_drum(2.0, 1.0), [], SR), [])

    def test_a_hit_at_the_very_start_does_not_reach_behind_the_file(self):
        # There is no "just before" to measure against at time zero.
        self.assertEqual(len(ps.split(_shaken(2.0, 0.0), [0.0], SR)), 1)


class TestWhyItMeasuresARiseAndNotALevel(unittest.TestCase):
    """The whole reason this module measures how much a band jumps rather
    than how loud it is. Measured on a real stem, the loudest thing in the
    low window near any onset was the kick's decay, not the hit - so a level
    reading called everything a drum and separated nothing."""

    def test_percussion_over_a_ringing_kick_is_still_percussion(self):
        # The low end here is enormous and has nothing to do with the hit.
        audio = _drone(2.0, 90.0, 0.8) + _shaken(2.0, 1.0, 0.5)

        self.assertEqual(ps.split(audio, [1.0], SR), [ps.PERCUSSION])

    def test_a_drum_over_a_bright_wash_is_still_a_drum(self):
        # And the mirror image: a cymbal ringing through doesn't turn a snare
        # into a tambourine.
        audio = _drone(2.0, 9000.0, 0.5) + _drum(2.0, 1.0, 1.0)

        self.assertEqual(ps.split(audio, [1.0], SR), [ps.SNARE])

    def test_the_verdict_does_not_depend_on_how_loud_the_mix_is(self):
        # Everything rises more in a sparse arrangement, which is a fact
        # about the mix and not about the instrument. A difference of two
        # rises cancels it; either rise on its own would not.
        quiet = ps.score(_shaken(2.0, 1.0, 0.02), [1.0], SR)[0]
        loud = ps.score(_shaken(2.0, 1.0, 1.0), [1.0], SR)[0]

        self.assertAlmostEqual(quiet, loud, delta=1.0)


if __name__ == "__main__":
    unittest.main()

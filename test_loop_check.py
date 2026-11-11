"""loop_check: does the drumming after a loop sound like the loop?"""

import os
import shutil
import tempfile
import unittest

import numpy as np
from scipy.io import wavfile

import loop_check

RATE = 22050


def _drums(tempo, bars, seed=0, jitter_sec=0.004):
    """A backbeat played by somebody slightly human: kick on 1, 3 and the
    and of 3 (so half a bar on isn't the same bar), snare on 2 and 4, hats
    on the eighths, each hit a few milliseconds off and a fresh burst of
    noise every time."""
    rng = np.random.default_rng(seed)
    beat = 60.0 / tempo
    out = np.zeros(int((bars * 4 + 2) * beat * RATE))

    def hit(at, sound):
        start = int((at + rng.uniform(-jitter_sec, jitter_sec)) * RATE)
        if 0 <= start < len(out):
            end = min(len(out), start + len(sound))
            out[start:end] += sound[: end - start]

    t_kick = np.arange(int(0.12 * RATE)) / RATE
    for bar in range(bars):
        at = bar * 4 * beat
        for when in (0, 2, 2.5):
            hit(at + when * beat, 0.9 * np.sin(2 * np.pi * 55 * t_kick) * np.exp(-t_kick * 30))
        for when in (1, 3):
            noise = rng.normal(0, 0.3, int(0.1 * RATE)) * np.exp(-np.arange(int(0.1 * RATE)) / RATE * 40)
            hit(at + when * beat, noise)
        for eighth in range(8):
            hats = np.diff(rng.normal(0, 0.2, int(0.03 * RATE) + 1)) * np.exp(-np.arange(int(0.03 * RATE)) / RATE * 150)
            hit(at + eighth * beat / 2, hats)
    return out


class TestRepeatLength(unittest.TestCase):
    def test_a_loop_marked_at_the_right_length_is_left_alone(self):
        samples = _drums(101.4, bars=12)
        got = loop_check.repeat_length(samples, RATE, 60 / 101.4, 16)

        self.assertIsNotNone(got)
        self.assertAlmostEqual(got["beats"], 16, delta=0.1)
        self.assertFalse(got["off"])

    def test_a_grid_four_percent_slow_is_caught(self):
        # Dry Rub: six bars of a 97.5 grid over drums at 101.4 hold almost
        # 25 of the drummer's beats. The drums repeat after 24 of them,
        # which is 23.1 of the grid's.
        samples = _drums(101.4, bars=16)
        got = loop_check.repeat_length(samples, RATE, 60 / 97.5, 24)

        self.assertTrue(got["off"])
        self.assertAlmostEqual(got["beats"], 24 * 97.5 / 101.4, delta=0.1)

    def test_a_loop_a_beat_long_is_caught(self):
        samples = _drums(120.0, bars=12)
        got = loop_check.repeat_length(samples, RATE, 0.5, 17)

        self.assertTrue(got["off"])
        self.assertAlmostEqual(got["beats"], 16, delta=0.1)

    def test_nothing_that_repeats_gets_no_opinion(self):
        samples = np.random.default_rng(1).normal(0, 0.2, 30 * RATE)
        self.assertIsNone(loop_check.repeat_length(samples, RATE, 0.5, 16))


class TestCheck(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_it_reads_the_loop_and_what_follows_out_of_a_stem(self):
        # 16-bit stereo, a few bars in, like a real stem.
        lead = np.zeros(int(3.0 * RATE))
        mono = np.concatenate([lead, _drums(120.0, bars=12)])
        stereo = (np.stack([mono, mono], axis=1) * 20000).astype(np.int16)
        path = os.path.join(self.tmp_dir, "stem.wav")
        wavfile.write(path, RATE, stereo)

        got = loop_check.check(path, 3.0, 0.5, 16)

        self.assertAlmostEqual(got["beats"], 16, delta=0.1)
        self.assertFalse(got["off"])

    def test_numpy_and_scipy_only(self):
        # Like pulse.py: liftable without the rest of Beat Thief.
        with open(loop_check.__file__) as source:
            imports = [line.split()[1] for line in source
                       if line.startswith(("import ", "from "))]
        for name in imports:
            self.assertIn(name.split(".")[0], ("__future__", "numpy", "scipy"))


if __name__ == "__main__":
    unittest.main()

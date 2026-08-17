#!/usr/bin/env python3
"""Tests for pulse.py.

Two of these matter more than the rest. The block-invariance test is what
lets this module be described as streaming-capable at all - without it that
is a claim rather than a fact. And the import test is what keeps it liftable
into another program (a lighting rig, one day) rather than quietly growing a
dependency on the rest of Beat Thief.
"""

from __future__ import annotations

import ast
import math
import os
import unittest

import numpy as np

import pulse


RATE = 4000


def _kick(rate=RATE, seconds=0.15, hz=60.0):
    """One low thump: a decaying sine, the way a kick reads under a 120 Hz
    lowpass. Sharp attack at sample zero, so "when it happened" is exact."""
    t = np.arange(int(seconds * rate)) / rate
    return (np.sin(2 * math.pi * hz * t) * np.exp(-t * 25.0)).astype(np.float32)


def _track(times, seconds, rate=RATE):
    """Silence with a kick starting at each of `times`."""
    signal = np.zeros(int(seconds * rate), dtype=np.float32)
    thump = _kick(rate)
    for at in times:
        start = int(round(at * rate))
        end = min(len(signal), start + len(thump))
        signal[start:end] += thump[:end - start]
    return signal


def _four_on_the_floor(tempo, bars, offset, rate=RATE):
    beat = 60.0 / tempo
    times = [offset + n * beat for n in range(bars * 4)]
    return _track(times, offset + bars * 4 * beat + 1.0, rate), times


class TestKicks(unittest.TestCase):
    def test_it_finds_thumps_where_they_are(self):
        times = [0.5, 1.2, 1.9, 2.6, 3.3]
        found = pulse.kicks(_track(times, 4.5), RATE)

        self.assertEqual(len(found), len(times))
        for at, expected in zip(found, times):
            self.assertAlmostEqual(at, expected, delta=0.010)

    def test_nothing_low_is_nothing_found(self):
        rng = np.random.default_rng(7)
        hiss = rng.normal(0, 0.2, int(4.0 * RATE)).astype(np.float32)
        # A lowpass on white noise still passes some low energy, so this is
        # the honest version of "no kicks": noise with no transients in it.
        self.assertEqual(pulse.kicks(hiss, RATE), [])

    def test_silence_is_nothing_found(self):
        self.assertEqual(pulse.kicks(np.zeros(4 * RATE, dtype=np.float32), RATE), [])

    def test_two_thumps_closer_than_the_gap_read_as_one(self):
        found = pulse.kicks(_track([1.0, 1.02], 3.0), RATE)
        self.assertEqual(len(found), 1)


class TestStreaming(unittest.TestCase):
    """Fed in pieces, it must say exactly what it says fed whole."""

    def setUp(self):
        self.times = [0.5, 0.93, 1.7, 2.11, 2.9, 3.44]
        self.signal = _track(self.times, 4.5)
        self.whole = pulse.kicks(self.signal, RATE)
        self.assertEqual(len(self.whole), len(self.times))  # the test is worth running

    def _streamed(self, block):
        detector = pulse.KickDetector(RATE)
        found = []
        for start in range(0, len(self.signal), block):
            found.extend(detector.feed(self.signal[start:start + block]))
        return found

    def test_any_block_size_gives_the_same_answer(self):
        # 1 sample is smaller than the smoothing window; 37 and 101 are prime
        # so they land mid-onset; 4000 is a whole second; 3000 straddles.
        for block in (1, 37, 101, 512, 3000, 4000, 100000):
            with self.subTest(block=block):
                streamed = self._streamed(block)
                self.assertEqual(len(streamed), len(self.whole))
                for got, expected in zip(streamed, self.whole):
                    self.assertAlmostEqual(got, expected, places=6)

    def test_a_block_boundary_on_an_onset_does_not_double_report(self):
        boundary = int(self.times[2] * RATE)
        detector = pulse.KickDetector(RATE)
        found = list(detector.feed(self.signal[:boundary]))
        found.extend(detector.feed(self.signal[boundary:]))

        self.assertEqual(len(found), len(self.whole))

    def test_reset_starts_it_over(self):
        detector = pulse.KickDetector(RATE)
        detector.feed(self.signal)
        detector.reset()
        found = list(detector.feed(self.signal))

        self.assertEqual(len(found), len(self.whole))


class TestGridPhase(unittest.TestCase):
    def test_it_finds_the_offset_a_grid_sits_on(self):
        grid = pulse.GridPhase(2.0, forget=1.0)
        for n in range(16):
            grid.feed(0.35 + n * 2.0)

        self.assertAlmostEqual(grid.phase, 0.35, delta=0.010)

    def test_it_averages_around_the_wrap_not_through_the_middle(self):
        # Hits a hair either side of the line: the answer is ~0, not ~half a
        # period, which is what a plain mean would say.
        grid = pulse.GridPhase(2.0, forget=1.0)
        for n in range(8):
            grid.feed(n * 2.0 + 0.02)
            grid.feed(n * 2.0 - 0.02 + 2.0)

        self.assertLess(min(grid.phase, 2.0 - grid.phase), 0.05)

    def test_scattered_hits_agree_on_nothing(self):
        rng = np.random.default_rng(3)
        grid = pulse.GridPhase(2.0, forget=1.0)
        for at in rng.uniform(0, 60, 200):
            grid.feed(float(at))

        self.assertLess(grid.confidence, 0.25)

    def test_a_steady_grid_is_confident(self):
        grid = pulse.GridPhase(2.0, forget=1.0)
        for n in range(16):
            grid.feed(0.35 + n * 2.0)

        self.assertGreater(grid.confidence, 0.9)

    def test_no_period_means_no_phase(self):
        grid = pulse.GridPhase(0.0)
        grid.feed(1.0)
        self.assertEqual(grid.phase, 0.0)
        self.assertEqual(grid.confidence, 0.0)

    def test_the_next_line_is_the_next_one(self):
        grid = pulse.GridPhase(2.0, forget=1.0)
        for n in range(8):
            grid.feed(0.5 + n * 2.0)

        self.assertAlmostEqual(grid.next_line(0.0), 0.5, delta=0.01)
        self.assertAlmostEqual(grid.next_line(0.5), 2.5, delta=0.01)
        self.assertAlmostEqual(grid.next_line(3.0), 4.5, delta=0.01)

    def test_forgetting_lets_a_moved_grid_win(self):
        grid = pulse.GridPhase(2.0, forget=0.8)
        for n in range(20):
            grid.feed(0.2 + n * 2.0)
        for n in range(20, 60):
            grid.feed(1.2 + n * 2.0)

        self.assertAlmostEqual(grid.phase, 1.2, delta=0.05)


class TestDownbeat(unittest.TestCase):
    def test_it_recovers_the_beat_grid_of_a_four_on_the_floor(self):
        # Every beat carries a kick, so nothing says which one is the "1".
        # The beat grid is still exactly right, and that is what it must
        # return - the earliest candidate, not a random one.
        _, times = _four_on_the_floor(120.0, bars=8, offset=0.37)
        self.assertAlmostEqual(pulse.downbeat(times, 120.0), 0.37, delta=0.010)

    def test_an_asymmetric_pattern_puts_the_bar_line_on_the_one(self):
        # Kick on 1, on 3, and on the "and" of 3: only beat 1 is unambiguous.
        beat = 0.5
        offset = 0.37
        times = []
        for bar in range(8):
            at = offset + bar * 4 * beat
            times += [at, at + 2 * beat, at + 2.5 * beat]

        self.assertAlmostEqual(pulse.downbeat(times, 120.0), offset, delta=0.010)

    def test_it_reads_a_real_signal_end_to_end(self):
        signal, _ = _four_on_the_floor(120.0, bars=8, offset=0.37)
        found = pulse.kicks(signal, RATE)
        self.assertAlmostEqual(pulse.downbeat(found, 120.0), 0.37, delta=0.015)

    def test_no_tempo_is_no_answer(self):
        self.assertEqual(pulse.downbeat([0.5, 1.0, 1.5], 0.0), 0.0)

    def test_too_few_to_go_on_is_no_answer(self):
        self.assertEqual(pulse.downbeat([0.5, 2.5], 120.0), 0.0)

    def test_agreement_on_nothing_is_no_answer(self):
        rng = np.random.default_rng(11)
        scattered = sorted(float(at) for at in rng.uniform(0, 120, 300))
        self.assertEqual(pulse.downbeat(scattered, 120.0), 0.0)

    def test_the_grid_comes_back_with_how_sure_it_is(self):
        _, times = _four_on_the_floor(120.0, bars=8, offset=0.37)
        step, phase, confidence, bar = pulse.fit_grid(times, 120.0)

        self.assertAlmostEqual(step, 0.125, delta=0.002)
        # A sixteenth at 120 BPM is 0.125s, and 0.37 is two of them plus a
        # remainder - the phase is that remainder, not the offset.
        self.assertAlmostEqual(phase, 0.37 % 0.125, delta=0.010)
        self.assertGreater(confidence, 0.9)
        self.assertAlmostEqual(bar, 0.37, delta=0.010)

    def test_it_finds_a_tempo_the_hint_got_wrong(self):
        # The hint is half a percent out, which is the size of error a
        # whole-song estimate really makes. The hits still say 120.
        _, times = _four_on_the_floor(120.0, bars=16, offset=0.37)
        step, _, confidence, _ = pulse.fit_grid(times, 119.4)

        self.assertAlmostEqual(60.0 / (step * 4), 120.0, delta=0.3)
        self.assertGreater(confidence, 0.9)


class TestGridTrack(unittest.TestCase):
    def test_it_covers_the_recording_in_segments(self):
        _, times = _four_on_the_floor(120.0, bars=64, offset=0.37)
        track = pulse.grid_track(times, 120.0)

        self.assertGreater(len(track), 2)
        for grid in track:
            self.assertAlmostEqual(grid.tempo, 120.0, delta=0.5)
            self.assertGreater(grid.confidence, 0.9)
        for earlier, later in zip(track, track[1:]):
            self.assertLessEqual(earlier.end, later.end)

    def test_nothing_to_go_on_is_no_track(self):
        self.assertEqual(pulse.grid_track([1.0, 2.0], 120.0), [])
        self.assertEqual(pulse.grid_track([], 120.0), [])
        _, times = _four_on_the_floor(120.0, bars=16, offset=0.0)
        self.assertEqual(pulse.grid_track(times, 0.0), [])

    def test_the_grid_for_a_moment_is_the_one_covering_it(self):
        _, times = _four_on_the_floor(120.0, bars=64, offset=0.37)
        track = pulse.grid_track(times, 120.0)

        middle = track[1]
        self.assertIs(pulse.grid_at(track, (middle.start + middle.end) / 2), middle)
        # Off either end it falls back to the nearest rather than nothing.
        self.assertIsNotNone(pulse.grid_at(track, -100.0))
        self.assertIsNotNone(pulse.grid_at(track, 10000.0))
        self.assertIsNone(pulse.grid_at([], 1.0))

    def test_a_line_can_be_asked_for_by_time(self):
        _, times = _four_on_the_floor(120.0, bars=32, offset=0.37)
        grid = pulse.grid_track(times, 120.0)[0]

        self.assertAlmostEqual(grid.snap(0.40), 0.37, delta=0.010)
        self.assertAlmostEqual(grid.snap(0.60), 0.37 + grid.step * 2, delta=0.010)


class TestItStaysLiftable(unittest.TestCase):
    """It is meant to be copied into another program whole. That only stays
    true if nothing in Beat Thief creeps into its imports."""

    ALLOWED = {"numpy", "scipy", "math", "__future__", "typing"}

    def test_it_imports_nothing_from_this_app(self):
        source = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pulse.py")
        with open(source) as handle:
            tree = ast.parse(handle.read())

        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])

        self.assertTrue(roots <= self.ALLOWED, f"unexpected imports: {roots - self.ALLOWED}")


if __name__ == "__main__":
    unittest.main()

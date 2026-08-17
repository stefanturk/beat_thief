#!/usr/bin/env python3
"""Tests for audition.py - the one decode that feeds everything the picker
draws, plays and snaps to.

These run real ffmpeg over real (tiny, generated) wav files. The module is
nothing but ffmpeg plumbing, so mocking it out would leave nothing worth
testing.
"""

from __future__ import annotations

import math
import os
import shutil
import struct
import tempfile
import unittest
import wave

import numpy as np

import audition


def _write_wav(path, samples, rate=44100):
    with wave.open(path, "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        clipped = np.clip(samples, -1.0, 1.0)
        handle.writeframes(b"".join(struct.pack("<h", int(v * 32000)) for v in clipped))


def _kicks_at(times, seconds, rate=44100, hz=60.0):
    """A low thump at each of `times`, and silence between."""
    audio = np.zeros(int(seconds * rate))
    shape_t = np.arange(int(0.15 * rate)) / rate
    shape = np.sin(2 * math.pi * hz * shape_t) * np.exp(-shape_t * 25.0)
    for at in times:
        start = int(at * rate)
        end = min(audio.size, start + shape.size)
        audio[start:end] += shape[: end - start]
    return audio


class TestPreview(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        audition._cache.clear()
        self.path = os.path.join(self.tmp_dir, "stem.wav")
        self.times = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        _write_wav(self.path, _kicks_at(self.times, 5.0))

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        audition._cache.clear()

    def test_it_brings_back_everything_the_page_needs(self):
        prepared = audition.preview(self.path)

        self.assertTrue(prepared["audio"].startswith("data:audio/mpeg;base64,"))
        self.assertEqual(len(prepared["peaks"]), audition.PEAK_BUCKETS)
        self.assertAlmostEqual(prepared["duration"], 5.0, delta=0.1)
        self.assertEqual(prepared["path"], self.path)

    def test_the_kicks_it_finds_are_where_the_kicks_are(self):
        found = audition.preview(self.path)["kicks"]

        self.assertEqual(len(found), len(self.times))
        for at, expected in zip(found, self.times):
            self.assertAlmostEqual(at, expected, delta=0.015)

    def test_one_decode_feeds_both_the_waveform_and_the_kicks(self):
        # The peaks and the kicks are two questions about the same samples,
        # and a six-minute stem decoded twice to answer them would be a
        # second of nothing for no reason.
        samples = audition._mono_samples(self.path)
        peaks = audition._peaks(samples)

        self.assertEqual(len(peaks), audition.PEAK_BUCKETS)
        self.assertGreater(max(peaks), 0.9)   # scaled against its own loudest

    def test_silence_draws_flat_and_finds_nothing(self):
        quiet = os.path.join(self.tmp_dir, "quiet.wav")
        _write_wav(quiet, np.zeros(int(3.0 * 44100)))

        prepared = audition.preview(quiet)
        self.assertEqual(prepared["kicks"], [])
        self.assertEqual(max(prepared["peaks"]), 0.0)

    def test_the_listening_copy_lines_up_with_the_stem(self):
        # ffmpeg strips the encoder's own delay back off, so there is
        # nothing to correct - but the point is that this is measured and
        # found to be zero, not assumed to be.
        self.assertAlmostEqual(audition.preview(self.path)["lead"], 0.0, delta=0.003)

    def test_a_copy_that_is_out_by_a_known_amount_is_caught(self):
        # The measurement has to be able to find an offset, or "it found
        # none" means nothing. Same audio, deliberately started 30ms late.
        shifted_path = os.path.join(self.tmp_dir, "shifted.wav")
        _write_wav(shifted_path, _kicks_at([t + 0.030 for t in self.times], 5.0))

        lead = audition._preview_lead(
            audition._mono_samples(self.path), audition._preview_mp3(shifted_path))

        self.assertAlmostEqual(lead, 0.030, delta=0.004)

    def test_two_things_with_nothing_in_common_line_up_nowhere(self):
        rng = np.random.default_rng(5)
        noise_path = os.path.join(self.tmp_dir, "noise.wav")
        _write_wav(noise_path, rng.normal(0, 0.3, int(5.0 * 44100)))

        lead = audition._preview_lead(
            audition._mono_samples(self.path), audition._preview_mp3(noise_path))

        self.assertEqual(lead, 0.0)

    def test_it_is_cached_by_path_and_mtime(self):
        first = audition.preview(self.path)
        self.assertIs(audition.preview(self.path), first)

    def test_a_file_that_is_not_there_is_an_error_worth_reading(self):
        with self.assertRaises(FileNotFoundError):
            audition.preview(os.path.join(self.tmp_dir, "nope.wav"))


if __name__ == "__main__":
    unittest.main()

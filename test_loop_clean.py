"""loop_clean: repair a loop by comparing it with the drummer's other passes."""

import unittest

import numpy as np
from scipy.signal import butter, sosfilt

import loop_clean

RATE = 22050
TEMPO = 100.0
BEAT = 60.0 / TEMPO
BAR = 4 * BEAT
BARS = 26
MARKED_BAR = 12          # the loop starts here and runs two bars
LOOP_BARS = 2
BAR_STARTS = [k * BAR for k in range(BARS)]


def _noise(rng, length, level, decay):
    t = np.arange(length) / RATE
    return rng.normal(0, level, length) * np.exp(-t * decay)


def _stem(mods=None, other_mods=None, bars=BARS, seed=0, parts=("kick", "snare", "hat")):
    """A backbeat played by somebody slightly human. Every hit is fresh noise
    a few ms off the grid. `mods(bar, name, sound)` may damage a hit in the
    marked bar; `other_mods(bar, name, sound)` damages hits in other bars.
    The random draws don't depend on the damage, so a stem built without
    damage is the truth for the same seed."""
    rng = np.random.default_rng(seed)
    out = np.zeros(int((bars * BAR + 3) * RATE))

    def hit(bar, at, name, sound):
        if name not in parts:
            return
        start = int((bar * BAR + at * BEAT + rng.uniform(-0.004, 0.004)) * RATE)
        if mods and bar == MARKED_BAR:
            sound = mods(bar, name, sound)
        elif other_mods and bar != MARKED_BAR:
            sound = other_mods(bar, name, sound)
        out[start:start + len(sound)] += sound[: len(out) - start]

    t = np.arange(int(0.12 * RATE)) / RATE
    for bar in range(bars):
        for when in (0, 2, 2.5):
            hit(bar, when, "kick", 0.9 * np.sin(2 * np.pi * 55 * t) * np.exp(-t * 30))
        for when in (1, 3):
            hit(bar, when, "snare", _noise(rng, int(0.12 * RATE), 0.4, 35))
        for eighth in range(8):
            hats = np.diff(rng.normal(0, 0.2, int(0.03 * RATE) + 1))
            hit(bar, eighth / 2, "hat", hats * np.exp(-np.arange(len(hats)) / RATE * 150))
    return out


def _clean(stem, **kwargs):
    return loop_clean.clean(
        stem.astype(np.float32), RATE, MARKED_BAR * BAR, LOOP_BARS * BAR, kwargs.get("bars", BAR_STARTS))


def _raw(stem):
    start = int(round(MARKED_BAR * BAR * RATE))
    return stem.astype(np.float32)[start:start + int(round(LOOP_BARS * BAR * RATE))]


def _band_db(x, low, high, begin, end):
    sos = butter(4, (low, high), btype="bandpass", fs=RATE, output="sos")
    part = sosfilt(sos, x[int(begin * RATE):int(end * RATE)])
    return 10 * np.log10((part ** 2).mean() + 1e-12)


def _bandstop(low, high):
    sos = butter(4, (low, high), btype="bandstop", fs=RATE, output="sos")
    return lambda sound: sosfilt(sos, sound)


class TestClean(unittest.TestCase):
    def test_an_ad_lib_in_one_pass_is_turned_down_and_the_drums_around_it_stay(self):
        def lib(bar, name, sound):
            return sound
        stem = _stem()
        tone_start = int((MARKED_BAR * BAR + 1.0) * RATE)
        t = np.arange(int(0.6 * RATE)) / RATE
        stem[tone_start:tone_start + len(t)] += 0.2 * np.sin(2 * np.pi * 440 * t)

        cleaned, report = _clean(stem)
        raw = _raw(stem)

        self.assertGreaterEqual(report["references"], 2)
        window = (1.1, 1.5)
        before = _band_db(raw, 400, 480, *window)
        after = _band_db(cleaned, 400, 480, *window)
        self.assertLessEqual(after, before - 10)
        # the second bar was never touched by the ad lib
        second = (BAR + 0.05, 2 * BAR - 0.05)
        self.assertLess(abs(_band_db(cleaned, 100, 9000, *second) - _band_db(raw, 100, 9000, *second)), 1.0)

    def test_a_hollow_snare_gets_its_body_back(self):
        notch = _bandstop(300, 1000)
        stem = _stem(lambda bar, name, s: notch(s) if name == "snare" else s, parts=("kick", "snare"))
        truth = _stem(parts=("kick", "snare"))

        cleaned, report = _clean(stem)

        window = (BEAT + 0.01, BEAT + 0.15)
        wanted = _band_db(_raw(truth), 450, 700, *window)
        self.assertLess(_band_db(_raw(stem), 450, 700, *window), wanted - 10)   # really hollow
        self.assertLess(abs(_band_db(cleaned, 450, 700, *window) - wanted), 3.0)

    def test_phasey_cymbals_lose_their_notches(self):
        def comb(bar, name, sound):
            if name != "hat":
                return sound
            delayed = np.concatenate([np.zeros(4), sound[:-4]])
            return sound + delayed          # notches at 2.7 and 8.3 kHz
        stem = _stem(comb)
        truth = _stem()

        cleaned, _ = _clean(stem)

        window = (0.9, 1.1)                 # hats only, between the kick and the snare
        wanted = _band_db(_raw(truth), 2500, 3000, *window)
        raw_gap = wanted - _band_db(_raw(stem), 2500, 3000, *window)
        clean_gap = wanted - _band_db(cleaned, 2500, 3000, *window)
        self.assertGreater(raw_gap, 6)
        self.assertLess(clean_gap, raw_gap - 6)

    def test_a_crash_that_only_happens_once_is_kept(self):
        stem = _stem()
        rng = np.random.default_rng(5)
        start = int((MARKED_BAR * BAR + 1.5 * BEAT) * RATE)
        crash = _noise(rng, int(1.0 * RATE), 0.4, 4)
        stem[start:start + len(crash)] += crash

        cleaned, _ = _clean(stem)
        raw = _raw(stem)

        window = (1.5 * BEAT, 1.5 * BEAT + 0.9)
        self.assertLess(abs(_band_db(cleaned, 150, 9000, *window) - _band_db(raw, 150, 9000, *window)), 1.0)

    def test_hollow_in_a_different_place_every_pass_is_still_repaired(self):
        rng = np.random.default_rng(9)
        marked = _bandstop(300, 1000)

        def elsewhere(bar, name, sound):
            if name != "snare":
                return sound
            low = rng.uniform(150, 3000)
            return _bandstop(low, low * 1.4)(sound)

        stem = _stem(lambda bar, name, s: marked(s) if name == "snare" else s, elsewhere,
                     parts=("kick", "snare"))
        truth = _stem(parts=("kick", "snare"))

        cleaned, report = _clean(stem)

        window = (BEAT + 0.01, BEAT + 0.15)
        wanted = _band_db(_raw(truth), 450, 700, *window)
        self.assertGreaterEqual(report["references"], 4)
        # Harder than one fixed notch: a quarter of the passes are hollow in this
        # very band, so the donors are noisier. Still ~20 dB better than raw.
        self.assertLess(abs(_band_db(cleaned, 450, 700, *window) - wanted), 5.0)

    def test_nothing_that_repeats_is_left_exactly_as_it_was(self):
        stem = np.random.default_rng(1).normal(0, 0.2, int((BARS * BAR + 3) * RATE))

        cleaned, report = _clean(stem)

        self.assertEqual(report["references"], 0)
        self.assertTrue(np.array_equal(cleaned, _raw(stem)))

    def test_without_bar_lines_there_is_nothing_to_compare_and_nothing_changes(self):
        stem = _stem()

        cleaned, report = _clean(stem, bars=[])

        self.assertEqual(report["references"], 0)
        self.assertTrue(np.array_equal(cleaned, _raw(stem)))

    def test_it_is_the_length_of_the_loop_and_lines_up_with_the_raw_cut(self):
        notch = _bandstop(300, 1000)
        stem = _stem(lambda bar, name, s: notch(s) if name == "snare" else s)
        stereo = np.stack([stem, stem * 0.8], axis=1).astype(np.float32)

        cleaned, _ = loop_clean.clean(stereo, RATE, MARKED_BAR * BAR, LOOP_BARS * BAR, BAR_STARTS)
        raw = _raw(stem)

        self.assertEqual(cleaned.shape, (len(raw), 2))
        self.assertEqual(cleaned.dtype, np.float32)
        corr = np.correlate(cleaned[:20000, 0].astype(np.float64), raw[:20000].astype(np.float64), "full")
        self.assertEqual(int(corr.argmax()) - 19999, 0)

    def test_the_stereo_image_stays_where_it_was(self):
        stem = _stem()
        stereo = np.stack([stem, stem * 0.5], axis=1).astype(np.float32)

        cleaned, _ = loop_clean.clean(stereo, RATE, MARKED_BAR * BAR, LOOP_BARS * BAR, BAR_STARTS)

        left, right = cleaned[:, 0].astype(np.float64), cleaned[:, 1].astype(np.float64)
        self.assertAlmostEqual(float(np.sqrt((right ** 2).sum() / (left ** 2).sum())), 0.5, delta=0.02)

    def test_numpy_and_scipy_only(self):
        with open(loop_clean.__file__) as source:
            imports = [line.split()[1] for line in source
                       if line.startswith(("import ", "from "))]
        for name in imports:
            self.assertIn(name.split(".")[0], ("__future__", "numpy", "scipy", "loop_check"))


if __name__ == "__main__":
    unittest.main()

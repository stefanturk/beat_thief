"""part_finder: break a song into its distinct beats, without naming them."""

import unittest

import numpy as np

import part_finder

RATE = 22050
TEMPO = 100.0
BEAT = 60.0 / TEMPO
BAR = 4 * BEAT

A = dict(kick=(0, 2, 2.5), snare=(1, 3), hats=tuple(i / 2 for i in range(8)))
B = dict(kick=(0, 1.75), snare=(2,), hats=(0, 1, 2, 3))                  # half-time
C = dict(kick=(0, 1, 2, 3), snare=(), hats=tuple(i / 4 for i in range(16)))      # four on the floor
FILL = dict(kick=(0,), snare=tuple(2.5 + i / 4 for i in range(6)), hats=())
REST = dict(kick=(), snare=(), hats=())


def _song(plan, seed=0, jitter_sec=0.004):
    """plan: [(pattern, bars)]. Backbeat-ish drums with fresh noise on every
    hit and a few ms of human timing."""
    rng = np.random.default_rng(seed)
    bars = sum(count for _, count in plan)
    out = np.zeros(int((bars * BAR + 2) * RATE))
    t_kick = np.arange(int(0.12 * RATE)) / RATE

    def hit(at, sound):
        start = int((at + rng.uniform(-jitter_sec, jitter_sec)) * RATE)
        if start >= 0:
            out[start:start + len(sound)] += sound[: len(out) - start]

    bar = 0
    for pattern, count in plan:
        for _ in range(count):
            base = bar * BAR
            for when in pattern["kick"]:
                hit(base + when * BEAT, 0.9 * np.sin(2 * np.pi * 55 * t_kick) * np.exp(-t_kick * 30))
            for when in pattern["snare"]:
                n = int(0.1 * RATE)
                hit(base + when * BEAT, rng.normal(0, 0.3, n) * np.exp(-np.arange(n) / RATE * 40))
            for when in pattern["hats"]:
                n = int(0.03 * RATE)
                hit(base + when * BEAT, np.diff(rng.normal(0, 0.2, n + 1)) * np.exp(-np.arange(n) / RATE * 150))
            bar += 1
    return out, [k * BAR for k in range(bars)]


PLAN = [(A, 8), (B, 8), (A, 8), (FILL, 1), (C, 4), (REST, 2)]


def _parts(plan=PLAN, **kwargs):
    samples, bar_starts = _song(plan, **kwargs)
    features = part_finder.drum_features(samples, RATE, bar_starts)
    return part_finder.find_parts(features, bar_starts), bar_starts


def _bar_of(seconds):
    return int(round(seconds / BAR))


class TestFindParts(unittest.TestCase):
    def test_a_song_of_three_beats_gives_three_parts(self):
        parts, _ = _parts()
        self.assertEqual(len(parts), 3)

    def test_the_beat_that_comes_back_is_one_part_with_two_runs(self):
        parts, _ = _parts()
        biggest = parts[0]
        self.assertEqual(biggest.bars, 16)
        self.assertEqual([(_bar_of(a), _bar_of(b)) for a, b in biggest.runs], [(0, 8), (16, 24)])

    def test_the_other_beats_are_found_where_they_play(self):
        parts, _ = _parts()
        runs = sorted(tuple((_bar_of(a), _bar_of(b)) for a, b in p.runs) for p in parts)
        self.assertEqual(runs, [((0, 8), (16, 24)), ((8, 16),), ((25, 29),)])

    def test_a_fill_and_the_silence_are_not_parts(self):
        parts, _ = _parts()
        covered = {bar for p in parts for a, b in p.runs for bar in range(_bar_of(a), _bar_of(b))}
        self.assertNotIn(24, covered)           # the fill
        self.assertFalse(covered & {29, 30})    # the rests

    def test_each_loop_starts_on_a_bar_line_and_stays_in_its_own_part(self):
        parts, bar_starts = _parts()
        for part in parts:
            start, bars = part.loop
            self.assertIn(round(start, 6), [round(b, 6) for b in bar_starts])
            self.assertGreaterEqual(bars, 4)
            self.assertEqual(bars % 4, 0)
            self.assertTrue(any(a - 1e-6 <= start and start + bars * BAR <= b + 1e-6 for a, b in part.runs))

    def test_a_loop_is_never_shorter_than_four_bars(self):
        parts, _ = _parts()
        self.assertEqual(sorted(p.loop[1] for p in parts), [4, 8, 8])
        # a beat that only ever plays two bars at a time still gets four
        parts, _ = _parts([(A, 8), (B, 2), (A, 8), (B, 2)])
        self.assertTrue(all(p.loop[1] >= 4 for p in parts))

    def test_a_loop_runs_through_the_passes_and_ends_on_the_fill(self):
        fill2 = dict(kick=(0, 1), snare=(2, 2.5, 3, 3.5), hats=())
        plan = [(A, 7), (FILL, 1), (A, 7), (fill2, 1), (A, 7), (FILL, 1)]
        parts, _ = _parts(plan)
        self.assertEqual(len(parts), 1)
        start, bars = parts[0].loop
        self.assertEqual((start, bars), (0.0, 24))          # three passes, three fills
        self.assertAlmostEqual(parts[0].loop_sec, 24 * BAR, places=3)

    def test_a_loop_is_never_longer_than_a_minute(self):
        parts, _ = _parts([(A, 40)])
        self.assertLessEqual(parts[0].loop_sec, part_finder.MAX_LOOP_SEC)
        self.assertEqual(parts[0].loop[1] % 4, 0)
        self.assertGreaterEqual(parts[0].loop[1], 20)

    def test_a_loop_starts_on_a_phrases_one_not_where_the_run_happened_to_start(self):
        # the beat changes on bars 8, 16, 24 (the ones); A's last run starts at
        # 27, three bars after the one, so its loop must wait for bar 28
        parts, _ = _parts([(B, 8), (A, 8), (B, 8), (C, 3), (A, 12)])
        a_loops = [_bar_of(p.loop[0]) for p in parts if any(_bar_of(x) == 27 for x, _ in p.runs)]
        self.assertEqual(len(a_loops), 1)
        self.assertEqual(a_loops[0] % 4, 0)

    def test_a_few_milliseconds_of_timing_does_not_split_a_beat(self):
        for seed in (1, 2, 3):
            parts, _ = _parts(seed=seed, jitter_sec=0.008)
            self.assertEqual(len(parts), 3, seed)

    def test_confidence_is_higher_for_beats_that_are_clearly_different(self):
        parts, _ = _parts()
        self.assertTrue(all(0.0 <= p.confidence <= 1.0 for p in parts))
        self.assertGreater(min(p.confidence for p in parts), 0.2)

    def test_it_works_on_any_features_not_just_drums(self):
        # The reusable core: one vector per bar, no audio.
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=12), rng.normal(size=12)
        rows = [a + rng.normal(0, 0.05, 12) for _ in range(6)] + [b + rng.normal(0, 0.05, 12) for _ in range(6)]
        parts = part_finder.find_parts(np.array(rows), [k * 2.0 for k in range(12)])
        self.assertEqual(sorted(p.bars for p in parts), [6, 6])

    def test_nothing_to_find_is_an_empty_list(self):
        self.assertEqual(part_finder.find_parts(np.zeros((0, 4)), []), [])
        self.assertEqual(part_finder.find_parts(np.zeros((3, 4)), [0.0, 2.0, 4.0]), [])
        silence, bar_starts = _song([(REST, 8)])
        self.assertEqual(part_finder.find_parts(part_finder.drum_features(silence, RATE, bar_starts), bar_starts), [])

    def test_a_part_serialises_for_the_page(self):
        parts, _ = _parts()
        d = parts[0].to_dict()
        self.assertEqual(set(d), {"id", "bars", "runs", "loop_start", "loop_bars", "loop_sec", "confidence"})

    def test_numpy_and_scipy_only(self):
        with open(part_finder.__file__) as source:
            imports = [line.split()[1] for line in source
                       if line.startswith(("import ", "from "))]
        for name in imports:
            self.assertIn(name.split(".")[0], ("__future__", "typing", "numpy", "scipy", "loop_check"))


if __name__ == "__main__":
    unittest.main()

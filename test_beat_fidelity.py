import unittest

import numpy as np

import beat_fidelity as bf
import beat_loop
import beat_writer

SR = 44100
TEMPO = 120.0
# Wide enough apart (1s at this tempo) that a burst's own ring doesn't bleed
# into the next step - a real loop's thirty-second grid is far denser, but
# these tests are about the separation logic, not the grid.
STEPS_PER_BAR = 8


def _burst(length_sec, at_sec, freq, decay_sec=0.02, amplitude=1.0):
    audio = np.zeros(int(length_sec * SR), dtype=np.float32)
    t = np.arange(int(decay_sec * 6 * SR)) / SR
    shape = (np.sin(2 * np.pi * freq * t) * np.exp(-t / decay_sec) * amplitude).astype(np.float32)
    start = int(at_sec * SR)
    end = min(start + shape.size, audio.size)
    audio[start:end] += shape[: end - start]
    return audio


def _jingle(length_sec, at_sec, amplitude=1.0):
    return _burst(length_sec, at_sec, 9000.0, 0.02, amplitude)


def _loop(piece, on_steps, bars=1, steps_per_bar=STEPS_PER_BAR, tempo=TEMPO, origin_sec=0.0):
    hits = tuple(beat_writer.Hit(piece, step, 100) for step in on_steps)
    beat = beat_writer.Beat(tempo=tempo, hits=hits, steps_per_bar=steps_per_bar, bars=bars)
    return beat_loop.Loop(
        beat=beat, bars=bars, origin_sec=origin_sec,
        hits_used=len(hits), hits_dropped=0, hits_inferred=0,
        tempo=tempo, song_tempo=tempo,
    )


class TestSeparation(unittest.TestCase):
    def test_clean_separation_no_suspects(self):
        on_steps = [0, 2, 4, 6]
        loop = _loop("tambourine", on_steps)
        step_sec = loop.beat.seconds_per_step

        audio = np.zeros(int(SR * loop.beat.duration_sec + SR), dtype=np.float32)
        for step in on_steps:
            audio += _jingle(len(audio) / SR, step * step_sec)

        [fidelity] = bf.check(loop, audio, SR)
        self.assertGreater(fidelity.separation_db, 20.0)
        self.assertEqual([s for s in fidelity.steps if s.suspect], [])

    def test_an_on_step_with_no_energy_is_a_suspect(self):
        on_steps = [0, 2, 4, 6]
        loop = _loop("tambourine", on_steps)
        step_sec = loop.beat.seconds_per_step

        audio = np.zeros(int(SR * loop.beat.duration_sec + SR), dtype=np.float32)
        for step in on_steps[1:]:  # step 0 gets nothing
            audio += _jingle(len(audio) / SR, step * step_sec)

        [fidelity] = bf.check(loop, audio, SR)
        suspect_steps = {s.step for s in fidelity.steps if s.suspect}
        self.assertIn(0, suspect_steps)
        self.assertTrue(next(s for s in fidelity.steps if s.step == 0).on)

    def test_an_off_step_with_energy_is_a_suspect(self):
        on_steps = [0, 2, 4, 6]
        loop = _loop("tambourine", on_steps)
        step_sec = loop.beat.seconds_per_step

        audio = np.zeros(int(SR * loop.beat.duration_sec + SR), dtype=np.float32)
        for step in on_steps:
            audio += _jingle(len(audio) / SR, step * step_sec)
        audio += _jingle(len(audio) / SR, 1 * step_sec)  # unexplained hit, no MIDI note

        [fidelity] = bf.check(loop, audio, SR)
        suspect = next(s for s in fidelity.steps if s.step == 1)
        self.assertTrue(suspect.suspect)
        self.assertFalse(suspect.on)

    def test_too_few_steps_gets_no_verdict_and_does_not_crash(self):
        loop = _loop("tambourine", [0, 8], bars=1, steps_per_bar=4)  # 4 total steps
        audio = np.zeros(int(SR * loop.beat.duration_sec + SR), dtype=np.float32)

        [fidelity] = bf.check(loop, audio, SR)
        self.assertIsNone(fidelity.separation_db)

    def test_pieces_sharing_a_band_are_reported_separately(self):
        hits = (beat_writer.Hit("tambourine", 0, 100), beat_writer.Hit("shaker", 8, 100))
        beat = beat_writer.Beat(tempo=TEMPO, hits=hits, steps_per_bar=STEPS_PER_BAR, bars=1)
        loop = beat_loop.Loop(
            beat=beat, bars=1, origin_sec=0.0, hits_used=2, hits_dropped=0, hits_inferred=0,
            tempo=TEMPO, song_tempo=TEMPO,
        )
        audio = np.zeros(int(SR * loop.beat.duration_sec + SR), dtype=np.float32)

        results = {f.piece: f for f in bf.check(loop, audio, SR)}
        self.assertIn("tambourine", results)
        self.assertIn("shaker", results)


if __name__ == "__main__":
    unittest.main()

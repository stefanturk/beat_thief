#!/usr/bin/env python3
"""The maths that turns a marked section into a loop.

None of these run the model. build() is exercised against a faked
transcription, so what's under test is the gridding, not ADTOF."""

import math
import os
import shutil
import tempfile
import unittest
from unittest import mock

import beat_loop
import beat_writer
import pretty_midi


def _note(pitch, start, velocity=100):
    return pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=start + 0.05)


class TestGridOrigin(unittest.TestCase):
    def test_hits_already_on_the_grid_need_no_offset(self):
        step = 0.1
        self.assertAlmostEqual(beat_loop._grid_origin([0.0, 0.1, 0.4, 0.7], step), 0.0, places=6)

    def test_a_consistent_lateness_is_found(self):
        step = 0.1
        late = [0.03, 0.13, 0.43, 0.73]
        self.assertAlmostEqual(beat_loop._grid_origin(late, step), 0.03, places=6)

    def test_it_averages_around_the_wrap_rather_than_through_the_middle(self):
        # Hits at 1% and 99% of a step are 2% apart. Averaging them as
        # plain numbers puts the grid at 50% - exactly out of phase, the
        # worst answer available. As angles it lands near zero.
        step = 0.1
        origin = beat_loop._grid_origin([0.001, 0.099, 0.101, 0.199], step)
        self.assertAlmostEqual(origin, 0.0, delta=0.01)

    def test_the_offset_is_signed_so_early_and_late_read_differently(self):
        # Being a hair early must not read as being a whole step late.
        step = 0.1
        early = beat_loop._grid_origin([0.098, 0.198, 0.398], step)
        late = beat_loop._grid_origin([0.002, 0.102, 0.302], step)
        self.assertLess(early, 0.0)
        self.assertGreater(late, 0.0)
        self.assertLessEqual(abs(early), step / 2)

    def test_nothing_to_go_on_gives_no_offset(self):
        self.assertEqual(beat_loop._grid_origin([], 0.1), 0.0)

    def test_hits_spread_evenly_across_a_step_agree_on_nothing(self):
        # Audio with no pulse should move the grid nowhere rather than
        # somewhere arbitrary.
        step = 0.1
        even = [i * step / 8 for i in range(8)]
        self.assertEqual(beat_loop._grid_origin(even, step), 0.0)


class TestBarCount(unittest.TestCase):
    def test_it_rounds_to_the_nearest_whole_bar(self):
        self.assertEqual(beat_loop._bar_count(2.02, 1.0), 2)
        self.assertEqual(beat_loop._bar_count(1.94, 1.0), 2)

    def test_a_section_shorter_than_a_bar_still_makes_one(self):
        # Somebody who marked a section wants a loop out of it.
        self.assertEqual(beat_loop._bar_count(0.1, 1.0), 1)

    def test_an_overlong_section_is_capped(self):
        self.assertEqual(beat_loop._bar_count(500.0, 1.0), beat_loop.MAX_BARS)


class TestDownbeatRotation(unittest.TestCase):
    """A section marked by ear starts where somebody clicked, so the
    quantized pattern is a rotation of the beat rather than the beat."""

    @staticmethod
    def _placed(hits):
        return {(step, piece): velocity for piece, step, velocity in hits}

    def test_a_beat_already_starting_on_one_is_left_alone(self):
        placed = self._placed([
            ("kick", 0, 110), ("snare", 4, 110), ("kick", 8, 110), ("snare", 12, 110),
        ])
        self.assertEqual(beat_loop._downbeat_rotation(placed, 16), 0)

    def test_a_beat_captured_from_the_backbeat_is_turned_round(self):
        # The same beat, marked starting on two: everything shifted by four
        # sixteenths. The rotation has to undo exactly that.
        placed = self._placed([
            ("snare", 0, 110), ("kick", 4, 110), ("snare", 8, 110), ("kick", 12, 110),
        ])
        self.assertEqual(beat_loop._downbeat_rotation(placed, 16), 12)

    def test_the_downbeat_is_never_placed_off_a_beat(self):
        # A busy hat pattern must not be able to drag the downbeat onto a
        # sixteenth - a downbeat is never a sixteenth off.
        placed = self._placed(
            [("kick", 0, 120), ("snare", 4, 120)]
            + [("closed hat", step, 70) for step in range(1, 16, 2)]
        )
        self.assertEqual(beat_loop._downbeat_rotation(placed, 16) % 4, 0)

    def test_nothing_placed_rotates_nowhere(self):
        self.assertEqual(beat_loop._downbeat_rotation({}, 16), 0)

    def test_a_crash_is_strong_evidence_for_the_downbeat(self):
        placed = self._placed([
            ("crash", 8, 120), ("kick", 8, 120), ("snare", 12, 100), ("snare", 4, 100),
        ])
        self.assertEqual(beat_loop._downbeat_rotation(placed, 16), 8)


class TestBuild(unittest.TestCase):
    """build() against a faked transcription - what's under test is the
    gridding, not the model."""

    TEMPO = 120.0          # a bar is 2.0s, a sixteenth is 0.125s

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.wav = os.path.join(self.tmp_dir, "stem.wav")
        with open(self.wav, "wb") as f:
            f.write(b"not really a wav")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _build(self, notes, start=10.0, span=4.0, lead=beat_loop._CONTEXT_SEC):
        # _section_wav returns how much padding it managed to put on the
        # front; transcribe() then sees the section shifted by that much.
        shifted = [_note(n.pitch, n.start + lead, n.velocity) for n in notes]
        with mock.patch("beat_loop._section_wav", return_value=lead), \
             mock.patch("drum_transcriber.transcribe", return_value=shifted):
            return beat_loop.build(self.wav, self.TEMPO, start, start + span)

    def test_a_straight_beat_comes_back_as_a_straight_beat(self):
        # Kick on one and three, snare on two and four, over two bars.
        notes = []
        for bar in (0.0, 2.0):
            notes += [_note(36, bar + 0.0), _note(38, bar + 0.5),
                      _note(36, bar + 1.0), _note(38, bar + 1.5)]
        loop = self._build(notes)

        self.assertEqual(loop.bars, 2)
        placed = {(hit.step, hit.piece) for hit in loop.beat.hits}
        self.assertEqual(placed, {
            (0, "kick"), (4, "snare"), (8, "kick"), (12, "snare"),
            (16, "kick"), (20, "snare"), (24, "kick"), (28, "snare"),
        })

    def test_hits_slightly_off_the_grid_are_pulled_onto_it(self):
        notes = [_note(36, 0.0), _note(38, 0.51), _note(36, 0.98), _note(38, 1.52)]
        loop = self._build(notes, span=2.0)

        for hit in loop.beat.hits:
            self.assertEqual(hit.step, round(hit.step))

    def test_the_padding_the_model_needed_is_not_part_of_the_loop(self):
        # A hit inside the lead-in is context, not content. If it survived,
        # every loop would start with a phantom note.
        notes = [_note(49, -1.5), _note(36, 0.0), _note(38, 0.5)]
        loop = self._build(notes, span=2.0)

        self.assertNotIn("crash", {hit.piece for hit in loop.beat.hits})

    def test_two_hits_of_one_piece_on_a_step_keep_the_louder(self):
        # A flam is two hits milliseconds apart; on a sixteenth grid it can
        # only be one note, and it should be the one you hear.
        notes = [_note(38, 0.500, 40), _note(38, 0.515, 118)]
        loop = self._build(notes, span=2.0)

        snares = [hit for hit in loop.beat.hits if hit.piece == "snare"]
        self.assertEqual(len(snares), 1)
        self.assertEqual(snares[0].velocity, 118)

    def test_velocity_survives(self):
        notes = [_note(36, 0.0, 120), _note(38, 0.5, 45)]
        loop = self._build(notes, span=2.0)

        by_piece = {hit.piece: hit.velocity for hit in loop.beat.hits}
        self.assertEqual(by_piece["kick"], 120)
        self.assertEqual(by_piece["snare"], 45)

    def test_every_piece_it_emits_is_a_name_beat_writer_knows(self):
        # A note number the map doesn't cover must be dropped, not passed
        # through as a piece name that would raise on write.
        notes = [_note(pitch, i * 0.5) for i, pitch in enumerate((36, 38, 42, 46, 47, 49, 99))]
        loop = self._build(notes)

        for hit in loop.beat.hits:
            self.assertIn(hit.piece, beat_writer.PIECES)
        self.assertGreaterEqual(loop.hits_dropped, 1)

    def test_an_end_before_the_start_is_refused(self):
        with self.assertRaises(ValueError):
            beat_loop.build(self.wav, self.TEMPO, 10.0, 10.0)

    def test_a_section_marked_exactly_right_agrees_with_the_song(self):
        # 4.0s at 120 BPM is exactly two bars, so measuring the section
        # against itself has to land back on the tempo we came in with.
        notes = [_note(36, 0.0), _note(38, 0.5), _note(36, 2.0), _note(38, 2.5)]
        loop = self._build(notes, span=4.0)

        self.assertEqual(loop.bars, 2)
        self.assertAlmostEqual(loop.tempo, self.TEMPO, places=6)
        self.assertAlmostEqual(loop.song_tempo, self.TEMPO, places=6)

    def test_a_section_marked_long_slows_the_loop_down_to_suit(self):
        # 4.4s is still nearest to two bars, but it's two bars of a slower
        # tempo - and the section is what counts. This is the whole point:
        # the beat determines the grid, not the song's average.
        notes = [_note(36, 0.0), _note(38, 0.55), _note(36, 2.2), _note(38, 2.75)]
        loop = self._build(notes, span=4.4)

        self.assertEqual(loop.bars, 2)
        self.assertAlmostEqual(loop.tempo, 240.0 / 2.2, places=6)   # ~109 BPM
        self.assertAlmostEqual(loop.song_tempo, self.TEMPO, places=6)

    def test_the_loop_carries_its_own_tempo_not_the_songs(self):
        # The tempo in the Beat is what reaches the MIDI header and the
        # filename, so it has to be the loop's rather than the stem's.
        notes = [_note(36, 0.0), _note(38, 0.55)]
        loop = self._build(notes, span=4.4)

        self.assertEqual(loop.beat.tempo, loop.tempo)
        self.assertNotAlmostEqual(loop.beat.tempo, self.TEMPO, places=1)

    def test_a_loop_is_exactly_as_long_as_the_section_that_was_marked(self):
        # What makes it loop seamlessly: bars * bar length == the span, by
        # construction, however far off the song's tempo the marking was.
        for span in (2.0, 4.0, 4.4, 7.6):
            with self.subTest(span=span):
                loop = self._build([_note(36, 0.0)], span=span)
                bar_sec = beat_writer.BEATS_PER_BAR * 60.0 / loop.tempo
                self.assertAlmostEqual(loop.bars * bar_sec, span, places=6)

    def test_the_reported_origin_points_back_into_the_stem(self):
        notes = [_note(36, 0.0), _note(38, 0.5), _note(36, 1.0), _note(38, 1.5)]
        loop = self._build(notes, start=10.0, span=2.0)

        self.assertGreaterEqual(loop.origin_sec, 10.0)
        self.assertLess(loop.origin_sec, 12.0)


class TestWrite(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _loop(self, bars=2):
        beat = beat_writer.Beat(
            tempo=120.0, hits=(beat_writer.Hit("kick", 0, 100),),
            steps_per_bar=16, bars=bars, name="Stolen Beat",
        )
        return beat_loop.Loop(
            beat=beat, bars=bars, origin_sec=10.0, hits_used=1, hits_dropped=0,
            tempo=120.0, song_tempo=120.0,
        )

    def test_the_filename_carries_the_tempo_and_the_length(self):
        path = beat_loop.write(self._loop(), self.tmp_dir, "Some Song")

        name = os.path.basename(path)
        self.assertIn("120 BPM", name)
        self.assertIn("2 bars", name)
        self.assertTrue(os.path.exists(path))

    def test_one_bar_is_not_called_1_bars(self):
        path = beat_loop.write(self._loop(bars=1), self.tmp_dir, "Some Song")

        self.assertIn("1 bar)", os.path.basename(path))

    def test_it_makes_the_folder_if_it_is_not_there(self):
        out_dir = os.path.join(self.tmp_dir, "new")

        path = beat_loop.write(self._loop(), out_dir, "Some Song")

        self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()

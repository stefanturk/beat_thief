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


# Steps to the beat. Expectations below are written in these rather than in
# step numbers, so that making the grid finer moves them all together instead
# of failing four tests that are about something else entirely.
BEAT = beat_loop.STEPS_PER_BAR // beat_writer.BEATS_PER_BAR


def _sixteenths(tempo, bars=2):
    """A plain beat played on the grid at `tempo`: hat on every sixteenth,
    kick on one and three, snare on the backbeat.

    Enough onsets to measure a tempo from - refine_tempo wants at least
    instrument_isolator._MIN_ONSETS_FOR_REFINEMENT of them and hands back
    the estimate it was given otherwise."""
    step = 60.0 / tempo / 4
    notes = []
    for step_index in range(bars * 16):
        at = step_index * step
        notes.append(_note(42, at))
        within = step_index % 16
        if within in (0, 8):
            notes.append(_note(36, at))
        elif within in (4, 12):
            notes.append(_note(38, at))
    return notes


def _straight(tempo, bars=4, offset=0.0):
    """A plain backbeat starting `offset` seconds into the section and
    running for `bars` bars: kick on one and three, snare on two and four,
    hats on the eighths.

    Longer than what gets marked, on purpose. A loop moved on to start at a
    kick has to take its last stretch from after the mark, so a fixture that
    stops at the mark can't tell whether it did."""
    beat = 60.0 / tempo
    notes = []
    for bar in range(bars):
        at = offset + bar * 4 * beat
        for eighth in range(8):
            notes.append(_note(42, at + eighth * beat / 2, 70))
        notes += [_note(36, at, 110), _note(38, at + beat, 100),
                  _note(36, at + 2 * beat, 110), _note(38, at + 3 * beat, 100)]
    return notes


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
            (0 * BEAT, "kick"), (1 * BEAT, "snare"),
            (2 * BEAT, "kick"), (3 * BEAT, "snare"),
            (4 * BEAT, "kick"), (5 * BEAT, "snare"),
            (6 * BEAT, "kick"), (7 * BEAT, "snare"),
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
        notes = [_note(36, 0.0, 120), _note(38, 0.5, 105)]
        loop = self._build(notes, span=2.0)

        by_piece = {hit.piece: hit.velocity for hit in loop.beat.hits}
        self.assertEqual(by_piece["kick"], 120)
        self.assertEqual(by_piece["snare"], 105)

    def test_a_quiet_snare_lands_on_the_ghost_pad_with_its_velocity_intact(self):
        notes = [_note(36, 0.0, 120), _note(38, 0.5, 30)]
        loop = self._build(notes, span=2.0)

        by_piece = {hit.piece: hit.velocity for hit in loop.beat.hits}
        self.assertEqual(by_piece["ghost snare"], 30)
        self.assertNotIn("snare", by_piece)

    def test_every_piece_it_emits_is_a_name_beat_writer_knows(self):
        # A note number the map doesn't cover must be dropped, not passed
        # through as a piece name that would raise on write.
        notes = [_note(pitch, i * 0.5) for i, pitch in enumerate((36, 38, 42, 46, 47, 49, 99))]
        loop = self._build(notes)

        for hit in loop.beat.hits:
            self.assertIn(hit.piece, beat_writer.PIECES)
        self.assertGreaterEqual(loop.hits_dropped, 1)

    def test_the_pads_the_snare_class_splits_into_reach_the_loop(self):
        # _PIECE_FOR_NOTE is a whitelist, so a pad missing from it is silently
        # dropped and the whole percussion split would appear to do nothing.
        notes = [_note(36, 0.0), _note(37, 0.5), _note(39, 1.0), _note(36, 2.0)]
        loop = self._build(notes)

        self.assertEqual(loop.hits_dropped, 0)
        for hit in loop.beat.hits:
            self.assertIn(hit.piece, beat_writer.PIECES)

    def test_a_percussion_voice_is_read_off_the_snare_and_completed(self):
        # A tambourine on every quarter, with the two under the backbeat
        # missing because the snare was louder there and only one onset was
        # detected. groove_reader is what puts those back.
        notes = []
        for bar in (0.0, 2.0):
            for beat in range(4):
                at = bar + beat * 0.5
                if beat in (1, 3):
                    notes.append(_note(38, at, 118))    # snare on two and four
                else:
                    notes.append(_note(39, at, 70))     # tambourine elsewhere
            notes.append(_note(36, bar))
        loop = self._build(notes)

        tambourine = {hit.step for hit in loop.beat.hits if hit.piece == "tambourine"}
        self.assertEqual(tambourine, {beat * BEAT for beat in range(8)})
        # ...without taking the backbeat away.
        self.assertEqual({hit.step for hit in loop.beat.hits if hit.piece == "snare"},
                         {beat * BEAT for beat in (1, 3, 5, 7)})
        self.assertEqual(loop.hits_inferred, 4)

    def test_a_beat_with_nothing_to_read_reports_nothing_inferred(self):
        notes = [_note(36, 0.0), _note(38, 0.5), _note(36, 1.0), _note(38, 1.5)]
        loop = self._build(notes, span=2.0)

        self.assertEqual(loop.hits_inferred, 0)

    def test_an_end_before_the_start_is_refused(self):
        with self.assertRaises(ValueError):
            beat_loop.build(self.wav, self.TEMPO, 10.0, 10.0)

    def test_a_loop_is_turned_to_start_on_a_kick(self):
        # Marked a beat early, on a snare, with the kick a beat later - the
        # ordinary result of clicking just before the phrase. A loop is
        # nearly always wanted starting on a kick, so it's turned onto one.
        notes = []
        for bar in (0.0, 2.0):
            notes += [_note(38, bar + 0.0), _note(36, bar + 0.5),
                      _note(38, bar + 1.0), _note(36, bar + 1.5)]
        loop = self._build(notes)

        placed = {(hit.step, hit.piece) for hit in loop.beat.hits}
        self.assertIn((0, "kick"), placed)

    def test_a_loop_that_already_starts_on_a_kick_is_left_alone(self):
        # The turn is only ever a correction. A marking that landed on the
        # one must come through untouched, even when a later kick is harder -
        # a beat whose kick on three is a shade louder than its kick on one
        # is completely ordinary, and turning it round would ruin it.
        notes = []
        for bar in (0.0, 2.0):
            notes += [_note(36, bar + 0.0, 100), _note(38, bar + 0.5),
                      _note(36, bar + 1.0, 108), _note(38, bar + 1.5)]
        loop = self._build(notes)

        placed = {(hit.step, hit.piece) for hit in loop.beat.hits}
        self.assertIn((0, "kick"), placed)
        self.assertIn((BEAT, "snare"), placed)
        self.assertAlmostEqual(loop.origin_sec, 10.0, delta=0.01)

    def test_a_loop_with_no_kick_in_it_is_not_turned(self):
        notes = [_note(38, i * 0.5) for i in range(8)]
        loop = self._build(notes)

        self.assertIn((0, "snare"), {(h.step, h.piece) for h in loop.beat.hits})

    def test_the_first_step_of_the_loop_stays_inside_the_marked_bar(self):
        # origin_sec is where step 0 sits in the stem. Turning onto a kick
        # can move it, but only within one bar of what was marked - never
        # far enough to be a different part of the song.
        notes = [_note(38, 0.0), _note(36, 0.5), _note(38, 1.0), _note(36, 1.5)]
        loop = self._build(notes, start=10.0, span=2.0)

        self.assertGreaterEqual(loop.origin_sec, 10.0 - 0.01)
        self.assertLess(loop.origin_sec, 12.0)

    def test_a_section_marked_exactly_right_agrees_with_the_song(self):
        # 4.0s at 120 BPM is exactly two bars, so measuring the section
        # against itself has to land back on the tempo we came in with.
        notes = [_note(36, 0.0), _note(38, 0.5), _note(36, 2.0), _note(38, 2.5)]
        loop = self._build(notes, span=4.0)

        self.assertEqual(loop.bars, 2)
        self.assertAlmostEqual(loop.tempo, self.TEMPO, places=6)
        self.assertAlmostEqual(loop.song_tempo, self.TEMPO, places=6)

    def test_a_sloppy_edge_does_not_move_the_grid(self):
        # The same drumming, marked 10% long. The tempo comes from the
        # playing, not from where the section was cut, so it doesn't budge.
        #
        # This is the one that matters. Deriving the tempo from span/bars
        # instead put a real two-bar section at 97.5 BPM against a song at
        # 105.4, and a third of its hits landed within a fifth of a step of
        # the halfway line - a coin toss for which sixteenth they got, which
        # is what a stacked kick and snare that should be apart looks like.
        exact = self._build(_sixteenths(self.TEMPO, bars=2), span=4.0)
        sloppy = self._build(_sixteenths(self.TEMPO, bars=2), span=4.4)

        self.assertAlmostEqual(sloppy.tempo, self.TEMPO, delta=1.0)
        self.assertAlmostEqual(sloppy.tempo, exact.tempo, delta=0.5)
        self.assertEqual(sloppy.bars, 2)

    def test_the_loop_carries_the_tempo_of_what_was_played(self):
        # Drumming at 110 inside a song estimated at 120. The tempo in the
        # Beat is what reaches the MIDI header and the filename, so it has
        # to follow the playing rather than the whole-song average.
        loop = self._build(_sixteenths(110.0, bars=2), span=4 * 60.0 / 110.0 * 2)

        self.assertEqual(loop.beat.tempo, loop.tempo)
        self.assertAlmostEqual(loop.tempo, 110.0, delta=1.0)
        self.assertAlmostEqual(loop.song_tempo, self.TEMPO, places=6)

    def test_a_loop_is_a_whole_number_of_bars_of_its_own_tempo(self):
        # What makes it seamless. It's the nearest whole number of bars to
        # what was marked, so it can differ from the marked span by up to
        # half a bar - and that half bar is the sloppy edge, not the music.
        for span in (2.0, 4.0, 4.4, 7.6):
            with self.subTest(span=span):
                loop = self._build(_sixteenths(self.TEMPO, bars=4), span=span)
                bar_sec = beat_writer.BEATS_PER_BAR * 60.0 / loop.tempo
                self.assertLess(abs(loop.bars * bar_sec - span), bar_sec / 2 + 1e-6)

    def test_the_loop_plays_all_the_way_to_its_end(self):
        # Marked two beats early, which is the ordinary way to click. The
        # loop moves on to the kick, and the bars it gains at the end have
        # to come out of the drumming that follows the mark.
        #
        # This is the one that matters. Moving on used to wrap instead, so
        # the silence at the front - the two beats you clicked early - came
        # round to the back, and the loop's last beats were empty while the
        # notes that belonged there had been transcribed and thrown away.
        loop = self._build(_straight(self.TEMPO, bars=4, offset=1.0), span=4.0)

        last = max(hit.step for hit in loop.beat.hits)
        # The last eighth of the last bar.
        self.assertEqual(last, loop.bars * beat_loop.STEPS_PER_BAR - BEAT // 2)

    def test_where_you_clicked_within_the_bar_does_not_change_the_loop(self):
        # The same drumming marked from three different places in the bar.
        # Once each is moved on to its kick they are the same two bars, so
        # they have to come out as the same loop - only origin_sec differs.
        loops = [self._build(_straight(self.TEMPO, bars=4, offset=off), span=4.0)
                 for off in (0.0, 0.5, 1.0)]
        placed = [{(h.step, h.piece, h.velocity) for h in loop.beat.hits} for loop in loops]

        self.assertEqual(placed[1], placed[0])
        self.assertEqual(placed[2], placed[0])
        self.assertAlmostEqual(loops[1].origin_sec - loops[0].origin_sec, 0.5, delta=0.02)
        self.assertAlmostEqual(loops[2].origin_sec - loops[0].origin_sec, 1.0, delta=0.02)

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
            beat=beat, bars=bars, origin_sec=10.0, hits_used=1, hits_dropped=0, hits_inferred=0,
            tempo=120.0, song_tempo=120.0,
        )

    def test_the_filename_is_the_song_and_the_tempo_to_set_ableton_to(self):
        path = beat_loop.write(self._loop(), self.tmp_dir, "Some Song")

        self.assertEqual(os.path.basename(path), "Some Song (Beat at 120 BPM).mid")
        self.assertTrue(os.path.exists(path))

    def test_the_track_inside_is_named_after_the_file(self):
        # Ableton names the clip you drag in after the track, so a constant
        # here means every beat from every song lands in Live called the
        # same thing.
        path = beat_loop.write(self._loop(), self.tmp_dir, "Some Song")

        midi = pretty_midi.PrettyMIDI(path)
        self.assertEqual(midi.instruments[0].name, "Some Song (Beat at 120 BPM)")

    def test_a_second_beats_track_carries_its_own_name(self):
        beat_loop.write(self._loop(), self.tmp_dir, "Some Song")
        second = beat_loop.write(self._loop(), self.tmp_dir, "Some Song")

        midi = pretty_midi.PrettyMIDI(second)
        self.assertEqual(midi.instruments[0].name, "Some Song (Beat at 120 BPM) (2)")

    def test_it_makes_the_folder_if_it_is_not_there(self):
        out_dir = os.path.join(self.tmp_dir, "new")

        path = beat_loop.write(self._loop(), out_dir, "Some Song")

        self.assertTrue(os.path.exists(path))

    def test_a_second_beat_does_not_replace_the_first(self):
        # A verse and a chorus of the same song can easily come out the same
        # number of bars at the same tempo, and the name carries nothing else.
        first = beat_loop.write(self._loop(), self.tmp_dir, "Some Song")
        second = beat_loop.write(self._loop(), self.tmp_dir, "Some Song")

        self.assertNotEqual(first, second)
        self.assertTrue(os.path.exists(first))
        self.assertTrue(os.path.exists(second))
        self.assertIn("(2)", os.path.basename(second))

    def test_it_keeps_counting_past_the_second(self):
        written = [beat_loop.write(self._loop(), self.tmp_dir, "Some Song") for _ in range(4)]

        self.assertEqual(len(set(written)), 4)
        self.assertTrue(all(name.endswith(".mid") for name in written))


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""What a written .mid actually contains.

Everything here reads the file back off disk rather than inspecting the Beat
that produced it - the whole point of this module is what survives the trip
through pretty_midi into a file, so checking the input twice would prove
nothing."""

import os
import shutil
import tempfile
import unittest

import pretty_midi

import beat_writer


class TestPieces(unittest.TestCase):
    def test_every_piece_is_a_pad_on_a_stock_drum_rack(self):
        # A note outside 36-51 exists in the clip but has no pad to play, so
        # it's silent AND invisible - the worst way for a note to be wrong.
        for piece, note in beat_writer.PIECES.items():
            with self.subTest(piece=piece):
                self.assertGreaterEqual(note, beat_writer.RACK_LOW)
                self.assertLessEqual(note, beat_writer.RACK_HIGH)

    def test_no_two_pieces_share_a_pad(self):
        notes = list(beat_writer.PIECES.values())
        self.assertEqual(len(notes), len(set(notes)))

    def test_an_unknown_piece_raises_rather_than_guessing(self):
        # A typo that quietly wrote note 0 would produce a clip that looks
        # right and plays nothing.
        with self.assertRaises(KeyError):
            beat_writer.note_for("cowbell")


class TestGrid(unittest.TestCase):
    def test_a_sixteenth_at_90_bpm_is_one_sixth_of_a_second(self):
        beat = beat_writer.Beat(tempo=90.0, hits=())
        self.assertAlmostEqual(beat.seconds_per_step, 60.0 / 90.0 / 4)

    def test_twelve_steps_a_bar_gives_eighth_note_triplets(self):
        beat = beat_writer.Beat(tempo=120.0, hits=(), steps_per_bar=12)
        # A bar is still four beats long whatever it's divided into.
        self.assertAlmostEqual(beat.seconds_per_step * 12, 4 * 60.0 / 120.0)


class WrittenFileTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _write(self, beat):
        path = os.path.join(self.tmp_dir, "beat.mid")
        beat_writer.write(beat, path)
        return pretty_midi.PrettyMIDI(path)


class TestWrite(WrittenFileTestCase):
    def test_notes_land_on_the_exact_step_they_were_given(self):
        beat = beat_writer.Beat(
            tempo=120.0,
            hits=(
                beat_writer.Hit("kick", 0),
                beat_writer.Hit("snare", 4),
                beat_writer.Hit("closed hat", 6),
            ),
            bars=1,
        )
        midi = self._write(beat)

        step = 60.0 / 120.0 / 4
        starts = [note.start for note in midi.instruments[0].notes]
        for expected, actual in zip([0 * step, 4 * step, 6 * step], starts):
            self.assertAlmostEqual(actual, expected, places=5)

    def test_a_swung_step_keeps_its_fraction(self):
        # Swing and triplets are the reason step is a float. Rounding one to
        # the nearest sixteenth would be exactly the quantization this is
        # meant to be able to avoid.
        #
        # A MIDI file stores times as whole ticks, so "keeps its fraction"
        # can only mean "to within one tick" - see beat_writer.RESOLUTION,
        # which is set high enough to put that under a millisecond.
        beat = beat_writer.Beat(tempo=120.0, hits=(beat_writer.Hit("closed hat", 1.66),), bars=1)
        midi = self._write(beat)

        one_tick = (60.0 / 120.0) / beat_writer.RESOLUTION
        self.assertLess(one_tick, 0.001)
        self.assertAlmostEqual(
            midi.instruments[0].notes[0].start, 1.66 * (60.0 / 120.0 / 4), delta=one_tick
        )

    def test_velocities_survive_the_trip_to_disk(self):
        beat = beat_writer.Beat(
            tempo=90.0,
            hits=tuple(beat_writer.Hit("snare", i, v) for i, v in enumerate((20, 50, 80, 110))),
            bars=1,
        )
        midi = self._write(beat)

        self.assertEqual([n.velocity for n in midi.instruments[0].notes], [20, 50, 80, 110])

    def test_velocity_is_clamped_into_range_rather_than_wrapping(self):
        beat = beat_writer.Beat(
            tempo=90.0,
            hits=(beat_writer.Hit("kick", 0, 400), beat_writer.Hit("snare", 1, 0)),
            bars=1,
        )
        midi = self._write(beat)

        self.assertEqual(sorted(n.velocity for n in midi.instruments[0].notes), [1, 127])

    def test_the_clip_is_a_whole_number_of_bars_even_when_it_ends_in_silence(self):
        # A file only runs as far as its last event, so without help a beat
        # whose last bar is empty would come in short and loop early.
        beat = beat_writer.Beat(tempo=90.0, hits=(beat_writer.Hit("kick", 0),), bars=2)
        midi = self._write(beat)

        bar_seconds = 4 * 60.0 / 90.0
        self.assertAlmostEqual(midi.get_end_time(), 2 * bar_seconds, places=4)

    def test_no_note_runs_past_the_end_of_the_clip(self):
        beat = beat_writer.Beat(
            tempo=90.0, hits=(beat_writer.Hit("crash", 15),), bars=1
        )
        midi = self._write(beat)

        self.assertLessEqual(midi.instruments[0].notes[0].end, beat.duration_sec + 1e-6)

    def test_it_is_one_drum_track(self):
        # Two instruments would import into Ableton as two tracks.
        beat = beat_writer.Beat(tempo=90.0, hits=(beat_writer.Hit("kick", 0),), bars=1)
        midi = self._write(beat)

        self.assertEqual(len(midi.instruments), 1)
        self.assertTrue(midi.instruments[0].is_drum)

    def test_the_tempo_and_time_signature_are_written_out(self):
        beat = beat_writer.Beat(tempo=143.0, hits=(beat_writer.Hit("kick", 0),), bars=1)
        midi = self._write(beat)

        # A MIDI header stores tempo as whole microseconds per beat, so it
        # can't hold every BPM exactly. Close is all the format allows.
        self.assertAlmostEqual(midi.get_tempo_changes()[1][0], 143.0, places=2)
        signature = midi.time_signature_changes[0]
        self.assertEqual((signature.numerator, signature.denominator), (4, 4))

    def test_hits_come_out_in_time_order_whatever_order_they_went_in(self):
        beat = beat_writer.Beat(
            tempo=90.0,
            hits=(beat_writer.Hit("snare", 8), beat_writer.Hit("kick", 0), beat_writer.Hit("kick", 4)),
            bars=1,
        )
        midi = self._write(beat)

        starts = [n.start for n in midi.instruments[0].notes]
        self.assertEqual(starts, sorted(starts))


class TestReferenceBeats(WrittenFileTestCase):
    """The two files that get dragged into Ableton by hand. They're built
    from PIECES, so these keep them from drifting away from it."""

    def test_all_pads_plays_every_piece_exactly_once(self):
        midi = self._write(beat_writer.reference_all_pads())

        played = sorted(n.pitch for n in midi.instruments[0].notes)
        self.assertEqual(played, sorted(beat_writer.PIECES.values()))

    def test_all_pads_walks_up_the_kit_one_beat_at_a_time(self):
        # Anything out of order or doubled here is a mapping bug, and this
        # is the file where it's meant to be obvious.
        beat = beat_writer.reference_all_pads()
        midi = self._write(beat)

        notes = midi.instruments[0].notes
        self.assertEqual([n.pitch for n in notes], sorted(n.pitch for n in notes))
        for earlier, later in zip(notes, notes[1:]):
            self.assertAlmostEqual(later.start - earlier.start, beat.seconds_per_step * 4, places=4)

    def test_all_pads_cycles_the_four_velocities(self):
        midi = self._write(beat_writer.reference_all_pads())

        self.assertEqual([n.velocity for n in midi.instruments[0].notes][:4], [20, 50, 80, 110])

    def test_the_groove_is_two_full_bars(self):
        beat = beat_writer.reference_groove()
        midi = self._write(beat)

        self.assertEqual(beat.bars, 2)
        self.assertAlmostEqual(midi.get_end_time(), 2 * 4 * 60.0 / beat.tempo, places=4)

    def test_the_groove_has_a_backbeat_on_two_and_four(self):
        beat = beat_writer.reference_groove()
        midi = self._write(beat)

        step = beat.seconds_per_step
        snares = {round(n.start / step) for n in midi.instruments[0].notes if n.pitch == beat_writer.PIECES["snare"]}
        self.assertTrue({4, 12, 20, 28} <= snares)

    def test_the_open_hat_replaces_the_closed_one_rather_than_stacking(self):
        # Two hats on the same step is a real mistake that sounds like a
        # flam and looks like a duplicate.
        beat = beat_writer.reference_groove()
        midi = self._write(beat)

        step = beat.seconds_per_step
        hats = beat_writer.PIECES["closed hat"], beat_writer.PIECES["open hat"]
        at_step = [round(n.start / step) for n in midi.instruments[0].notes if n.pitch in hats]
        self.assertEqual(len(at_step), len(set(at_step)))

    def test_no_piece_is_hit_twice_on_the_same_step(self):
        # Two notes of the same pitch at the same instant draw as one note
        # and play as a doubled trigger. Caught exactly this in the groove's
        # second bar, where a pickup kick was written onto a step that
        # already had one.
        for beat in (beat_writer.reference_groove(), beat_writer.reference_all_pads()):
            midi = self._write(beat)
            placed = [(round(n.start / beat.seconds_per_step), n.pitch) for n in midi.instruments[0].notes]
            with self.subTest(beat=beat.name):
                self.assertEqual(len(placed), len(set(placed)))

    def test_every_reference_note_is_on_a_rack_pad(self):
        for beat in (beat_writer.reference_groove(), beat_writer.reference_all_pads()):
            midi = self._write(beat)
            for note in midi.instruments[0].notes:
                with self.subTest(beat=beat.name, pitch=note.pitch):
                    self.assertGreaterEqual(note.pitch, beat_writer.RACK_LOW)
                    self.assertLessEqual(note.pitch, beat_writer.RACK_HIGH)


class TestFilename(unittest.TestCase):
    def test_the_tempo_is_in_the_filename(self):
        # Ableton won't read it out of the file, so this is the only place
        # the number is available when you need to type it into Live.
        beat = beat_writer.Beat(tempo=90.0, hits=())
        self.assertEqual(beat_writer.filename_for(beat, "Reference Groove"), "Reference Groove (90 BPM).mid")

    def test_a_fractional_tempo_is_not_rounded_away(self):
        beat = beat_writer.Beat(tempo=157.6, hits=())
        self.assertIn("157.6", beat_writer.filename_for(beat, "Stolen"))


class TestStolenBeatFilename(unittest.TestCase):
    def test_it_is_the_song_and_the_tempo(self):
        beat = beat_writer.Beat(tempo=104.862, hits=(), bars=4)
        self.assertEqual(
            beat_writer.stolen_beat_filename(beat, "Officially Missing You - Brasstracks"),
            "Officially Missing You - Brasstracks (Beat at 104.862 BPM).mid",
        )

    def test_the_bar_count_is_not_in_it(self):
        # Two brackets deep the tempo was the half that got clipped first,
        # and the bar count is the one number in the name nothing acts on.
        beat = beat_writer.Beat(tempo=120.0, hits=(), bars=4)
        self.assertNotIn("bar", beat_writer.stolen_beat_filename(beat, "Song"))

    def test_a_beat_is_recognised_by_its_filename(self):
        beat = beat_writer.Beat(tempo=120.0, hits=())
        self.assertTrue(beat_writer.is_stolen_beat(beat_writer.stolen_beat_filename(beat, "Song")))

    def test_a_beat_stolen_under_the_old_name_is_still_a_beat(self):
        # Otherwise a folder full of beats taken before the rename reads as
        # a song nobody has ever taken a loop out of.
        self.assertTrue(beat_writer.is_stolen_beat("Song (Stolen Beat, 4 bars) (120 BPM).mid"))

    def test_a_stem_is_not_a_beat(self):
        self.assertFalse(beat_writer.is_stolen_beat("Song (Isolated Drums at 120 BPM).wav"))


class TestWriteReferences(WrittenFileTestCase):
    def test_it_writes_both_files_and_makes_the_folder(self):
        out_dir = os.path.join(self.tmp_dir, "new", "folder")

        written = beat_writer.write_references(out_dir)

        self.assertEqual(len(written), 2)
        for path in written:
            self.assertTrue(os.path.exists(path))
            self.assertTrue(path.endswith(".mid"))


if __name__ == "__main__":
    unittest.main()

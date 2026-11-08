"""Every file's name: label first, and every older name still read."""
import os
import unittest

import names


class TestNewNames(unittest.TestCase):
    def test_isolated_stems_say_what_they_are_first(self):
        self.assertEqual(names.isolated("drums", "Song - Artist", 104.53), "Drums Isolated - Song - Artist - 104.5 BPM")
        self.assertEqual(names.isolated("bass", "Song - Artist", 120.0), "Bass Isolated - Song - Artist - 120 BPM")
        self.assertEqual(names.isolated("vocals", "Song - Artist", 120.0), "Vocals Isolated - Song - Artist")
        self.assertEqual(names.isolated("harmony", "Song - Artist"), "Harmony Isolated - Song - Artist")
        self.assertEqual(names.isolated("instrumental", "Song - Artist"), "Instrumental Isolated - Song - Artist")

    def test_loops_are_beats_of_their_stem(self):
        self.assertEqual(names.beat("drums", "Song - Artist", 104.53), "Drum Beat - Song - Artist - 104.5 BPM")
        self.assertEqual(names.beat("vocals", "Song - Artist", 98, 2), "Vocal Beat - Song - Artist - 98 BPM 2")

    def test_a_song_carries_its_tempo_last(self):
        self.assertEqual(names.song("Song - Artist", 104.53), "Song - Artist - 104.5 BPM")
        self.assertEqual(names.song("Song - Artist (98 BPM)", 104.5), "Song - Artist - 104.5 BPM")

    def test_new_names_read_back(self):
        for stem in names.STEMS:
            n = names.parse(names.beat(stem, "004 - Song - Artist", 104.5, 3) + ".wav")
            self.assertEqual((n.kind, n.stem, n.title, n.tempo, n.nth), ("beat", stem, "004 - Song - Artist", 104.5, 3))
            n = names.parse(names.isolated(stem, "Song - Artist", 120) + ".wav")
            self.assertEqual((n.kind, n.stem, n.title), ("isolated", stem, "Song - Artist"))

    def test_japanese_titles_read_back(self):
        n = names.parse("Drums Isolated - 夜に駆ける - YOASOBI - 130 BPM.wav")
        self.assertEqual((n.stem, n.title, n.tempo), ("drums", "夜に駆ける - YOASOBI", 130.0))


class TestOldNames(unittest.TestCase):
    def test_every_older_name_still_reads(self):
        cases = {
            "Song - Artist (Isolated Drums at 105.373 BPM).wav": ("isolated", "drums", "Song - Artist", 105.373, 1),
            "Song - Artist (Isolated Vocals).wav": ("isolated", "vocals", "Song - Artist", None, 1),
            "Song (Remix) - Artist (Isolated Harmony).wav": ("isolated", "harmony", "Song (Remix) - Artist", None, 1),
            "Song - Artist (Beat at 104.5 BPM).mid": ("beat", "drums", "Song - Artist", 104.5, 1),
            "Song - Artist (Beat at 104.5 BPM) (2).wav": ("beat", "drums", "Song - Artist", 104.5, 2),
            "Song - Artist (Vocals beat at 120 BPM).wav": ("beat", "vocals", "Song - Artist", 120.0, 1),
            "Song - Artist (Stolen Beat, 4 bars) (120 BPM).mid": ("beat", "drums", "Song - Artist", 120.0, 1),
            "Song - Artist (104.5 BPM).mp3": ("song", None, "Song - Artist", 104.5, 1),
        }
        for name, want in cases.items():
            self.assertEqual(tuple(names.parse(name)), want, name)

    def test_a_drum_loop_of_either_age_gives_its_stems_new_names(self):
        self.assertEqual(os.path.basename(names.stem_loop_of("/x/Song (Beat at 100 BPM) (2).wav", "bass")),
                         "Bass Beat - Song - 100 BPM 2.wav")
        self.assertEqual(os.path.basename(names.stem_loop_of("/x/Drum Beat - Song - 100 BPM.wav", "vocals")),
                         "Vocal Beat - Song - 100 BPM.wav")
        with self.assertRaises(ValueError):
            names.stem_loop_of("/x/Bass Beat - Song - 100 BPM.wav", "vocals")


class TestBelongsTo(unittest.TestCase):
    def test_two_songs_in_one_folder_keep_their_own_files(self):
        self.assertTrue(names.belongs_to("Drums Isolated - 001 - One - A - 98 BPM.wav", "001 - One - A"))
        self.assertFalse(names.belongs_to("Drums Isolated - 002 - Two - A - 98 BPM.wav", "001 - One - A"))
        self.assertTrue(names.belongs_to("001 - One - A (Isolated Bass at 98 BPM).wav", "001 - One - A"))
        self.assertFalse(names.belongs_to("001 - One - A - 98 BPM.mp3", "001 - One - A"))


if __name__ == "__main__":
    unittest.main()

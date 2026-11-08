"""Japanese (and any other script) titles: never mistaken for duplicates,
matched on the artist, tidied, and fit to be file names on any disk."""
import unittest

import names
import pipeline
import song_sanitizer
import youtube_match


class TestDuplicates(unittest.TestCase):
    def test_two_songs_by_one_artist_are_not_one_song(self):
        self.assertEqual(song_sanitizer.find_duplicate_pairs(
            ["夜に駆ける - YOASOBI.mp3", "群青 - YOASOBI.mp3"]), [])

    def test_two_titles_with_no_latin_letters_are_not_one_song(self):
        self.assertEqual(song_sanitizer.find_duplicate_pairs(
            ["残酷な天使のテーゼ.mp3", "千本桜.mp3"]), [])

    def test_the_same_japanese_song_twice_still_is(self):
        self.assertEqual(len(song_sanitizer.find_duplicate_pairs(
            ["夜に駆ける - YOASOBI.mp3", "夜に駆ける - YOASOBI - 130 BPM.mp3"])), 1)


class TestMatching(unittest.TestCase):
    def test_a_japanese_artist_is_recognised(self):
        self.assertTrue(youtube_match.same_artist("米津玄師", ["米津玄師"]))
        self.assertTrue(youtube_match.same_artist("米津玄師", ["米津玄師 - Topic"]))
        self.assertFalse(youtube_match.same_artist("米津玄師", ["YOASOBI"]))

    def test_japanese_live_and_cover_uploads_are_other_versions(self):
        for title in ("夜に駆ける (ライブ)", "夜に駆ける 歌ってみた", "夜に駆ける カバー"):
            self.assertTrue(youtube_match.looks_like_another_version(title), title)
        self.assertFalse(youtube_match.looks_like_another_version("夜に駆ける"))


class TestTidyNames(unittest.TestCase):
    def test_fullwidth_tags_come_off(self):
        self.assertEqual(song_sanitizer.clean_title("【MV】夜に駆ける（Official Music Video）"), "夜に駆ける")

    def test_the_song_in_corner_brackets_is_the_title(self):
        title, artist, _ = song_sanitizer._derive_title_artist(
            "YOASOBI「夜に駆ける」Official Music Video - Ayase.mp3")
        self.assertEqual((title, artist), ("夜に駆ける", "YOASOBI"))

    def test_a_slash_splits_only_when_one_side_is_the_channel(self):
        self.assertEqual(song_sanitizer.split_title_artist("夜に駆ける／YOASOBI - YOASOBI"), ("夜に駆ける", "YOASOBI"))
        self.assertEqual(song_sanitizer.split_title_artist("A／B - Someone"), ("A／B", "Someone"))


class TestSafeNames(unittest.TestCase):
    def test_characters_windows_refuses_are_taken_out(self):
        self.assertEqual(pipeline.file_name_for('What? "Now" <1|2>*', "A/B"), "What Now 1 2 - A B")

    def test_a_long_japanese_name_is_cut_on_a_character(self):
        name = names.safe_name("夜に駆ける" * 40)
        self.assertLessEqual(len(name.encode("utf-8")), names.MAX_TITLE_BYTES)
        name.encode("utf-8").decode("utf-8")   # whole characters only
        self.assertTrue(name.startswith("夜に駆ける"))

    def test_japanese_survives_untouched(self):
        self.assertEqual(pipeline.file_name_for("夜に駆ける", "YOASOBI"), "夜に駆ける - YOASOBI")


if __name__ == "__main__":
    unittest.main()

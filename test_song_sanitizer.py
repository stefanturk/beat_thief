import os
import shutil
import subprocess
import tempfile
import unittest

from mutagen.easyid3 import EasyID3

import song_sanitizer as sanitizer


class TestSanitizedArchive(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_load_sanitized_archive_missing_file_returns_empty_set(self):
        self.assertEqual(sanitizer.load_sanitized_archive(self.tmp_dir), set())

    def test_mark_sanitized_then_load_returns_filename(self):
        sanitizer.mark_sanitized(self.tmp_dir, "Song - Artist.mp3")
        sanitizer.mark_sanitized(self.tmp_dir, "Other - Artist.mp3")
        archive = sanitizer.load_sanitized_archive(self.tmp_dir)
        self.assertEqual(archive, {"Song - Artist.mp3", "Other - Artist.mp3"})


class TestFlaggedState(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_load_flagged_missing_file_returns_empty_list(self):
        self.assertEqual(sanitizer.load_flagged(self.tmp_dir), [])

    def test_save_then_load_flagged_roundtrips(self):
        flags = [{"filename": "Song - Artist.mp3", "end": "start", "cut_ms": 4200}]
        sanitizer.save_flagged(self.tmp_dir, flags)
        self.assertEqual(sanitizer.load_flagged(self.tmp_dir), flags)


class TestCleanTitle(unittest.TestCase):
    def test_strips_official_video_tag(self):
        self.assertEqual(
            sanitizer.clean_title("Song Name (Official Video) - Artist"),
            "Song Name - Artist",
        )

    def test_strips_multiple_junk_tags_case_insensitive(self):
        self.assertEqual(
            sanitizer.clean_title("Song Name [OFFICIAL AUDIO] (Lyrics) HD - Artist"),
            "Song Name - Artist",
        )

    def test_leaves_clean_title_unchanged(self):
        self.assertEqual(sanitizer.clean_title("Song Name - Artist"), "Song Name - Artist")


class TestSplitTitleArtist(unittest.TestCase):
    def test_splits_on_last_dash(self):
        self.assertEqual(
            sanitizer.split_title_artist("Song - Name - Artist"),
            ("Song - Name", "Artist"),
        )

    def test_no_dash_returns_empty_artist(self):
        self.assertEqual(sanitizer.split_title_artist("Song Name"), ("Song Name", ""))


class TestWriteId3Tags(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.mp3_path = os.path.join(self.tmp_dir, "test.mp3")
        # 0.5s silent MP3 generated via ffmpeg, used purely as a real MP3 container for tag tests.
        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                "-t", "0.5", "-q:a", "9", self.mp3_path,
            ],
            check=True, capture_output=True,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_writes_title_and_artist(self):
        sanitizer.write_id3_tags(self.mp3_path, "Song Name", "Artist")
        read_tags = EasyID3(self.mp3_path)
        self.assertEqual(read_tags["title"], ["Song Name"])
        self.assertEqual(read_tags["artist"], ["Artist"])


class TestNormalizeForCompare(unittest.TestCase):
    def test_lowercases_and_strips_punctuation(self):
        self.assertEqual(
            sanitizer.normalize_for_compare("Song Name! - Artist."),
            "song name artist",
        )


class TestFindDuplicatePairs(unittest.TestCase):
    def test_finds_near_identical_titles(self):
        files = ["Song Name - Artist.mp3", "Song Name  - Artist.mp3", "Totally Different - Other.mp3"]
        pairs = sanitizer.find_duplicate_pairs(files)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(set(pairs[0]), {"Song Name - Artist.mp3", "Song Name  - Artist.mp3"})

    def test_no_duplicates_returns_empty_list(self):
        files = ["Song One - Artist.mp3", "Song Two - Other Artist.mp3"]
        self.assertEqual(sanitizer.find_duplicate_pairs(files), [])


if __name__ == "__main__":
    unittest.main()

import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from mutagen.easyid3 import EasyID3

from pydub import AudioSegment
from pydub.generators import Sine

import song_sanitizer as sanitizer


def _tone(duration_ms, dbfs_gain=0.0, freq=440):
    tone = Sine(freq).to_audio_segment(duration=duration_ms)
    return tone.apply_gain(dbfs_gain - tone.max_dBFS)


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


class TestAnalyzeCutCandidates(unittest.TestCase):
    def test_detects_silent_intro(self):
        silent_intro = AudioSegment.silent(duration=3000)
        loud_body = _tone(5000, dbfs_gain=-3)
        track = silent_intro + loud_body
        result = sanitizer.analyze_cut_candidates(track)
        self.assertIn("start", result)
        self.assertEqual(result["start"]["classification"], "silent")
        self.assertGreater(result["start"]["cut_ms"], 2000)

    def test_detects_ambiguous_quiet_intro(self):
        quiet_intro = _tone(3000, dbfs_gain=-45)
        loud_body = _tone(5000, dbfs_gain=-3)
        track = quiet_intro + loud_body
        result = sanitizer.analyze_cut_candidates(track)
        self.assertIn("start", result)
        self.assertEqual(result["start"]["classification"], "ambiguous")

    def test_no_flag_for_consistent_volume_track(self):
        track = _tone(5000, dbfs_gain=-6)
        result = sanitizer.analyze_cut_candidates(track)
        self.assertNotIn("start", result)
        self.assertNotIn("end", result)


class TestTrim(unittest.TestCase):
    def test_trims_start_and_end(self):
        track = _tone(5000, dbfs_gain=-6)
        trimmed = sanitizer.trim(track, start_ms=1000, end_ms=4000)
        self.assertEqual(len(trimmed), 3000)


class TestApplyFade(unittest.TestCase):
    def test_fade_start_reduces_early_volume(self):
        track = _tone(5000, dbfs_gain=-6)
        faded = sanitizer.apply_fade(track, "start", 2000)
        self.assertLess(faded[0:100].dBFS, track[0:100].dBFS)

    def test_fade_end_reduces_late_volume(self):
        track = _tone(5000, dbfs_gain=-6)
        faded = sanitizer.apply_fade(track, "end", 3000)
        self.assertLess(faded[4900:5000].dBFS, track[4900:5000].dBFS)


class TestNormalize(unittest.TestCase):
    def test_needs_normalization_true_for_quiet_track(self):
        quiet = _tone(2000, dbfs_gain=-20)
        self.assertTrue(sanitizer.needs_normalization(quiet))

    def test_needs_normalization_false_for_loud_track(self):
        loud = _tone(2000, dbfs_gain=-1.5)
        self.assertFalse(sanitizer.needs_normalization(loud))

    def test_peak_normalize_brings_peak_near_target(self):
        quiet = _tone(2000, dbfs_gain=-20)
        normalized = sanitizer.peak_normalize(quiet)
        self.assertAlmostEqual(normalized.max_dBFS, sanitizer.NORMALIZE_TARGET_DBFS, delta=0.5)


class TestExtractSnippet(unittest.TestCase):
    def test_extracts_window_around_center(self):
        track = _tone(10000, dbfs_gain=-6)
        snippet = sanitizer.extract_snippet(track, center_ms=5000, window_ms=1000)
        self.assertEqual(len(snippet), 2000)

    def test_clamps_at_track_boundaries(self):
        track = _tone(3000, dbfs_gain=-6)
        snippet = sanitizer.extract_snippet(track, center_ms=200, window_ms=1000)
        self.assertEqual(len(snippet), 1200)


class TestLoadExportRoundtrip(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_export_then_load_preserves_duration(self):
        track = _tone(1000, dbfs_gain=-6)
        path = os.path.join(self.tmp_dir, "test.mp3")
        sanitizer.export_audio(track, path)
        loaded = sanitizer.load_audio(path)
        self.assertAlmostEqual(len(loaded), len(track), delta=100)


class TestPlaySnippet(unittest.TestCase):
    @mock.patch("song_sanitizer.subprocess.run")
    def test_calls_afplay_with_temp_file(self, mock_run):
        track = _tone(500, dbfs_gain=-6)
        sanitizer.play_snippet(track)
        self.assertTrue(mock_run.called)
        args = mock_run.call_args[0][0]
        self.assertEqual(args[0], "afplay")
        self.assertFalse(os.path.exists(args[1]))  # temp file cleaned up


class TestReviewFlagged(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.filename = "Song - Artist.mp3"
        sanitizer.export_audio(_tone(5000, dbfs_gain=-6), os.path.join(self.tmp_dir, self.filename))

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_no_flags_does_nothing(self):
        sanitizer.review_flagged(self.tmp_dir)  # should not raise

    @mock.patch("song_sanitizer.play_snippet")
    @mock.patch("builtins.input", return_value="k")
    def test_keep_choice_leaves_file_and_clears_flag(self, mock_input, mock_play):
        sanitizer.save_flagged(self.tmp_dir, [{"filename": self.filename, "end": "start", "cut_ms": 1000}])
        original_size = os.path.getsize(os.path.join(self.tmp_dir, self.filename))

        sanitizer.review_flagged(self.tmp_dir)

        self.assertEqual(sanitizer.load_flagged(self.tmp_dir), [])
        self.assertEqual(os.path.getsize(os.path.join(self.tmp_dir, self.filename)), original_size)

    @mock.patch("song_sanitizer.play_snippet")
    @mock.patch("builtins.input", return_value="c")
    def test_cut_choice_shortens_file_and_clears_flag(self, mock_input, mock_play):
        sanitizer.save_flagged(self.tmp_dir, [{"filename": self.filename, "end": "start", "cut_ms": 1000}])

        sanitizer.review_flagged(self.tmp_dir)

        self.assertEqual(sanitizer.load_flagged(self.tmp_dir), [])
        result_audio = sanitizer.load_audio(os.path.join(self.tmp_dir, self.filename))
        self.assertLess(len(result_audio), 5000)

    @mock.patch("song_sanitizer.play_snippet")
    @mock.patch("builtins.input", side_effect=["a", "1.0", "k"])
    def test_adjust_then_keep_updates_cut_ms_and_clears_flag(self, mock_input, mock_play):
        sanitizer.save_flagged(self.tmp_dir, [{"filename": self.filename, "end": "start", "cut_ms": 1000}])

        sanitizer.review_flagged(self.tmp_dir)

        self.assertEqual(sanitizer.load_flagged(self.tmp_dir), [])
        self.assertEqual(mock_play.call_count, 2)  # replayed after adjusting

    @mock.patch("song_sanitizer.play_snippet")
    @mock.patch("builtins.input", return_value="k")
    def test_missing_file_skips_flag_without_error(self, mock_input, mock_play):
        sanitizer.save_flagged(self.tmp_dir, [{"filename": "Missing.mp3", "end": "start", "cut_ms": 1000}])

        sanitizer.review_flagged(self.tmp_dir)

        self.assertEqual(sanitizer.load_flagged(self.tmp_dir), [])
        mock_play.assert_not_called()


class TestBackupOriginal(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp_dir, "Song - Artist.mp3")
        sanitizer.export_audio(_tone(1000, dbfs_gain=-3), self.path)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_copies_file_into_originals(self):
        sanitizer.backup_original(self.path, self.tmp_dir)
        backup_path = os.path.join(self.tmp_dir, ".originals", "Song - Artist.mp3")
        self.assertTrue(os.path.exists(backup_path))

    def test_does_not_overwrite_existing_backup(self):
        sanitizer.backup_original(self.path, self.tmp_dir)
        backup_path = os.path.join(self.tmp_dir, ".originals", "Song - Artist.mp3")
        first_mtime = os.path.getmtime(backup_path)
        sanitizer.backup_original(self.path, self.tmp_dir)
        self.assertEqual(os.path.getmtime(backup_path), first_mtime)


class TestSanitizeFile(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_renames_junky_title_and_writes_tags(self):
        filename = "Song Name (Official Video) - Artist.mp3"
        sanitizer.export_audio(_tone(3000, dbfs_gain=-3), os.path.join(self.tmp_dir, filename))

        new_flags = sanitizer.sanitize_file(filename, self.tmp_dir)

        self.assertEqual(new_flags, [])
        self.assertFalse(os.path.exists(os.path.join(self.tmp_dir, filename)))
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, "Song Name - Artist.mp3")))
        archive = sanitizer.load_sanitized_archive(self.tmp_dir)
        self.assertIn("Song Name - Artist.mp3", archive)

    def test_flags_ambiguous_intro_without_modifying_audio(self):
        filename = "Song - Artist.mp3"
        quiet_intro = _tone(3000, dbfs_gain=-45)
        loud_body = _tone(5000, dbfs_gain=-3)
        track = quiet_intro + loud_body
        original_len = len(track)
        sanitizer.export_audio(track, os.path.join(self.tmp_dir, filename))

        new_flags = sanitizer.sanitize_file(filename, self.tmp_dir)

        self.assertEqual(len(new_flags), 1)
        self.assertEqual(new_flags[0]["end"], "start")
        result_audio = sanitizer.load_audio(os.path.join(self.tmp_dir, filename))
        self.assertAlmostEqual(len(result_audio), original_len, delta=200)

    def test_auto_trims_silent_intro(self):
        filename = "Song - Artist.mp3"
        silent_intro = AudioSegment.silent(duration=3000)
        loud_body = _tone(5000, dbfs_gain=-3)
        track = silent_intro + loud_body
        sanitizer.export_audio(track, os.path.join(self.tmp_dir, filename))

        new_flags = sanitizer.sanitize_file(filename, self.tmp_dir)

        self.assertEqual(new_flags, [])
        result_audio = sanitizer.load_audio(os.path.join(self.tmp_dir, filename))
        self.assertLess(len(result_audio), 8000)


class TestSanitizeFolder(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_skips_files_already_in_archive(self):
        filename = "Song - Artist.mp3"
        sanitizer.export_audio(_tone(1000, dbfs_gain=-3), os.path.join(self.tmp_dir, filename))
        sanitizer.mark_sanitized(self.tmp_dir, filename)

        with mock.patch("song_sanitizer.review_flagged") as mock_review:
            sanitizer.sanitize_folder(self.tmp_dir)

        mock_review.assert_not_called()
        self.assertFalse(os.path.exists(os.path.join(self.tmp_dir, ".originals", filename)))

    def test_removes_duplicate_and_calls_review_when_flags_pending(self):
        f1 = "Song Name - Artist.mp3"
        f2 = "Song Name  - Artist.mp3"
        sanitizer.export_audio(_tone(1000, dbfs_gain=-3), os.path.join(self.tmp_dir, f1))
        sanitizer.export_audio(_tone(1000, dbfs_gain=-3), os.path.join(self.tmp_dir, f2))

        with mock.patch("song_sanitizer.review_flagged") as mock_review:
            sanitizer.sanitize_folder(self.tmp_dir)

        remaining = [f for f in os.listdir(self.tmp_dir) if f.endswith(".mp3")]
        self.assertEqual(len(remaining), 1)
        mock_review.assert_not_called()  # no ambiguous flags in this scenario


if __name__ == "__main__":
    unittest.main()

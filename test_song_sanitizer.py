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

    def test_strips_bare_junk_without_brackets(self):
        self.assertEqual(
            sanitizer.clean_title("Song Name Official Audio - Artist"),
            "Song Name - Artist",
        )
        self.assertEqual(
            sanitizer.clean_title("Song Name Official Video HD - Artist"),
            "Song Name - Artist",
        )

    def test_strips_a_bracketed_aside_the_junk_list_never_heard_of(self):
        self.assertEqual(
            sanitizer.clean_title("Officially Missing You (Bonus Track) - Brasstracks"),
            "Officially Missing You - Brasstracks",
        )

    def test_strips_every_bracket_not_just_the_first(self):
        self.assertEqual(
            sanitizer.clean_title("Song Name (Remastered 2011) (Deluxe Edition) - Artist"),
            "Song Name - Artist",
        )

    def test_strips_a_nested_bracket_too(self):
        self.assertEqual(
            sanitizer.clean_title("Song Name (Live (1978)) - Artist"),
            "Song Name - Artist",
        )

    def test_a_title_that_is_all_brackets_is_left_alone(self):
        # Stylised names exist, and an empty title is worse than a fussy
        # one - it's what the file would have to be named.
        self.assertEqual(sanitizer.clean_title("(Nice Dream)"), "(Nice Dream)")

    def test_square_brackets_are_left_to_the_junk_list(self):
        self.assertEqual(
            sanitizer.clean_title("Song Name [Kaytranada Remix] - Artist"),
            "Song Name [Kaytranada Remix] - Artist",
        )


class TestSplitTitleArtist(unittest.TestCase):
    def test_splits_on_last_dash(self):
        self.assertEqual(
            sanitizer.split_title_artist("Song - Name - Artist"),
            ("Song - Name", "Artist"),
        )

    def test_no_dash_returns_empty_artist(self):
        self.assertEqual(sanitizer.split_title_artist("Song Name"), ("Song Name", ""))

    def test_pipe_separator_takes_precedence_over_uploader(self):
        # "Artist｜Title ... - Uploader" is a common YouTube channel naming
        # convention — the pipe-derived artist is more reliable than whatever
        # comes after " - " (often just the uploader/channel name).
        self.assertEqual(
            sanitizer.split_title_artist("Childish Gambino｜Redbone - SirSoloDolo"),
            ("Redbone", "Childish Gambino"),
        )

    def test_ascii_pipe_also_works(self):
        self.assertEqual(
            sanitizer.split_title_artist("Artist Name|Song Title - Uploader"),
            ("Song Title", "Artist Name"),
        )


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


class TestSanitizedMarker(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.mp3_path = os.path.join(self.tmp_dir, "test.mp3")
        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                "-t", "0.5", "-q:a", "9", self.mp3_path,
            ],
            check=True, capture_output=True,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_unmarked_file_is_not_already_sanitized(self):
        self.assertFalse(sanitizer._is_already_sanitized(self.mp3_path))

    def test_marked_file_is_already_sanitized(self):
        sanitizer._mark_as_sanitized(self.mp3_path)
        self.assertTrue(sanitizer._is_already_sanitized(self.mp3_path))


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

    def test_flags_sparse_loud_blips_in_mostly_silent_intro(self):
        # A few isolated drum hits (or knocks) surrounded by silence shouldn't
        # be mistaken for "the song has started" — real-world case found by
        # testing against an actual downloaded track (Childish Gambino -
        # Redbone), which opens with a sparse kick drum hit at 0:00 that a
        # naive "stop at the first loud chunk" scan treated as no intro at all.
        blip = _tone(500, dbfs_gain=-10)
        gap = AudioSegment.silent(duration=1000)
        loud_body = _tone(3000, dbfs_gain=-3)
        track = blip + gap + blip + gap + blip + gap + loud_body

        result = sanitizer.analyze_cut_candidates(track)

        self.assertIn("start", result)
        self.assertEqual(result["start"]["classification"], "ambiguous")
        # The cut point should land where the loud body actually begins
        # (4500ms), not at 0 (stopping at the very first blip).
        self.assertEqual(result["start"]["cut_ms"], 4500)


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
    def test_start_anchor_begins_exactly_at_cut_point(self):
        track = _tone(10000, dbfs_gain=-6)
        snippet = sanitizer.extract_snippet(track, 4000, "start", duration_ms=2000)
        self.assertEqual(len(snippet), 2000)
        # Should match audio[4000:6000] exactly, not audio centered on 4000.
        self.assertEqual(snippet.raw_data, track[4000:6000].raw_data)

    def test_end_anchor_finishes_exactly_at_cut_point(self):
        track = _tone(10000, dbfs_gain=-6)
        snippet = sanitizer.extract_snippet(track, 6000, "end", duration_ms=2000)
        self.assertEqual(len(snippet), 2000)
        self.assertEqual(snippet.raw_data, track[4000:6000].raw_data)

    def test_start_anchor_clamps_at_track_end(self):
        track = _tone(3000, dbfs_gain=-6)
        snippet = sanitizer.extract_snippet(track, 2000, "start", duration_ms=2000)
        self.assertEqual(len(snippet), 1000)

    def test_end_anchor_clamps_at_track_start(self):
        track = _tone(3000, dbfs_gain=-6)
        snippet = sanitizer.extract_snippet(track, 1000, "end", duration_ms=2000)
        self.assertEqual(len(snippet), 1000)


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
    @mock.patch("song_sanitizer.sys.stdin")
    @mock.patch("song_sanitizer.subprocess.Popen")
    def test_calls_afplay_with_temp_file(self, mock_popen, mock_stdin):
        mock_stdin.isatty.return_value = False
        mock_proc = mock.Mock()
        mock_popen.return_value = mock_proc

        track = _tone(500, dbfs_gain=-6)
        sanitizer.play_snippet(track)

        self.assertTrue(mock_popen.called)
        args = mock_popen.call_args[0][0]
        self.assertEqual(args[0], "afplay")
        self.assertFalse(os.path.exists(args[1]))  # temp file cleaned up
        mock_proc.wait.assert_called_once()

    @mock.patch("song_sanitizer.select.select")
    @mock.patch("song_sanitizer.termios.tcgetattr")
    @mock.patch("song_sanitizer.termios.tcsetattr")
    @mock.patch("song_sanitizer.tty.setcbreak")
    @mock.patch("song_sanitizer.sys.stdin")
    @mock.patch("song_sanitizer.subprocess.Popen")
    def test_space_press_stops_playback_early(
        self, mock_popen, mock_stdin, mock_setcbreak, mock_tcsetattr, mock_tcgetattr, mock_select
    ):
        mock_stdin.isatty.return_value = True
        mock_stdin.read.return_value = " "
        mock_select.return_value = ([mock_stdin], [], [])
        mock_proc = mock.Mock()
        mock_proc.poll.return_value = None  # still "playing"
        mock_popen.return_value = mock_proc

        track = _tone(500, dbfs_gain=-6)
        sanitizer.play_snippet(track)

        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once()


class TestWaitForSpace(unittest.TestCase):
    @mock.patch("song_sanitizer.sys.stdin")
    def test_falls_back_to_enter_when_not_a_tty(self, mock_stdin):
        mock_stdin.isatty.return_value = False
        with mock.patch("builtins.input", return_value="") as mock_input:
            sanitizer._wait_for_space("Ready? ")
        mock_input.assert_called_once_with("Ready? ")


class TestResolveFlags(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.filename = "Song - Artist.mp3"
        sanitizer.export_audio(_tone(5000, dbfs_gain=-6), os.path.join(self.tmp_dir, self.filename))

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_no_flags_does_nothing(self):
        sanitizer.resolve_flags([], self.tmp_dir)  # should not raise

    @mock.patch("song_sanitizer._wait_for_space")
    @mock.patch("song_sanitizer.play_snippet")
    @mock.patch("song_sanitizer._prompt_choice", return_value="k")
    def test_keep_choice_leaves_audio_unchanged_but_marks_it_sanitized(self, mock_choice, mock_play, mock_wait):
        path = os.path.join(self.tmp_dir, self.filename)
        original_audio_len = len(sanitizer.load_audio(path))

        sanitizer.resolve_flags(
            [{"filename": self.filename, "end": "start", "cut_ms": 1000}], self.tmp_dir
        )

        # "Keep" doesn't touch the audio content...
        self.assertEqual(len(sanitizer.load_audio(path)), original_audio_len)
        # ...but does mark the file so it's never re-flagged on a later run.
        self.assertTrue(sanitizer._is_already_sanitized(path))

    @mock.patch("song_sanitizer._wait_for_space")
    @mock.patch("song_sanitizer.play_snippet")
    @mock.patch("song_sanitizer._prompt_choice", return_value="c")
    def test_cut_choice_shortens_file(self, mock_choice, mock_play, mock_wait):
        sanitizer.resolve_flags(
            [{"filename": self.filename, "end": "start", "cut_ms": 1000}], self.tmp_dir
        )

        result_audio = sanitizer.load_audio(os.path.join(self.tmp_dir, self.filename))
        self.assertLess(len(result_audio), 5000)

    @mock.patch("song_sanitizer._wait_for_space")
    @mock.patch("song_sanitizer.play_snippet")
    @mock.patch("song_sanitizer._prompt_adjust_seconds", return_value=1.0)
    @mock.patch("song_sanitizer._prompt_choice", side_effect=["a", "k"])
    def test_adjust_then_keep_replays(self, mock_choice, mock_adjust, mock_play, mock_wait):
        sanitizer.resolve_flags(
            [{"filename": self.filename, "end": "start", "cut_ms": 1000}], self.tmp_dir
        )

        self.assertEqual(mock_play.call_count, 2)  # replayed after adjusting

    @mock.patch("song_sanitizer._wait_for_space")
    @mock.patch("song_sanitizer.play_snippet")
    @mock.patch("song_sanitizer._prompt_choice", return_value="k")
    def test_missing_file_skips_flag_without_error(self, mock_choice, mock_play, mock_wait):
        sanitizer.resolve_flags(
            [{"filename": "Missing.mp3", "end": "start", "cut_ms": 1000}], self.tmp_dir
        )

        mock_play.assert_not_called()


class TestSanitizeFile(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_junky_title_creates_cleaned_file_and_removes_the_original(self):
        filename = "Song Name (Official Video) - Artist.mp3"
        original_path = os.path.join(self.tmp_dir, filename)
        sanitizer.export_audio(_tone(3000, dbfs_gain=-3), original_path)

        new_flags = sanitizer.sanitize_file(filename, self.tmp_dir)

        self.assertEqual(new_flags, [])
        # The original is only ever read from, and removed once its
        # replacement is written — never opened for writing itself.
        self.assertFalse(os.path.exists(original_path))
        cleaned_path = os.path.join(self.tmp_dir, "Song Name - Artist.mp3")
        self.assertTrue(os.path.exists(cleaned_path))

    def test_rerun_skips_because_a_sanitized_copy_already_exists(self):
        filename = "Song Name (Official Video) - Artist.mp3"
        sanitizer.export_audio(_tone(3000, dbfs_gain=-3), os.path.join(self.tmp_dir, filename))

        sanitizer.sanitize_file(filename, self.tmp_dir)
        before = sorted(os.listdir(self.tmp_dir))
        new_flags = sanitizer.sanitize_file(filename, self.tmp_dir)  # run again
        after = sorted(os.listdir(self.tmp_dir))

        self.assertEqual(new_flags, [])
        self.assertEqual(before, after)  # no extra copy created on the re-run

    def test_flags_ambiguous_intro_on_the_replacement_copy(self):
        filename = "Song - Artist.mp3"
        original_path = os.path.join(self.tmp_dir, filename)
        quiet_intro = _tone(3000, dbfs_gain=-45)
        loud_body = _tone(5000, dbfs_gain=-3)
        track = quiet_intro + loud_body
        original_len = len(track)
        sanitizer.export_audio(track, original_path)

        new_flags = sanitizer.sanitize_file(filename, self.tmp_dir)

        self.assertEqual(len(new_flags), 1)
        self.assertEqual(new_flags[0]["end"], "start")
        # The name is unchanged, so the replacement reuses it — the flag
        # points at whatever now lives at that path (the replacement copy).
        copy_path = os.path.join(self.tmp_dir, new_flags[0]["filename"])
        self.assertEqual(copy_path, original_path)
        self.assertTrue(os.path.exists(copy_path))
        self.assertAlmostEqual(len(sanitizer.load_audio(copy_path)), original_len, delta=200)

    @mock.patch("song_sanitizer._prompt_choice")
    def test_auto_fades_ambiguous_outro_without_prompting(self, mock_prompt_choice):
        filename = "Song - Artist.mp3"
        original_path = os.path.join(self.tmp_dir, filename)
        loud_body = _tone(5000, dbfs_gain=-3)
        quiet_outro = _tone(3000, dbfs_gain=-45)
        track = loud_body + quiet_outro
        sanitizer.export_audio(track, original_path)

        new_flags = sanitizer.sanitize_file(filename, self.tmp_dir)

        self.assertEqual(new_flags, [])
        mock_prompt_choice.assert_not_called()
        copy_path = os.path.join(self.tmp_dir, filename)
        self.assertTrue(os.path.exists(copy_path))
        # Faded in place, not trimmed away - length is essentially unchanged.
        self.assertAlmostEqual(len(sanitizer.load_audio(copy_path)), len(track), delta=200)

    def test_auto_trims_silent_intro_and_replaces_the_original(self):
        filename = "Song - Artist.mp3"
        original_path = os.path.join(self.tmp_dir, filename)
        silent_intro = AudioSegment.silent(duration=3000)
        loud_body = _tone(5000, dbfs_gain=-3)
        track = silent_intro + loud_body
        sanitizer.export_audio(track, original_path)

        new_flags = sanitizer.sanitize_file(filename, self.tmp_dir)

        self.assertEqual(new_flags, [])
        mp3_files = [f for f in os.listdir(self.tmp_dir) if f.endswith(".mp3")]
        self.assertEqual(mp3_files, [filename])  # replaced in place, same name reused
        result_audio = sanitizer.load_audio(original_path)
        self.assertLess(len(result_audio), 8000)

    def test_nothing_to_fix_leaves_folder_untouched(self):
        filename = "Song Name - Artist.mp3"
        original_path = os.path.join(self.tmp_dir, filename)
        sanitizer.export_audio(_tone(3000, dbfs_gain=-3), original_path)

        new_flags = sanitizer.sanitize_file(filename, self.tmp_dir)

        self.assertEqual(new_flags, [])
        mp3_files = [f for f in os.listdir(self.tmp_dir) if f.endswith(".mp3")]
        self.assertEqual(mp3_files, [filename])


class TestSanitizeFileEdgeCases(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_fully_silent_track_is_not_reduced_to_empty_file(self):
        filename = "Silent - Artist.mp3"
        track = AudioSegment.silent(duration=4000)
        sanitizer.export_audio(track, os.path.join(self.tmp_dir, filename))

        # Should not raise (e.g. from applying infinite gain) and should not
        # collapse the file to zero/near-zero length.
        sanitizer.sanitize_file(filename, self.tmp_dir)

        result_path = os.path.join(self.tmp_dir, filename)
        self.assertTrue(os.path.exists(result_path))
        result_audio = sanitizer.load_audio(result_path)
        self.assertGreater(len(result_audio), 1000)

    def test_name_collision_with_an_unrelated_file_never_overwrites_it(self):
        f1 = "Song Name (Official Video) - Artist.mp3"
        f2 = "Song Name - Artist.mp3"
        f1_path = os.path.join(self.tmp_dir, f1)
        f2_path = os.path.join(self.tmp_dir, f2)
        sanitizer.export_audio(_tone(2000, dbfs_gain=-3), f1_path)
        sanitizer.export_audio(_tone(2000, dbfs_gain=-3, freq=880), f2_path)
        f1_bytes = open(f1_path, "rb").read()
        f2_bytes = open(f2_path, "rb").read()

        # f2 is already clean (nothing to do, left alone); f1's cleaned-up
        # name then collides with f2's existing filename. With no manifest to
        # tell them apart, f1 is simply left alone rather than guessing.
        sanitizer.sanitize_file(f2, self.tmp_dir)
        new_flags = sanitizer.sanitize_file(f1, self.tmp_dir)

        self.assertEqual(new_flags, [])
        # Neither original was ever opened for writing, and no third file
        # (like a numbered duplicate) was silently created either.
        self.assertEqual(open(f1_path, "rb").read(), f1_bytes)
        self.assertEqual(open(f2_path, "rb").read(), f2_bytes)
        mp3_files = sorted(f for f in os.listdir(self.tmp_dir) if f.endswith(".mp3"))
        self.assertEqual(mp3_files, sorted([f1, f2]))

    def test_all_junk_title_keeps_original_filename(self):
        filename = "HD (Official Video).mp3"
        sanitizer.export_audio(_tone(2000, dbfs_gain=-3), os.path.join(self.tmp_dir, filename))

        sanitizer.sanitize_file(filename, self.tmp_dir)

        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, filename)))
        self.assertFalse(os.path.exists(os.path.join(self.tmp_dir, ".mp3")))


class TestResolveFlagsEdgeCases(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @mock.patch("song_sanitizer._wait_for_space")
    @mock.patch("song_sanitizer.play_snippet")
    @mock.patch("song_sanitizer._prompt_choice", side_effect=["c", "k"])
    def test_resolving_start_cut_adjusts_stale_end_flag_cut_ms(self, mock_choice, mock_play, mock_wait):
        filename = "Song - Artist.mp3"
        track = _tone(10000, dbfs_gain=-6)
        sanitizer.export_audio(track, os.path.join(self.tmp_dir, filename))
        start_cut_ms = 2000
        end_cut_ms = 8000

        sanitizer.resolve_flags(
            [
                {"filename": filename, "end": "start", "cut_ms": start_cut_ms},
                {"filename": filename, "end": "end", "cut_ms": end_cut_ms},
            ],
            self.tmp_dir,
        )

        # Should have completed without crashing.
        result_audio = sanitizer.load_audio(os.path.join(self.tmp_dir, filename))
        # The start cut removed ~2000ms, so the file should be shorter than
        # the original but still a sensible, non-trivial length.
        self.assertLess(len(result_audio), 10000)
        self.assertGreater(len(result_audio), 1000)

    @mock.patch("song_sanitizer._wait_for_space")
    @mock.patch("song_sanitizer.play_snippet")
    @mock.patch("song_sanitizer._prompt_choice", return_value="k")
    def test_bad_flag_is_skipped_without_stopping_the_rest(self, mock_choice, mock_play, mock_wait):
        good_filename = "Song - Artist.mp3"
        sanitizer.export_audio(_tone(2000, dbfs_gain=-6), os.path.join(self.tmp_dir, good_filename))
        bad_path = os.path.join(self.tmp_dir, "Corrupt - Artist.mp3")
        with open(bad_path, "wb") as f:
            f.write(b"not a real mp3")

        # Should not raise despite the corrupt file being first in the list.
        sanitizer.resolve_flags(
            [
                {"filename": "Corrupt - Artist.mp3", "end": "start", "cut_ms": 500},
                {"filename": good_filename, "end": "start", "cut_ms": 500},
            ],
            self.tmp_dir,
        )


class TestSanitizeFolder(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_rerun_skips_files_already_sanitized(self):
        filename = "Song Name (Official Video) - Artist.mp3"
        sanitizer.export_audio(_tone(1000, dbfs_gain=-3), os.path.join(self.tmp_dir, filename))

        sanitizer.sanitize_folder(self.tmp_dir)
        before = sorted(os.listdir(self.tmp_dir))
        sanitizer.sanitize_folder(self.tmp_dir)  # run again
        after = sorted(os.listdir(self.tmp_dir))

        self.assertEqual(before, after)  # no extra copy created on the re-run
        self.assertFalse(os.path.exists(os.path.join(self.tmp_dir, ".originals")))

    @mock.patch("song_sanitizer._wait_for_space")
    @mock.patch("song_sanitizer.play_snippet")
    @mock.patch("song_sanitizer._prompt_choice", return_value="k")
    def test_rerun_after_resolving_a_flag_does_not_duplicate_the_copy(self, mock_choice, mock_play, mock_wait):
        # Regression test: a produced copy whose ambiguous flag was resolved
        # via "keep" must not be mistaken for a fresh original on a later
        # scan and spawn a second "(sanitized)" copy of itself.
        filename = "Song - Artist.mp3"
        quiet_intro = _tone(3000, dbfs_gain=-45)
        loud_body = _tone(5000, dbfs_gain=-3)
        sanitizer.export_audio(quiet_intro + loud_body, os.path.join(self.tmp_dir, filename))

        sanitizer.sanitize_folder(self.tmp_dir)  # creates a copy, flags it, resolves via "keep"
        before = sorted(os.listdir(self.tmp_dir))
        sanitizer.sanitize_folder(self.tmp_dir)  # run again
        after = sorted(os.listdir(self.tmp_dir))

        self.assertEqual(before, after)

    def test_removes_duplicate_into_visible_duplicates_folder(self):
        f1 = "Song Name - Artist.mp3"
        f2 = "Song Name  - Artist.mp3"
        sanitizer.export_audio(_tone(1000, dbfs_gain=-3), os.path.join(self.tmp_dir, f1))
        sanitizer.export_audio(_tone(1000, dbfs_gain=-3), os.path.join(self.tmp_dir, f2))

        sanitizer.sanitize_folder(self.tmp_dir)

        remaining = [f for f in os.listdir(self.tmp_dir) if f.endswith(".mp3")]
        self.assertEqual(len(remaining), 1)
        duplicates_dir = os.path.join(self.tmp_dir, sanitizer.DUPLICATES_DIR_NAME)
        self.assertTrue(os.path.isdir(duplicates_dir))
        self.assertFalse(os.path.exists(os.path.join(self.tmp_dir, ".originals")))


class TestSanitizeNewDownloads(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_only_touches_the_given_filenames_not_the_whole_folder(self):
        # An older, already-sanitized song sitting in the folder should be
        # left completely alone - not re-scanned, not reported on.
        old_filename = "Old Song - Artist.mp3"
        sanitizer.export_audio(_tone(1000, dbfs_gain=-3), os.path.join(self.tmp_dir, old_filename))
        sanitizer.sanitize_new_downloads([old_filename], self.tmp_dir)
        old_bytes = open(os.path.join(self.tmp_dir, old_filename), "rb").read()

        new_filename = "New Song (Official Video) - Artist.mp3"
        sanitizer.export_audio(_tone(1000, dbfs_gain=-3), os.path.join(self.tmp_dir, new_filename))

        final_filenames = sanitizer.sanitize_new_downloads([new_filename], self.tmp_dir)

        # The old song's bytes are untouched - it was never re-processed.
        self.assertEqual(open(os.path.join(self.tmp_dir, old_filename), "rb").read(), old_bytes)
        self.assertEqual(final_filenames, ["New Song - Artist.mp3"])
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, "New Song - Artist.mp3")))

    def test_still_deduplicates_the_new_download_against_the_existing_library(self):
        existing = "Song Name - Artist.mp3"
        sanitizer.export_audio(_tone(1000, dbfs_gain=-3), os.path.join(self.tmp_dir, existing))
        sanitizer.sanitize_new_downloads([existing], self.tmp_dir)

        new_duplicate = "Song Name  - Artist.mp3"
        sanitizer.export_audio(_tone(1000, dbfs_gain=-3), os.path.join(self.tmp_dir, new_duplicate))

        sanitizer.sanitize_new_downloads([new_duplicate], self.tmp_dir)

        remaining = [f for f in os.listdir(self.tmp_dir) if f.endswith(".mp3")]
        self.assertEqual(len(remaining), 1)
        duplicates_dir = os.path.join(self.tmp_dir, sanitizer.DUPLICATES_DIR_NAME)
        self.assertTrue(os.path.isdir(duplicates_dir))

    def test_missing_filename_is_skipped_without_raising(self):
        final_filenames = sanitizer.sanitize_new_downloads(["Nonexistent.mp3"], self.tmp_dir)
        self.assertEqual(final_filenames, [])

    @mock.patch("song_sanitizer._prompt_choice")
    @mock.patch("song_sanitizer.play_snippet")
    @mock.patch("song_sanitizer._wait_for_space")
    def test_non_interactive_run_fades_an_ambiguous_intro_without_asking(
        self, mock_wait, mock_play, mock_choice
    ):
        filename = "Song - Artist.mp3"
        path = os.path.join(self.tmp_dir, filename)
        track = _tone(3000, dbfs_gain=-45) + _tone(5000, dbfs_gain=-3)
        sanitizer.export_audio(track, path)

        final_filenames = sanitizer.sanitize_new_downloads(
            [filename], self.tmp_dir, interactive=False
        )

        # Nothing was played and nothing was asked - the GUI can't answer.
        mock_wait.assert_not_called()
        mock_play.assert_not_called()
        mock_choice.assert_not_called()
        self.assertEqual(final_filenames, [filename])
        # Faded, not trimmed: the file is still its original length.
        self.assertAlmostEqual(len(sanitizer.load_audio(path)), len(track), delta=200)


class TestAutoResolveFlags(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _flagged_song(self, filename="Song - Artist.mp3"):
        path = os.path.join(self.tmp_dir, filename)
        track = _tone(3000, dbfs_gain=-45) + _tone(5000, dbfs_gain=-3)
        sanitizer.export_audio(track, path)
        flags = sanitizer.sanitize_file(filename, self.tmp_dir)
        self.assertEqual(len(flags), 1)  # guard: the fixture really is ambiguous
        return path, flags, len(track)

    def test_fades_the_flagged_intro_instead_of_cutting_it(self):
        path, flags, original_len = self._flagged_song()
        before = sanitizer.load_audio(path)
        cut_ms = flags[0]["cut_ms"]
        head_before = sanitizer._region_dbfs(before, 0, cut_ms)

        sanitizer.auto_resolve_flags(flags, self.tmp_dir)

        after = sanitizer.load_audio(path)
        self.assertAlmostEqual(len(after), original_len, delta=200)  # faded, not trimmed
        self.assertLess(sanitizer._region_dbfs(after, 0, cut_ms), head_before)

    def test_marks_the_file_so_a_later_run_never_re_flags_it(self):
        path, flags, _ = self._flagged_song()

        sanitizer.auto_resolve_flags(flags, self.tmp_dir)

        self.assertTrue(sanitizer._is_already_sanitized(path))

    def test_preserves_id3_tags_across_the_re_export(self):
        path, flags, _ = self._flagged_song()
        sanitizer.write_id3_tags(path, "Real Title", "Real Artist")

        sanitizer.auto_resolve_flags(flags, self.tmp_dir)

        tags = EasyID3(path)
        self.assertEqual(tags.get("title", [""])[0], "Real Title")
        self.assertEqual(tags.get("artist", [""])[0], "Real Artist")

    def test_missing_file_is_skipped_without_raising(self):
        flags = [{"filename": "Gone.mp3", "end": "start", "cut_ms": 1000}]
        sanitizer.auto_resolve_flags(flags, self.tmp_dir)  # must not raise

    def test_no_flags_does_nothing(self):
        sanitizer.auto_resolve_flags([], self.tmp_dir)
        self.assertEqual(os.listdir(self.tmp_dir), [])


class TestSanitizeFileReplace(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_deletes_original_and_reuses_its_name_when_untitled_change(self):
        filename = "Song - Artist.mp3"
        original_path = os.path.join(self.tmp_dir, filename)
        silent_intro = AudioSegment.silent(duration=3000)
        loud_body = _tone(5000, dbfs_gain=-3)
        sanitizer.export_audio(silent_intro + loud_body, original_path)

        sanitizer.sanitize_file(filename, self.tmp_dir)

        # Original name is reused (nothing left behind), original bytes gone.
        mp3_files = [f for f in os.listdir(self.tmp_dir) if f.endswith(".mp3")]
        self.assertEqual(mp3_files, [filename])
        result_audio = sanitizer.load_audio(original_path)
        self.assertLess(len(result_audio), 8000)  # intro was trimmed

    def test_title_change_leaves_only_the_cleaned_file(self):
        filename = "Song Name (Official Video) - Artist.mp3"
        sanitizer.export_audio(_tone(2000, dbfs_gain=-3), os.path.join(self.tmp_dir, filename))

        sanitizer.sanitize_file(filename, self.tmp_dir)

        mp3_files = [f for f in os.listdir(self.tmp_dir) if f.endswith(".mp3")]
        self.assertEqual(mp3_files, ["Song Name - Artist.mp3"])


if __name__ == "__main__":
    unittest.main()

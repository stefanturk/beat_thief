import os
import shutil
import tempfile
import unittest
from unittest import mock

import beat_writer
import history
import instrument_isolator
import pipeline
import song_sanitizer
import spotify


class _FakeYoutubeDL:
    """Stands in for yt_dlp.YoutubeDL: fires the hooks a real download would
    fire, without any network. Set `entries` to control what the url looks
    like it resolves to."""

    entries = [{"title": "Some Song"}]
    # What a particular url resolves to, for tests with a queue of them -
    # a playlist downloads one url at a time and each has to be its own
    # song, or every track would arrive under the same filename.
    by_url = {}
    downloads = []

    @classmethod
    def _entries_for(cls, url):
        return list(cls.by_url.get(url, cls.entries))

    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    # What yt-dlp calls a playlist, at the top level of its info - empty
    # for everything but the tests that are about a playlist's name.
    playlist_title = ""

    def extract_info(self, url, download=False):
        return {"entries": self._entries_for(url), "title": self.playlist_title}

    def _stem(self, entry):
        """What yt-dlp would call the file: the upload's title and channel,
        unless the run named it (a song Spotify already named)."""
        template = os.path.basename(self.opts.get("outtmpl", ""))
        if "%(title)s" in template:
            return entry["title"] + " - Artist"
        return template[:-len(".%(ext)s")].replace("%%", "%")

    def prepare_filename(self, entry):
        return os.path.join(
            os.path.dirname(self.opts.get("outtmpl", "")), self._stem(entry) + ".webm"
        )

    def download(self, urls):
        output_dir = os.path.dirname(self.opts["outtmpl"])
        entries = self._entries_for(urls[0] if urls else None)
        for number, entry in enumerate(entries, start=1):
            filename = self._stem(entry) + ".mp3"
            path = os.path.join(output_dir, filename)
            with open(path, "wb") as f:
                f.write(b"fake mp3")
            # The mp3 is what ends up on disk, but ExtractAudio's hook
            # reports the file it converted *from* - the .webm or .mp4 that
            # came down the wire, which no longer exists by the time the
            # hook fires. Faking the mp3 here hid a real bug for months:
            # the sanitizer was being handed a filename that wasn't there.
            info = {"title": entry["title"], "filepath": os.path.splitext(path)[0] + ".webm"}
            if len(entries) > 1:
                info["playlist_index"] = number
            for hook in self.opts.get("progress_hooks", []):
                hook({"status": "downloading", "info_dict": info,
                      "total_bytes": 100, "downloaded_bytes": 50})
                hook({"status": "finished", "info_dict": info})
            for hook in self.opts.get("postprocessor_hooks", []):
                hook({"status": "finished", "postprocessor": "ExtractAudio", "info_dict": info})
        _FakeYoutubeDL.downloads.append(urls)
        return 0


class PipelineTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        _FakeYoutubeDL.downloads = []
        _FakeYoutubeDL.by_url = {}
        self.events = []
        patcher = mock.patch("yt_dlp.YoutubeDL", _FakeYoutubeDL)
        patcher.start()
        self.addCleanup(patcher.stop)
        # Real sanitizing needs real audio; these tests are about
        # orchestration, so the sanitizer is stood in for throughout.
        sanitize = mock.patch(
            "song_sanitizer.sanitize_new_downloads",
            side_effect=lambda filenames, output_dir, interactive=True, review=None, renamed=None, **_: list(filenames),
        )
        self.mock_sanitize = sanitize.start()
        self.addCleanup(sanitize.stop)
        # Every run through the pipeline records what it downloaded, and
        # without this that lands in the real history file in Application
        # Support. It caps at 200 entries, so a few test runs were enough
        # to push every song the user actually owns out of it and empty
        # the app's Files panel.
        recorded = mock.patch("history.HISTORY_PATH", os.path.join(self.tmp_dir, "history.json"))
        recorded.start()
        self.addCleanup(recorded.stop)
        # No YouTube Music release by default, so every Spotify test goes
        # through the ordinary search it mocks - and none of them touch the
        # network. The tests about the release patch this back.
        released = mock.patch("youtube_match.music_match", return_value=None)
        self.mock_released = released.start()
        self.addCleanup(released.stop)
        # Reading a tempo needs real audio too, so by default there isn't
        # one and names stay as the sanitizer left them. The tests about
        # the tempo in the name patch this back.
        tempo = mock.patch("pipeline._read_alignment", return_value=None)
        self.mock_tempo = tempo.start()
        self.addCleanup(tempo.stop)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _run(self, **kwargs):
        kwargs.setdefault("output_dir", self.tmp_dir)
        kwargs.setdefault("on_event", self.events.append)
        return pipeline.run("https://example.com/song", **kwargs)

    def _stages(self):
        return [e["stage"] for e in self.events]


class TestReviewingATrim(PipelineTestCase):
    """An ambiguous intro or outro has somewhere to go for an answer."""

    def test_a_reviewer_reaches_the_sanitizer(self):
        def review(flag):
            return {"action": "keep"}

        self._run(on_review=review)

        self.assertIs(self.mock_sanitize.call_args.kwargs["review"], review)

    def test_without_one_the_sanitizer_is_left_to_decide_alone(self):
        self._run()

        self.assertIsNone(self.mock_sanitize.call_args.kwargs["review"])


class TestSongOnlyRun(PipelineTestCase):
    def test_downloads_the_song_and_reports_it(self):
        result = self._run()

        self.assertEqual(result["downloaded"], 1)
        self.assertEqual(result["failed"], 0)
        self.assertEqual([os.path.basename(p) for p in result["songs"]], ["Some Song - Artist.mp3"])
        self.assertIn("done", self._stages())

    def test_reports_progress_in_order(self):
        self._run()

        stages = self._stages()
        self.assertLess(stages.index("looking-up"), stages.index("downloading"))
        self.assertLess(stages.index("downloading"), stages.index("download-summary"))
        self.assertEqual(stages[-1], "done")

    def test_no_instruments_means_no_isolation(self):
        with mock.patch("drum_isolator.isolate_drums_for_single_file") as mock_drums:
            self._run()

        mock_drums.assert_not_called()

    def test_works_without_an_event_callback(self):
        result = pipeline.run("https://example.com/song", output_dir=self.tmp_dir)

        self.assertEqual(result["downloaded"], 1)

    def test_found_names_the_song_for_a_single_song_link(self):
        self._run()

        found = next(e for e in self.events if e["stage"] == "found")
        self.assertEqual(found["song"], "Some Song")

    def test_found_does_not_name_a_song_for_a_playlist(self):
        _FakeYoutubeDL.entries = [{"title": "One"}, {"title": "Two"}]
        self.addCleanup(setattr, _FakeYoutubeDL, "entries", [{"title": "Some Song"}])

        self._run()

        found = next(e for e in self.events if e["stage"] == "found")
        self.assertIsNone(found["song"])


class TestInstrumentRuns(PipelineTestCase):
    def test_isolates_each_requested_instrument_for_the_song(self):
        with mock.patch("drum_isolator.isolate_drums_for_single_file") as mock_drums, \
             mock.patch("harmony_isolator.isolate_harmony_for_single_file") as mock_harmony:
            self._run(instruments=["harmony", "drums"])

        expected = os.path.join(self.tmp_dir, "Some Song - Artist", "Some Song - Artist.mp3")
        self.assertEqual(mock_drums.call_args.args[0], expected)
        self.assertEqual(mock_harmony.call_args.args[0], expected)

    def test_instruments_run_in_a_fixed_order_regardless_of_request_order(self):
        order = []
        with mock.patch("drum_isolator.isolate_drums_for_single_file",
                        side_effect=lambda *a, **k: order.append("drums")), \
             mock.patch("bass_isolator.isolate_bass_for_single_file",
                        side_effect=lambda *a, **k: order.append("bass")), \
             mock.patch("harmony_isolator.isolate_harmony_for_single_file",
                        side_effect=lambda *a, **k: order.append("harmony")), \
             mock.patch("vocals_isolator.isolate_vocals_for_single_file",
                        side_effect=lambda *a, **k: order.append("vocals")):
            self._run(instruments=["vocals", "harmony", "bass", "drums"])

        self.assertEqual(order, ["drums", "bass", "harmony", "vocals"])

    def test_isolates_vocals_when_asked_for(self):
        with mock.patch("vocals_isolator.isolate_vocals_for_single_file") as mock_vocals:
            self._run(instruments=["vocals"])

        self.assertEqual(
            mock_vocals.call_args.args[0], os.path.join(self.tmp_dir, "Some Song - Artist", "Some Song - Artist.mp3")
        )

    def test_non_interactive_choice_reaches_both_the_sanitizer_and_the_isolators(self):
        with mock.patch("drum_isolator.isolate_drums_for_single_file") as mock_drums:
            self._run(instruments=["drums"], interactive=False)

        self.assertIs(self.mock_sanitize.call_args.kwargs["interactive"], False)
        self.assertIs(mock_drums.call_args.kwargs["context"].interactive, False)

    def test_isolation_progress_is_reported_per_instrument(self):
        def fake_isolate(path, context=None):
            context.on_percent(40)
            context.on_percent(100)

        with mock.patch("drum_isolator.isolate_drums_for_single_file", side_effect=fake_isolate):
            self._run(instruments=["drums"])

        percents = [e["percent"] for e in self.events if e["stage"] == "isolating"]
        self.assertEqual(percents, [None, 40, 100])
        isolating = [e for e in self.events if e["stage"] == "isolating"]
        self.assertEqual(isolating[0]["instrument"], "drums")
        self.assertEqual(isolating[0]["song"], "Some Song - Artist")

    def test_produced_files_are_collected_as_outputs(self):
        song_dir = os.path.join(self.tmp_dir, "Some Song - Artist")
        wav = os.path.join(song_dir, "Some Song - Artist (Isolated Drums at 120.000 BPM).wav")

        def fake_isolate(path, context=None):
            os.makedirs(song_dir, exist_ok=True)
            with open(wav, "wb") as f:
                f.write(b"x")

        with mock.patch("drum_isolator.isolate_drums_for_single_file", side_effect=fake_isolate):
            result = self._run(instruments=["drums"])

        self.assertIn(wav, result["outputs"])
        isolated = [e for e in self.events if e["stage"] == "isolated"][0]
        self.assertEqual(isolated["outputs"], [wav])

    def test_a_leftover_mid_from_an_older_version_is_not_offered_as_output(self):
        # Isolators write nothing but a wav now. A .mid sitting next to one
        # is a stale file from a version that wrote whole-song MIDI, and
        # handing it back as something just produced would be a lie.
        song_dir = os.path.join(self.tmp_dir, "Some Song - Artist")
        basename = "Some Song - Artist (Isolated Drums at 120.000 BPM)"
        wav = os.path.join(song_dir, basename + ".wav")
        stale_mid = os.path.join(song_dir, basename + ".mid")

        def fake_isolate(path, context=None):
            os.makedirs(song_dir, exist_ok=True)
            for target in (wav, stale_mid):
                with open(target, "wb") as f:
                    f.write(b"x")

        with mock.patch("drum_isolator.isolate_drums_for_single_file", side_effect=fake_isolate):
            result = self._run(instruments=["drums"])

        isolated = [e for e in self.events if e["stage"] == "isolated"][0]
        self.assertEqual(isolated["outputs"], [wav])
        self.assertNotIn(stale_mid, result["outputs"])


class TestAlreadyDownloadedSongs(PipelineTestCase):
    def test_a_song_skipped_by_the_archive_is_still_isolated(self):
        # yt-dlp fires no hooks for a song it skips, so the fresh-download
        # list is empty - but asking to isolate it is still a real request.
        class SkippingYoutubeDL(_FakeYoutubeDL):
            def download(self, urls):
                return 0

        existing = os.path.join(self.tmp_dir, "Some Song - Artist", "Some Song - Artist.mp3")
        os.makedirs(os.path.dirname(existing))
        with open(existing, "wb") as f:
            f.write(b"already here")

        with mock.patch("yt_dlp.YoutubeDL", SkippingYoutubeDL), \
             mock.patch("drum_isolator.isolate_drums_for_single_file") as mock_drums:
            result = self._run(instruments=["drums"])

        self.assertEqual(result["downloaded"], 0)
        mock_drums.assert_called_once()
        self.assertEqual(mock_drums.call_args.args[0], existing)

    def test_a_song_already_on_disk_is_reported_even_with_nothing_armed(self):
        # Pasting the link of a song you already have has to put it back in
        # front of you. This used to be scoped to runs that asked for an
        # instrument, so a link pasted with only Song armed came back with
        # nothing at all - which is exactly what somebody does when the
        # stash has lost track of a song.
        class SkippingYoutubeDL(_FakeYoutubeDL):
            def download(self, urls):
                return 0

        existing = os.path.join(self.tmp_dir, "Some Song - Artist", "Some Song - Artist.mp3")
        os.makedirs(os.path.dirname(existing))
        with open(existing, "wb") as f:
            f.write(b"already here")

        with mock.patch("yt_dlp.YoutubeDL", SkippingYoutubeDL):
            result = self._run()

        self.assertEqual(result["songs"], [existing])

    def test_an_archive_entry_for_a_song_that_is_gone_is_not_taken_at_its_word(self):
        # The archive lists video ids, not files, so it goes on claiming a
        # song that has since been deleted or was never filed. yt-dlp then
        # skips it and the link produces nothing, forever. The file on disk
        # is the authority: nothing there means go and get it.
        passes = []

        class ArchiveSkippingYoutubeDL(_FakeYoutubeDL):
            def download(self, urls):
                archived = "download_archive" in self.opts
                passes.append(archived)
                if archived:
                    return 0            # "you already have this one"
                return super().download(urls)

        with mock.patch("yt_dlp.YoutubeDL", ArchiveSkippingYoutubeDL):
            result = self._run()

        self.assertEqual(passes, [True, False])
        self.assertEqual(result["downloaded"], 1)
        self.assertEqual(
            result["songs"],
            [os.path.join(self.tmp_dir, "Some Song - Artist", "Some Song - Artist.mp3")],
        )

    def test_a_song_that_is_there_is_not_downloaded_a_second_time(self):
        # The other half of it: a skip with the file present is the archive
        # doing its job, and re-downloading would undo the point of it.
        passes = []

        class SkippingYoutubeDL(_FakeYoutubeDL):
            def download(self, urls):
                passes.append("download_archive" in self.opts)
                return 0

        existing = os.path.join(self.tmp_dir, "Some Song - Artist", "Some Song - Artist.mp3")
        os.makedirs(os.path.dirname(existing))
        with open(existing, "wb") as f:
            f.write(b"already here")

        with mock.patch("yt_dlp.YoutubeDL", SkippingYoutubeDL):
            self._run()

        self.assertEqual(passes, [True])

    def test_a_link_that_resolves_to_nothing_is_not_retried(self):
        # Unknowable isn't the same as missing. A link that can't be
        # resolved at all - offline, or dead - must not turn into a second
        # download attempt.
        passes = []

        class UnresolvableYoutubeDL(_FakeYoutubeDL):
            entries = []

            def download(self, urls):
                passes.append("download_archive" in self.opts)
                return 0

        with mock.patch("yt_dlp.YoutubeDL", UnresolvableYoutubeDL):
            self._run()

        self.assertEqual(passes, [True])


class TestTheNameHandedToTheSanitizer(PipelineTestCase):
    def test_it_is_the_mp3_that_exists_not_the_download_it_came_from(self):
        # ExtractAudio's hook reports the .webm or .mp4 it converted, which
        # is gone by the time anyone could look for it. Handing that name on
        # meant the sanitizer found nothing, returned nothing, and the song
        # was never filed, never remembered, and never showed up in the app.
        self._run()

        self.assertEqual(
            self.mock_sanitize.call_args.args[0], ["Some Song - Artist.mp3"]
        )


class TestSkippingTheTidyUp(PipelineTestCase):
    """Sanitizing is the only part of a run that stops to ask anything, and
    a long playlist is a lot of small questions. Turned off, the download is
    kept exactly as it landed and nothing is asked."""

    def test_the_sanitizer_is_never_called(self):
        self._run(sanitize=False)
        self.mock_sanitize.assert_not_called()
        self.assertNotIn("sanitizing", self._stages())

    def test_the_song_is_still_filed_and_still_isolated(self):
        """Skipping the tidy-up must not mean losing track of the download:
        what yt-dlp wrote is what the rest of the run chains onto."""
        with mock.patch("drum_isolator.isolate_drums_for_single_file") as mock_drums:
            result = self._run(sanitize=False, instruments=["drums"])
        self.assertEqual(len(result["songs"]), 1)
        self.assertIn("Some Song - Artist.mp3", result["songs"][0])
        mock_drums.assert_called_once()

    def test_the_run_still_finishes(self):
        result = self._run(sanitize=False)
        self.assertEqual(self._stages()[-1], "done")
        self.assertFalse(result.get("error"))

    def test_sanitizing_is_what_happens_when_nobody_says_otherwise(self):
        self._run()
        self.mock_sanitize.assert_called_once()


class TestTheTempoGoesInTheName(PipelineTestCase):
    """A sanitized song says its tempo in its filename and its BPM tag, so
    it can be dropped into Live or sorted in Finder without isolating
    anything first."""

    def test_a_sanitized_song_is_named_and_tagged_with_its_tempo(self):
        self.mock_tempo.return_value = (0, 104.46)
        result = self._run()

        expected = os.path.join(self.tmp_dir, "Some Song - Artist (104.5 BPM)",
                                "Some Song - Artist (104.5 BPM).mp3")
        self.assertEqual(result["songs"], [expected])
        from mutagen.easyid3 import EasyID3
        self.assertEqual(EasyID3(expected)["bpm"], ["104.5"])
        self.assertIn("tempo", self._stages())

    def test_the_tempo_read_is_kept_for_the_isolators(self):
        # Reading a tempo takes seconds and the renamed file is a new key
        # to song_alignment's cache, so without this the isolators would
        # read it all over again - and could land on a different answer.
        self.mock_tempo.return_value = (250, 104.46)
        result = self._run()
        self.assertEqual(instrument_isolator.song_alignment(result["songs"][0], interactive=False),
                         (250, 104.46))
        instrument_isolator._alignment_cache.clear()

    def test_a_whole_number_tempo_has_no_decimal(self):
        self.mock_tempo.return_value = (0, 120.0)
        result = self._run()
        self.assertTrue(result["songs"][0].endswith("Some Song - Artist (120 BPM).mp3"))

    def test_left_alone_when_bpm_is_off(self):
        result = self._run(sanitize=False, bpm=False)
        self.mock_tempo.assert_not_called()
        self.assertTrue(result["songs"][0].endswith("Some Song - Artist.mp3"))

    def test_read_without_sanitizing_when_only_bpm_is_on(self):
        """The tempo needs nothing but the song itself - no tidying, no
        drums - so it can be had on its own."""
        self.mock_tempo.return_value = (0, 98.0)
        result = self._run(sanitize=False, bpm=True)
        self.mock_sanitize.assert_not_called()
        self.assertTrue(result["songs"][0].endswith("Some Song - Artist (98 BPM).mp3"))

    def test_a_tempo_that_cannot_be_read_leaves_the_name_as_it_was(self):
        result = self._run()
        self.assertTrue(result["songs"][0].endswith("Some Song - Artist.mp3"))
        self.assertEqual(self._stages()[-1], "done")

    def test_pasting_the_link_again_finds_the_renamed_song(self):
        # yt-dlp only knows the name it gave the file. The song on disk now
        # has its tempo in the name, and a re-paste still has to find it
        # rather than download it all over again.
        class SkippingYoutubeDL(_FakeYoutubeDL):
            def download(self, urls):
                _FakeYoutubeDL.downloads.append(urls)
                return 0

        existing = os.path.join(self.tmp_dir, "Some Song - Artist (104.5 BPM)",
                                "Some Song - Artist (104.5 BPM).mp3")
        os.makedirs(os.path.dirname(existing))
        with open(existing, "wb") as f:
            f.write(b"already here")

        with mock.patch("yt_dlp.YoutubeDL", SkippingYoutubeDL):
            result = self._run()

        self.assertEqual(result["songs"], [existing])
        self.assertEqual(len(_FakeYoutubeDL.downloads), 1, "downloaded again without the archive")


class TestCancelling(PipelineTestCase):
    def test_cancelling_before_isolation_stops_the_run(self):
        with mock.patch("drum_isolator.isolate_drums_for_single_file") as mock_drums:
            result = self._run(instruments=["drums"], should_cancel=lambda: True)

        mock_drums.assert_not_called()
        self.assertTrue(result["cancelled"])
        self.assertIn("cancelled", self._stages())
        self.assertNotIn("done", self._stages())

    def test_cancel_raised_from_inside_demucs_is_reported_not_swallowed(self):
        def cancel_midway(path, context=None):
            raise instrument_isolator.Cancelled()

        with mock.patch("drum_isolator.isolate_drums_for_single_file", side_effect=cancel_midway):
            result = self._run(instruments=["drums"])

        self.assertTrue(result["cancelled"])
        self.assertIn("cancelled", self._stages())

    def test_a_completed_run_is_not_marked_cancelled(self):
        with mock.patch("drum_isolator.isolate_drums_for_single_file"):
            result = self._run(instruments=["drums"], should_cancel=lambda: False)

        self.assertFalse(result["cancelled"])


class TestSeparatedAudioIsCleanedUp(PipelineTestCase):
    """A shared demucs pass is hundreds of megabytes of temp files. However a
    run ends, they go."""

    def test_after_a_finished_run(self):
        with mock.patch("drum_isolator.isolate_drums_for_single_file"), \
             mock.patch("instrument_isolator.clear_stem_cache") as mock_clear:
            self._run(instruments=["drums"])

        mock_clear.assert_called_once()

    def test_after_a_cancelled_run(self):
        def cancel_midway(path, context=None):
            raise instrument_isolator.Cancelled()

        with mock.patch("drum_isolator.isolate_drums_for_single_file", side_effect=cancel_midway), \
             mock.patch("instrument_isolator.clear_stem_cache") as mock_clear:
            self._run(instruments=["drums"])

        mock_clear.assert_called_once()

    def test_after_an_isolator_blows_up(self):
        def explode(path, context=None):
            raise MemoryError("out of room")

        with mock.patch("drum_isolator.isolate_drums_for_single_file", side_effect=explode), \
             mock.patch("instrument_isolator.clear_stem_cache") as mock_clear:
            with self.assertRaises(MemoryError):
                self._run(instruments=["drums"])

        mock_clear.assert_called_once()


class TestRemembersWhereSongsCameFrom(PipelineTestCase):
    def setUp(self):
        super().setUp()
        self.history_path = os.path.join(self.tmp_dir, "history.json")
        patcher = mock.patch("history.HISTORY_PATH", self.history_path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_downloaded_song_records_the_link_it_came_from(self):
        self._run()

        song = os.path.join(self.tmp_dir, "Some Song - Artist", "Some Song - Artist.mp3")
        self.assertEqual(history.url_for(song, self.history_path), "https://example.com/song")

    def test_a_song_no_longer_on_disk_does_not_take_up_a_slot(self):
        # History caps at 200 and library() shows 20. A run of entries for
        # songs that have been deleted used to fill that window and leave
        # the panel empty even though real songs were sitting behind them.
        real = os.path.join(self.tmp_dir, "Real Song - Artist.mp3")
        with open(real, "wb") as f:
            f.write(b"x")
        history.remember("https://example.com/real", [real])
        history.remember(
            "https://example.com/gone",
            [os.path.join(self.tmp_dir, f"gone-{i}.mp3") for i in range(30)],
        )

        listed = pipeline.library(limit=5)

        self.assertEqual([song["song"] for song in listed], [real])

    def test_a_song_whose_folder_cannot_be_read_does_not_blank_the_rest(self):
        # macOS blocks a packaged app from listing ~/Downloads without a
        # permission grant it can't reliably get (see make_app.sh) - a song
        # the terminal front end downloaded there is unreadable from the
        # GUI even though the mp3 itself still exists. That used to raise
        # out of the whole function and blank every other song in the list;
        # now it's dropped like a song that isn't on disk, and the readable
        # ones still come back.
        blocked_dir = os.path.join(self.tmp_dir, "Blocked Song - Artist")
        blocked_song = os.path.join(blocked_dir, "Blocked Song - Artist.mp3")
        os.makedirs(blocked_dir)
        with open(blocked_song, "wb") as f:
            f.write(b"x")

        real = os.path.join(self.tmp_dir, "Real Song - Artist.mp3")
        with open(real, "wb") as f:
            f.write(b"x")

        history.remember("https://example.com/blocked", [blocked_song])
        history.remember("https://example.com/real", [real])

        real_listdir = os.listdir

        def blocked_listdir(path):
            if path == blocked_dir:
                raise PermissionError("Operation not permitted")
            return real_listdir(path)

        with mock.patch("os.listdir", side_effect=blocked_listdir):
            listed = pipeline.library()

        self.assertEqual([song["song"] for song in listed], [real])

    def test_the_library_lists_the_song_with_its_link_and_its_files(self):
        song_dir = os.path.join(self.tmp_dir, "Some Song - Artist")
        wav = os.path.join(song_dir, "Some Song - Artist (Isolated Drums at 120.000 BPM).wav")

        def fake_isolate(path, context=None):
            os.makedirs(song_dir, exist_ok=True)
            with open(wav, "wb") as f:
                f.write(b"x")

        with mock.patch("drum_isolator.isolate_drums_for_single_file", side_effect=fake_isolate):
            self._run(instruments=["drums"])

        library = pipeline.library()

        self.assertEqual(len(library), 1)
        self.assertEqual(library[0]["title"], "Some Song - Artist")
        self.assertEqual(library[0]["url"], "https://example.com/song")
        self.assertIn(wav, library[0]["files"])
        self.assertIn(os.path.join(self.tmp_dir, "Some Song - Artist", "Some Song - Artist.mp3"), library[0]["files"])
        # The folder, so a front end can offer the whole song at once
        # rather than a row per wav inside it.
        self.assertEqual(library[0]["dir"], song_dir)

    def test_the_isolators_own_marker_files_are_not_listed_as_output(self):
        song_dir = os.path.join(self.tmp_dir, "Some Song - Artist")

        def fake_isolate(path, context=None):
            os.makedirs(song_dir, exist_ok=True)
            for name in ("stem.wav", ".drums_source.json"):
                with open(os.path.join(song_dir, name), "wb") as f:
                    f.write(b"x")

        with mock.patch("drum_isolator.isolate_drums_for_single_file", side_effect=fake_isolate):
            self._run(instruments=["drums"])

        listed = [os.path.basename(p) for p in pipeline.library()[0]["files"]]

        self.assertIn("stem.wav", listed)
        self.assertNotIn(".drums_source.json", listed)

    def test_a_song_with_nothing_isolated_yet_still_lists_its_mp3(self):
        self._run()

        self.assertEqual(
            pipeline.library()[0]["files"],
            [os.path.join(self.tmp_dir, "Some Song - Artist", "Some Song - Artist.mp3")],
        )


class TestOneFolderPerSong(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _mp3(self, name="Song - Artist.mp3", where=None):
        path = os.path.join(where or self.tmp_dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"mp3")
        return path

    def test_a_downloaded_song_is_moved_into_a_folder_of_its_own(self):
        flat = self._mp3()

        filed = pipeline.file_into_own_folder(flat)

        self.assertEqual(filed, os.path.join(self.tmp_dir, "Song - Artist", "Song - Artist.mp3"))
        self.assertTrue(os.path.exists(filed))
        self.assertFalse(os.path.exists(flat))

    def test_a_song_already_in_its_folder_is_left_where_it_is(self):
        already = self._mp3(where=os.path.join(self.tmp_dir, "Song - Artist"))

        self.assertEqual(pipeline.file_into_own_folder(already), already)
        self.assertTrue(os.path.exists(already))

    def test_a_song_that_cannot_be_moved_is_still_returned(self):
        # Losing a song to a tidying step would be a much worse trade than
        # leaving it where it is.
        flat = self._mp3()

        with mock.patch("shutil.move", side_effect=OSError("read-only")):
            self.assertEqual(pipeline.file_into_own_folder(flat), flat)

        self.assertTrue(os.path.exists(flat))

    def test_the_stems_land_in_the_same_folder_as_the_song(self):
        filed = pipeline.file_into_own_folder(self._mp3())

        self.assertEqual(
            instrument_isolator.song_output_dir(filed),
            os.path.join(self.tmp_dir, "Song - Artist"),
        )

    def test_a_song_is_found_whether_it_has_been_filed_yet_or_not(self):
        # A download lands flat and is filed afterwards, so between those
        # two moments both places are correct answers.
        flat = self._mp3()
        self.assertEqual(pipeline.existing_song(self.tmp_dir, "Song - Artist.mp3"), flat)

        filed = pipeline.file_into_own_folder(flat)
        self.assertEqual(pipeline.existing_song(self.tmp_dir, "Song - Artist.mp3"), filed)

    def test_a_song_is_found_under_its_tempo_name(self):
        folder = os.path.join(self.tmp_dir, "Song - Artist (98 BPM)")
        os.makedirs(folder)
        renamed = os.path.join(folder, "Song - Artist (98 BPM).mp3")
        with open(renamed, "wb") as f:
            f.write(b"x")
        self.assertEqual(pipeline.existing_song(self.tmp_dir, "Song - Artist.mp3"), renamed)

    def test_a_song_is_found_under_the_name_the_sanitizer_gave_it(self):
        # yt-dlp's "Song - Artist (Official Video)" is tidied to
        # "Song - Artist" - the same derivation finds it again.
        folder = os.path.join(self.tmp_dir, "Song - Artist (98 BPM)")
        os.makedirs(folder)
        renamed = os.path.join(folder, "Song - Artist (98 BPM).mp3")
        with open(renamed, "wb") as f:
            f.write(b"x")
        self.assertEqual(
            pipeline.existing_song(self.tmp_dir, "Song - Artist (Official Video).mp3"), renamed)

    def test_a_numbered_song_is_found_under_its_number(self):
        folder = os.path.join(self.tmp_dir, "004 - Song - Artist (98 BPM)")
        os.makedirs(folder)
        renamed = os.path.join(folder, "004 - Song - Artist (98 BPM).mp3")
        with open(renamed, "wb") as f:
            f.write(b"x")
        self.assertEqual(pipeline.existing_song(self.tmp_dir, "Song - Artist.mp3"), renamed)

    def test_a_different_song_with_a_tempo_is_not_mistaken_for_this_one(self):
        folder = os.path.join(self.tmp_dir, "Song - Artist Two (98 BPM)")
        os.makedirs(folder)
        with open(os.path.join(folder, "Song - Artist Two (98 BPM).mp3"), "wb") as f:
            f.write(b"x")
        self.assertEqual(pipeline.existing_song(self.tmp_dir, "Song - Artist.mp3"), "")

    def test_a_song_that_is_not_there_is_reported_as_missing(self):
        self.assertEqual(pipeline.existing_song(self.tmp_dir, "Nothing.mp3"), "")


class TestWhatASongHas(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.song = os.path.join(self.tmp_dir, "Song - Artist.mp3")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _have(self, *names):
        files = [self.song] + [os.path.join(self.tmp_dir, n) for n in names]
        return pipeline._what_a_song_has(self.song, files)

    def test_the_song_itself_always_counts(self):
        self.assertEqual(self._have(), {"song": self.song})

    def test_each_stem_is_found_by_its_own_label(self):
        have = self._have(
            "Song - Artist (Isolated Drums at 120.000 BPM).wav",
            "Song - Artist (Isolated Vocals).wav",
        )

        self.assertEqual(set(have), {"song", "drums", "vocals"})
        self.assertTrue(have["drums"].endswith("Drums at 120.000 BPM).wav"))

    def test_a_stolen_wav_counts_as_a_beat(self):
        have = self._have("Song - Artist (Beat at 120 BPM).wav")

        self.assertIn("beat", have)
        self.assertNotIn("midi", have)

    def test_a_stolen_mid_counts_as_midi(self):
        have = self._have("Song - Artist (Beat at 120 BPM).mid")

        self.assertIn("midi", have)
        self.assertNotIn("beat", have)

    def test_a_loop_stolen_under_the_old_name_still_counts(self):
        # Beats written before the rename are sitting in people's folders,
        # and a green square going out is the same as losing the file.
        have = self._have("Song - Artist (Stolen Beat, 2 bars) (120 BPM).mid")

        self.assertIn("midi", have)

    def test_the_newest_of_several_beats_is_the_one_offered(self):
        # A song can have several stolen out of it, and the one you want to
        # reach for is the one you just made - not whichever sorted first.
        older = os.path.join(self.tmp_dir, "Song - Artist (Stolen Beat, 2 bars) (120 BPM).mid")
        newer = os.path.join(self.tmp_dir, "Song - Artist (Stolen Beat, 4 bars) (98 BPM).mid")
        for path in (older, newer):
            open(path, "wb").close()
        os.utime(older, (0, 0))

        have = pipeline._what_a_song_has(self.song, [self.song, older, newer])

        self.assertEqual(have["midi"], newer)

    def test_the_newest_of_several_beat_wavs_is_the_one_offered(self):
        older = os.path.join(self.tmp_dir, "Song - Artist (Beat at 98 BPM).wav")
        newer = os.path.join(self.tmp_dir, "Song - Artist (Beat at 120 BPM).wav")
        for path in (older, newer):
            open(path, "wb").close()
        os.utime(older, (0, 0))

        have = pipeline._what_a_song_has(self.song, [self.song, older, newer])

        self.assertEqual(have["beat"], newer)

    def test_the_beat_label_is_the_one_beat_loop_actually_writes(self):
        # pipeline names the file by a constant it doesn't build itself, so
        # this is what stops the two drifting apart.
        import beat_loop
        loop = beat_loop.Loop(
            beat=beat_writer.Beat(tempo=120.0, hits=(beat_writer.Hit("kick", 0),)),
            bars=2, origin_sec=0.0, hits_used=1, hits_dropped=0, hits_inferred=0,
            tempo=120.0, song_tempo=120.0,
        )
        written = beat_loop.write(loop, self.tmp_dir, "Song - Artist")

        self.assertIn("midi", pipeline._what_a_song_has(self.song, [self.song, written]))

    def test_everything_it_can_report_is_a_name_the_app_shows(self):
        have = self._have(
            "Song - Artist (Isolated Drums at 120.000 BPM).wav",
            "Song - Artist (Isolated Bass at 120.000 BPM).wav",
            "Song - Artist (Isolated Harmony).wav",
            "Song - Artist (Isolated Vocals).wav",
            "Song - Artist (Beat at 120 BPM).wav",
            "Song - Artist (Beat at 120 BPM).mid",
            "Song - Artist (Bass beat at 120 BPM).wav",
            "Song - Artist (Harmony beat at 120 BPM).wav",
            "Song - Artist (Vocals beat at 120 BPM).wav",
        )

        self.assertEqual(set(have), set(pipeline.STASH_ORDER))

    def test_each_stem_loop_is_found_and_none_is_taken_for_the_drum_loop(self):
        have = self._have(
            "Song - Artist (Isolated Bass at 120.000 BPM).wav",
            "Song - Artist (Bass beat at 120 BPM).wav",
        )

        self.assertTrue(have["bass_beat"].endswith("(Bass beat at 120 BPM).wav"))
        self.assertTrue(have["bass"].endswith("(Isolated Bass at 120.000 BPM).wav"))
        self.assertNotIn("beat", have)

    def test_the_stem_loop_names_are_the_ones_beat_loop_writes(self):
        import beat_loop
        beat = os.path.join(self.tmp_dir, "Song - Artist (Beat at 120 BPM).wav")
        names = [os.path.basename(beat_loop.stem_loop_path(beat, stem)) for stem in beat_loop.STEM_LOOP_LABELS]

        have = self._have(*names)

        self.assertEqual({"bass_beat", "harmony_beat", "vocals_beat"}, set(have) - {"song"})


class TestLoopSpanInTheLibrary(unittest.TestCase):
    """Whether a song's newest drum loop remembers its bars - which is what
    lets the other stems be looped without marking again."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.beat = os.path.join(self.tmp_dir, "Song (Beat at 120 BPM).wav")
        open(self.beat, "wb").close()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_no_sidecar_no_span(self):
        self.assertFalse(pipeline.has_loop_span({"beat": self.beat}))

    def test_no_beat_no_span(self):
        self.assertFalse(pipeline.has_loop_span({}))

    def test_a_sidecar_is_a_span(self):
        import beat_loop
        loop = beat_loop.Loop(
            beat=beat_writer.Beat(tempo=120.0, hits=(beat_writer.Hit("kick", 0),)),
            bars=2, origin_sec=1.0, hits_used=1, hits_dropped=0, hits_inferred=0,
            tempo=120.0, song_tempo=120.0,
        )
        beat_loop.write_span(self.beat, loop)
        self.assertTrue(pipeline.has_loop_span({"beat": self.beat}))


class TestIsolateWithoutDownloading(unittest.TestCase):
    """Re-taking a stem from a song already on disk must not need the
    internet - and for a song whose link was never recorded, there's no
    link to go back to."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.song = os.path.join(self.tmp_dir, "Song - Artist", "Song - Artist.mp3")
        os.makedirs(os.path.dirname(self.song))
        with open(self.song, "wb") as f:
            f.write(b"mp3")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_it_isolates_without_touching_yt_dlp(self):
        with mock.patch("yt_dlp.YoutubeDL", side_effect=AssertionError("no network here")), \
             mock.patch("drum_isolator.isolate_drums_for_single_file") as mock_drums:
            result = pipeline.isolate([self.song], instruments=["drums"])

        mock_drums.assert_called_once()
        self.assertEqual(mock_drums.call_args.args[0], self.song)
        self.assertFalse(result["cancelled"])
        self.assertEqual(result["downloaded"], 0)

    def test_it_reports_progress_the_same_way_a_download_run_does(self):
        events = []
        with mock.patch("drum_isolator.isolate_drums_for_single_file"):
            pipeline.isolate([self.song], instruments=["drums"], on_event=events.append)

        stages = [e["stage"] for e in events]
        self.assertEqual(stages, ["isolating", "isolated", "done"])

    def test_a_song_that_is_not_there_is_skipped(self):
        with mock.patch("drum_isolator.isolate_drums_for_single_file") as mock_drums:
            result = pipeline.isolate(
                [os.path.join(self.tmp_dir, "gone.mp3")], instruments=["drums"]
            )

        mock_drums.assert_not_called()
        self.assertEqual(result["songs"], [])

    def test_cancelling_stops_it(self):
        with mock.patch("drum_isolator.isolate_drums_for_single_file") as mock_drums:
            result = pipeline.isolate(
                [self.song], instruments=["drums"], should_cancel=lambda: True
            )

        mock_drums.assert_not_called()
        self.assertTrue(result["cancelled"])


class TestMissingFfmpeg(PipelineTestCase):
    def test_says_so_up_front_instead_of_downloading_something_it_cannot_convert(self):
        # Without this check the run looks like it worked - full progress bar,
        # a stray .mp4 on disk, and a baffling "nothing came back".
        with mock.patch("shutil.which", return_value=None), \
             mock.patch("drum_isolator.isolate_drums_for_single_file") as mock_drums:
            result = self._run(instruments=["drums"])

        self.assertIn("ffmpeg", result["error"])
        self.assertEqual(_FakeYoutubeDL.downloads, [])  # never even started
        mock_drums.assert_not_called()
        self.assertEqual(self._stages(), ["error"])

    def test_a_present_ffmpeg_does_not_get_in_the_way(self):
        with mock.patch("shutil.which", return_value="/opt/homebrew/bin/ffmpeg"):
            result = self._run()

        self.assertNotIn("error", result)
        self.assertEqual(result["downloaded"], 1)


class TestFailures(PipelineTestCase):
    def test_a_download_error_is_reported_and_stops_the_run(self):
        import yt_dlp

        class FailingYoutubeDL(_FakeYoutubeDL):
            def download(self, urls):
                raise yt_dlp.utils.DownloadError("ffmpeg not found")

        with mock.patch("yt_dlp.YoutubeDL", FailingYoutubeDL), \
             mock.patch("drum_isolator.isolate_drums_for_single_file") as mock_drums:
            result = self._run(instruments=["drums"])

        self.assertIn("ffmpeg not found", result["error"])
        mock_drums.assert_not_called()
        self.assertEqual(self._stages()[-1], "error")

    def test_a_sanitizer_failure_does_not_lose_the_download(self):
        self.mock_sanitize.side_effect = RuntimeError("bad audio")

        with mock.patch("drum_isolator.isolate_drums_for_single_file") as mock_drums:
            result = self._run(instruments=["drums"])

        self.assertIn("warning", self._stages())
        # The song is still on disk and still gets isolated.
        mock_drums.assert_called_once()
        self.assertEqual(result["downloaded"], 1)


class TestSpotifyLinks(PipelineTestCase):
    """A Spotify link is swapped for the YouTube link of the same recording
    at the top of the run. Everything below that point never learns Spotify
    was involved - which is the whole reason the integration is one line."""

    SPOTIFY = "https://open.spotify.com/track/4PTG3Z6ehGkBFwjybzWkR8"
    SONG = {"title": "Never Gonna Give You Up", "artist": "Rick Astley",
            "duration_sec": 213.573, "query": "Rick Astley Never Gonna Give You Up"}
    MATCH = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def _found(self, *titles, offset=0.2):
        return [
            {"url": self.MATCH + str(i), "title": title, "channel": "Rick Astley",
             "duration": 213.573 + offset, "offset": offset}
            for i, title in enumerate(titles)
        ]

    def _unsure(self):
        """Two results, both a few seconds out - nothing here is obviously the
        recording asked for, so the run has to put it to somebody."""
        return self._found("Official Video", "2022 Remaster", offset=3.0)

    def test_what_gets_downloaded_is_the_youtube_link(self):
        with mock.patch("spotify.track", return_value=self.SONG), \
             mock.patch("youtube_match.candidates", return_value=self._found("Official Video")):
            pipeline.run(self.SPOTIFY, output_dir=self.tmp_dir, on_event=self.events.append)
        self.assertEqual(_FakeYoutubeDL.downloads, [[self.MATCH + "0"]])

    def test_the_wait_is_announced_before_it_starts(self):
        """Looking a track up is a couple of seconds of nothing, so the page
        gets told what's happening rather than sitting blank."""
        with mock.patch("spotify.track", return_value=self.SONG), \
             mock.patch("youtube_match.candidates", return_value=self._found("Official Video")):
            pipeline.run(self.SPOTIFY, output_dir=self.tmp_dir, on_event=self.events.append)
        self.assertEqual(self._stages()[0], "looking-up")

    def test_an_ordinary_link_never_touches_spotify(self):
        with mock.patch("spotify.track") as looked_up:
            self._run()
        looked_up.assert_not_called()
        self.assertEqual(_FakeYoutubeDL.downloads, [["https://example.com/song"]])

    def test_a_clear_match_is_not_put_to_anybody(self):
        """Several results for a famous song are usually the same recording
        uploaded three times. Asking which copy to take is a question with no
        wrong answer, so the run doesn't stop to ask it."""
        asked = []
        with mock.patch("spotify.track", return_value=self.SONG), \
             mock.patch("youtube_match.candidates",
                        return_value=self._found("Official Video", "Lyric Video",
                                                 "Rick Astley - Topic")):
            pipeline.run(self.SPOTIFY, output_dir=self.tmp_dir,
                         on_event=self.events.append,
                         on_choose=lambda request: asked.append(request) or {"url": None})
        self.assertEqual(asked, [])
        self.assertEqual(_FakeYoutubeDL.downloads, [[self.MATCH + "0"]])
        self.assertNotIn("choosing", self._stages())

    def test_an_uncertain_match_is_put_to_whoever_can_answer(self):
        asked = []

        def choose(request):
            asked.append(request)
            return {"url": self.MATCH + "1"}

        with mock.patch("spotify.track", return_value=self.SONG), \
             mock.patch("youtube_match.candidates", return_value=self._unsure()):
            pipeline.run(self.SPOTIFY, output_dir=self.tmp_dir,
                         on_event=self.events.append, on_choose=choose)

        self.assertEqual(len(asked), 1)
        self.assertEqual(asked[0]["title"], "Never Gonna Give You Up")
        self.assertEqual(len(asked[0]["candidates"]), 2)
        # The one that was picked is the one that gets downloaded, not the
        # one that happened to sort first.
        self.assertEqual(_FakeYoutubeDL.downloads, [[self.MATCH + "1"]])

    def test_with_nobody_to_ask_it_takes_the_best_and_says_which(self):
        """The CLI has no way to put a list on screen, so it carries on -
        but it has to name what it assumed, or a wrong recording arrives
        with no explanation."""
        with mock.patch("spotify.track", return_value=self.SONG), \
             mock.patch("youtube_match.candidates", return_value=self._unsure()):
            pipeline.run(self.SPOTIFY, output_dir=self.tmp_dir, on_event=self.events.append)
        warnings = [e for e in self.events if e["stage"] == "warning"]
        self.assertTrue(warnings)
        self.assertIn("Official Video", warnings[0]["message"])
        self.assertEqual(_FakeYoutubeDL.downloads, [[self.MATCH + "0"]])

    def test_picking_none_of_them_stops_the_run(self):
        with mock.patch("spotify.track", return_value=self.SONG), \
             mock.patch("youtube_match.candidates", return_value=self._unsure()):
            result = pipeline.run(self.SPOTIFY, output_dir=self.tmp_dir,
                                  on_event=self.events.append,
                                  on_choose=lambda request: {"url": None})
        self.assertEqual(_FakeYoutubeDL.downloads, [])
        self.assertTrue(result.get("error"))

    def test_cancelling_while_being_asked_is_a_cancel_not_an_error(self):
        with mock.patch("spotify.track", return_value=self.SONG), \
             mock.patch("youtube_match.candidates", return_value=self._unsure()):
            result = pipeline.run(self.SPOTIFY, output_dir=self.tmp_dir,
                                  on_event=self.events.append,
                                  should_cancel=lambda: True,
                                  on_choose=lambda request: {"url": None})
        self.assertTrue(result["cancelled"])
        self.assertFalse(result.get("error"))
        self.assertEqual(_FakeYoutubeDL.downloads, [])

    def test_a_spotify_link_that_cannot_be_read_says_so_plainly(self):
        import spotify as spotify_module

        with mock.patch("spotify.track",
                        side_effect=spotify_module.SpotifyUnavailable("Spotify changed their page.")):
            result = pipeline.run(self.SPOTIFY, output_dir=self.tmp_dir, on_event=self.events.append)
        errors = [e for e in self.events if e["stage"] == "error"]
        self.assertEqual(errors[0]["message"], "Spotify changed their page.")
        self.assertTrue(result.get("error"))
        self.assertEqual(_FakeYoutubeDL.downloads, [])

    def test_a_song_youtube_simply_does_not_have(self):
        with mock.patch("spotify.track", return_value=self.SONG), \
             mock.patch("youtube_match.candidates", return_value=[]):
            result = pipeline.run(self.SPOTIFY, output_dir=self.tmp_dir, on_event=self.events.append)
        errors = [e for e in self.events if e["stage"] == "error"]
        self.assertIn("Rick Astley Never Gonna Give You Up", errors[0]["message"])
        self.assertTrue(result.get("error"))

    def test_history_remembers_the_link_that_can_be_used_again(self):
        """The Spotify link can't be re-downloaded from; the YouTube one can."""
        with mock.patch("spotify.track", return_value=self.SONG), \
             mock.patch("youtube_match.candidates", return_value=self._found("Official Video")), \
             mock.patch("history.remember") as remembered:
            pipeline.run(self.SPOTIFY, output_dir=self.tmp_dir, on_event=self.events.append)
        self.assertEqual(remembered.call_args[0][0], self.MATCH + "0")


class TestSpotifyPlaylists(PipelineTestCase):
    """A playlist is a queue of ordinary songs. Everything that already
    worked per song keeps working per song - history, the stale-archive
    retry, a failure - because the queue is downloaded one url at a time
    rather than handed to yt-dlp as a list."""

    PLAYLIST = "https://open.spotify.com/playlist/4jFldPjGeA3jiOj6U6PaIW"

    @staticmethod
    def _songs(*titles):
        return [{"title": title, "artist": "Someone", "duration_sec": 200.0,
                 "query": f"Someone {title}"} for title in titles]

    @staticmethod
    def _youtube(title):
        return "https://www.youtube.com/watch?v=" + title.replace(" ", "")

    def _matching(self, missing=()):
        """A YouTube result per song, exactly the right length so nothing
        has to be asked about - except the titles named in `missing`, which
        YouTube simply doesn't have."""
        def candidates(query, want_sec):
            title = query.split(" ", 1)[1]
            if title in missing:
                return []
            return [{"url": self._youtube(title), "title": title, "channel": "Someone",
                     "duration": want_sec, "offset": 0.0}]
        return candidates

    def _collection(self, name, *titles):
        _FakeYoutubeDL.by_url = {
            self._youtube(title): [{"title": title}] for title in titles
        }
        return {"name": name, "songs": self._songs(*titles)}

    def _play(self, name="Misco", titles=("Them Changes", "Bad Bad News"),
              missing=(), **kwargs):
        collection = self._collection(name, *titles)
        kwargs.setdefault("output_dir", self.tmp_dir)
        kwargs.setdefault("on_event", self.events.append)
        with mock.patch("spotify.collection", return_value=collection), \
             mock.patch("youtube_match.candidates", side_effect=self._matching(missing)):
            return pipeline.run(self.PLAYLIST, **kwargs)

    def test_every_song_is_downloaded_one_url_at_a_time(self):
        """One url per download, not one download of a list: that's what
        keeps history, retries and failures attached to the right song."""
        result = self._play()
        self.assertEqual(_FakeYoutubeDL.downloads,
                         [[self._youtube("Them Changes")], [self._youtube("Bad Bad News")]])
        self.assertEqual(result["downloaded"], 2)

    def test_numbered_songs_keep_their_place_in_the_spotify_list(self):
        # Numbered by where the song sits on Spotify, not by what YouTube
        # happened to have: a song that couldn't be found leaves a gap
        # rather than shuffling everything after it up one.
        result = self._play(titles=("Them Changes", "Missing One", "Bad Bad News"),
                            missing=("Missing One",), number=True)
        self.assertEqual(sorted(os.path.basename(p) for p in result["songs"]),
                         ["001 - Them Changes - Someone.mp3", "003 - Bad Bad News - Someone.mp3"])

    def test_a_single_song_is_never_numbered(self):
        _FakeYoutubeDL.by_url = {}
        result = self._run(number=True)
        self.assertTrue(os.path.basename(result["songs"][0]).startswith("Some Song"))

    def test_songs_alone_land_in_one_folder_named_after_the_playlist(self):
        """A playlist arrives as one thing and should land as one thing.
        Nobody asked for stems, so nothing needs a folder of its own -
        burying each song in one would only make them harder to use."""
        result = self._play(name="Misco")
        folder = os.path.join(self.tmp_dir, "Misco")
        self.assertEqual(result["output_dir"], folder)
        self.assertEqual(sorted(os.path.basename(p) for p in result["songs"]),
                         ["Bad Bad News - Someone.mp3", "Them Changes - Someone.mp3"])
        for path in result["songs"]:
            self.assertEqual(os.path.dirname(path), folder)

    def test_asking_for_a_stem_gives_each_song_its_own_folder_again(self):
        """Stems and MIDI need somewhere to live, so the per-song folders
        come back - inside the playlist's folder, not scattered."""
        with mock.patch("drum_isolator.isolate_drums_for_single_file"):
            result = self._play(instruments=["drums"])
        folder = os.path.join(self.tmp_dir, "Misco")
        for path in result["songs"]:
            title = os.path.splitext(os.path.basename(path))[0]
            self.assertEqual(os.path.dirname(path), os.path.join(folder, title))

    def test_a_single_track_link_still_lands_where_it_always_did(self):
        """One song is not a playlist, whatever its album is called."""
        with mock.patch("spotify.collection",
                        return_value={"name": "", "songs": self._songs("Them Changes")}), \
             mock.patch("youtube_match.candidates", side_effect=self._matching()):
            _FakeYoutubeDL.by_url = {self._youtube("Them Changes"): [{"title": "Them Changes"}]}
            result = pipeline.run("https://open.spotify.com/track/abc",
                                  output_dir=self.tmp_dir, on_event=self.events.append)
        self.assertEqual(result["output_dir"], self.tmp_dir)
        self.assertEqual(os.path.dirname(result["songs"][0]),
                         os.path.join(self.tmp_dir, "Them Changes - Someone"))

    def test_a_song_youtube_does_not_have_is_skipped_not_fatal(self):
        result = self._play(titles=("Them Changes", "Obscure B-Side", "Bad Bad News"),
                            missing=("Obscure B-Side",))
        self.assertEqual(result["downloaded"], 2)
        self.assertFalse(result.get("error"))
        warnings = [e["message"] for e in self.events if e["stage"] == "warning"]
        self.assertTrue(any("Obscure B-Side" in m for m in warnings))

    def test_a_song_left_out_by_hand_is_skipped_not_a_cancelled_run(self):
        """"None of these" is about one song. The other thirty-nine are
        still wanted, which is what makes it Skip rather than Cancel."""
        asked = []

        def choose(request):
            asked.append(request["title"])
            return {"url": None}

        def candidates(query, want_sec):
            title = query.split(" ", 1)[1]
            if title == "Them Changes":
                # Two results, both a few seconds out: nothing obvious.
                return [{"url": self._youtube(title), "title": title, "channel": "Someone",
                         "duration": want_sec + 3.0, "offset": 3.0},
                        {"url": self._youtube(title) + "b", "title": title + " (Live)",
                         "channel": "Someone", "duration": want_sec + 4.0, "offset": 4.0}]
            return self._matching()(query, want_sec)

        collection = self._collection("Misco", "Them Changes", "Bad Bad News")
        with mock.patch("spotify.collection", return_value=collection), \
             mock.patch("youtube_match.candidates", side_effect=candidates):
            result = pipeline.run(self.PLAYLIST, output_dir=self.tmp_dir,
                                  on_event=self.events.append, on_choose=choose)

        self.assertEqual(asked, ["Them Changes"])
        self.assertEqual(_FakeYoutubeDL.downloads, [[self._youtube("Bad Bad News")]])
        self.assertFalse(result.get("error"))
        self.assertFalse(result["cancelled"])

    def test_the_card_says_which_song_it_is_asking_about(self):
        asked = []

        def choose(request):
            asked.append((request["index"], request["total"]))
            return {"url": None}

        def candidates(query, want_sec):
            return [{"url": "https://youtu.be/a", "title": "A", "channel": "C",
                     "duration": want_sec + 3.0, "offset": 3.0},
                    {"url": "https://youtu.be/b", "title": "B", "channel": "C",
                     "duration": want_sec + 4.0, "offset": 4.0}]

        collection = self._collection("Misco", "One", "Two")
        with mock.patch("spotify.collection", return_value=collection), \
             mock.patch("youtube_match.candidates", side_effect=candidates):
            pipeline.run(self.PLAYLIST, output_dir=self.tmp_dir,
                         on_event=self.events.append, on_choose=choose)
        self.assertEqual(asked, [(1, 2), (2, 2)])

    def test_matching_reports_which_song_it_is_on(self):
        """Resolving forty tracks is a minute of nothing to look at."""
        self._play(titles=("Them Changes", "Bad Bad News"))
        resolving = [e for e in self.events if e["stage"] == "resolving"]
        self.assertEqual([(e["index"], e["total"], e["song"]) for e in resolving],
                         [(1, 2, "Them Changes"), (2, 2, "Bad Bad News")])

    def test_history_remembers_each_song_against_its_own_link(self):
        """Coming back later for a stem has to work for every song in the
        playlist, not just whichever one happened to be first."""
        with mock.patch("history.remember") as remembered:
            self._play()
        remembered_urls = [call.args[0] for call in remembered.call_args_list]
        self.assertEqual(remembered_urls,
                         [self._youtube("Them Changes"), self._youtube("Bad Bad News")])
        for call in remembered.call_args_list:
            self.assertEqual(len(call.args[1]), 1)

    def test_a_download_that_blows_up_does_not_take_the_rest_with_it(self):
        import yt_dlp

        real_download = _FakeYoutubeDL.download

        def explode(self, urls):
            if urls == [TestSpotifyPlaylists._youtube("Them Changes")]:
                raise yt_dlp.utils.DownloadError("video unavailable")
            return real_download(self, urls)

        with mock.patch.object(_FakeYoutubeDL, "download", explode):
            result = self._play()
        self.assertEqual(_FakeYoutubeDL.downloads, [[self._youtube("Bad Bad News")]])
        self.assertEqual(result["downloaded"], 1)
        self.assertFalse(result.get("error"))

    def test_cancelling_partway_keeps_what_already_finished(self):
        stop = []

        def should_cancel():
            return bool(stop)

        collection = self._collection("Misco", "Them Changes", "Bad Bad News")
        real_download = _FakeYoutubeDL.download

        def one_then_stop(self, urls):
            outcome = real_download(self, urls)
            stop.append(True)
            return outcome

        with mock.patch("spotify.collection", return_value=collection), \
             mock.patch("youtube_match.candidates", side_effect=self._matching()), \
             mock.patch.object(_FakeYoutubeDL, "download", one_then_stop):
            result = pipeline.run(self.PLAYLIST, output_dir=self.tmp_dir,
                                  on_event=self.events.append, should_cancel=should_cancel)

        self.assertTrue(result["cancelled"])
        self.assertEqual(_FakeYoutubeDL.downloads, [[self._youtube("Them Changes")]])
        # The song that did finish is a real song: it's in the playlist's
        # folder and the stash knows it, rather than sitting there
        # unremembered because the run stopped before filing anything.
        finished = os.path.join(self.tmp_dir, "Misco", "Them Changes - Someone.mp3")
        self.assertEqual(result["songs"], [finished])
        self.assertEqual(history.url_for(finished), self._youtube("Them Changes"))

    def test_a_playlist_youtube_only_had_one_song_from_is_still_a_folder(self):
        result = self._play(titles=("Them Changes", "Obscure B-Side"),
                            missing=("Obscure B-Side",))
        folder = os.path.join(self.tmp_dir, "Misco")
        self.assertEqual(result["output_dir"], folder)
        self.assertEqual(os.path.dirname(result["songs"][0]), folder)

    def test_a_release_youtube_music_is_sure_of_skips_the_search(self):
        self.mock_released.side_effect = lambda query, want, artist: {
            "url": "https://www.youtube.com/watch?v=release" + query.split(" ", 1)[1][:3]}
        collection = self._collection("Misco", "Them Changes")
        _FakeYoutubeDL.by_url = {"https://www.youtube.com/watch?v=releaseThe": [{"title": "Them Changes"}]}
        with mock.patch("spotify.collection", return_value=collection), \
             mock.patch("youtube_match.candidates") as searched:
            pipeline.run(self.PLAYLIST, output_dir=self.tmp_dir, on_event=self.events.append)
        searched.assert_not_called()
        self.assertEqual(_FakeYoutubeDL.downloads, [["https://www.youtube.com/watch?v=releaseThe"]])
        self.mock_released.assert_called_once_with("Someone Them Changes", 200.0, "Someone")

    def test_a_full_page_of_songs_says_it_might_not_be_all_of_them(self):
        """The embed page stops listing at its limit. Told to take what's
        listed, the run does - and says what it left behind."""
        titles = tuple(f"Song {i}" for i in range(spotify.EMBED_ROW_LIMIT))
        with mock.patch("spotify.song_count", return_value=150):
            self._play(titles=titles)
        warnings = [e["message"] for e in self.events if e["stage"] == "warning"]
        self.assertTrue(any("100" in m and "150" in m for m in warnings))

    def test_a_longer_playlist_stops_to_say_paste_it_instead(self):
        """The app would rather have the whole playlist than 100 songs of
        it: nothing is downloaded, and the page is told why."""
        titles = tuple(f"Song {i}" for i in range(spotify.EMBED_ROW_LIMIT))
        with mock.patch("spotify.song_count", return_value=150):
            result = self._play(titles=titles, take_listed=False)
        self.assertEqual(result["too_long"], {"listed": 100, "count": 150, "name": "Misco"})
        self.assertEqual(_FakeYoutubeDL.downloads, [])

    def test_a_playlist_of_exactly_one_hundred_is_not_too_long(self):
        titles = tuple(f"Song {i}" for i in range(spotify.EMBED_ROW_LIMIT))
        with mock.patch("spotify.song_count", return_value=100):
            result = self._play(titles=titles[:3] + titles[3:], take_listed=False)
        self.assertNotIn("too_long", result)

    def test_when_the_count_cannot_be_read_a_full_page_is_assumed_cut_short(self):
        titles = tuple(f"Song {i}" for i in range(spotify.EMBED_ROW_LIMIT))
        with mock.patch("spotify.song_count", return_value=None):
            result = self._play(titles=titles, take_listed=False)
        self.assertEqual(result["too_long"]["count"], None)

    def test_an_ordinary_link_is_downloaded_exactly_as_it_always_was(self):
        """The regression that matters most: a YouTube link is a queue of
        one and takes the same path it took before any of this existed."""
        result = self._run()
        self.assertEqual(_FakeYoutubeDL.downloads, [["https://example.com/song"]])
        self.assertEqual(result["output_dir"], self.tmp_dir)
        self.assertEqual(os.path.dirname(result["songs"][0]),
                         os.path.join(self.tmp_dir, "Some Song - Artist"))


class TestSanitizingSongsAlreadyOnDisk(PipelineTestCase):
    """Songs downloaded before sanitizing was on - or before it read the
    tempo - tidied where they sit, in any folder."""

    def _put(self, *parts):
        path = os.path.join(self.tmp_dir, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"x")
        return path

    def _tidy(self, **kwargs):
        kwargs.setdefault("on_event", self.events.append)
        return pipeline.sanitize_existing(self.tmp_dir, **kwargs)

    def test_every_song_is_found_flat_in_its_own_folder_or_in_a_playlist(self):
        self._put("Flat - Artist.mp3")
        self._put("Filed - Artist", "Filed - Artist.mp3")
        self._put("Misco", "One - Artist.mp3")
        self._put("Misco", "Two - Artist.mp3")
        self._tidy()
        tidied = sorted(call.args[0][0] for call in self.mock_sanitize.call_args_list)
        self.assertEqual(tidied, ["Filed - Artist.mp3", "Flat - Artist.mp3",
                                  "One - Artist.mp3", "Two - Artist.mp3"])

    def test_duplicates_and_hidden_folders_are_left_alone(self):
        self._put("Duplicates", "Old - Artist.mp3")
        self._put(".stash", "Hidden - Artist.mp3")
        result = self._tidy()
        self.mock_sanitize.assert_not_called()
        self.assertEqual(result["songs"], [])

    def test_a_song_gets_its_tempo_and_its_folder_follows(self):
        self.mock_tempo.return_value = (0, 98.0)
        self._put("Filed - Artist", "Filed - Artist.mp3")
        stem = self._put("Filed - Artist", "Filed - Artist (Isolated Drums at 98 BPM).wav")
        result = self._tidy()
        folder = os.path.join(self.tmp_dir, "Filed - Artist (98 BPM)")
        self.assertEqual(result["songs"], [os.path.join(folder, "Filed - Artist (98 BPM).mp3")])
        self.assertTrue(os.path.exists(os.path.join(folder, os.path.basename(stem))),
                        "the stems moved with the song")

    def test_a_playlist_folder_is_not_renamed_after_one_of_its_songs(self):
        self.mock_tempo.return_value = (0, 98.0)
        self._put("Misco", "Misco.mp3")
        self._put("Misco", "Two - Artist.mp3")
        result = self._tidy()
        self.assertTrue(all(os.path.dirname(p) == os.path.join(self.tmp_dir, "Misco")
                            for p in result["songs"]))

    def test_a_song_that_already_has_its_tempo_is_not_read_again(self):
        self._put("Song - Artist (98 BPM)", "Song - Artist (98 BPM).mp3")
        self._tidy()
        self.mock_tempo.assert_not_called()

    def test_the_stash_keeps_track_of_a_renamed_song(self):
        self.mock_tempo.return_value = (0, 98.0)
        old = self._put("Filed - Artist", "Filed - Artist.mp3")
        history.remember("https://youtube.com/watch?v=x", [old])
        result = self._tidy()
        self.assertEqual(history.url_for(result["songs"][0]), "https://youtube.com/watch?v=x")

    def test_says_which_song_it_is_on(self):
        self._put("A - Artist.mp3")
        self._put("B - Artist.mp3")
        self._tidy()
        tidying = [e for e in self.events if e["stage"] == "tidying"]
        self.assertEqual([(e["index"], e["total"]) for e in tidying], [(1, 2), (2, 2)])
        self.assertEqual(self.events[-1]["stage"], "done")

    def test_stopping_stops_between_songs(self):
        self._put("A - Artist.mp3")
        self._put("B - Artist.mp3")
        result = self._tidy(should_cancel=lambda: self.mock_sanitize.call_count >= 1)
        self.assertTrue(result["cancelled"])
        self.assertEqual(self.mock_sanitize.call_count, 1)

    def test_a_trim_question_goes_to_whoever_can_answer(self):
        self._put("A - Artist.mp3")
        reviewer = mock.Mock()
        self._tidy(on_review=reviewer)
        self.assertIs(self.mock_sanitize.call_args.kwargs["review"], reviewer)


class TestIsPlaylistLink(unittest.TestCase):
    """Whether the page offers to number the songs - from the link's shape
    alone, since it's asked on every keystroke and can't wait on YouTube."""

    def test_playlists_and_albums(self):
        for url in ("https://open.spotify.com/playlist/4jFldPjGeA3jiOj6U6PaIW",
                    "https://open.spotify.com/album/1bt6q2SruMsBtcerNVtpZB",
                    "https://music.youtube.com/playlist?list=PLabc",
                    "https://www.youtube.com/playlist?list=PLabc"):
            self.assertTrue(pipeline.is_playlist_link(url), url)

    def test_single_songs(self):
        for url in ("https://open.spotify.com/track/4uLU6hMCjMI75M1A2tKUQC",
                    "https://www.youtube.com/watch?v=abc",
                    # A song clicked inside a playlist is still one song -
                    # noplaylist downloads just it.
                    "https://www.youtube.com/watch?v=abc&list=PLabc",
                    "", "not a link"):
            self.assertFalse(pipeline.is_playlist_link(url), url)


class TestYoutubePlaylists(PipelineTestCase):
    """A YouTube playlist lands the way a Spotify one does: in one folder
    named after it. It used to be filed song by song at the top level of
    the downloads folder, because only a Spotify link carried a playlist
    name - found by a 17-song YouTube Music playlist scattered among
    everything else."""

    PLAYLIST = "https://music.youtube.com/playlist?list=PLabc"

    def setUp(self):
        super().setUp()
        _FakeYoutubeDL.by_url = {
            self.PLAYLIST: [{"title": "One", "url": "https://www.youtube.com/watch?v=one"},
                            {"title": "Two", "url": "https://www.youtube.com/watch?v=two"}],
            "https://www.youtube.com/watch?v=one": [{"title": "One"}],
            "https://www.youtube.com/watch?v=two": [{"title": "Two"}],
        }
        playlist_name = mock.patch.object(_FakeYoutubeDL, "playlist_title", "Funk Night", create=True)
        playlist_name.start()
        self.addCleanup(playlist_name.stop)

    def _play(self, **kwargs):
        kwargs.setdefault("output_dir", self.tmp_dir)
        kwargs.setdefault("on_event", self.events.append)
        return pipeline.run(self.PLAYLIST, **kwargs)

    def test_songs_land_in_one_folder_named_after_the_playlist(self):
        result = self._play()
        folder = os.path.join(self.tmp_dir, "Funk Night")
        self.assertEqual(result["output_dir"], folder)
        self.assertEqual(sorted(os.path.basename(p) for p in result["songs"]),
                         ["One - Artist.mp3", "Two - Artist.mp3"])
        for path in result["songs"]:
            self.assertEqual(os.path.dirname(path), folder)

    def test_numbered_songs_say_where_they_sit_in_the_playlist(self):
        result = self._play(number=True)
        self.assertEqual(sorted(os.path.basename(p) for p in result["songs"]),
                         ["001 - One - Artist.mp3", "002 - Two - Artist.mp3"])

    def test_the_number_goes_in_front_of_the_tempo_name(self):
        self.mock_tempo.return_value = (0, 98.0)
        result = self._play(number=True)
        self.assertEqual(sorted(os.path.basename(p) for p in result["songs"]),
                         ["001 - One - Artist (98 BPM).mp3", "002 - Two - Artist (98 BPM).mp3"])

    def test_numbering_is_off_unless_asked_for(self):
        result = self._play()
        self.assertFalse(any(os.path.basename(p)[0].isdigit() for p in result["songs"]))

    def test_the_sanitizer_renaming_a_song_does_not_lose_its_number(self):
        # The number is yt-dlp's to know, keyed by the name it gave the
        # file - and sanitizing hands back a different name.
        def tidy(filenames, output_dir, interactive=True, review=None, renamed=None, **_):
            out = []
            for name in filenames:
                tidied = "Tidied " + name
                os.rename(os.path.join(output_dir, name), os.path.join(output_dir, tidied))
                if renamed is not None:
                    renamed[name] = tidied
                out.append(tidied)
            return out
        self.mock_sanitize.side_effect = tidy
        result = self._play(number=True)
        self.assertEqual(sorted(os.path.basename(p) for p in result["songs"]),
                         ["001 - Tidied One - Artist.mp3", "002 - Tidied Two - Artist.mp3"])

    def test_a_playlist_with_no_name_still_gets_a_folder(self):
        with mock.patch.object(_FakeYoutubeDL, "playlist_title", ""):
            result = self._play()
        self.assertNotEqual(result["output_dir"], self.tmp_dir)
        self.assertTrue(os.path.basename(result["output_dir"]).startswith("Playlist"))
        for path in result["songs"]:
            self.assertEqual(os.path.dirname(path), result["output_dir"])

    def test_asking_for_a_stem_puts_each_songs_folder_inside_the_playlists(self):
        with mock.patch("drum_isolator.isolate_drums_for_single_file"):
            result = self._play(instruments=["drums"])
        folder = os.path.join(self.tmp_dir, "Funk Night")
        for path in result["songs"]:
            title = os.path.splitext(os.path.basename(path))[0]
            self.assertEqual(os.path.dirname(path), os.path.join(folder, title))

    def test_a_single_video_with_a_list_in_its_link_is_still_one_song(self):
        """A watch link off a playlist carries list= too, but it's one video
        (noplaylist), and one song lands where one song always has."""
        url = "https://www.youtube.com/watch?v=abc&list=PLabc"
        _FakeYoutubeDL.by_url = {url: [{"title": "Just This"}]}
        result = pipeline.run(url, output_dir=self.tmp_dir, on_event=self.events.append)
        self.assertEqual(result["output_dir"], self.tmp_dir)
        self.assertEqual(os.path.dirname(result["songs"][0]),
                         os.path.join(self.tmp_dir, "Just This - Artist"))


class TestBigPlaylists(PipelineTestCase):
    """Thousands of songs: taken one at a time all the way through, written
    down as each finishes, and picked up from where a run stopped."""

    PLAYLIST = "https://open.spotify.com/playlist/4jFldPjGeA3jiOj6U6PaIW"
    IDS = ["A" * 22, "B" * 22, "C" * 22]

    def _song(self, title, track_id=None):
        song = {"title": title, "artist": "Someone", "duration_sec": 200.0,
                "query": f"Someone {title}"}
        if track_id:
            song["id"] = track_id
        return song

    def _matching(self, query, want_sec):
        title = query.split(" ", 1)[1]
        return [{"url": "https://www.youtube.com/watch?v=" + title, "title": title + " (Official)",
                 "channel": "SomeoneVEVO", "duration": want_sec, "offset": 0.0}]

    def _setup_youtube(self, *titles):
        _FakeYoutubeDL.by_url = {"https://www.youtube.com/watch?v=" + t: [{"title": t}] for t in titles}

    def _play(self, url=None, songs=None, **kwargs):
        songs = songs or [self._song(t, i) for t, i in zip(("One", "Two", "Three"), self.IDS)]
        self._setup_youtube(*(s["title"] for s in songs))
        kwargs.setdefault("output_dir", self.tmp_dir)
        kwargs.setdefault("on_event", self.events.append)
        with mock.patch("spotify.collection", return_value={"name": "Misco", "songs": [dict(s) for s in songs]}), \
             mock.patch("youtube_match.candidates", side_effect=self._matching):
            return pipeline.run(url or self.PLAYLIST, **kwargs)

    def test_each_song_is_finished_before_the_next_is_downloaded(self):
        order = []
        real_download = _FakeYoutubeDL.download

        def download(self_, urls):
            order.append(("download", urls[0].rsplit("=", 1)[1]))
            return real_download(self_, urls)

        def tidy(filenames, output_dir, **_):
            order.append(("sanitize", filenames[0].split(" - ")[0]))
            return list(filenames)

        self.mock_sanitize.side_effect = tidy
        with mock.patch.object(_FakeYoutubeDL, "download", download):
            self._play()
        self.assertEqual(order, [("download", "One"), ("sanitize", "One"),
                                 ("download", "Two"), ("sanitize", "Two"),
                                 ("download", "Three"), ("sanitize", "Three")])

    def test_a_stopped_run_picks_up_where_it_left_off(self):
        stop = []
        real_download = _FakeYoutubeDL.download

        def two_then_stop(self_, urls):
            outcome = real_download(self_, urls)
            if len(_FakeYoutubeDL.downloads) == 2:
                stop.append(True)
            return outcome

        with mock.patch.object(_FakeYoutubeDL, "download", two_then_stop):
            first = self._play(should_cancel=lambda: bool(stop))
        self.assertTrue(first["cancelled"])

        _FakeYoutubeDL.downloads = []
        self.events.clear()
        second = self._play()
        # Song one was finished; song two was only downloaded when the stop
        # came, so it's finished now (the archive finds it on disk) - and
        # three is new.
        self.assertNotIn([["https://www.youtube.com/watch?v=One"]], _FakeYoutubeDL.downloads)
        self.assertIn(["https://www.youtube.com/watch?v=Three"], _FakeYoutubeDL.downloads)
        self.assertEqual(len(second["songs"]), 3)
        resuming = [e for e in self.events if e["stage"] == "resuming"]
        self.assertEqual(resuming, [{"stage": "resuming", "done": 1, "total": 3}])

    def test_songs_already_finished_are_not_looked_up_again(self):
        self._play()
        with mock.patch("youtube_match.candidates") as searched, \
             mock.patch("youtube_match.music_match") as released, \
             mock.patch("spotify.collection",
                        return_value={"name": "Misco", "songs": [self._song(t, i) for t, i in
                                                                 zip(("One", "Two", "Three"), self.IDS)]}):
            _FakeYoutubeDL.downloads = []
            result = pipeline.run(self.PLAYLIST, output_dir=self.tmp_dir)
        searched.assert_not_called()
        released.assert_not_called()
        self.assertEqual(_FakeYoutubeDL.downloads, [])
        self.assertEqual(len(result["songs"]), 3)

    def test_a_song_whose_file_is_gone_is_downloaded_again(self):
        first = self._play()
        os.remove(first["songs"][0])
        _FakeYoutubeDL.downloads = []
        self._play()
        self.assertTrue(_FakeYoutubeDL.downloads)

    def test_files_are_named_and_tagged_from_spotify(self):
        result = self._play()
        names = sorted(os.path.basename(p) for p in result["songs"])
        self.assertEqual(names, ["One - Someone.mp3", "Three - Someone.mp3", "Two - Someone.mp3"])
        tags = song_sanitizer.write_id3_tags
        with mock.patch("song_sanitizer.write_id3_tags") as written:
            _FakeYoutubeDL.downloads = []
            shutil.rmtree(result["output_dir"])
            self._play()
        self.assertIn(mock.call(mock.ANY, "One", "Someone"), written.call_args_list)
        self.assertIs(tags, song_sanitizer.write_id3_tags)

    def test_the_sanitizer_is_told_to_keep_spotifys_name(self):
        self._play()
        self.assertTrue(self.mock_sanitize.call_args.kwargs["keep_name"])

    def test_sources_csv_says_which_upload_each_song_came_from(self):
        import csv
        result = self._play()
        with open(os.path.join(result["output_dir"], "Sources.csv"), encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([r["Song"] for r in rows], ["One", "Two", "Three"])
        self.assertEqual(rows[0]["Matched YouTube title"], "One (Official)")
        self.assertEqual(rows[0]["Channel"], "SomeoneVEVO")
        self.assertEqual(rows[0]["YouTube link"], "https://www.youtube.com/watch?v=One")
        self.assertEqual(rows[0]["Spotify link"], "https://open.spotify.com/track/" + self.IDS[0])
        self.assertEqual(rows[0]["File"], "One - Someone.mp3")

    def test_a_guessed_match_is_marked_to_be_checked(self):
        def unsure(query, want_sec):
            title = query.split(" ", 1)[1]
            return [{"url": "https://www.youtube.com/watch?v=" + title, "title": title, "channel": "C",
                     "duration": want_sec + 3.0, "offset": 3.0},
                    {"url": "https://www.youtube.com/watch?v=" + title + "x", "title": title + " (Live)",
                     "channel": "C", "duration": want_sec + 4.0, "offset": 4.0}]
        songs = [self._song("One", self.IDS[0])]
        self._setup_youtube("One")
        with mock.patch("spotify.collection", return_value={"name": "", "songs": songs}), \
             mock.patch("youtube_match.candidates", side_effect=unsure):
            pipeline.run("https://open.spotify.com/track/" + self.IDS[0], output_dir=self.tmp_dir)
        import sources
        record = sources.Sources(self.tmp_dir).get(sources.spotify_key(self.IDS[0]))
        self.assertTrue(record["check"])

    def test_a_pasted_list_reads_each_song_when_its_turn_comes(self):
        pasted = "\n".join("https://open.spotify.com/track/" + i + "?si=x" for i in self.IDS)
        looked_up = []

        def track(url):
            track_id = url.rsplit("/", 1)[1]
            looked_up.append(track_id)
            title = {self.IDS[0]: "One", self.IDS[1]: "Two", self.IDS[2]: "Three"}[track_id]
            return self._song(title, track_id)

        self._setup_youtube("One", "Two", "Three")
        with mock.patch("spotify.track", side_effect=track), \
             mock.patch("spotify.collection") as listed, \
             mock.patch("youtube_match.candidates", side_effect=self._matching):
            result = pipeline.run(pasted, output_dir=self.tmp_dir, folder_name="Road Trip",
                                  on_event=self.events.append)
        listed.assert_not_called()
        self.assertEqual(looked_up, self.IDS)
        self.assertEqual(result["output_dir"], os.path.join(self.tmp_dir, "Road Trip"))
        self.assertEqual(len(result["songs"]), 3)

    def test_a_pasted_list_with_no_name_carries_on_in_the_folder_it_started(self):
        """The first 100 came from the playlist's link; the paste of all of
        it lands beside them rather than in a new folder."""
        self._play()
        pasted = " ".join("spotify:track:" + i for i in self.IDS + ["D" * 22])
        self._setup_youtube("One", "Two", "Three", "Four")
        with mock.patch("spotify.track", return_value=self._song("Four", "D" * 22)), \
             mock.patch("youtube_match.candidates", side_effect=self._matching):
            _FakeYoutubeDL.downloads = []
            result = pipeline.run(pasted, output_dir=self.tmp_dir)
        self.assertEqual(result["output_dir"], os.path.join(self.tmp_dir, "Misco"))
        self.assertEqual(_FakeYoutubeDL.downloads, [["https://www.youtube.com/watch?v=Four"]])

    def test_auto_never_asks(self):
        """No on_choose and no on_review is the app's Auto: nothing is put
        to anybody, however unsure."""
        self._play(on_choose=None, on_review=None)
        self.assertIsNone(self.mock_sanitize.call_args.kwargs["review"])

    def test_a_youtube_playlist_is_taken_one_video_at_a_time(self):
        playlist = "https://www.youtube.com/playlist?list=PLx"
        _FakeYoutubeDL.by_url = {
            playlist: [{"title": "One", "url": "https://www.youtube.com/watch?v=1"},
                       {"title": "Two", "url": "https://www.youtube.com/watch?v=2"}],
            "https://www.youtube.com/watch?v=1": [{"title": "One"}],
            "https://www.youtube.com/watch?v=2": [{"title": "Two"}],
        }
        result = pipeline.run(playlist, output_dir=self.tmp_dir)
        self.assertEqual(_FakeYoutubeDL.downloads,
                         [["https://www.youtube.com/watch?v=1"], ["https://www.youtube.com/watch?v=2"]])
        self.assertEqual(len(result["songs"]), 2)


if __name__ == "__main__":
    unittest.main()

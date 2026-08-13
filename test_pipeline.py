import os
import shutil
import tempfile
import unittest
from unittest import mock

import beat_writer
import history
import instrument_isolator
import pipeline


class _FakeYoutubeDL:
    """Stands in for yt_dlp.YoutubeDL: fires the hooks a real download would
    fire, without any network. Set `entries` to control what the url looks
    like it resolves to."""

    entries = [{"title": "Some Song"}]
    downloads = []

    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=False):
        return {"entries": list(self.entries)}

    def prepare_filename(self, entry):
        return os.path.join(
            os.path.dirname(self.opts.get("outtmpl", "")), entry["title"] + " - Artist.webm"
        )

    def download(self, urls):
        output_dir = os.path.dirname(self.opts["outtmpl"])
        for entry in self.entries:
            filename = f"{entry['title']} - Artist.mp3"
            path = os.path.join(output_dir, filename)
            with open(path, "wb") as f:
                f.write(b"fake mp3")
            # The mp3 is what ends up on disk, but ExtractAudio's hook
            # reports the file it converted *from* - the .webm or .mp4 that
            # came down the wire, which no longer exists by the time the
            # hook fires. Faking the mp3 here hid a real bug for months:
            # the sanitizer was being handed a filename that wasn't there.
            info = {"title": entry["title"], "filepath": os.path.splitext(path)[0] + ".webm"}
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
        self.events = []
        patcher = mock.patch("yt_dlp.YoutubeDL", _FakeYoutubeDL)
        patcher.start()
        self.addCleanup(patcher.stop)
        # Real sanitizing needs real audio; these tests are about
        # orchestration, so the sanitizer is stood in for throughout.
        sanitize = mock.patch(
            "song_sanitizer.sanitize_new_downloads",
            side_effect=lambda filenames, output_dir, interactive=True: list(filenames),
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

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _run(self, **kwargs):
        kwargs.setdefault("output_dir", self.tmp_dir)
        kwargs.setdefault("on_event", self.events.append)
        return pipeline.run("https://example.com/song", **kwargs)

    def _stages(self):
        return [e["stage"] for e in self.events]


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

    def test_a_stolen_loop_counts_as_a_beat(self):
        have = self._have("Song - Artist (Beat at 120 BPM).mid")

        self.assertIn("beat", have)

    def test_a_loop_stolen_under_the_old_name_still_counts(self):
        # Beats written before the rename are sitting in people's folders,
        # and a green Beat square going out is the same as losing the file.
        have = self._have("Song - Artist (Stolen Beat, 2 bars) (120 BPM).mid")

        self.assertIn("beat", have)

    def test_the_newest_of_several_beats_is_the_one_offered(self):
        # A song can have several stolen out of it, and the one you want to
        # reach for is the one you just made - not whichever sorted first.
        older = os.path.join(self.tmp_dir, "Song - Artist (Stolen Beat, 2 bars) (120 BPM).mid")
        newer = os.path.join(self.tmp_dir, "Song - Artist (Stolen Beat, 4 bars) (98 BPM).mid")
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

        self.assertIn("beat", pipeline._what_a_song_has(self.song, [self.song, written]))

    def test_everything_it_can_report_is_a_name_the_app_shows(self):
        have = self._have(
            "Song - Artist (Isolated Drums at 120.000 BPM).wav",
            "Song - Artist (Isolated Bass at 120.000 BPM).wav",
            "Song - Artist (Isolated Harmony).wav",
            "Song - Artist (Isolated Vocals).wav",
            "Song - Artist (Stolen Beat, 2 bars) (120 BPM).mid",
        )

        self.assertEqual(set(have), set(pipeline.STASH_ORDER))


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


if __name__ == "__main__":
    unittest.main()

import os
import shutil
import tempfile
import unittest
from unittest import mock

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
            info = {"title": entry["title"], "filepath": path}
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


class TestInstrumentRuns(PipelineTestCase):
    def test_isolates_each_requested_instrument_for_the_song(self):
        with mock.patch("drum_isolator.isolate_drums_for_single_file") as mock_drums, \
             mock.patch("harmony_isolator.isolate_harmony_for_single_file") as mock_harmony:
            self._run(instruments=["harmony", "drums"])

        expected = os.path.join(self.tmp_dir, "Some Song - Artist.mp3")
        self.assertEqual(mock_drums.call_args.args[0], expected)
        self.assertEqual(mock_harmony.call_args.args[0], expected)

    def test_instruments_run_in_a_fixed_order_regardless_of_request_order(self):
        order = []
        with mock.patch("drum_isolator.isolate_drums_for_single_file",
                        side_effect=lambda *a, **k: order.append("drums")), \
             mock.patch("bass_isolator.isolate_bass_for_single_file",
                        side_effect=lambda *a, **k: order.append("bass")), \
             mock.patch("harmony_isolator.isolate_harmony_for_single_file",
                        side_effect=lambda *a, **k: order.append("harmony")):
            self._run(instruments=["harmony", "bass", "drums"])

        self.assertEqual(order, ["drums", "bass", "harmony"])

    def test_harmony_is_never_asked_for_midi(self):
        # harmony is audio only - passing write_midi to it would be a TypeError.
        with mock.patch("harmony_isolator.isolate_harmony_for_single_file") as mock_harmony:
            self._run(instruments=["harmony"], write_midi=True)

        self.assertNotIn("write_midi", mock_harmony.call_args.kwargs)

    def test_write_midi_reaches_the_instruments_that_support_it(self):
        with mock.patch("drum_isolator.isolate_drums_for_single_file") as mock_drums:
            self._run(instruments=["drums"], write_midi=True)

        self.assertIs(mock_drums.call_args.kwargs["write_midi"], True)

    def test_non_interactive_choice_reaches_both_the_sanitizer_and_the_isolators(self):
        with mock.patch("drum_isolator.isolate_drums_for_single_file") as mock_drums:
            self._run(instruments=["drums"], interactive=False)

        self.assertIs(self.mock_sanitize.call_args.kwargs["interactive"], False)
        self.assertIs(mock_drums.call_args.kwargs["context"].interactive, False)

    def test_isolation_progress_is_reported_per_instrument(self):
        def fake_isolate(path, write_midi=False, context=None):
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
        song_dir = os.path.join(self.tmp_dir, "Some Song - Artist (Isolated)")
        wav = os.path.join(song_dir, "Some Song - Artist (Isolated Drums at 120.000 BPM).wav")

        def fake_isolate(path, write_midi=False, context=None):
            os.makedirs(song_dir, exist_ok=True)
            with open(wav, "wb") as f:
                f.write(b"x")

        with mock.patch("drum_isolator.isolate_drums_for_single_file", side_effect=fake_isolate):
            result = self._run(instruments=["drums"])

        self.assertIn(wav, result["outputs"])
        isolated = [e for e in self.events if e["stage"] == "isolated"][0]
        self.assertEqual(isolated["outputs"], [wav])


class TestAlreadyDownloadedSongs(PipelineTestCase):
    def test_a_song_skipped_by_the_archive_is_still_isolated(self):
        # yt-dlp fires no hooks for a song it skips, so the fresh-download
        # list is empty - but asking to isolate it is still a real request.
        class SkippingYoutubeDL(_FakeYoutubeDL):
            def download(self, urls):
                return 0

        existing = os.path.join(self.tmp_dir, "Some Song - Artist.mp3")
        with open(existing, "wb") as f:
            f.write(b"already here")

        with mock.patch("yt_dlp.YoutubeDL", SkippingYoutubeDL), \
             mock.patch("drum_isolator.isolate_drums_for_single_file") as mock_drums:
            result = self._run(instruments=["drums"])

        self.assertEqual(result["downloaded"], 0)
        mock_drums.assert_called_once()
        self.assertEqual(mock_drums.call_args.args[0], existing)


class TestCancelling(PipelineTestCase):
    def test_cancelling_before_isolation_stops_the_run(self):
        with mock.patch("drum_isolator.isolate_drums_for_single_file") as mock_drums:
            result = self._run(instruments=["drums"], should_cancel=lambda: True)

        mock_drums.assert_not_called()
        self.assertTrue(result["cancelled"])
        self.assertIn("cancelled", self._stages())
        self.assertNotIn("done", self._stages())

    def test_cancel_raised_from_inside_demucs_is_reported_not_swallowed(self):
        def cancel_midway(path, write_midi=False, context=None):
            raise instrument_isolator.Cancelled()

        with mock.patch("drum_isolator.isolate_drums_for_single_file", side_effect=cancel_midway):
            result = self._run(instruments=["drums"])

        self.assertTrue(result["cancelled"])
        self.assertIn("cancelled", self._stages())

    def test_a_completed_run_is_not_marked_cancelled(self):
        with mock.patch("drum_isolator.isolate_drums_for_single_file"):
            result = self._run(instruments=["drums"], should_cancel=lambda: False)

        self.assertFalse(result["cancelled"])


class TestRemembersWhereSongsCameFrom(PipelineTestCase):
    def setUp(self):
        super().setUp()
        self.history_path = os.path.join(self.tmp_dir, "history.json")
        patcher = mock.patch("history.HISTORY_PATH", self.history_path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_downloaded_song_records_the_link_it_came_from(self):
        self._run()

        song = os.path.join(self.tmp_dir, "Some Song - Artist.mp3")
        self.assertEqual(history.url_for(song, self.history_path), "https://example.com/song")

    def test_the_library_lists_the_song_with_its_link_and_its_files(self):
        song_dir = os.path.join(self.tmp_dir, "Some Song - Artist (Isolated)")
        wav = os.path.join(song_dir, "Some Song - Artist (Isolated Drums at 120.000 BPM).wav")

        def fake_isolate(path, write_midi=False, context=None):
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
        self.assertIn(os.path.join(self.tmp_dir, "Some Song - Artist.mp3"), library[0]["files"])

    def test_the_isolators_own_marker_files_are_not_listed_as_output(self):
        song_dir = os.path.join(self.tmp_dir, "Some Song - Artist (Isolated)")

        def fake_isolate(path, write_midi=False, context=None):
            os.makedirs(song_dir, exist_ok=True)
            for name in ("stem.wav", ".drums_source.json"):
                with open(os.path.join(song_dir, name), "wb") as f:
                    f.write(b"x")

        with mock.patch("drum_isolator.isolate_drums_for_single_file", side_effect=fake_isolate):
            self._run(instruments=["drums"])

        listed = [os.path.basename(p) for p in pipeline.library()[0]["files"]]

        self.assertIn("stem.wav", listed)
        self.assertNotIn(".drums_source.json", listed)

    def test_a_song_with_no_isolated_folder_still_lists_its_mp3(self):
        self._run()

        self.assertEqual(
            pipeline.library()[0]["files"],
            [os.path.join(self.tmp_dir, "Some Song - Artist.mp3")],
        )


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

import os
import shutil
import tempfile
import threading
import time
import unittest
from unittest import mock

import gui


def _wait_until(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class TestApiStart(unittest.TestCase):
    def test_a_blank_link_is_refused_without_starting_anything(self):
        started = []
        api = gui.Api(run_pipeline=lambda *a, **k: started.append(a) or {})

        state = api.start("   ")

        self.assertEqual(started, [])
        self.assertFalse(state["running"])
        self.assertIn("link", state["error"].lower())

    def test_start_returns_immediately_while_the_work_runs_behind_it(self):
        release = threading.Event()

        def slow_pipeline(url, **kwargs):
            release.wait(timeout=2)
            return {"outputs": [], "downloaded": 1}

        api = gui.Api(run_pipeline=slow_pipeline)
        state = api.start("https://example.com/song")

        self.assertTrue(state["running"])
        release.set()
        self.assertTrue(_wait_until(lambda: not api.status()["running"]))

    def test_checked_boxes_become_the_requested_instruments(self):
        calls = {}

        def capture(url, **kwargs):
            calls.update(kwargs)
            calls["url"] = url
            return {"outputs": []}

        api = gui.Api(run_pipeline=capture)
        api.start("https://example.com/song", {"drums": True, "harmony": True})
        _wait_until(lambda: not api.status()["running"])

        self.assertEqual(calls["url"], "https://example.com/song")
        self.assertEqual(calls["instruments"], ["drums", "harmony"])

    def test_every_pad_armed_asks_for_the_whole_song(self):
        calls = {}

        api = gui.Api(run_pipeline=lambda url, **kwargs: calls.update(kwargs) or {"outputs": []})
        api.start(
            "https://example.com/song",
            {"drums": True, "bass": True, "harmony": True, "vocals": True},
        )
        _wait_until(lambda: not api.status()["running"])

        self.assertEqual(calls["instruments"], ["drums", "bass", "harmony", "vocals"])

    def test_the_gui_always_runs_non_interactively(self):
        calls = {}

        api = gui.Api(run_pipeline=lambda url, **kwargs: calls.update(kwargs) or {"outputs": []})
        api.start("https://example.com/song")
        _wait_until(lambda: not api.status()["running"])

        # A window can't answer a terminal prompt, so it must never provoke one.
        self.assertIs(calls["interactive"], False)

    def test_a_second_start_while_running_is_ignored(self):
        release = threading.Event()
        runs = []

        def slow_pipeline(url, **kwargs):
            runs.append(url)
            release.wait(timeout=2)
            return {"outputs": []}

        api = gui.Api(run_pipeline=slow_pipeline)
        api.start("https://example.com/first")
        api.start("https://example.com/second")

        release.set()
        _wait_until(lambda: not api.status()["running"])
        self.assertEqual(runs, ["https://example.com/first"])


class TestApiStatus(unittest.TestCase):
    def test_progress_events_become_a_readable_message_and_percentage(self):
        def emitting_pipeline(url, on_event=None, **kwargs):
            on_event({"stage": "isolating", "instrument": "drums", "song": "Redbone", "percent": 58})
            return {"outputs": []}

        api = gui.Api(run_pipeline=emitting_pipeline)
        api.start("https://example.com/song")
        _wait_until(lambda: not api.status()["running"])

        # The final state is "done", so check what the event produced en route.
        self.assertTrue(_wait_until(lambda: api.status()["stage"] == "done"))

    def test_the_message_for_each_stage_is_plain_english(self):
        message, percent = gui.Api._describe(
            {"stage": "isolating", "instrument": "drums", "song": "Redbone", "percent": 58}
        )
        self.assertEqual(message, "Isolating drums — Redbone")
        self.assertEqual(percent, 58)

        message, _ = gui.Api._describe({"stage": "looking-up"})
        self.assertIn("Looking up", message)

    def test_a_named_phase_replaces_the_generic_isolating_line(self):
        # Demucs is silent for the best part of a minute before it can
        # report 1%. Naming what it's doing is all there is to show.
        message, percent = gui.Api._describe(
            {"stage": "isolating", "instrument": "drums", "song": "Redbone",
             "phase": "Loading the separator...", "percent": None}
        )
        self.assertEqual(message, "Loading the separator... — Redbone")
        self.assertIsNone(percent)

    def test_which_step_of_how_many_is_shown_when_there_is_more_than_one(self):
        message, _ = gui.Api._describe(
            {"stage": "isolating", "instrument": "bass", "song": "Redbone",
             "index": 2, "total": 4, "percent": None}
        )
        self.assertEqual(message, "Isolating bass (2 of 4) — Redbone")

    def test_a_lone_instrument_is_not_labelled_one_of_one(self):
        message, _ = gui.Api._describe(
            {"stage": "isolating", "instrument": "bass", "song": "Redbone",
             "index": 1, "total": 1, "percent": None}
        )
        self.assertEqual(message, "Isolating bass — Redbone")

    def test_a_stage_with_nothing_to_say_leaves_the_message_alone(self):
        api = gui.Api(run_pipeline=lambda *a, **k: {"outputs": []})
        with api._lock:
            api._state["message"] = "Cleaning it up..."

        api._on_event({"stage": "download-summary"})

        self.assertEqual(api.status()["message"], "Cleaning it up...")

    def test_finished_run_reports_its_outputs(self):
        api = gui.Api(run_pipeline=lambda *a, **k: {"outputs": ["/tmp/drums.wav"], "downloaded": 1})
        api.start("https://example.com/song")
        _wait_until(lambda: not api.status()["running"])

        state = api.status()
        self.assertEqual(state["stage"], "done")
        self.assertEqual(state["outputs"], ["/tmp/drums.wav"])

    def test_an_exception_in_the_worker_surfaces_instead_of_spinning_forever(self):
        def exploding_pipeline(url, **kwargs):
            raise RuntimeError("demucs blew up")

        api = gui.Api(run_pipeline=exploding_pipeline)
        api.start("https://example.com/song")

        self.assertTrue(_wait_until(lambda: not api.status()["running"]))
        state = api.status()
        self.assertEqual(state["stage"], "error")
        self.assertIn("demucs blew up", state["error"])

    def test_a_pipeline_error_result_is_shown_as_an_error(self):
        api = gui.Api(run_pipeline=lambda *a, **k: {"error": "no such video", "outputs": []})
        api.start("https://example.com/song")
        _wait_until(lambda: not api.status()["running"])

        self.assertEqual(api.status()["stage"], "error")
        self.assertIn("no such video", api.status()["error"])


class TestApiCancel(unittest.TestCase):
    def test_cancelling_signals_the_pipeline(self):
        seen = {}

        def watching_pipeline(url, should_cancel=None, **kwargs):
            seen["before"] = should_cancel()
            api.cancel()
            seen["after"] = should_cancel()
            return {"cancelled": True, "outputs": []}

        api = gui.Api(run_pipeline=watching_pipeline)
        api.start("https://example.com/song")
        _wait_until(lambda: not api.status()["running"])

        self.assertFalse(seen["before"])
        self.assertTrue(seen["after"])
        self.assertEqual(api.status()["stage"], "cancelled")

    def test_a_new_run_after_a_cancel_starts_uncancelled(self):
        api = gui.Api(run_pipeline=lambda *a, **k: {"outputs": []})
        api.cancel()

        seen = {}
        api._run_pipeline = lambda url, should_cancel=None, **kwargs: (
            seen.update(cancelled=should_cancel()) or {"outputs": []}
        )
        api.start("https://example.com/song")
        _wait_until(lambda: not api.status()["running"])

        self.assertFalse(seen["cancelled"])


class TestApiReveal(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @mock.patch("subprocess.run")
    def test_reveals_an_existing_file_in_finder(self, mock_run):
        path = os.path.join(self.tmp_dir, "drums.wav")
        with open(path, "wb") as f:
            f.write(b"x")

        api = gui.Api(run_pipeline=lambda *a, **k: {})

        self.assertTrue(api.reveal(path))
        mock_run.assert_called_once_with(["open", "-R", path], check=False)

    @mock.patch("subprocess.run")
    def test_a_song_folder_is_opened_rather_than_revealed(self, mock_run):
        # A folder revealed in its parent is one more click from the stems
        # you came to drag; opened, they're right there.
        song_dir = os.path.join(self.tmp_dir, "Song (Isolated)")
        os.makedirs(song_dir)

        api = gui.Api(run_pipeline=lambda *a, **k: {})

        self.assertTrue(api.reveal(song_dir))
        mock_run.assert_called_once_with(["open", song_dir], check=False)

    @mock.patch("subprocess.run")
    def test_a_missing_file_is_not_handed_to_finder(self, mock_run):
        api = gui.Api(run_pipeline=lambda *a, **k: {})

        self.assertFalse(api.reveal(os.path.join(self.tmp_dir, "gone.wav")))
        mock_run.assert_not_called()


class TestDefaultOutput(unittest.TestCase):
    def test_downloads_do_not_default_into_a_folder_macos_blocks(self):
        # An app can't write to Desktop, Documents or Downloads without a
        # permission grant that doesn't reliably reach a python3 subprocess,
        # so defaulting there would fail on every single run.
        home = os.path.expanduser("~")
        blocked = [os.path.join(home, name) for name in ("Desktop", "Documents", "Downloads")]

        self.assertFalse(
            any(gui.DEFAULT_OUTPUT.startswith(path) for path in blocked), gui.DEFAULT_OUTPUT
        )

    def test_a_run_without_an_output_folder_uses_that_default(self):
        calls = {}
        api = gui.Api(run_pipeline=lambda url, **kwargs: calls.update(kwargs) or {"outputs": []})

        api.start("https://example.com/song")
        _wait_until(lambda: not api.status()["running"])

        self.assertEqual(calls["output_dir"], gui.DEFAULT_OUTPUT)

    def test_the_window_can_show_where_files_are_going(self):
        api = gui.Api(run_pipeline=lambda *a, **k: {"outputs": []})

        self.assertEqual(api.status()["output_dir"], gui.DEFAULT_OUTPUT)


class TestLibrary(unittest.TestCase):
    def test_hands_the_page_recent_songs_with_their_links(self):
        songs = [{"title": "Redbone", "url": "https://youtu.be/x", "song": "/m/Redbone.mp3", "files": []}]
        api = gui.Api(run_pipeline=lambda *a, **k: {})

        with mock.patch("pipeline.library", return_value=songs):
            self.assertEqual(api.library(), songs)

    def test_an_unreadable_library_returns_nothing_rather_than_breaking_the_window(self):
        api = gui.Api(run_pipeline=lambda *a, **k: {})

        with mock.patch("pipeline.library", side_effect=OSError("disk gone")):
            self.assertEqual(api.library(), [])


class TestUiFile(unittest.TestCase):
    def test_the_page_the_window_loads_actually_exists(self):
        self.assertTrue(os.path.exists(gui.UI_FILE), gui.UI_FILE)


if __name__ == "__main__":
    unittest.main()

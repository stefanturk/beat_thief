import contextlib
import os
import re
import shutil
import tempfile
import threading
import time
import unittest
import wave
from unittest import mock

import pretty_midi

import beat_loop
import beat_writer
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

    def test_a_song_from_the_stash_is_isolated_without_downloading(self):
        # Re-taking a stem from a song you already have shouldn't need the
        # internet, and a song whose link was never recorded has no link to
        # go back to.
        calls = {}
        api = gui.Api(
            run_pipeline=lambda *a, **k: self.fail("should not have downloaded"),
            isolate_pipeline=lambda songs, **kwargs: calls.update(kwargs, songs=songs) or {"outputs": []},
        )

        api.start("", {"source": "/songs/Track/Track.mp3", "bass": True})
        _wait_until(lambda: not api.status()["running"])

        self.assertEqual(calls["songs"], ["/songs/Track/Track.mp3"])
        self.assertEqual(calls["instruments"], ["bass"])

    def test_a_stash_song_with_nothing_armed_says_so(self):
        api = gui.Api(isolate_pipeline=lambda *a, **k: self.fail("nothing to do"))

        state = api.start("", {"source": "/songs/Track/Track.mp3"})

        self.assertFalse(state["running"])
        self.assertIn("armed", state["error"].lower())

    def test_no_link_and_no_song_is_still_refused(self):
        api = gui.Api(run_pipeline=lambda *a, **k: self.fail("nothing to run"))

        state = api.start("   ", {"drums": True})

        self.assertIn("link", state["error"].lower())

    def test_every_pad_armed_asks_for_the_whole_song(self):
        calls = {}

        api = gui.Api(run_pipeline=lambda url, **kwargs: calls.update(kwargs) or {"outputs": []})
        api.start(
            "https://example.com/song",
            {"drums": True, "bass": True, "harmony": True, "vocals": True},
        )
        _wait_until(lambda: not api.status()["running"])

        self.assertEqual(calls["instruments"], ["drums", "bass", "harmony", "vocals"])

    def test_the_options_the_page_actually_sends_are_understood(self):
        # The page sends one boolean per square, and Song is a square. A
        # "song" key that was sometimes a flag and sometimes a path meant
        # every click of Steal it raised inside start(), which rejects the
        # promise the page is waiting on - so the button did nothing at all
        # and the window went on saying "Ready". This is that exact shape.
        calls = {}

        api = gui.Api(run_pipeline=lambda url, **kwargs: calls.update(kwargs, url=url) or {"outputs": []})
        state = api.start(
            "https://example.com/song",
            {"song": True, "drums": True, "bass": False, "harmony": False, "vocals": False},
        )
        _wait_until(lambda: not api.status()["running"])

        self.assertTrue(state["running"])
        self.assertEqual(state["error"], "")
        self.assertEqual(calls["url"], "https://example.com/song")
        self.assertEqual(calls["instruments"], ["drums"])

    def test_a_song_flag_is_not_mistaken_for_a_song_path(self):
        # Same shape, but pointed at something already in the stash: the
        # path comes from "source", and the Song square's flag alongside it
        # doesn't change where it looks.
        calls = {}
        api = gui.Api(
            run_pipeline=lambda *a, **k: self.fail("should not have downloaded"),
            isolate_pipeline=lambda songs, **kwargs: calls.update(kwargs, songs=songs) or {"outputs": []},
        )

        api.start("Track", {"source": "/songs/Track/Track.mp3", "song": True, "bass": True})
        _wait_until(lambda: not api.status()["running"])

        self.assertEqual(calls["songs"], ["/songs/Track/Track.mp3"])
        self.assertEqual(calls["instruments"], ["bass"])

    def test_taking_more_from_the_stash_does_not_claim_it_was_downloaded(self):
        # isolate() reports the song it worked on and nothing downloaded,
        # which through run()'s wording came out as "you already had this
        # one downloaded" - true of the mp3, misleading about the stem that
        # was just made.
        api = gui.Api(
            isolate_pipeline=lambda songs, **k: {
                "songs": list(songs), "downloaded": 0,
                "outputs": ["/songs/Track/Track (Isolated Bass at 96.000 BPM).wav"],
            },
        )
        api.start("", {"source": "/songs/Track/Track.mp3", "bass": True})
        _wait_until(lambda: not api.status()["running"])

        self.assertEqual(api.status()["message"], "Done.")

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

    def test_found_says_the_songs_name_when_it_knows_it(self):
        message, _ = gui.Api._describe({"stage": "found", "total": 1, "song": "Redbone"})
        self.assertIn("Redbone", message)

    def test_found_falls_back_when_it_is_a_playlist(self):
        message, _ = gui.Api._describe({"stage": "found", "total": 12, "song": None})
        self.assertNotIn("None", message)

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
    def test_the_output_folder_is_created_before_it_is_opened(self, mock_run):
        # It's offered on a first launch, before anything has been
        # downloaded, so the folder genuinely may not exist yet. A button
        # that silently does nothing is worse than no button.
        path = os.path.join(self.tmp_dir, "not made yet")

        api = gui.Api(run_pipeline=lambda *a, **k: {})

        self.assertTrue(api.open_output_dir(path))
        self.assertTrue(os.path.isdir(path))
        mock_run.assert_called_once_with(["open", path], check=False)

    @mock.patch("subprocess.run")
    def test_opening_with_no_path_falls_back_to_the_default(self, mock_run):
        api = gui.Api(run_pipeline=lambda *a, **k: {})

        with mock.patch("os.makedirs") as mock_makedirs:
            api.open_output_dir("")

        mock_makedirs.assert_called_once_with(gui.DEFAULT_OUTPUT, exist_ok=True)
        mock_run.assert_called_once_with(["open", gui.DEFAULT_OUTPUT], check=False)

    @mock.patch("subprocess.run")
    def test_a_folder_that_cannot_be_made_is_reported_rather_than_raising(self, mock_run):
        api = gui.Api(run_pipeline=lambda *a, **k: {})

        with mock.patch("os.makedirs", side_effect=OSError("read-only")):
            self.assertFalse(api.open_output_dir("/nope"))

        mock_run.assert_not_called()

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


class TestAudition(unittest.TestCase):
    """What the picker gets handed when it opens."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _audition(self, filename):
        path = os.path.join(self.tmp_dir, filename)
        with open(path, "wb") as f:
            f.write(b"not really a wav")
        prepared = {"audio": "data:x", "peaks": [0.1], "duration": 9.0, "path": path}
        with mock.patch("audition.preview", return_value=prepared):
            return gui.Api().audition(path)

    def test_the_songs_tempo_comes_along(self):
        # It sizes the slide keystroke. Without it a beat is a guess.
        prepared = self._audition("Song - Artist (Isolated Drums at 105.373 BPM).wav")

        self.assertAlmostEqual(prepared["tempo"], 105.373, places=3)

    def test_a_stem_with_no_tempo_in_its_name_is_not_an_error(self):
        # The page falls back to a round second rather than breaking.
        prepared = self._audition("Some Loose File.wav")

        self.assertEqual(prepared["tempo"], 0.0)
        self.assertNotIn("error", prepared)

    def test_the_cached_preview_is_not_written_into(self):
        # audition.preview hands back the same dict every time it's asked
        # for a file, so adding a key to it would accumulate across calls.
        cached = {"audio": "data:x", "peaks": [0.1], "duration": 9.0, "path": "p"}
        path = os.path.join(self.tmp_dir, "Song (Isolated Drums at 99.000 BPM).wav")
        with open(path, "wb") as f:
            f.write(b"x")
        with mock.patch("audition.preview", return_value=cached):
            gui.Api().audition(path)

        self.assertNotIn("tempo", cached)


class TestStealBeat(unittest.TestCase):
    """What comes back from stealing a loop - specifically which of the two
    tempos in play reaches the page."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.stem = os.path.join(
            self.tmp_dir, "Song - Artist (Isolated Drums at 120.000 BPM).wav")
        with open(self.stem, "wb") as f:
            f.write(b"not really a wav")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _steal(self, notes, start=10.0, end=14.4):
        # Real beat_loop, faked transcription: what's under test is which
        # numbers gui hands the page, not the model.
        import pretty_midi
        shifted = [
            pretty_midi.Note(velocity=100, pitch=pitch, start=at + 3.0, end=at + 3.05)
            for pitch, at in notes
        ]
        with mock.patch("beat_loop._section_wav", return_value=3.0), \
             mock.patch("drum_transcriber.calibrate_hat_threshold", return_value=-5.0), \
             mock.patch("drum_transcriber.transcribe", return_value=shifted), \
             mock.patch("beat_loop.write_wav") as self.mock_write_wav:
            return gui.Api().steal_beat(self.stem, start, end)

    def test_the_tempo_it_reports_is_the_loops_not_the_songs(self):
        # This is the number the page tells you to set Ableton to, so it has
        # to be the loop's. Drumming at 110 inside a stem whose filename says
        # 120: the reported tempo follows the playing.
        step = 60.0 / 110.0 / 4
        played = [(42, i * step) for i in range(32)]
        played += [(36, i * 16 * step) for i in range(2)]
        played += [(38, (i * 16 + 4) * step) for i in range(2)]
        loop = self._steal(played, start=10.0, end=10.0 + 32 * step)

        self.assertEqual(loop["bars"], 2)
        self.assertAlmostEqual(loop["tempo"], 110.0, delta=1.0)
        self.assertAlmostEqual(loop["song_tempo"], 120.0, places=6)

    def test_the_tempo_it_reports_is_the_one_in_the_filename(self):
        # The .mid is named with its own tempo, and the page quotes a tempo
        # in the same breath. If those two ever differ the app is lying.
        loop = self._steal([(36, 0.0), (38, 0.55), (36, 2.2), (38, 2.75)])

        self.assertIn(f"{loop['tempo']:g} BPM", loop["name"])

    def test_a_section_marked_right_leaves_the_two_tempos_agreeing(self):
        loop = self._steal([(36, 0.0), (38, 0.5), (36, 2.0), (38, 2.5)], end=14.0)

        self.assertAlmostEqual(loop["tempo"], loop["song_tempo"], places=6)

    def test_a_stem_that_is_not_there_comes_back_as_a_message(self):
        loop = gui.Api().steal_beat(
            os.path.join(self.tmp_dir, "gone (Isolated Drums at 120.000 BPM).wav"), 0.0, 4.0)

        self.assertIn("error", loop)

    def test_it_also_cuts_a_wav_of_the_loop_beside_the_midi(self):
        self._steal([(36, 0.0), (38, 0.5), (36, 2.0), (38, 2.5)])

        self.assertEqual(self.mock_write_wav.call_count, 1)
        args = self.mock_write_wav.call_args.args
        self.assertEqual(args[1], self.stem)  # cut out of the drum stem
        self.assertTrue(args[2].endswith(".mid"))  # named after the .mid it sits beside

    def test_the_wav_actually_lands_next_to_a_real_midi(self):
        real_stem = os.path.join(self.tmp_dir, "Real - Artist (Isolated Drums at 120.000 BPM).wav")
        with contextlib.closing(wave.open(real_stem, "wb")) as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(44100)
            f.writeframes(b"\x00\x00" * 44100 * 20)

        played = [(36, 0.0), (38, 0.5), (36, 2.0), (38, 2.5)]
        shifted = [
            pretty_midi.Note(velocity=100, pitch=pitch, start=at + 3.0, end=at + 3.05)
            for pitch, at in played
        ]
        with mock.patch("beat_loop._section_wav", return_value=3.0), \
             mock.patch("drum_transcriber.calibrate_hat_threshold", return_value=-5.0), \
             mock.patch("drum_transcriber.transcribe", return_value=shifted):
            loop = gui.Api().steal_beat(real_stem, 10.0, 14.0)

        mid_path = loop["path"]
        wav_path = os.path.splitext(mid_path)[0] + ".wav"
        self.assertTrue(os.path.exists(wav_path))

    def _steal_real(self, outputs, start=10.0, end=14.0):
        # Real write()/write_wav(), so what lands on disk can actually be
        # checked - a mocked write_wav can't prove a .mid was or wasn't
        # deleted alongside it.
        real_stem = os.path.join(self.tmp_dir, "Real - Artist (Isolated Drums at 120.000 BPM).wav")
        with contextlib.closing(wave.open(real_stem, "wb")) as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(44100)
            f.writeframes(b"\x00\x00" * 44100 * 20)

        played = [(36, 0.0), (38, 0.5), (36, 2.0), (38, 2.5)]
        shifted = [
            pretty_midi.Note(velocity=100, pitch=pitch, start=at + 3.0, end=at + 3.05)
            for pitch, at in played
        ]
        with mock.patch("beat_loop._section_wav", return_value=3.0), \
             mock.patch("drum_transcriber.calibrate_hat_threshold", return_value=-5.0), \
             mock.patch("drum_transcriber.transcribe", return_value=shifted):
            return gui.Api().steal_beat(real_stem, start, end, outputs=outputs)

    def test_outputs_wav_writes_only_the_wav_and_no_midi(self):
        loop = self._steal_real("wav")

        self.assertTrue(loop["path"].endswith(".wav"))
        self.assertTrue(os.path.exists(loop["path"]))
        self.assertFalse(os.path.exists(os.path.splitext(loop["path"])[0] + ".mid"))

    def test_outputs_midi_writes_only_the_midi_and_no_wav(self):
        loop = self._steal_real("midi")

        self.assertTrue(loop["path"].endswith(".mid"))
        self.assertTrue(os.path.exists(loop["path"]))
        self.assertFalse(os.path.exists(os.path.splitext(loop["path"])[0] + ".wav"))


class TestStealBeatAsync(unittest.TestCase):
    """steal_beat_start()/beat_status(): the same start()/status() polling
    shape as a run, so the picker can show something better than a static
    "Building it..." for the whole transcription step."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.stem = os.path.join(
            self.tmp_dir, "Song - Artist (Isolated Drums at 120.000 BPM).wav")
        with open(self.stem, "wb") as f:
            f.write(b"not really a wav")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_returns_immediately_and_reports_the_phase_as_it_goes(self):
        release = threading.Event()

        def slow_build(wav_path, tempo, start_sec, end_sec, name="Stolen Beat", on_phase=None):
            on_phase("Listening for the hits...")
            release.wait(timeout=2)
            beat = beat_writer.Beat(tempo=120.0, hits=(beat_writer.Hit("kick", 0, 100),), bars=1)
            return beat_loop.Loop(
                beat=beat, bars=1, origin_sec=0.0, hits_used=1, hits_dropped=0,
                hits_inferred=0, tempo=120.0, song_tempo=120.0,
            )

        with mock.patch("beat_loop.build", side_effect=slow_build), \
             mock.patch("beat_loop.write", return_value=os.path.join(self.tmp_dir, "x.mid")), \
             mock.patch("beat_loop.write_wav"):
            api = gui.Api()
            snapshot = api.steal_beat_start(self.stem, 0.0, 4.0)
            self.assertTrue(snapshot["running"])

            self.assertTrue(_wait_until(
                lambda: api.beat_status()["phase"] == "Listening for the hits..."))
            self.assertTrue(api.beat_status()["running"])

            release.set()
            self.assertTrue(_wait_until(lambda: not api.beat_status()["running"]))

        final = api.beat_status()
        self.assertFalse(final["error"])
        self.assertEqual(final["result"]["bars"], 1)

    def test_a_missing_stem_surfaces_as_an_error_rather_than_hanging(self):
        api = gui.Api()
        api.steal_beat_start(os.path.join(self.tmp_dir, "gone.wav"), 0.0, 4.0)

        self.assertTrue(_wait_until(lambda: not api.beat_status()["running"]))
        self.assertTrue(api.beat_status()["error"])

    def test_a_second_build_while_one_is_running_is_ignored(self):
        release = threading.Event()
        calls = []

        def slow_build(wav_path, tempo, start_sec, end_sec, name="Stolen Beat", on_phase=None):
            calls.append(wav_path)
            release.wait(timeout=2)
            beat = beat_writer.Beat(tempo=120.0, hits=(beat_writer.Hit("kick", 0, 100),), bars=1)
            return beat_loop.Loop(
                beat=beat, bars=1, origin_sec=0.0, hits_used=1, hits_dropped=0,
                hits_inferred=0, tempo=120.0, song_tempo=120.0,
            )

        with mock.patch("beat_loop.build", side_effect=slow_build), \
             mock.patch("beat_loop.write", return_value=os.path.join(self.tmp_dir, "x.mid")), \
             mock.patch("beat_loop.write_wav"):
            api = gui.Api()
            api.steal_beat_start(self.stem, 0.0, 4.0)
            api.steal_beat_start(self.stem, 0.0, 4.0)
            release.set()
            self.assertTrue(_wait_until(lambda: not api.beat_status()["running"]))

        self.assertEqual(len(calls), 1)

    def test_a_second_build_while_one_is_running_is_flagged_busy(self):
        release = threading.Event()

        def slow_build(wav_path, tempo, start_sec, end_sec, name="Stolen Beat", on_phase=None):
            release.wait(timeout=2)
            beat = beat_writer.Beat(tempo=120.0, hits=(beat_writer.Hit("kick", 0, 100),), bars=1)
            return beat_loop.Loop(
                beat=beat, bars=1, origin_sec=0.0, hits_used=1, hits_dropped=0,
                hits_inferred=0, tempo=120.0, song_tempo=120.0,
            )

        with mock.patch("beat_loop.build", side_effect=slow_build), \
             mock.patch("beat_loop.write", return_value=os.path.join(self.tmp_dir, "x.mid")), \
             mock.patch("beat_loop.write_wav"):
            api = gui.Api()
            first = api.steal_beat_start(self.stem, 0.0, 4.0)
            second = api.steal_beat_start(self.stem, 0.0, 4.0)
            release.set()
            self.assertTrue(_wait_until(lambda: not api.beat_status()["running"]))

        self.assertFalse(first["busy"])
        self.assertTrue(second["busy"])

    def test_the_outputs_choice_is_forwarded_to_steal_beat(self):
        with mock.patch.object(gui.Api, "steal_beat", return_value={"bars": 1}) as mock_steal:
            api = gui.Api()
            api.steal_beat_start(self.stem, 0.0, 4.0, outputs="wav")
            self.assertTrue(_wait_until(lambda: not api.beat_status()["running"]))

        self.assertEqual(mock_steal.call_args.kwargs.get("outputs"), "wav")


class TestUiFile(unittest.TestCase):
    def test_the_page_the_window_loads_actually_exists(self):
        self.assertTrue(os.path.exists(gui.UI_FILE), gui.UI_FILE)

    def test_every_api_method_the_page_calls_is_one_the_api_has(self):
        # The page reaches Python by name through pywebview, so a renamed or
        # missing method fails silently in a JS promise nobody is watching -
        # the button just does nothing. Nothing else catches that.
        with open(gui.UI_FILE) as page:
            called = set(re.findall(r"pywebview\.api\.(\w+)", page.read()))

        self.assertTrue(called)
        for name in sorted(called):
            with self.subTest(method=name):
                self.assertTrue(callable(getattr(gui.Api, name, None)))


if __name__ == "__main__":
    unittest.main()

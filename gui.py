#!/usr/bin/env python3
"""beat_thief's window: paste a link, tick what you want, watch it run.

There's no web server and no port here. pywebview opens a real macOS window
rendering ui/index.html, and hands that page an Api object it can call
directly - so "the page asks Python to do something" is an ordinary method
call, not a request over a socket. The window is the app: it starts when you
click the icon and everything exits when you close it.

The actual work is pipeline.run(), exactly the same code path beat_thief.py
takes from the terminal. This file only moves messages between that pipeline
and the page.

Nothing here asks questions. The pipeline's two interactive moments (the
quiet-intro review and the tempo-drift picker) are given deterministic
defaults via interactive=False - a window has no way to answer them, and
silently blocking forever on an invisible prompt would be worse than the
default. The terminal front end keeps both prompts."""

from __future__ import annotations

import os
import subprocess
import threading

import audition
import beat_loop
import instrument_isolator
import pipeline
import pulse

UI_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "index.html")

APP_NAME = "Beat Thief"
WINDOW_SIZE = (560, 760)

# Not the CLI's ~/Downloads/Song Downloads: macOS blocks apps from writing to
# Downloads (as it does Desktop and Documents) without a permission grant that
# doesn't reliably apply to a python3 subprocess, so an app defaulting there
# would fail on every run. ~/Music isn't protected - and for a tool whose
# output goes straight into a DAW, it's the more natural home anyway.
DEFAULT_OUTPUT = os.path.join(os.path.expanduser("~"), "Music", "Beat Thief")


class Api:
    """What the page can call. Every method returns immediately - the slow
    work happens on a worker thread and the page polls status() for it.

    run_pipeline and isolate_pipeline are injectable so this is testable
    against fakes without downloading anything or importing pywebview."""

    def __init__(self, run_pipeline=pipeline.run, isolate_pipeline=pipeline.isolate):
        self._run_pipeline = run_pipeline
        self._isolate_pipeline = isolate_pipeline
        self._lock = threading.Lock()
        self._thread = None
        self._cancel = threading.Event()
        self._state = self._idle_state()
        self._beat_lock = threading.Lock()
        self._beat_thread = None
        self._beat_state = self._idle_beat_state()
        # A trim the run is waiting on somebody to answer. The worker thread
        # blocks on the event; the page answers through resolve_trim.
        self._review_answered = threading.Event()
        self._review_decision = None
        # The same arrangement for a Spotify link whose YouTube match isn't
        # obvious: the worker blocks, the page answers through resolve_match.
        self._choice_answered = threading.Event()
        self._choice_decision = None

    @staticmethod
    def _idle_state() -> dict:
        return {
            "running": False,
            "stage": "idle",
            "message": "",
            "percent": None,
            "outputs": [],
            "error": "",
            "cancelled": False,
            "output_dir": DEFAULT_OUTPUT,
            # The intro or outro currently waiting on a decision, if any.
            "review": None,
            # The Spotify match currently waiting to be picked, if any.
            "choice": None,
        }

    @staticmethod
    def _idle_beat_state() -> dict:
        return {"running": False, "phase": "", "error": "", "result": None}

    # --- called from the page ------------------------------------------

    def start(self, url: str, options: dict | None = None) -> dict:
        """Begin a run. Returns the state the page should show right away,
        so a click feels immediate rather than waiting on a network probe.

        options may carry a "source": the path of something already in the
        stash. Given one, this takes more from that song and never goes
        near the network - no download to do, and for a song whose link was
        never recorded there'd be no link to use anyway.

        It's "source" rather than "song" because the rest of options is one
        armed/not flag per square, and one of the squares is called Song -
        a single key can't be both a boolean and a path."""
        options = options or {}
        source = options.get("source")
        song = source.strip() if isinstance(source, str) else ""
        url = (url or "").strip()
        if not song and not url:
            return self._fail("Paste a link first.")

        # Checked unless the page says otherwise, so a caller that predates
        # the switch - or a page that fails to send it - keeps tidying.
        sanitize = options.get("sanitize", True) is not False

        instruments = [name for name in pipeline.INSTRUMENT_ORDER if options.get(name)]
        if song and not instruments:
            return self._fail("Nothing armed - pick what to take.")

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return dict(self._state)
            self._cancel.clear()
            self._state = self._idle_state()
            self._state.update({"running": True, "stage": "starting", "message": "Getting ready..."})
            state_snapshot = dict(self._state)

        output_dir = options.get("output_dir") or DEFAULT_OUTPUT

        self._thread = threading.Thread(
            target=self._work,
            args=(url, output_dir, instruments, song, sanitize),
            daemon=True,
        )
        self._thread.start()
        return state_snapshot

    def status(self) -> dict:
        """The current state, polled by the page a few times a second."""
        with self._lock:
            return dict(self._state)

    def cancel(self) -> dict:
        """Ask the run to stop. The pipeline checks between stages and during
        demucs, so this takes effect within a second or so rather than
        instantly - the page shows "Stopping..." in the meantime."""
        self._cancel.set()
        # A run parked on a trim decision is blocked in _on_review, which
        # polls this flag - without the nudge it would sit there for another
        # fifth of a second doing nothing, and with a longer wait it would
        # sit there for that.
        self._review_answered.set()
        # Same again for a run parked on a Spotify match - forgetting this
        # one is how a cancel turns into a wedged window.
        self._choice_answered.set()
        with self._lock:
            if self._state["running"]:
                self._state["message"] = "Stopping..."
            return dict(self._state)

    def reveal(self, path: str) -> bool:
        """Show what a song produced in Finder, ready to drag into a DAW - a
        page can't hand a file to another app itself.

        A folder is opened so its stems are there to drag; a file is
        revealed in its parent, selected."""
        if not path or not os.path.exists(path):
            return False
        subprocess.run(["open", path] if os.path.isdir(path) else ["open", "-R", path], check=False)
        return True

    def open_output_dir(self, path: str = "") -> bool:
        """Open the folder everything is saved to, creating it first if it
        isn't there yet.

        Separate from reveal() because this one is offered before anything
        has been downloaded - on a first launch the folder genuinely doesn't
        exist, and a button that silently does nothing is worse than no
        button. reveal() must keep refusing a path that isn't there, since
        for a file that means it was deleted or moved."""
        path = path or DEFAULT_OUTPUT
        try:
            os.makedirs(path, exist_ok=True)
        except OSError:
            return False
        subprocess.run(["open", path], check=False)
        return True

    def library(self) -> list:
        """Recently downloaded songs with their files and their original
        links, so the page can offer to go back for another stem."""
        try:
            return pipeline.library()
        except Exception:
            # A listing that can't be built is worth nothing, but it's never
            # worth breaking the window over.
            return []

    def default_output_dir(self) -> str:
        return DEFAULT_OUTPUT

    # --- deciding where a song starts and ends ---------------------------

    def review_audio(self, path: str) -> dict:
        """The song itself, ready to play and draw - the same preparation the
        beat picker gets, pointed at an mp3 instead of a drum stem.

        audition.preview() is ffmpeg all the way down and never cared what
        kind of file it was handed, so this is the whole of what a second
        waveform needs. The kicks it works out along the way are simply not
        used here; a fade point has nothing to do with the drummer."""
        try:
            return audition.preview(path)
        except Exception as e:
            return {"error": str(e)}

    def resolve_trim(self, cut_sec: float, action: str = "fade") -> dict:
        """Answer the intro or outro the run is waiting on.

        Returns the state the page should show, which is the run carrying on
        - answering is the thing that unblocks it."""
        with self._lock:
            pending = self._state.get("review")
        if not pending:
            # Nothing is waiting: a second click on the same button, or a
            # decision that arrived after a cancel. Saying so beats
            # unblocking something that isn't there.
            return self.status()

        self._review_decision = {
            "action": "fade" if action == "fade" else "keep",
            "cut_ms": int(round(max(0.0, float(cut_sec)) * 1000)),
        }
        self._review_answered.set()
        return self.status()

    def skip_match(self) -> dict:
        """Leave this song out and carry on with the rest.

        Separate from cancel, which stops everything: in a playlist "none of
        these" is about one song, and the other thirty-nine are still
        wanted. For a single link the two come to the same thing, because
        skipping the only song leaves nothing to do."""
        return self.resolve_match("")

    def resolve_match(self, url: str = "") -> dict:
        """Answer the Spotify match the run is waiting on.

        An empty url means "none of these", which skips that song. For a
        single link that ends the run, since there's nothing else to go
        on with."""
        with self._lock:
            pending = self._state.get("choice")
        if not pending:
            # Nothing is waiting: a double click, or an answer that arrived
            # after a cancel.
            return self.status()

        self._choice_decision = {"url": url or None}
        self._choice_answered.set()
        return self.status()

    # --- stealing a beat out of a stem ----------------------------------

    def audition(self, wav_path: str) -> dict:
        """The drum stem, ready to play and draw in the page.

        Slow the first time (a transcode and a decode, about three
        seconds), instant afterwards - see audition.preview. pywebview runs
        these calls off the UI thread, so the window stays alive through
        it, and the page shows its own "getting the audio" state.

        The song's tempo rides along, read out of the stem's filename. The
        picker draws no grid with it - what it's for is sizing one keystroke,
        so sliding a section by a beat is exact rather than a guess at how
        many tenths of a second a beat is."""
        try:
            prepared = dict(audition.preview(wav_path))   # preview() caches its dict
        except Exception as e:
            return {"error": str(e) or e.__class__.__name__}

        basename = os.path.splitext(os.path.basename(wav_path))[0]
        try:
            prepared["tempo"] = instrument_isolator.parse_tempo_from_basename(basename)
        except ValueError:
            prepared["tempo"] = 0.0

        # The grid the kicks agree on, fitted in local windows rather than
        # once across the song. Not a nicety: the filename's tempo is a rough
        # whole-song estimate, and being a few tenths of a percent out puts
        # the end of a three-minute stem several sixteenths away from where a
        # single grid would say. Measured on real stems, one grid for a whole
        # song agrees with the hits about as well as a random number does.
        # Fitted here rather than in the page so the arithmetic lives in one
        # place (pulse.py) that can be lifted out whole.
        prepared["grid"] = [grid._asdict() for grid in
                            pulse.grid_track(prepared.get("kicks") or [], prepared["tempo"])]
        return prepared

    def steal_beat(self, wav_path: str, start_sec: float, end_sec: float,
                    outputs: str = "both", on_phase=None) -> dict:
        """Turn the marked section into a loop and save it next to the stem.

        outputs picks what's kept: "wav" for just the trimmed loop audio,
        "midi" for just the .mid, "both" for the pair (the default, so a
        direct caller that doesn't care - the tests, a script - still gets
        everything). The picker always asks for one or the other now, since
        a click on "Beat" or "Midi" only ever means the one file its label
        says; writing both and throwing one away is cheaper than a second
        write path and keeps _free_path's " (2)" numbering in one place
        (see beat_loop.write_wav's docstring).

        Two tempos come back and they're different things. "tempo" is the
        loop's, measured off the section that was marked, and it's the
        number to set Ableton to. "song_tempo" is the whole song's rough
        estimate, read back out of the stem's filename rather than
        re-estimated here; it's only useful for noticing that the two
        disagree, which means the phrase was marked long or short.

        on_phase is optional and only used by steal_beat_start - a direct
        caller (the tests, or a script) has no poller reading it, so it
        defaults to doing nothing."""
        try:
            song_dir = os.path.dirname(wav_path)
            basename = os.path.splitext(os.path.basename(wav_path))[0]
            tempo = instrument_isolator.parse_tempo_from_basename(basename)
            title = basename.split(" (Isolated")[0]

            loop = beat_loop.build(wav_path, tempo, float(start_sec), float(end_sec), on_phase=on_phase)
            if on_phase is not None:
                on_phase("Saving the loop...")
            mid_path = beat_loop.write(loop, song_dir, title)
            path = mid_path
            if outputs != "midi":
                wav_path_out = beat_loop.write_wav(loop, wav_path, mid_path)
                if outputs == "wav":
                    os.remove(mid_path)
                    path = wav_path_out
        except Exception as e:
            return {"error": str(e) or e.__class__.__name__}

        return {
            "path": path,
            "name": os.path.basename(path),
            "bars": loop.bars,
            # Where the cut actually landed in the stem, and how long it
            # runs. Reported because the picker marks a start and the build
            # is still free to move it (see beat_loop._kick_downbeat), and a
            # move nobody is told about is how a loop ends up starting
            # somewhere you never heard it start.
            "origin": round(loop.origin_sec, 4),
            "duration": round(loop.beat.duration_sec, 4),
            "tempo": round(loop.tempo, 3),
            "song_tempo": round(loop.song_tempo, 3),
            "hits": loop.hits_used,
            # Hits nothing was detected for, worked out from the pulse the
            # rest of that voice is on (see groove_reader). Reported because
            # a stage that adds notes to your beat shouldn't do it quietly.
            "inferred": loop.hits_inferred,
            "pieces": sorted({hit.piece for hit in loop.beat.hits}),
        }

    def steal_beat_start(self, wav_path: str, start_sec: float, end_sec: float,
                          outputs: str = "both") -> dict:
        """Like steal_beat, but returns immediately and reports progress
        through beat_status() - the page polls it exactly the way it polls
        status() for a run.

        A state dict of its own rather than self._state: stealing a beat
        happens after a run has already finished, and starting one
        shouldn't overwrite that run's status while it's still on screen."""
        with self._beat_lock:
            if self._beat_thread is not None and self._beat_thread.is_alive():
                # A build already going wins - the lock only ever guards one
                # thread at a time - but returning its state bare looked
                # identical to a fresh snapshot, so a click swallowed here
                # read as nothing having happened at all. Flagging it lets
                # the page say so instead of staying silent.
                busy = dict(self._beat_state)
                busy["busy"] = True
                return busy
            self._beat_state = {
                "running": True, "phase": "Cutting the section...", "error": "", "result": None,
            }
            snapshot = dict(self._beat_state)
            snapshot["busy"] = False

        self._beat_thread = threading.Thread(
            target=self._steal_beat_work,
            args=(wav_path, start_sec, end_sec, outputs),
            daemon=True,
        )
        self._beat_thread.start()
        return snapshot

    def beat_status(self) -> dict:
        """The current state of a steal_beat_start() build, polled by the
        page a few times a second while the picker is waiting on it."""
        with self._beat_lock:
            return dict(self._beat_state)

    def _set_beat_phase(self, phase: str) -> None:
        with self._beat_lock:
            self._beat_state["phase"] = phase

    def _steal_beat_work(self, wav_path, start_sec, end_sec, outputs="both"):
        result = self.steal_beat(wav_path, start_sec, end_sec, outputs=outputs, on_phase=self._set_beat_phase)
        with self._beat_lock:
            self._beat_state["running"] = False
            if "error" in result:
                self._beat_state["error"] = result["error"]
            else:
                self._beat_state["result"] = result

    # --- internals -----------------------------------------------------

    def _fail(self, message: str) -> dict:
        with self._lock:
            self._state = self._idle_state()
            self._state["error"] = message
            return dict(self._state)

    def _on_review(self, flag: dict) -> dict:
        """Put one ambiguous intro or outro to the page and wait for it.

        This runs on the worker thread and blocks it on purpose: the whole
        point is that nothing gets isolated from a song whose start or end is
        still in question, because every stem, tempo and beat taken from it
        would inherit the answer.

        A cancel releases it as "keep" - a run being stopped must never be
        the thing that rewrites a file."""
        self._review_decision = None
        self._review_answered.clear()
        with self._lock:
            self._state["stage"] = "reviewing"
            self._state["message"] = (
                "Where does the song start?" if flag["end"] == "start"
                else "Where does the song end?")
            self._state["review"] = {
                "path": flag.get("path", flag["filename"]),
                "filename": flag["filename"],
                "end": flag["end"],
                "cut_sec": round(flag["cut_ms"] / 1000.0, 3),
            }

        # Polled rather than waited on outright, so a cancel gets us out of
        # here even though it has no way to set this event itself.
        while not self._review_answered.wait(0.2):
            if self._cancel.is_set():
                break

        decision = self._review_decision or {"action": "keep"}
        self._review_decision = None
        with self._lock:
            self._state["review"] = None
        return decision

    def _on_choose(self, request: dict) -> dict:
        """Put the YouTube candidates for a Spotify track to the page and
        wait for one to be picked.

        Blocks the worker thread deliberately, for the same reason the trim
        review does: the wrong recording here - a remaster, a live take - has
        its own tempo and its own beat 1, and every grid made from it further
        down would be built on that.

        A cancel releases it with nothing picked, which stops the run."""
        self._choice_decision = None
        self._choice_answered.clear()
        with self._lock:
            self._state["stage"] = "choosing"
            self._state["message"] = "Which one is it?"
            self._state["choice"] = {
                "query": request.get("query", ""),
                "title": request.get("title", ""),
                "artist": request.get("artist", ""),
                "want_sec": request.get("want_sec"),
                "candidates": list(request.get("candidates") or []),
                # Which song of how many, when it's a playlist - so the card
                # can say what it's asking about rather than just asking.
                "index": request.get("index"),
                "total": request.get("total"),
            }

        while not self._choice_answered.wait(0.2):
            if self._cancel.is_set():
                break

        decision = self._choice_decision or {"url": None}
        self._choice_decision = None
        with self._lock:
            self._state["choice"] = None
        return decision

    def _work(self, url, output_dir, instruments, song="", sanitize=True):
        try:
            if song:
                result = self._isolate_pipeline(
                    [song],
                    instruments=instruments,
                    on_event=self._on_event,
                    should_cancel=self._cancel.is_set,
                    interactive=False,
                )
            else:
                result = self._run_pipeline(
                    url,
                    output_dir=output_dir,
                    instruments=instruments,
                    on_event=self._on_event,
                    should_cancel=self._cancel.is_set,
                    interactive=False,
                    on_review=self._on_review if sanitize else None,
                    on_choose=self._on_choose,
                    sanitize=sanitize,
                )
        except BaseException as e:
            # Includes Cancelled and anything a dependency throws: a worker
            # thread dying silently would leave the page spinning forever.
            with self._lock:
                self._state["running"] = False
                self._state["stage"] = "error"
                self._state["error"] = str(e) or e.__class__.__name__
            return

        with self._lock:
            self._state["running"] = False
            self._state["outputs"] = result.get("outputs", [])
            self._state["cancelled"] = bool(result.get("cancelled"))
            if result.get("error"):
                self._state["stage"] = "error"
                self._state["error"] = result["error"]
            elif result.get("cancelled"):
                self._state["stage"] = "cancelled"
                self._state["message"] = "Stopped. Anything already finished is saved."
            else:
                self._state["stage"] = "done"
                self._state["percent"] = 100
                self._state["message"] = self._done_message(result, bool(song))

    @staticmethod
    def _done_message(result: dict, from_stash: bool = False) -> str:
        if from_stash:
            # Nothing was downloaded because there was nothing to download,
            # so neither counter means here what it means after a run.
            return "Done." if result.get("outputs") else "Nothing came back for that song."
        if result.get("downloaded"):
            return "Done."
        if result.get("songs"):
            return "Done (you already had this one downloaded)."
        return "Nothing came back for that link."

    def _on_event(self, event):
        stage = event["stage"]
        message, percent = self._describe(event)
        with self._lock:
            if message is not None:
                self._state["message"] = message
            self._state["stage"] = stage
            self._state["percent"] = percent
            if stage == "error":
                self._state["error"] = event.get("message", "")

    @staticmethod
    def _describe(event) -> tuple[str | None, float | None]:
        """One line of plain English for the page, plus a percentage when
        there's a real one to show. Returning None for the message leaves
        whatever was there - better than flickering to a blank line."""
        stage = event["stage"]

        if stage == "looking-up":
            return "Looking up that link...", None
        if stage == "resolving":
            # A playlist is looked up one song at a time and that takes a
            # while, so it says which song rather than sitting on one line.
            return f"Matching {event['index']} of {event['total']} — {event['song']}", None
        if stage == "found":
            song = event.get("song")
            return (f"Found {song}" if song else "Downloading..."), None
        if stage == "downloading":
            return f"Downloading {event['song']}", event.get("percent")
        if stage == "downloaded":
            return f"Downloaded {event['song']}", 100
        if stage == "download-failed":
            return f"Couldn't download {event['song']}", None
        if stage == "download-summary":
            return None, None
        if stage == "sanitizing":
            return "Cleaning it up...", None
        if stage == "isolating":
            # "2 of 4" because the slow parts have no percentage of their
            # own for minutes at a time, and knowing which step you're on is
            # the only progress there is to report during them.
            total = event.get("total") or 0
            step = f" ({event['index']} of {total})" if total > 1 else ""
            phase = event.get("phase")
            what = phase or f"Isolating {event['instrument']}"
            return f"{what}{step} — {event['song']}", event.get("percent")
        if stage == "isolated":
            return f"Finished {event['instrument']}", 100
        if stage == "warning":
            return event.get("message"), None
        return None, None


def _name_the_menu_bar() -> None:
    """Make the macOS menu bar say "Beat Thief" rather than "Python".

    The menu bar takes its name from the running executable's bundle, and
    that executable is /usr/bin/python3 - so it reads "Python" no matter what
    the .app around it is called. Overwriting the loaded bundle's info
    dictionary before AppKit builds the menu is the standard way to fix this
    for a Python app; there's no supported API for it. Failing is harmless,
    so a missing PyObjC or an OS change only costs the nicer name."""
    try:
        from Foundation import NSBundle

        bundle = NSBundle.mainBundle()
        info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
        if info is not None:
            info["CFBundleName"] = APP_NAME
            info["CFBundleDisplayName"] = APP_NAME
    except Exception:
        pass


def main() -> None:
    import webview  # imported here so the Api above stays testable without it

    _name_the_menu_bar()

    api = Api()
    webview.create_window(
        APP_NAME,
        UI_FILE,
        js_api=api,
        width=WINDOW_SIZE[0],
        height=WINDOW_SIZE[1],
        min_size=(420, 520),
    )
    webview.start()


if __name__ == "__main__":
    main()

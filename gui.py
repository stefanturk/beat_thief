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
        }

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
            args=(url, output_dir, instruments, song),
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
        return prepared

    def steal_beat(self, wav_path: str, start_sec: float, end_sec: float) -> dict:
        """Turn the marked section into a looping .mid next to the stem.

        Two tempos come back and they're different things. "tempo" is the
        loop's, measured off the section that was marked, and it's the
        number to set Ableton to. "song_tempo" is the whole song's rough
        estimate, read back out of the stem's filename rather than
        re-estimated here; it's only useful for noticing that the two
        disagree, which means the phrase was marked long or short."""
        try:
            song_dir = os.path.dirname(wav_path)
            basename = os.path.splitext(os.path.basename(wav_path))[0]
            tempo = instrument_isolator.parse_tempo_from_basename(basename)
            title = basename.split(" (Isolated")[0]

            loop = beat_loop.build(wav_path, tempo, float(start_sec), float(end_sec))
            path = beat_loop.write(loop, song_dir, title)
        except Exception as e:
            return {"error": str(e) or e.__class__.__name__}

        return {
            "path": path,
            "name": os.path.basename(path),
            "bars": loop.bars,
            "tempo": round(loop.tempo, 3),
            "song_tempo": round(loop.song_tempo, 3),
            "hits": loop.hits_used,
            "pieces": sorted({hit.piece for hit in loop.beat.hits}),
        }

    # --- internals -----------------------------------------------------

    def _fail(self, message: str) -> dict:
        with self._lock:
            self._state = self._idle_state()
            self._state["error"] = message
            return dict(self._state)

    def _work(self, url, output_dir, instruments, song=""):
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
        if stage == "found":
            return "Downloading...", None
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

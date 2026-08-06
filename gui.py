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

    run_pipeline is injectable so this is testable against a fake without
    downloading anything or importing pywebview."""

    def __init__(self, run_pipeline=pipeline.run):
        self._run_pipeline = run_pipeline
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
        so a click feels immediate rather than waiting on a network probe."""
        options = options or {}
        url = (url or "").strip()
        if not url:
            return self._fail("Paste a link first.")

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return dict(self._state)
            self._cancel.clear()
            self._state = self._idle_state()
            self._state.update({"running": True, "stage": "starting", "message": "Getting ready..."})
            state_snapshot = dict(self._state)

        instruments = [name for name in pipeline.INSTRUMENT_ORDER if options.get(name)]
        # MIDI is asked for per instrument, so a page that only wants drum
        # MIDI doesn't also get the rough bass one. Intersected with what was
        # actually requested, since the page leaves a tick behind when a pad
        # is switched back off.
        midi_for = frozenset(options.get("midi") or ()) & pipeline.TAKES_MIDI & set(instruments)
        output_dir = options.get("output_dir") or DEFAULT_OUTPUT

        self._thread = threading.Thread(
            target=self._work,
            args=(url, output_dir, instruments, midi_for),
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
        """Show a produced file in Finder, selected and ready to drag into a
        DAW - a page can't hand a file to another app itself."""
        if not path or not os.path.exists(path):
            return False
        subprocess.run(["open", "-R", path], check=False)
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

    # --- internals -----------------------------------------------------

    def _fail(self, message: str) -> dict:
        with self._lock:
            self._state = self._idle_state()
            self._state["error"] = message
            return dict(self._state)

    def _work(self, url, output_dir, instruments, midi_for):
        try:
            result = self._run_pipeline(
                url,
                output_dir=output_dir,
                instruments=instruments,
                midi_for=midi_for,
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
                self._state["message"] = self._done_message(result)

    @staticmethod
    def _done_message(result: dict) -> str:
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
            return f"Isolating {event['instrument']} — {event['song']}", event.get("percent")
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

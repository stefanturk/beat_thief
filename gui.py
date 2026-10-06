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

import json
import os
import shutil
import subprocess
import threading
import time

import audition
import beat_loop
import instrument_isolator
import pipeline
import pulse
import sources

UI_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "index.html")

APP_NAME = "Beat Thief"
WINDOW_SIZE = (560, 760)

# Not the CLI's ~/Downloads/Song Downloads: macOS blocks apps from writing to
# Downloads (as it does Desktop and Documents) without a permission grant that
# doesn't reliably apply to a python3 subprocess, so an app defaulting there
# would fail on every run. ~/Music isn't protected - and for a tool whose
# output goes straight into a DAW, it's the more natural home anyway.
# How many of a run's problems the page keeps to show.
MAX_PROBLEMS = 50

# A playlist that was still going when the app quit, crashed or lost its
# power - so the next launch can offer to carry on with it (see
# Api.unfinished). Beside history.json, in Application Support.
RUN_STATE_PATH = os.path.join(os.path.expanduser("~"), "Library", "Application Support",
                              "Beat Thief", "unfinished_run.json")
# How often (in songs) the saved progress is brought up to date. It's only
# for the banner's "stopped at": the folder's own record is what resumes.
RUN_STATE_EVERY = 10

# Whether a run holds off idle sleep (see _keep_awake). Off in tests.
KEEP_AWAKE = True


def _keep_awake():
    """Hold off idle sleep for as long as this app is running a job -
    `caffeinate -i -w <this process>`, so even a crash lets the Mac sleep
    again. A closed lid still sleeps; a run picks up after it wakes."""
    if not KEEP_AWAKE or not shutil.which("caffeinate"):
        return None
    try:
        return subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        return None


def _let_sleep(process) -> None:
    if process is not None:
        try:
            process.terminate()
        except OSError:
            pass


def _save_run(state: dict) -> None:
    try:
        os.makedirs(os.path.dirname(RUN_STATE_PATH), exist_ok=True)
        tmp = RUN_STATE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f)
        os.replace(tmp, RUN_STATE_PATH)
    except OSError:
        pass


def _load_run() -> dict | None:
    try:
        with open(RUN_STATE_PATH, encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, ValueError):
        return None
    return state if isinstance(state, dict) and state.get("url") else None


def _forget_run() -> None:
    try:
        os.remove(RUN_STATE_PATH)
    except OSError:
        pass

DEFAULT_OUTPUT = os.path.join(os.path.expanduser("~"), "Music", "Beat Thief")


class Api:
    """What the page can call. Every method returns immediately - the slow
    work happens on a worker thread and the page polls status() for it.

    run_pipeline and isolate_pipeline are injectable so this is testable
    against fakes without downloading anything or importing pywebview."""

    def __init__(self, run_pipeline=pipeline.run, isolate_pipeline=pipeline.isolate,
                 sanitize_existing=pipeline.sanitize_existing, choose_folder=None,
                 review_pending=pipeline.review_pending):
        self._run_pipeline = run_pipeline
        self._review_pending = review_pending
        self._isolate_pipeline = isolate_pipeline
        self._sanitize_existing = sanitize_existing
        self._choose_folder = choose_folder or _choose_folder
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
        # The link and options of the run in hand, and - once it turns out
        # to be a playlist - what's saved about it for the next launch.
        self._started = None
        self._saved = None

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
            # A Spotify playlist longer than its link lists:
            # {"listed", "count", "name"} - see pipeline.run(take_listed).
            "too_long": None,
            # Songs this run couldn't get, newest last - kept, since each
            # one's message is otherwise gone the moment the next song
            # starts. Capped: the full list is in the folder's Sources.csv.
            "problems": [],
            "problem_count": 0,
            # The folder's Sources.csv, once a playlist run has one.
            "sources_csv": "",
            # Guessed matches a finished run left to check, and in which
            # folder - offered as "Check them" (see check_guesses).
            "to_check": 0,
            "checks_in": "",
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

        # Off, Auto or Ask. Auto unless the page says otherwise, so a caller
        # that predates the switch - or a page that fails to send it - keeps
        # tidying. The old checkbox sent true, which was asking.
        sanitize = options.get("sanitize", "auto")
        if sanitize is True:
            sanitize = "ask"
        elif sanitize is False:
            sanitize = "off"
        elif sanitize not in ("off", "auto", "ask"):
            sanitize = "auto"
        run_options = {
            "sanitize": sanitize,
            "bpm": options.get("bpm", True) is not False,
            "steps": [step for step in (options.get("sanitize_steps") or [])
                      if isinstance(step, str)] or None,
            "folder_name": options.get("folder_name") if isinstance(options.get("folder_name"), str) else "",
            # Asked for after being told the link only lists 100: "just
            # these". Without it, a longer playlist stops and says paste.
            "take_listed": options.get("take_listed") is True,
        }
        # Off unless asked for: a filename that starts with a number is a
        # surprise to somebody who didn't tick the box.
        number = options.get("number") is True

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
            self._started = None if song else {"url": url, "options": dict(options)}
            self._saved = None

        output_dir = options.get("output_dir") or DEFAULT_OUTPUT

        self._thread = threading.Thread(
            target=self._work,
            args=(url, output_dir, instruments, song, number, run_options),
            daemon=True,
        )
        self._thread.start()
        return state_snapshot

    def tidy_folder(self) -> dict:
        """Ask which folder, then sanitize every song in it where it sits
        (pipeline.sanitize_existing) - on the same worker, status and Stop
        as a download, so a trim question comes up the way it always does.
        Closing the folder picker starts nothing."""
        folder = self._choose_folder()
        with self._lock:
            if not folder or (self._thread is not None and self._thread.is_alive()):
                return dict(self._state)
            self._cancel.clear()
            self._state = self._idle_state()
            self._state.update({"running": True, "stage": "starting", "message": "Getting ready...",
                                "output_dir": folder})
            state_snapshot = dict(self._state)
            self._thread = threading.Thread(target=self._work_tidy, args=(folder,), daemon=True)
            self._thread.start()
        return state_snapshot

    def check_guesses(self) -> dict:
        """Put the matches a finished run guessed at to you, one by one -
        the same questions Ask mode asks at the end of a playlist, for a run
        that was on Auto (see pipeline.review_pending)."""
        with self._lock:
            folder = self._state.get("checks_in")
            if not folder or (self._thread is not None and self._thread.is_alive()):
                return dict(self._state)
            self._cancel.clear()
            self._state = self._idle_state()
            self._state.update({"running": True, "stage": "checking", "message": "Getting ready...",
                                "output_dir": folder})
            state_snapshot = dict(self._state)
            self._thread = threading.Thread(target=self._work_check, args=(folder,), daemon=True)
            self._thread.start()
        return state_snapshot

    def _work_check(self, folder):
        try:
            outcome = self._review_pending(folder, on_choose=self._on_choose, on_review=self._on_review,
                                           on_event=self._on_event, should_cancel=self._cancel.is_set)
            left = pipeline.pending_questions(folder)
        except BaseException as e:
            with self._lock:
                self._state.update(running=False, stage="error", error=str(e) or e.__class__.__name__)
            return
        with self._lock:
            swapped = outcome.get("swapped") or 0
            message = (f"Checked. {swapped} song{'' if swapped == 1 else 's'} swapped for a better upload."
                       if swapped else "Checked.")
            if left:
                message += f" {left} still to check."
            self._state.update(running=False, stage="done", percent=100, message=message,
                               outputs=list(outcome.get("moved", {}).values()),
                               to_check=left, checks_in=folder if left else "",
                               cancelled=self._cancel.is_set())

    def is_playlist(self, url: str) -> bool:
        """Whether the page should offer to number the songs this link
        brings - asked as the link is typed, so it goes by the link's shape
        and never the network (see pipeline.is_playlist_link)."""
        return pipeline.is_playlist_link(url)

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

    def unfinished(self) -> dict | None:
        """The playlist a run was still working through when the app last
        quit, crashed or lost power - {"name", "done", "total"} - or None.

        How far it got is counted from the playlist folder's own record, so
        it's right however the run ended. A folder that's gone (an external
        drive not plugged in) isn't offered: there'd be nowhere to carry
        on into."""
        with self._lock:
            if self._state.get("running"):
                return None
        saved = _load_run()
        if not saved or not os.path.isdir(saved.get("folder") or ""):
            return None
        record = sources.Sources(saved["folder"])
        done = sum(1 for key in record.songs if record.finished(key))
        total = saved.get("total") or 0
        if total and done >= total:
            _forget_run()
            return None
        return {"name": saved.get("name") or os.path.basename(saved["folder"]),
                "done": done, "total": total}

    def resume_unfinished(self) -> dict:
        """Carry on with unfinished(): the same link (or the same pasted
        list) with the same options, which skips every song already done."""
        saved = _load_run()
        if not saved:
            return self._fail("There's nothing to carry on with.")
        return self.start(saved["url"], dict(saved.get("options") or {}))

    def forget_unfinished(self) -> bool:
        _forget_run()
        return True

    def quit_warning(self) -> str:
        """What to say when the window is closed mid-run, or "" if nothing's
        running. A playlist says where it got to and that it'll pick up."""
        with self._lock:
            state = dict(self._state)
        if not state.get("running"):
            return ""
        index, total = state.get("index"), state.get("total")
        if index and total and total > 1:
            return (f"Beat Thief is on song {index:,} of {total:,}. Quit anyway? "
                    "Everything finished is saved, and it'll offer to carry on next time.")
        return "Beat Thief is still working. Quit anyway?"

    def _remember_progress(self, event) -> None:
        """Keep the saved run (RUN_STATE_PATH) and the quit warning up to
        date as a playlist goes."""
        stage = event["stage"]
        if stage == "folder" and self._started:
            self._saved = dict(self._started, folder=event["output_dir"], name=event.get("name"),
                               total=event.get("total"), done=0, started=time.time())
            _save_run(self._saved)
        index, total = event.get("index"), event.get("total") or event.get("queue_total")
        if index and total and stage in ("resolving", "found", "offline", "blocked", "disk-full"):
            with self._lock:
                self._state["index"], self._state["total"] = index, total
            if self._saved and index % RUN_STATE_EVERY == 0 and self._saved.get("done") != index - 1:
                self._saved["done"] = index - 1
                _save_run(self._saved)

    def open_file(self, path: str) -> bool:
        """Open a file in whatever opens it - Sources.csv in Numbers."""
        if not path or not os.path.isfile(path):
            return False
        subprocess.run(["open", path], check=False)
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

    def audition_song(self, song_path: str) -> dict:
        """The whole song, to swap in for the drum stem while marking.

        offset is where stem time's zero sits in the song - the beat-1 trim
        every stem had cut off its front (instrument_isolator.song_trim_ms).
        The page plays the song from stem time + offset, so the marks stay in
        stem time and the swap lands on the same beat. Only what playing it
        needs goes back; the song's peaks and kicks are never drawn."""
        try:
            prepared = audition.preview(song_path)
            offset_ms = instrument_isolator.song_trim_ms(song_path)
        except Exception as e:
            return {"error": str(e) or e.__class__.__name__}
        return {
            "audio": prepared["audio"],
            "lead": prepared["lead"],
            "duration": prepared["duration"],
            "offset": offset_ms / 1000.0,
        }

    def steal_beat(self, wav_path: str, start_sec: float, end_sec: float,
                    outputs: str = "both", on_phase=None, stems=()) -> dict:
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
        defaults to doing nothing.

        stems names the other stems - "bass", "harmony", "vocals" - to loop
        over the same bars, cut from the loop's own span (after any nudge
        onto a kick), so they line up with the drum loop to the sample. The
        span is kept beside the loop either way, for loop_stems later."""
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
            beat_loop.write_span(mid_path, loop)
            loops, missing = self._cut_stem_loops(
                song_dir, mid_path, loop.origin_sec, loop.beat.duration_sec, stems)
        except Exception as e:
            return {"error": str(e) or e.__class__.__name__}

        return {
            "path": path,
            "name": os.path.basename(path),
            "loops": loops,
            "missing": missing,
            "bars": loop.bars,
            # Where the cut actually landed in the stem, and how long it
            # runs. Reported because the picker marks a start and the build
            # is still free to move it (see beat_loop._kick_downbeat), and a
            # move nobody is told about is how a loop ends up starting
            # somewhere you never heard it start.
            "origin": round(loop.origin_sec, 4),
            "duration": round(loop.beat.duration_sec, 4),
            "tempo": round(loop.tempo, 1),
            "song_tempo": round(loop.song_tempo, 1),
            "hits": loop.hits_used,
            # Hits nothing was detected for, worked out from the pulse the
            # rest of that voice is on (see groove_reader). Reported because
            # a stage that adds notes to your beat shouldn't do it quietly.
            "inferred": loop.hits_inferred,
            "pieces": sorted({hit.piece for hit in loop.beat.hits}),
        }

    def steal_beat_start(self, wav_path: str, start_sec: float, end_sec: float,
                          outputs: str = "both", stems=None) -> dict:
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
            args=(wav_path, start_sec, end_sec, outputs, list(stems or [])),
            daemon=True,
        )
        self._beat_thread.start()
        return snapshot

    def loop_stems(self, song_path: str, stems: list) -> dict:
        """Loop stems over the bars of the song's newest drum loop, without
        marking anything: the span that loop was cut from is kept beside it
        (see steal_beat). Quick - it's a cut per stem, nothing is built - so
        it answers directly rather than through a poller."""
        try:
            song_dir = instrument_isolator.song_output_dir(song_path)
            files = [os.path.join(song_dir, n) for n in os.listdir(song_dir)]
            beat_path = pipeline._newest_stolen_beat(files, ".wav") or pipeline._newest_stolen_beat(files, ".mid")
            span = beat_loop.read_span(beat_path) if beat_path else None
            if span is None:
                return {"error": "That drum loop doesn't remember its bars — mark the beat again to loop the other stems over it."}
            loops, missing = self._cut_stem_loops(
                song_dir, beat_path, span["origin_sec"], span["duration_sec"], stems)
        except Exception as e:
            return {"error": str(e) or e.__class__.__name__}
        return {"loops": loops, "missing": missing, "bars": span["bars"], "tempo": round(span["tempo"], 1)}

    @staticmethod
    def _cut_stem_loops(song_dir, beat_path, origin_sec, duration_sec, stems):
        """Cut each of stems over [origin_sec, +duration_sec], named after
        the drum loop at beat_path. Returns (paths written, stems that have
        no isolated wav to cut from)."""
        names = sorted(os.listdir(song_dir))
        loops, missing = [], []
        for stem in stems:
            label = pipeline._INSTRUMENTS[stem][1]
            source = next((os.path.join(song_dir, n) for n in names
                           if label in n and n.endswith(".wav")), None)
            if source is None:
                missing.append(stem)
                continue
            loops.append(beat_loop.cut(source, origin_sec, duration_sec,
                                       beat_loop.stem_loop_path(beat_path, stem)))
        return loops, missing

    def beat_status(self) -> dict:
        """The current state of a steal_beat_start() build, polled by the
        page a few times a second while the picker is waiting on it."""
        with self._beat_lock:
            return dict(self._beat_state)

    def _set_beat_phase(self, phase: str) -> None:
        with self._beat_lock:
            self._beat_state["phase"] = phase

    def _steal_beat_work(self, wav_path, start_sec, end_sec, outputs="both", stems=()):
        result = self.steal_beat(wav_path, start_sec, end_sec, outputs=outputs,
                                 on_phase=self._set_beat_phase, stems=stems)
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

    def _work(self, *args, **kwargs):
        """A run, with the Mac kept awake for it - and, once it's over, the
        saved playlist forgotten if it got all the way through."""
        awake = _keep_awake()
        try:
            self._do_work(*args, **kwargs)
        finally:
            _let_sleep(awake)
        with self._lock:
            stage = self._state.get("stage")
        if self._saved and stage in ("done", "too-long"):
            _forget_run()

    def _do_work(self, url, output_dir, instruments, song="", number=False, run_options=None):
        run_options = dict(run_options or {})
        sanitize = run_options.pop("sanitize", "auto")
        # Ask puts every doubt to the page - which recording, where the song
        # starts. Auto and Off ask nothing: the best match and the
        # algorithm's own trim, so a long playlist can run unattended.
        asking = sanitize == "ask"
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
                    on_review=self._on_review if asking else None,
                    on_choose=self._on_choose if asking else None,
                    sanitize=sanitize != "off",
                    number=number,
                    # A playlist in Ask mode keeps going, and puts its
                    # questions to you at the end rather than at song 3.
                    ask_later=asking,
                    **run_options,
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
            self._state["sources_csv"] = result.get("sources_csv") or ""
            self._state["to_check"] = result.get("to_check") or 0
            self._state["checks_in"] = result.get("output_dir") or "" if result.get("to_check") else ""
            if result.get("error"):
                self._state["stage"] = "error"
                self._state["error"] = result["error"]
            elif result.get("too_long"):
                # Not an error: the page explains how to paste the whole
                # playlist, or offers to take the songs the link did list.
                too_long = result["too_long"]
                self._state["stage"] = "too-long"
                self._state["too_long"] = too_long
                count = too_long.get("count")
                self._state["message"] = (
                    f"This playlist has {count:,} songs - Spotify's link only gives {too_long['listed']}."
                    if count else
                    f"Spotify's link stops at {too_long['listed']} songs - this playlist may be longer.")
            elif result.get("cancelled"):
                self._state["stage"] = "cancelled"
                self._state["message"] = ("Stopped. Anything already finished is saved - paste "
                                          "the same link again to carry on."
                                          if (result.get("total") or 0) > 1 else
                                          "Stopped. Anything already finished is saved.")
            else:
                self._state["stage"] = "done"
                self._state["percent"] = 100
                self._state["message"] = self._done_message(result, bool(song))

    def _work_tidy(self, folder):
        awake = _keep_awake()
        try:
            self._do_work_tidy(folder)
        finally:
            _let_sleep(awake)

    def _do_work_tidy(self, folder):
        try:
            result = self._sanitize_existing(
                folder,
                on_event=self._on_event,
                on_review=self._on_review,
                should_cancel=self._cancel.is_set,
                interactive=False,
            )
        except BaseException as e:
            with self._lock:
                self._state["running"] = False
                self._state["stage"] = "error"
                self._state["error"] = str(e) or e.__class__.__name__
            return

        with self._lock:
            self._state["running"] = False
            self._state["outputs"] = result.get("outputs", [])
            self._state["cancelled"] = bool(result.get("cancelled"))
            if result.get("cancelled"):
                self._state["stage"] = "cancelled"
                self._state["message"] = "Stopped. The songs already tidied stay tidied."
            else:
                self._state["stage"] = "done"
                self._state["percent"] = 100
                count = len(result.get("songs", []))
                self._state["message"] = (f"Tidied {count} song{'' if count == 1 else 's'}."
                                          if count else "No songs in that folder.")

    @staticmethod
    def _done_message(result: dict, from_stash: bool = False) -> str:
        if from_stash:
            # Nothing was downloaded because there was nothing to download,
            # so neither counter means here what it means after a run.
            return "Done." if result.get("outputs") else "Nothing came back for that song."
        total = result.get("total") or 0
        if total > 1:
            # A playlist: how many of its songs are here now, and what
            # didn't work - so a run that met trouble doesn't just say Done.
            have = (result.get("finished") or 0) + (result.get("already") or 0)
            failed, to_check = result.get("failed") or 0, result.get("to_check") or 0
            message = f"Done - all {total:,} songs." if have >= total else f"Done - {have:,} of {total:,} songs."
            if failed:
                message += f" {failed:,} couldn't be had - pasting it again tries them again."
            if to_check:
                message += f" {to_check:,} {'match is a guess' if to_check == 1 else 'matches are guesses'} worth checking."
            return message
        if result.get("downloaded"):
            return "Done."
        if result.get("songs"):
            return "Done (you already had this one downloaded)."
        return "Nothing came back for that link."

    def _on_event(self, event):
        stage = event["stage"]
        self._remember_progress(event)
        message, percent = self._describe(event)
        with self._lock:
            if message is not None:
                self._state["message"] = message
            self._state["stage"] = stage
            self._state["percent"] = percent
            if stage == "error":
                self._state["error"] = event.get("message", "")
            if event.get("problem"):
                self._state["problem_count"] += 1
                self._state["problems"] = (self._state["problems"] + [event["message"]])[-MAX_PROBLEMS:]

    @staticmethod
    def _which_song(event) -> str:
        index, total = event.get("index"), event.get("total")
        return f" with song {index:,} of {total:,}" if index and total and total > 1 else ""

    @staticmethod
    def _so_far(event) -> float | None:
        index, total = event.get("index"), event.get("total")
        return (index - 1) / total * 100 if index and total and total > 1 else None

    @staticmethod
    def _describe(event) -> tuple[str | None, float | None]:
        """One line of plain English for the page, plus a percentage when
        there's a real one to show. Returning None for the message leaves
        whatever was there - better than flickering to a blank line."""
        stage = event["stage"]

        if stage == "looking-up":
            return "Looking up that link...", None
        if stage == "resuming":
            # Picking a long playlist back up: say how far it already got
            # before the first song that still needs doing.
            done, total = event.get("done") or 0, event.get("total") or 0
            return f"Picking up at song {done + 1:,} of {total:,}", done / total * 100 if total else None
        if stage == "resolving":
            # A playlist is looked up one song at a time and that takes a
            # while, so it says which song rather than sitting on one line.
            return f"Matching {event['index']} of {event['total']} — {event['song']}", None
        if stage == "found":
            song = event.get("song")
            index, total = event.get("index"), event.get("queue_total")
            if index and total and total > 1:
                # Between two songs of a playlist the bar holds its place
                # rather than going blank and starting over.
                return f"Song {index} of {total} — {song or 'next song'}", (index - 1) / total * 100
            return (f"Found {song}" if song else "Downloading..."), None
        if stage in ("downloading", "downloaded"):
            index, total = event.get("index"), event.get("total")
            percent = 100.0 if stage == "downloaded" else event.get("percent")
            if not (index and total and total > 1):
                verb = "Downloaded" if stage == "downloaded" else "Downloading"
                return f"{verb} {event['song']}", percent
            # A playlist: one bar for the whole thing, since each song's
            # own 0-100 told you nothing about how far through twelve of
            # them you were. The song's own progress stays in the line.
            overall = ((index - 1) + (percent or 0) / 100) / total * 100
            of_it = f" ({round(percent)}%)" if stage == "downloading" and percent is not None else ""
            return f"Song {index} of {total} — {event['song']}{of_it}", overall
        if stage == "checking":
            return (f"Checking {event['index']:,} of {event['total']:,} — {event['song']}",
                    (event["index"] - 1) / event["total"] * 100 if event.get("total") else None)
        if stage == "offline":
            where = Api._which_song(event)
            return f"No internet - waiting to carry on{where}. It'll pick up by itself.", Api._so_far(event)
        if stage == "blocked":
            minutes, seconds = divmod(int(event.get("seconds") or 0), 60)
            return (f"YouTube asked for a break - trying again in {minutes}:{seconds:02d}"
                    f"{Api._which_song(event)}."), Api._so_far(event)
        if stage == "disk-full":
            free = event.get("free")
            left = f" ({free / 1e9:.1f} GB left)" if free is not None else ""
            return (f"Your disk is nearly full{left} - free up some space and it'll "
                    "carry on by itself."), Api._so_far(event)
        if stage == "download-failed":
            return f"Couldn't download {event['song']}", None
        if stage == "download-summary":
            return None, None
        if stage == "sanitizing":
            return "Cleaning it up...", None
        if stage == "tempo":
            return "Reading the tempo...", None
        if stage == "tidying":
            total = event.get("total") or 0
            index = event.get("index") or 0
            percent = (index - 1) / total * 100 if total else None
            return f"Tidying {index} of {total} — {event['song']}", percent
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


def _choose_folder() -> str:
    """The folder picked in a Finder dialog, starting in the stash, or ""
    if it was closed."""
    import webview  # here, not at the top, so the Api stays testable without it
    if not webview.windows:
        return ""
    chosen = webview.windows[0].create_file_dialog(webview.FileDialog.FOLDER,
                                                    directory=DEFAULT_OUTPUT)
    if not chosen:
        return ""
    return chosen[0] if isinstance(chosen, (list, tuple)) else str(chosen)


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
    window = webview.create_window(
        APP_NAME,
        UI_FILE,
        js_api=api,
        width=WINDOW_SIZE[0],
        height=WINDOW_SIZE[1],
        min_size=(420, 520),
    )
    window.events.closing += lambda: _may_close(api)
    webview.start()


def _may_close(api) -> bool:
    """Asked as the window closes: False keeps it open. Mid-run, closing
    is checked with whoever's closing it - a 5,000-song playlist shouldn't
    stop because of a stray ⌘Q, though it would carry on next time.

    Called on the main thread, where pywebview's own confirmation dialog
    would wait on the main thread forever, so the alert is put up directly."""
    warning = api.quit_warning()
    if not warning:
        return True
    try:
        from webview.platforms.cocoa import BrowserView
        return bool(BrowserView.display_confirmation_dialog("Quit", "Keep going", warning))
    except Exception:
        return True


if __name__ == "__main__":
    main()

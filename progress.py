"""Where a run is, as one bar that never loses its place, and how long is left.

The pipeline reports what it's doing one step at a time - downloading song
3, cleaning it up, reading its tempo, separating its drums - and most of
those steps say nothing about which song they belong to or how far through
the whole run that is. Shown as they come, a playlist's bar went blank and
"Song 3 of 12" vanished every time a song moved on from downloading.

RunProgress is fed every one of those events and keeps the picture whole:
which song of how many, which step of that song (and which come next), one
percentage for the whole run that only ever goes forwards, and an estimate
of the time left.

The estimate starts from rough guesses at how long each step takes and
replaces them with what this run actually measured as soon as there's
something measured, so it settles after the first song. Time spent waiting -
offline, YouTube asking for a break, a full disk, a question on the page -
isn't counted as a step taking long. Nothing here knows about the page;
gui.py turns it into state.
"""

from __future__ import annotations

import time

# A first guess, in seconds, at how long each step takes on a song, until
# this run has timed one of its own. The first instrument of a song is the
# demucs pass; every later one reuses that pass (see
# instrument_isolator.separated_stems), so it's quick.
PRIORS = {
    "match": 3.0,
    "download": 12.0,
    "clean": 5.0,
    "tempo": 4.0,
    "first-instrument": 90.0,
    "later-instrument": 10.0,
}

LABELS = {
    "match": "Match",
    "download": "Download",
    "clean": "Clean up",
    "tempo": "Tempo",
    "drums": "Drums",
    "bass": "Bass",
    "harmony": "Harmony",
    "vocals": "Vocals",
}

INSTRUMENTS = ("drums", "bass", "harmony", "vocals")

# A step with no percentage of its own is guessed at from how long it's
# been going - but never past this much of itself, so a step that runs
# long sits still rather than claiming to be finished.
GUESS_CAP = 0.9

# One song's estimate isn't worth showing until it's had this long to
# measure anything.
SINGLE_SONG_SETTLE_SEC = 10.0

# What each pipeline event means for the step in hand.
_STEP_OF = {
    "resolving": "match",
    "found": "download",
    "downloading": "download",
    "downloaded": "download",
    "sanitizing": "clean",
    "tidying": "clean",
    "tempo": "tempo",
}

# Waiting on something outside the run, not the run being slow.
_WAITS = ("offline", "blocked", "disk-full")


def steps_for(sanitize: bool, bpm: bool, instruments=(), download: bool = True) -> list[str]:
    """The steps one song goes through, for a run with these options.
    Match isn't here: only a Spotify playlist matches, and the tracker adds
    it the first time one does."""
    steps = []
    if download:
        steps.append("download")
        if sanitize:
            steps.append("clean")
        if bpm:
            steps.append("tempo")
    steps += [name for name in INSTRUMENTS if name in set(instruments)]
    return steps


def eta_text(seconds: float | None) -> str:
    """Loosely, because it's an estimate: "about 12 min left"."""
    if seconds is None:
        return ""
    if seconds < 10:
        return "almost done"
    if seconds < 60:
        return f"about {max(10, int(round(seconds / 10.0)) * 10)} sec left"
    minutes = int(round(seconds / 60.0))
    if minutes < 60:
        return f"about {minutes} min left"
    hours, minutes = divmod(minutes, 60)
    return f"about {hours} hr {minutes} min left" if minutes else f"about {hours} hr left"


class RunProgress:
    """Fed every event of one run (feed) and asked what to show (view).

    clock is injectable so tests can move time by hand."""

    def __init__(self, steps, total: int = 1, then: str = "", clock=time.monotonic):
        self.steps = list(steps)
        self.total = max(int(total or 1), 1)
        # A stage after the last song - Ask mode's questions - that's
        # mentioned but not timed: how long it takes is up to whoever's
        # answering.
        self.then = then
        self._clock = clock

        self.index = 1
        self.song = ""
        self.step = None            # the step in hand, or None before the first
        self.step_percent = None    # its own percentage, when it has one
        self.manual = False         # on the questions at the end
        self._measured = {}         # category -> [seconds, ...]
        self._songs_timed = 0

        self._paused_since = None
        self._paused_total = 0.0
        self._started = self._active()
        self._step_started = self._started
        self._best = 0.0

    # ------------------------------------------------------------- time

    def _active(self) -> float:
        """Seconds of the run that weren't spent waiting."""
        now = self._clock()
        paused = self._paused_total + (now - self._paused_since if self._paused_since is not None else 0.0)
        return now - paused

    def pause(self) -> None:
        if self._paused_since is None:
            self._paused_since = self._clock()

    def resume(self) -> None:
        if self._paused_since is not None:
            self._paused_total += self._clock() - self._paused_since
            self._paused_since = None

    # ------------------------------------------------------------ steps

    def _category(self, step: str) -> str:
        if step in INSTRUMENTS:
            first = next((s for s in self.steps if s in INSTRUMENTS), None)
            return "first-instrument" if step == first else "later-instrument"
        return step

    def expected(self, step: str) -> float:
        category = self._category(step)
        measured = self._measured.get(category)
        if measured:
            return sum(measured) / len(measured)
        return PRIORS.get(category, PRIORS["download"])

    def _per_song(self) -> float:
        return sum(self.expected(step) for step in self.steps) or 1.0

    def _finish_step(self) -> None:
        """Time the step that just ended."""
        if self.step is not None:
            spent = self._active() - self._step_started
            self._measured.setdefault(self._category(self.step), []).append(max(spent, 0.0))

    def _enter(self, step: str) -> None:
        if step == self.step:
            return
        if step not in self.steps:
            if step != "match":
                return
            # A Spotify playlist: every song is matched before it's fetched.
            self.steps.insert(0, "match")
        self._finish_step()
        self.step = step
        self.step_percent = None
        self._step_started = self._active()

    def _next_song(self, index: int) -> None:
        self._finish_step()
        if self.step is not None:
            self._songs_timed += 1
        self.index = index
        self.step = None
        self.step_percent = None
        self.song = ""

    # ------------------------------------------------------------- feed

    def feed(self, event: dict) -> None:
        stage = event.get("stage")
        if stage in _WAITS:
            self.pause()
            return
        self.resume()

        if stage == "checking":
            # The questions put off to the end: all the songs are done.
            self._finish_step()
            self.manual = True
            self.index, self.step, self.step_percent = self.total, None, None
            self._best = 100.0
            return
        if stage == "resuming":
            done = int(event.get("done") or 0)
            if done:
                self.index = min(done + 1, self.total)
            return

        index = event.get("index")
        total = event.get("total") or event.get("queue_total")
        if stage == "isolating" or stage == "isolated":
            # Their index is the instrument's place, not the song's.
            index = total = None
        if total and total > 1 and total != self.total and stage != "found":
            self.total = int(total)
        if stage == "found" and event.get("queue_total"):
            self.total = int(event["queue_total"])
        if index and (self.total > 1) and int(index) != self.index and int(index) <= self.total:
            if int(index) > self.index:
                self._next_song(int(index))
        if event.get("song"):
            self.song = event["song"]

        if stage in ("isolating", "isolated"):
            step = event.get("instrument")
        else:
            step = _STEP_OF.get(stage)
        if step is None:
            return
        self._enter(step)
        if self.step != step:
            return
        if stage in ("downloaded", "isolated"):
            self.step_percent = 100.0
        elif stage in ("downloading", "isolating"):
            percent = event.get("percent")
            self.step_percent = None if percent is None else max(0.0, min(float(percent), 100.0))

    # ------------------------------------------------------------- view

    def _song_fraction(self) -> float:
        if self.step is None:
            return 0.0
        position = self.steps.index(self.step)
        before = sum(self.expected(step) for step in self.steps[:position])
        return (before + self.expected(self.step) * self._step_fraction()) / self._per_song()

    def _step_fraction(self) -> float:
        if self.step_percent is not None:
            return self.step_percent / 100.0
        spent = self._active() - self._step_started
        return min(spent / self.expected(self.step), GUESS_CAP)

    def percent(self) -> float | None:
        """The whole run, 0-100, never going backwards. None until there's
        a step to go on, while the link itself is still being looked up."""
        if self.manual:
            return None
        if self.step is None and self.index == 1 and self._best == 0.0:
            return None
        overall = ((self.index - 1) + self._song_fraction()) / self.total * 100.0
        self._best = max(self._best, min(overall, 100.0))
        return self._best

    def seconds_left(self) -> float | None:
        """None until the estimate is worth showing: after the first song of
        a playlist, or a few seconds into a single song."""
        if self.manual:
            return None
        if self.total > 1:
            if not self._songs_timed:
                return None
        elif self._active() - self._started < SINGLE_SONG_SETTLE_SEC:
            return None
        per_song = self._per_song()
        this_song = per_song * (1.0 - self._song_fraction())
        return this_song + per_song * (self.total - self.index)

    def step_states(self) -> list[dict]:
        if self.manual:
            return []
        current = self.steps.index(self.step) if self.step in self.steps else -1
        return [{"name": LABELS.get(step, step.title()),
                 "state": "done" if i < current else "now" if i == current else "next"}
                for i, step in enumerate(self.steps)]

    def view(self) -> dict:
        return {
            "percent": self.percent(),
            "step_percent": None if self.manual else self.step_percent,
            "steps": self.step_states(),
            "then": "" if self.manual else self.then,
            "eta": eta_text(self.seconds_left()),
            "index": self.index,
            "total": self.total,
            "song": self.song,
        }

"""The songs waiting their turn while another one is being stolen.

Paste a link, set the pads, click Add to queue: each entry keeps the link
and the settings it was added with, and they run one at a time once the
song in hand is done. A playlist link is one entry - it runs as a playlist
when its turn comes.

Saved to disk on every change, so a quit or a crash doesn't lose it. The
entry being worked on is saved too (as "current") until it's finished, so
after a crash it comes back first rather than vanishing: the pipeline picks
it up from its own record, so nothing finished is done twice.

Not called queue.py: that would shadow the standard library's queue, which
concurrent.futures and urllib3 import."""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from urllib.parse import parse_qs, urlparse

QUEUE_PATH = os.path.join(os.path.expanduser("~"), "Library", "Application Support",
                          "Beat Thief", "queue.json")

_TEMPO_IN_NAME = re.compile(r" \([\d.]+ BPM\)$")


def short_label(url: str, source: str = "") -> str:
    """Something to call an entry before its real title is known."""
    if source:
        name = os.path.splitext(os.path.basename(source))[0]
        return _TEMPO_IN_NAME.sub("", name)
    text = (url or "").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return f"{lines[0]} (and more)"
    parsed = urlparse(text)
    host = parsed.netloc.lower()
    if "youtu.be" in host:
        return f"YouTube {parsed.path.strip('/')}"
    if "youtube" in host:
        video = parse_qs(parsed.query).get("v", [""])[0]
        if video:
            return f"YouTube {video}"
        if "list" in parse_qs(parsed.query):
            return "YouTube playlist"
        return "YouTube link"
    if "spotify" in host:
        kind = next((part for part in parsed.path.split("/")
                     if part in ("track", "album", "playlist")), "link")
        return f"Spotify {kind}"
    return text[:60]


class SongQueue:
    """An ordered list of entries - plain dicts with "id", "url", "options",
    "label", "after" and "added" - kept on disk at path (or only in memory,
    for a path of None)."""

    def __init__(self, path: str | None = QUEUE_PATH):
        self._path = path
        self._lock = threading.Lock()
        self._entries: list[dict] = []
        self._current: dict | None = None
        self._load()

    # --- reading ---------------------------------------------------------

    def entries(self) -> list[dict]:
        with self._lock:
            return [dict(e) for e in self._entries]

    def next(self) -> dict | None:
        with self._lock:
            return dict(self._entries[0]) if self._entries else None

    def last_added(self) -> dict | None:
        with self._lock:
            if not self._entries:
                return None
            return dict(max(self._entries, key=lambda e: (e.get("added") or 0, e.get("seq") or 0)))

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    # --- changing --------------------------------------------------------

    def add(self, entry: dict) -> str:
        with self._lock:
            entry = dict(entry)
            entry.setdefault("id", uuid.uuid4().hex[:12])
            entry.setdefault("label", short_label(entry.get("url", ""), entry.get("source", "")))
            entry.setdefault("after", {})
            entry.setdefault("added", time.time())
            # Two adds in the same clock tick still know which came last.
            entry["seq"] = max([e.get("seq") or 0 for e in self._entries] + [0]) + 1
            self._entries.append(entry)
            self._save()
            return entry["id"]

    def begin(self, entry: dict) -> dict:
        """Work on entry straight away, ahead of anything waiting - saved
        as the one in hand, like a popped entry."""
        with self._lock:
            entry = dict(entry)
            entry.setdefault("id", uuid.uuid4().hex[:12])
            entry.setdefault("label", short_label(entry.get("url", ""), entry.get("source", "")))
            entry.setdefault("after", {})
            entry.setdefault("added", time.time())
            self._current = entry
            self._save()
            return dict(entry)

    def pop_next(self) -> dict | None:
        """Take the first entry to work on. It stays saved as the one in
        hand until finish_current() or put_back_current()."""
        with self._lock:
            if not self._entries:
                return None
            self._current = self._entries.pop(0)
            self._save()
            return dict(self._current)

    def finish_current(self) -> None:
        with self._lock:
            self._current = None
            self._save()

    def put_back_current(self) -> None:
        """Stopped partway: the entry in hand goes back to the front."""
        with self._lock:
            if self._current is not None:
                self._entries.insert(0, self._current)
                self._current = None
                self._save()

    def remove(self, entry_id: str) -> bool:
        with self._lock:
            before = len(self._entries)
            self._entries = [e for e in self._entries if e.get("id") != entry_id]
            if len(self._entries) == before:
                return False
            self._save()
            return True

    def move(self, entry_id: str, delta: int) -> bool:
        with self._lock:
            index = next((i for i, e in enumerate(self._entries) if e.get("id") == entry_id), None)
            if index is None:
                return False
            to = max(0, min(len(self._entries) - 1, index + int(delta)))
            if to != index:
                self._entries.insert(to, self._entries.pop(index))
                self._save()
            return True

    def relabel(self, entry_id: str, label: str) -> None:
        with self._lock:
            for entry in self._entries + ([self._current] if self._current else []):
                if entry.get("id") == entry_id and label:
                    entry["label"] = label
                    self._save()

    def clear(self) -> None:
        with self._lock:
            self._entries = []
            self._current = None
            self._save()

    # --- disk ------------------------------------------------------------

    def _load(self) -> None:
        if not self._path:
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                saved = json.load(f)
        except (OSError, ValueError):
            return
        if not isinstance(saved, dict):
            return
        entries = [e for e in saved.get("entries") or [] if isinstance(e, dict) and e.get("id")]
        current = saved.get("current")
        # Whatever was in hand when the app went away is first in line.
        if isinstance(current, dict) and current.get("id"):
            entries.insert(0, current)
        self._entries = entries

    def _save(self) -> None:
        if not self._path:
            return
        try:
            if not self._entries and self._current is None:
                if os.path.exists(self._path):
                    os.remove(self._path)
                return
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            tmp = self._path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"entries": self._entries, "current": self._current}, f)
            os.replace(tmp, self._path)
        except OSError:
            pass

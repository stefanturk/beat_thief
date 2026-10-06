#!/usr/bin/env python3
"""Why a song couldn't be had - and whether waiting would fix it.

A 5,000-song run spends a day or two online, so it will meet a dropped
connection, YouTube asking it to slow down, and a disk filling up. Each of
those is temporary and means "wait, then try the same song again"; a video
that's been taken down is not, and means "note it and move on". Treating
the first kind like the second is how a lost connection used to burn
through the rest of a playlist in minutes and then say "Done."

Kept to the standard library: pipeline imports this, and pipeline is the
part of the app that has to start without torch."""

from __future__ import annotations

import os
import shutil
import urllib.error
import urllib.request

# Something tiny that's always there when YouTube is: an answer of any kind
# means the connection works.
PROBE_URL = "https://www.youtube.com/generate_204"
PROBE_TIMEOUT_SEC = 6

# Below this much free space a run stops and waits rather than half-writing
# songs. A song is ~10 MB, but demucs wants hundreds of MB of scratch space
# for a song with stems, and macOS itself gets unhappy near empty.
MIN_FREE_BYTES = int(float(os.environ.get("BEAT_THIEF_MIN_FREE_GB", "1")) * 1_000_000_000)

OFFLINE = "offline"
BLOCKED = "blocked"
DISK = "disk"
GONE = "gone"
OTHER = "other"

# Matched against yt-dlp's (lowercased) error text.
_BLOCKED = ("http error 429", "too many requests", "not a bot", "rate limit",
            "rate-limit", "unusual traffic")
_GONE = ("video unavailable", "this video is not available", "private video",
         "has been removed", "has been terminated", "no longer available",
         "not available in your country", "members-only", "join this channel",
         "confirm your age", "age-restricted", "copyright")
_DISK = ("no space left", "errno 28", "disk full", "not enough space")


def online(timeout: float = PROBE_TIMEOUT_SEC) -> bool:
    """Whether YouTube can be reached at all right now."""
    request = urllib.request.Request(PROBE_URL, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            return True
    except urllib.error.HTTPError:
        # It answered, just not with a 2xx - that's a connection.
        return True
    except Exception:
        return False


def free_bytes(path: str) -> int | None:
    """Free space on the disk path is on (or would be on), or None if that
    can't be told."""
    probe = path
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        return shutil.disk_usage(probe or "/").free
    except OSError:
        return None


def disk_nearly_full(path: str) -> bool:
    free = free_bytes(path)
    return free is not None and free < MIN_FREE_BYTES


def classify(error_text: str, folder: str) -> str:
    """Which kind of problem stopped a song, from what went wrong (yt-dlp's
    or Spotify's error text, "" if there wasn't any) and where it was going.

    Space and connection are checked for real rather than read off the
    message: a full disk or a dead connection breaks things in many ways
    that say so in no particular words."""
    text = (error_text or "").lower()
    if any(s in text for s in _DISK) or disk_nearly_full(folder):
        return DISK
    if any(s in text for s in _BLOCKED):
        return BLOCKED
    if any(s in text for s in _GONE):
        return GONE
    if not online():
        return OFFLINE
    return OTHER

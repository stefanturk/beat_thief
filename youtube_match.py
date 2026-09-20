#!/usr/bin/env python3
"""Finding the YouTube recording a Spotify track names.

Searching is the easy half. The hard half is that YouTube will happily offer
five plausible results for one song, and several of them are the wrong
recording: a remaster, a live cut, a sped-up edit. Verified while designing
this - a search for a trivially famous track returned four results within
0.6 s of Spotify's duration, one of them a 2022 remaster.

For this app that distinction is not cosmetic. A remaster or a live take has
its own tempo and its own beat 1, and every grid, loop and MIDI file made
downstream inherits them. So this module's job is not to pick a winner - it's
to be honest about whether there is an obvious one. pick() takes the silent
shortcut when one result is plainly the recording asked for, and otherwise
hands the list back for a person to choose from.

What it deliberately does not do is treat duplicates as doubt. A famous song
comes back as the official video, the Topic upload and a lyric video, all the
same audio to the second - asking which copy to take is a question with no
wrong answer, so it isn't asked.

A remaster is a case of its own. At the same length it is the same
performance with the audio cleaned up - same tempo, same beat 1 - and it is
usually the master Spotify itself is serving, so it is preferred over a
murkier upload of the same song. At a different length it is a different
edit, and gets the same suspicion as everything else here."""

from __future__ import annotations

import re

import yt_dlp

# How far from Spotify's duration a result can be and still be worth showing.
# Four seconds covers the ordinary differences - a video's silent lead-in, a
# fade held a beat longer - without letting in a different arrangement.
DURATION_TOLERANCE_SEC = 4.0

# How close it has to be to be taken without asking when it's the only thing
# nearby. Tighter, because this is the threshold at which nobody gets a say.
CONFIDENT_TOLERANCE_SEC = 2.0

# How close counts as the same recording rather than merely a near one. Inside
# this, a title with nothing odd about it is taken even if other results are
# also close - they're almost always copies of it.
EXACT_TOLERANCE_SEC = 1.0

# Words that mean "this is a different performance of that song" - a new take
# played at its own tempo, with its own beat 1. Any of them in a title is
# enough to ask rather than assume, even on a perfect duration match.
# "Remaster" is not among them: see REMASTER below.
DISQUALIFYING = (
    "live",
    "remix",
    "cover",
    "sped up",
    "spedup",
    "slowed",
    "8d",
    "karaoke",
    "instrumental",
    "reverb",
)

SEARCH_LIMIT = 5


def _search_opts() -> dict:
    # Imported here rather than at the top because pipeline imports this
    # module: at module level the two would chase each other's tails. One
    # silent logger shared is still better than a second copy of it.
    from pipeline import _SilentLogger

    return {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
        "noplaylist": True,
        "extractor_args": {"youtube": {"player_client": ["android"]}},
        "logger": _SilentLogger(),
    }


def candidates(query: str, want_sec: float | None, limit: int = SEARCH_LIMIT) -> list[dict]:
    """What YouTube offers for this query, nearest the wanted duration first.

    Metadata only - nothing is downloaded here. A candidate with no duration
    reported keeps an offset of None and sorts last, rather than being
    dropped: it's still a real result, just one nothing is known about."""
    try:
        with yt_dlp.YoutubeDL(_search_opts()) as ydl:
            info = ydl.extract_info(f"ytsearch{int(limit)}:{query}", download=False)
    except Exception:
        return []

    found = []
    for entry in (info or {}).get("entries") or []:
        if not entry:
            continue
        url = entry.get("url") or entry.get("webpage_url")
        if not url:
            video_id = entry.get("id")
            url = f"https://www.youtube.com/watch?v={video_id}" if video_id else None
        if not url:
            continue
        duration = entry.get("duration")
        try:
            duration = float(duration) if duration is not None else None
        except (TypeError, ValueError):
            duration = None
        offset = (
            abs(duration - want_sec)
            if duration is not None and want_sec is not None
            else None
        )
        found.append(
            {
                "url": url,
                "title": entry.get("title") or "(untitled)",
                "channel": entry.get("uploader") or entry.get("channel") or "",
                "duration": duration,
                "offset": offset,
            }
        )

    # None sorts last: an unknown offset is not a good offset.
    found.sort(key=lambda c: (c["offset"] is None, c["offset"] or 0.0))
    return found


# Whole words, with the ordinary endings allowed: "remastered" counts,
# "Can't Live Without You" and "Deliver Me" do not. A plain substring test
# disqualifies real songs by accident, and every one it catches wrongly is a
# question put to somebody who had nothing to decide.
_DISQUALIFYING_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in DISQUALIFYING) + r")(?:s|d|ed|ing)?\b"
)


def looks_like_another_version(title: str) -> bool:
    return bool(_DISQUALIFYING_RE.search((title or "").lower()))


# A remaster is the same players in the same room; only the mastering differs.
# So at the right length it's wanted rather than avoided - Spotify is usually
# serving the remaster itself, and it's the cleanest audio of the two.
_REMASTER_RE = re.compile(r"\bre-?master(?:s|ed)?\b")


def looks_remastered(title: str) -> bool:
    return bool(_REMASTER_RE.search((title or "").lower()))


def pick(candidates_: list[dict]) -> tuple[str | None, bool]:
    """(url, needs_asking).

    Takes a result silently when its length is within EXACT_TOLERANCE_SEC and
    its title carries no sign of being a different performance - preferring a
    remaster, which at that length is the same performance better mastered -
    or when it is the only result inside CONFIDENT_TOLERANCE_SEC at all.
    Anything else - only a remaster of some other length, two near misses,
    nothing close - comes back needing a person, with the best guess as the
    url so a caller that can't ask still has something to fall back on."""
    if not candidates_:
        return None, True

    # Already sorted nearest-first, so the first of any group is the closest.
    usable = [
        c for c in candidates_
        if c["offset"] is not None and not looks_like_another_version(c["title"])
    ]
    right_length = [c for c in usable if c["offset"] <= EXACT_TOLERANCE_SEC]
    if right_length:
        remastered = [c for c in right_length if looks_remastered(c["title"])]
        return (remastered or right_length)[0]["url"], False

    close = [
        c for c in candidates_
        if c["offset"] is not None and c["offset"] <= CONFIDENT_TOLERANCE_SEC
    ]
    if (
        len(close) == 1
        and not looks_like_another_version(close[0]["title"])
        # A remaster this far out isn't the same master, whatever it says.
        and not looks_remastered(close[0]["title"])
    ):
        return close[0]["url"], False

    return candidates_[0]["url"], True


def worth_offering(candidates_: list[dict], limit: int = SEARCH_LIMIT) -> list[dict]:
    """The candidates to put in front of a person: everything within
    DURATION_TOLERANCE_SEC, plus the next best two so a track whose duration
    is simply mislabelled isn't a dead end."""
    inside = [
        c for c in candidates_
        if c["offset"] is not None and c["offset"] <= DURATION_TOLERANCE_SEC
    ]
    rest = [c for c in candidates_ if c not in inside]
    return (inside + rest[:2])[:limit]

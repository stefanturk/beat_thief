"""What every file Beat Thief makes is called, in one place.

The label always comes first, then the song, then the tempo:

    Drums Isolated - Song - Artist - 104.5 BPM.wav
    Vocals Isolated - Song - Artist.wav
    Drum Beat - Song - Artist - 104.5 BPM.wav       (and the .mid beside it)
    Bass Beat - Song - Artist - 104.5 BPM 2.wav     (a second loop out of one song)
    Song - Artist - 104.5 BPM.mp3                   (the song itself: no label)

Isolated stems are "<Stem> Isolated" and loops are "<Stem> Beat", so all
the same kind of thing reads the same way and sorts together in Finder.

parse() also knows every name Beat Thief has used before, so a folder made
by an older version still reads correctly, and nothing gets redone just
because it was named the old way:

    Song - Artist (104.5 BPM).mp3
    Song - Artist (Isolated Drums at 104.5 BPM).wav
    Song - Artist (Isolated Vocals).wav
    Song - Artist (Beat at 104.5 BPM) (2).wav
    Song - Artist (Bass beat at 104.5 BPM).wav
    Song - Artist (Stolen Beat, 4 bars) (120 BPM).mid

Kept free of imports beyond the standard library: the library listing
reads names without loading any of the audio code.
"""
from __future__ import annotations

import os
import re
import unicodedata
from typing import NamedTuple, Optional

STEMS = ("drums", "bass", "harmony", "vocals", "instrumental")

ISOLATED = {"drums": "Drums Isolated", "bass": "Bass Isolated", "harmony": "Harmony Isolated",
            "vocals": "Vocals Isolated", "instrumental": "Instrumental Isolated"}
BEAT = {"drums": "Drum Beat", "bass": "Bass Beat", "harmony": "Harmony Beat",
        "vocals": "Vocal Beat", "instrumental": "Instrumental Beat"}

# Stems whose isolated file carries the song's tempo. The others are the
# same length as the song and are dragged in as audio, not onto a grid.
TEMPO_IN_ISOLATED = ("drums", "bass")


class Name(NamedTuple):
    kind: str             # "isolated", "beat" or "song"
    stem: Optional[str]   # one of STEMS; None for a song
    title: str            # "Song - Artist" (a playlist number stays on, if it had one)
    tempo: Optional[float]
    nth: int              # 1, or 2, 3... for a second loop out of one song


def bpm_text(tempo: float) -> str:
    """A tempo as it goes into a name: one decimal at most, none for a
    whole number."""
    return f"{round(tempo, 1):g}"


def _join(label: str, title: str, tempo: Optional[float], nth: int = 1) -> str:
    base = f"{label} - {title}" if label else title
    if tempo is not None:
        base += f" - {bpm_text(tempo)} BPM"
    if nth > 1:
        base += f" {nth}"
    return base


def isolated(stem: str, title: str, tempo: Optional[float] = None) -> str:
    """The basename (no extension) of a song's isolated stem."""
    return _join(ISOLATED[stem], title, tempo if stem in TEMPO_IN_ISOLATED else None)


def beat(stem: str, title: str, tempo: float, nth: int = 1) -> str:
    """The basename (no extension) of a loop cut out of a stem."""
    return _join(BEAT[stem], title, tempo, nth)


def song(title: str, tempo: Optional[float]) -> str:
    """The basename of a song named for its tempo."""
    return _join("", strip_song_tempo(title), tempo)


# ---- reading names back -------------------------------------------------

_LABELS = {label: ("isolated", stem) for stem, label in ISOLATED.items()}
_LABELS.update({label: ("beat", stem) for stem, label in BEAT.items()})
_NEW_RE = re.compile(
    r"^(?P<label>" + "|".join(re.escape(l) for l in sorted(_LABELS, key=len, reverse=True)) + r")"
    r" - (?P<title>.+?)(?: - (?P<bpm>\d+(?:\.\d+)?) BPM(?: (?P<nth>\d+))?)?$")

_OLD_STEM_WORDS = {"drums": "drums", "bass": "bass", "harmony": "harmony", "vocals": "vocals",
                   "instrumental": "instrumental"}
_OLD_ISOLATED_RE = re.compile(
    r"^(?P<title>.+?) \(Isolated (?P<stem>Drums|Bass|Harmony|Vocals|Instrumental)"
    r"(?: at (?P<bpm>\d+(?:\.\d+)?) BPM)?\)$")
_OLD_BEAT_RE = re.compile(
    r"^(?P<title>.+?) \((?:(?P<stem>Bass|Harmony|Vocals|Instrumental) beat|Beat) at "
    r"(?P<bpm>\d+(?:\.\d+)?) BPM\)(?: \((?P<nth>\d+)\))?$")
_OLD_STOLEN_RE = re.compile(
    r"^(?P<title>.+?) \(Stolen Beat[^)]*\)(?: \((?P<bpm>\d+(?:\.\d+)?) BPM\))?(?: \((?P<nth>\d+)\))?$")

_SONG_TEMPO_RE = re.compile(r"(?: \((?P<old>\d+(?:\.\d+)?) BPM\)| - (?P<new>\d+(?:\.\d+)?) BPM)$")


def _float(text):
    return float(text) if text else None


def parse(filename: str) -> Name:
    """What a file is, from its name alone. Takes a basename with or
    without its extension, or a whole path. Anything that isn't a stem or
    a loop reads as a song."""
    base = os.path.basename(filename)
    root, ext = os.path.splitext(base)
    if ext.lower() in (".wav", ".mid", ".mp3", ".json"):
        base = root
    m = _NEW_RE.match(base)
    if m:
        kind, stem = _LABELS[m.group("label")]
        return Name(kind, stem, m.group("title"), _float(m.group("bpm")), int(m.group("nth") or 1))
    m = _OLD_ISOLATED_RE.match(base)
    if m:
        return Name("isolated", m.group("stem").lower(), m.group("title"), _float(m.group("bpm")), 1)
    m = _OLD_BEAT_RE.match(base)
    if m:
        stem = (m.group("stem") or "drums").lower()
        return Name("beat", stem, m.group("title"), _float(m.group("bpm")), int(m.group("nth") or 1))
    m = _OLD_STOLEN_RE.match(base)
    if m:
        return Name("beat", "drums", m.group("title"), _float(m.group("bpm")), int(m.group("nth") or 1))
    t = _SONG_TEMPO_RE.search(base)
    tempo = _float((t.group("old") or t.group("new"))) if t else None
    return Name("song", None, strip_song_tempo(base), tempo, 1)


def strip_song_tempo(title: str) -> str:
    """"Song - Artist - 104.5 BPM" or "Song - Artist (104.5 BPM)" -> "Song - Artist"."""
    return _SONG_TEMPO_RE.sub("", title)


def is_isolated(filename: str, stem: Optional[str] = None) -> bool:
    n = parse(filename)
    return n.kind == "isolated" and (stem is None or n.stem == stem)


def is_beat(filename: str, stem: Optional[str] = None) -> bool:
    n = parse(filename)
    return n.kind == "beat" and (stem is None or n.stem == stem)


def belongs_to(filename: str, title: str) -> bool:
    """Whether a stem or loop was made from the song called title. This is
    what keeps two songs whose files share a folder apart."""
    n = parse(filename)
    return n.kind != "song" and n.title == title


def stem_loop_of(drum_beat_path: str, stem: str) -> str:
    """The path of stem's loop over the same bars as a drum loop, keeping
    its number so a second marking's stems stay paired with it:
    "Drum Beat - Song - 104.5 BPM 2.wav" -> "Bass Beat - Song - 104.5 BPM 2.wav".
    Works from an old-style drum loop name too (the new loop gets a new name)."""
    directory, name = os.path.split(drum_beat_path)
    n = parse(name)
    if n.kind != "beat" or n.stem != "drums":
        raise ValueError(f"not a drum loop: {name}")
    return os.path.join(directory, beat(stem, n.title, n.tempo if n.tempo is not None else 120.0, n.nth) + ".wav")


# ---- text in any script ---------------------------------------------------

# Characters a file name can't hold on Windows (and a slash anywhere).
_UNSAFE = set('/\\:*?"<>|')

# Room left for what goes around a title: "Instrumental Isolated - " in
# front, " - 104.5 BPM 2.wav" behind, a hidden ".loop.json" span file, a
# "004 - " playlist number. A Japanese character is three bytes, and the
# disk allows 255 bytes per name.
MAX_TITLE_BYTES = 150


def fold(text: str) -> str:
    """text reduced to what matters when comparing two names: letters and
    digits of any script, lower case, one space between words. "夜に駆ける"
    stays "夜に駆ける" - it used to come out empty, and two empty names
    look identical."""
    text = unicodedata.normalize("NFKC", text or "").casefold()
    out = "".join(c if c.isalnum() else " " for c in text)
    return " ".join(out.split())


def squash(text: str) -> str:
    """fold() with the spaces taken out too."""
    return fold(text).replace(" ", "")


def safe_name(name: str, fallback: str = "Song", max_bytes: int = MAX_TITLE_BYTES) -> str:
    """name fit to be a file or folder name on a Mac and on Windows: no
    slashes or other characters Windows refuses, no control characters, no
    leading or trailing dots or spaces (a leading dot hides it, a trailing
    one Windows drops), and short enough in bytes that every file named
    after it still fits."""
    name = unicodedata.normalize("NFC", name or "")
    cleaned = "".join(" " if (c in _UNSAFE or unicodedata.category(c) == "Cc") else c for c in name)
    cleaned = " ".join(cleaned.split()).strip(" .")
    if len(cleaned.encode("utf-8")) > max_bytes:
        cut = cleaned.encode("utf-8")[:max_bytes].decode("utf-8", "ignore")
        cleaned = cut.rstrip(" .-")
    return cleaned or fallback

#!/usr/bin/env python3
"""The whole beat_thief job, start to finish: download a link, sanitize what
came down, isolate whichever instruments were asked for.

This lives apart from beat_thief.py so the terminal and the GUI can run the
exact same sequence rather than each maintaining its own copy of it - a
divergence between them would show up as the GUI quietly producing different
audio than the CLI for the same song, which is precisely the bug worth
designing out.

Neither front end is assumed: nothing here prints, and nothing here asks
questions. Progress is reported by calling on_event with plain dicts, and the
caller decides whether that becomes a line of terminal text or a progress bar
in a window."""

from __future__ import annotations

import os
import re
import shutil
import time

import yt_dlp

import bass_isolator
import beat_writer
import drum_isolator
import harmony_isolator
import history
import instrument_isolator
import song_sanitizer
import sources
import spotify
import vocals_isolator
import youtube_match

# Which module handles each instrument, and the filename marker its outputs
# carry (see instrument_isolator.find_existing_basename). Keyed by the same
# names the CLI accepts as bare arguments and the GUI shows as checkboxes.
_INSTRUMENTS = {
    "drums": (drum_isolator, drum_isolator._LABEL),
    "bass": (bass_isolator, bass_isolator._LABEL),
    "harmony": (harmony_isolator, harmony_isolator._LABEL),
    "vocals": (vocals_isolator, vocals_isolator._LABEL),
}

# The order instruments are worked through, so two runs asking for the same
# set produce the same sequence regardless of how the set was built.
#
# These four are the whole song between them: demucs separates a mix into
# exactly these sources, so nothing belongs to two of them and nothing to
# none of them. Playing all four gives the song back - not bit-exactly
# (separation is a guess; measured residual on a real track is about -20 dB)
# but with nothing dropped.
INSTRUMENT_ORDER = ("drums", "bass", "harmony", "vocals")

DEFAULT_OUTPUT = os.path.join(os.path.expanduser("~"), "Downloads", "Song Downloads")

# yt-dlp's record of what's already been taken, so pasting the same link
# twice doesn't download it twice. It lists video ids, not files - so it can
# and does go stale (see run), and the file on disk is the authority.
ARCHIVE_FILENAME = ".downloaded_archive.txt"


class _SilentLogger:
    """Swallow yt-dlp's own chatter - progress is reported through on_event."""

    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass


def file_name_for(title: str, artist: str) -> str:
    """"Song - Artist", fit to be a filename, from what Spotify calls the
    song. A slash would make a path of it and a leading dot would hide it;
    a colon shows up in Finder as a slash."""
    name = f"{title} - {artist}" if artist else title
    cleaned = "".join(" " if c in "/\\:" else c for c in name)
    return re.sub(r"\s+", " ", cleaned).strip(" .") or "Song"


def _base_ydl_opts(output_dir: str, name: str | None = None) -> dict:
    """yt-dlp's settings for one download. Given a name, the file is saved
    under it rather than under whatever the upload calls itself - so a
    song Spotify named arrives with Spotify's name on it."""
    stem = name.replace("%", "%%") if name else "%(title)s - %(uploader)s"
    return {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, stem + ".%(ext)s"),
        "noplaylist": True,
        "extractor_args": {"youtube": {"player_client": ["android"]}},
        "quiet": True,
        "no_warnings": True,
        "logger": _SilentLogger(),
    }


def _probe_info(url: str) -> dict | None:
    """yt-dlp's metadata for url, flat - no download, and no per-video
    lookups inside a playlist. None if it can't be had."""
    probe_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
        "noplaylist": True,
        "extractor_args": {"youtube": {"player_client": ["android"]}},
        "logger": _SilentLogger(),
    }
    try:
        with yt_dlp.YoutubeDL(probe_opts) as ydl:
            return ydl.extract_info(url, download=False)
    except Exception:
        return None


def _probe(url: str) -> tuple[int | None, str | None]:
    """How many songs this url resolves to, and the name of the one worth
    saying - only when there's exactly one, since a playlist has no single
    "the" song to name. Metadata only - nothing is downloaded."""
    info = _probe_info(url)
    if info is None:
        return None, None
    entries = info.get("entries")
    if entries is None:
        return 1, (info.get("title") if info else None)
    title = entries[0].get("title") if len(entries) == 1 and entries[0] else None
    return len(entries), title


def youtube_playlist_name(url: str) -> str | None:
    """The name of the playlist this link is, "" for one with no name, or
    None when it isn't a playlist at all.

    Only a link with list= in it is asked about - that's every playlist
    link, and asking about anything else would put a network round trip in
    front of every single-song download for nothing. A watch link taken off
    a playlist carries list= as well, but noplaylist makes it one video, so
    it's the count that decides, not the link."""
    if "list=" not in url:
        return None
    info = _probe_info(url)
    entries = (info or {}).get("entries")
    if entries is None or len(entries) < 2:
        return None
    return info.get("title") or ""


def is_playlist_link(url: str) -> bool:
    """Whether this link looks like more than one song, from its shape
    alone - no network. What the page asks to decide whether numbering the
    songs is worth offering, on every keystroke, so it can't wait on
    YouTube. A watch link taken off a playlist is one song (noplaylist),
    so list= only counts on a link that isn't a single video."""
    url = (url or "").strip()
    found = spotify.spotify_id(url)
    if found:
        return found[0] in ("playlist", "album")
    return "list=" in url and "v=" not in url


def count_entries(url: str) -> int | None:
    """How many songs this url resolves to, or None if that can't be
    determined quickly. Metadata only - nothing is downloaded."""
    return _probe(url)[0]


def requested_mp3_filenames(url: str, output_dir: str, name: str | None = None) -> list[str]:
    """The mp3 filename(s) this url resolves to, whether they were just
    downloaded this run or already sat on disk from a previous one (skipped
    via the download archive).

    Used to scope isolation to what was actually asked for: the list of
    fresh downloads alone misses a request for a song you already have,
    since yt-dlp's archive skip means no download hook ever fires for it."""
    try:
        with yt_dlp.YoutubeDL(_base_ydl_opts(output_dir, name)) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                return []
            entries = info.get("entries") if info.get("entries") is not None else [info]
            filenames = []
            for entry in entries:
                if not entry:
                    continue
                try:
                    raw_path = ydl.prepare_filename(entry)
                except Exception:
                    continue
                filenames.append(os.path.splitext(os.path.basename(raw_path))[0] + ".mp3")
            return filenames
    except Exception:
        return []


class _Download:
    """One yt-dlp run, translating its hooks into on_event calls.

    A class rather than the module-level counters this grew out of: the GUI
    can start a second run in the same process after the first finishes, and
    leftover counters from the previous one would make its summary wrong."""

    def __init__(self, url: str, output_dir: str, on_event, use_archive: bool = True,
                 queue_index: int | None = None, queue_total: int | None = None,
                 known_title: str | None = None, name: str | None = None):
        self.url = url
        # What to save the file as, when that's already known (see
        # _base_ydl_opts) - None leaves it to the upload's own title.
        self.name = name
        self.output_dir = output_dir
        self.on_event = on_event
        self.use_archive = use_archive
        # Where this download sits in a queue of them, when there is one. A
        # Spotify playlist is downloaded one url at a time, so without this
        # every track would report itself as "1 of 1" and the window would
        # count to one eight times.
        self.queue_index = queue_index
        self.queue_total = queue_total
        # What Spotify already said this track is. Knowing it means the
        # metadata probe below can be skipped - a whole network round trip
        # per track, spent learning something already known.
        self.known_title = known_title
        self.total = None
        self.active_title = None
        self.song_number = 0
        self.downloaded = 0
        self.failed = 0
        self.filenames: list[str] = []
        # Where each downloaded file sat in the playlist it came from, by
        # the name it was saved under - for numbering them in order.
        self.numbers: dict[str, int] = {}

    def _position(self) -> tuple[int | None, int | None]:
        """Which song of how many. Counted across the queue when there is
        one, and within this download when there isn't."""
        if self.queue_total:
            return self.queue_index, self.queue_total
        return self.song_number, self.total

    def _progress_hook(self, d):
        title = d.get("info_dict", {}).get("title", "Unknown")
        index, total = self._position()

        if d["status"] == "downloading":
            if title != self.active_title:
                self.active_title = title
                self.song_number += 1
            total_bytes = d.get("total_bytes") or d.get("total_bytes_estimate")
            percent = None
            if total_bytes:
                percent = min(d.get("downloaded_bytes", 0) / total_bytes, 1.0) * 100
            self.on_event(
                {
                    "stage": "downloading",
                    "song": title,
                    "index": index,
                    "total": total,
                    "percent": percent,
                }
            )
        elif d["status"] == "finished":
            self.on_event({"stage": "downloading", "song": title, "index": index,
                           "total": total, "percent": 100.0})
        elif d["status"] == "error":
            self.failed += 1
            self.active_title = None
            self.on_event({"stage": "download-failed", "song": title})

    def _postprocessor_hook(self, d):
        if d["status"] != "finished" or d.get("postprocessor") != "ExtractAudio":
            return
        if self.active_title is None:
            return
        info = d.get("info_dict", {})
        self.downloaded += 1
        filepath = info.get("filepath")
        if filepath:
            # ExtractAudio reports the file it converted *from* - the .mp4 or
            # .webm that came down the wire, which by now has been replaced by
            # the mp3 beside it. Passing that name on meant the sanitizer was
            # handed a file that doesn't exist, quietly returned nothing, and
            # the song was never filed, never remembered and never appeared in
            # the stash. The extension is ours to know: we asked for mp3.
            filename = os.path.splitext(os.path.basename(filepath))[0] + ".mp3"
            self.filenames.append(filename)
            if info.get("playlist_index"):
                self.numbers[filename] = int(info["playlist_index"])
        index, total = self._position()
        self.on_event({"stage": "downloaded", "song": info.get("title", "Unknown"),
                       "index": index, "total": total})
        self.active_title = None

    def run(self) -> int:
        os.makedirs(self.output_dir, exist_ok=True)

        if self.known_title:
            self.total, title = 1, self.known_title
        else:
            self.on_event({"stage": "looking-up"})
            self.total, title = _probe(self.url)
        self.on_event({"stage": "found", "total": self.total, "song": title,
                       "index": self.queue_index, "queue_total": self.queue_total})

        opts = _base_ydl_opts(self.output_dir, self.name)
        if self.use_archive:
            opts["download_archive"] = os.path.join(self.output_dir, ARCHIVE_FILENAME)
        opts.update(
            {
                "ignoreerrors": True,
                "noprogress": True,
                "progress_hooks": [self._progress_hook],
                "postprocessor_hooks": [self._postprocessor_hook],
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "320",
                    }
                ],
            }
        )
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.download([self.url])


def _nothing_on_disk_for(url: str, output_dir: str, name: str | None = None) -> bool:
    """True when this url resolves to songs and none of them are here.

    The question behind the stale-archive retry in run(). Unknowable is not
    the same as missing: if the url can't be resolved to any filename at all
    (offline, a dead link) this says False, so a network problem can't turn
    into a second download attempt."""
    filenames = requested_mp3_filenames(url, output_dir, name)
    if not filenames:
        return False
    return not any(existing_song(output_dir, filename) for filename in filenames)


def song_folder(output_dir: str, title: str) -> str:
    """Where everything for one song lives - the mp3 itself, its stems and
    its MIDI, all together. One folder per song rather than a flat pile of
    mp3s next to a parallel pile of "(Isolated)" folders."""
    return os.path.join(output_dir, title)


# What numbering puts in front of a song: "004 - ".
_NUMBER_RE = re.compile(r"^\d{3} - ")


def existing_song(output_dir: str, filename: str) -> str:
    """Where this song's mp3 actually is, or "" if it isn't there.

    Checks its own folder first and then the top level, because a download
    lands flat and is filed afterwards (see file_into_own_folder) - so
    between those two moments both are correct answers.

    filename is what yt-dlp calls the song, which is all a link can tell
    us. Sanitizing may since have renamed it - tidied the title, then put
    the tempo on the end - so each of those names is looked for too, worked
    out the same way the sanitizer did it."""
    title = os.path.splitext(filename)[0]
    _, _, tidied = song_sanitizer._derive_title_artist(filename)
    titles = [title] + ([tidied] if tidied and tidied != title else [])
    for name in titles:
        for candidate in (os.path.join(song_folder(output_dir, name), name + ".mp3"),
                          os.path.join(output_dir, name + ".mp3")):
            if os.path.exists(candidate):
                return candidate
    try:
        names = sorted(os.listdir(output_dir))
    except OSError:
        return ""
    # A folder or a flat mp3 whose name is one of those titles with a tempo
    # on the end. (Not splitext on a folder: "(104.5 BPM)" has a dot in it.)
    # Numbered in playlist order too, perhaps: "004 - Song - Artist".
    for name in names:
        stem = name[:-len(".mp3")] if name.endswith(".mp3") else name
        bare = instrument_isolator.song_title(_NUMBER_RE.sub("", stem, count=1) + ".mp3")
        if bare == stem or bare not in titles:
            continue
        for candidate in (os.path.join(output_dir, stem, stem + ".mp3"),
                          os.path.join(output_dir, stem + ".mp3")):
            if os.path.isfile(candidate):
                return candidate
    return ""


def file_into_own_folder(mp3_path: str) -> str:
    """Move a freshly downloaded mp3 into a folder of its own, and return
    where it ended up.

    Downloading and sanitizing both work on a flat directory - the
    sanitizer renames, dedupes and compares across the whole set of mp3s at
    once - so filing happens afterwards rather than by downloading straight
    into place. Everything downstream then reads the folder off the mp3's
    own path (see instrument_isolator.song_output_dir).

    Already-filed and can't-be-filed both return the path unchanged: a song
    is worth isolating either way, and losing one to a tidying step would
    be a poor trade."""
    output_dir = os.path.dirname(os.path.abspath(mp3_path))
    filename = os.path.basename(mp3_path)
    title = os.path.splitext(filename)[0]
    if os.path.basename(output_dir) == title:
        return mp3_path

    destination = os.path.join(song_folder(output_dir, title), filename)
    if os.path.exists(destination):
        return destination
    try:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.move(mp3_path, destination)
    except OSError:
        return mp3_path
    return destination


def _instrument_outputs(mp3_path: str, label: str) -> list[str]:
    """The files an isolator just produced for this song and instrument, so
    a caller can offer them directly instead of making someone go looking."""
    song_dir = instrument_isolator.song_output_dir(mp3_path)
    basename = instrument_isolator.find_existing_basename(song_dir, label)
    if basename is None:
        return []
    # Just the wav. An isolator produces nothing else now that whole-song
    # MIDI is gone - and a leftover .mid from an older version alongside it
    # is exactly the stale file that shouldn't be offered as fresh output.
    wav_path = os.path.join(song_dir, basename + ".wav")
    return [wav_path] if os.path.exists(wav_path) else []


# The things a song can have, in the order they're shown. "song" is the
# download itself; "beat" is a stolen loop's trimmed .wav and "midi" is its
# .mid - the same steal produces either or both, but as two separate files
# now (see gui.Api.steal_beat's outputs param), so each gets its own square.
# The middle four are the stems. This order is the app's, top to bottom and
# left to right.
#
# Each "<stem>_beat" is a loop of that stem cut over a drum loop's bars (see
# beat_writer.stem_loop_path).
STASH_ORDER = ("song", "drums", "beat", "midi", "bass", "harmony", "vocals",
               "bass_beat", "harmony_beat", "vocals_beat")


def _newest_stolen_beat(files: list[str], ext: str) -> str | None:
    """The most recently made stolen-loop file with this extension, or None.

    A song can have several beats stolen out of it, and the one worth
    offering is the one just made - so the newest wins rather than whichever
    sorted first. A file that went away between the listing and here sorts
    last rather than raising: this is a description of a folder, and a
    missing file is a thing to leave out, not to fail over."""
    matches = [p for p in files if p.endswith(ext) and beat_writer.is_stolen_beat(os.path.basename(p))]
    return max(matches, key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0.0) if matches else None


def _what_a_song_has(song_path: str, files: list[str]) -> dict:
    """Which of STASH_ORDER this song already has, and the file for each.

    A path rather than a flag, because a front end wants both: which
    squares to fill in, and what to reveal when one is clicked."""
    have = {"song": song_path}
    for name, (_module, label) in _INSTRUMENTS.items():
        match = next((p for p in files if label in os.path.basename(p) and p.endswith(".wav")), None)
        if match:
            have[name] = match
    wav_beat = _newest_stolen_beat(files, ".wav")
    if wav_beat:
        have["beat"] = wav_beat
    midi_beat = _newest_stolen_beat(files, ".mid")
    if midi_beat:
        have["midi"] = midi_beat
    for stem, label in beat_writer.STEM_LOOP_LABELS.items():
        marker = f"({label} at "
        loops = [p for p in files if marker in os.path.basename(p) and p.endswith(".wav")]
        if loops:
            have[stem + "_beat"] = max(loops, key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0.0)
    return have


def has_loop_span(have: dict) -> bool:
    """Whether the song's newest drum loop remembers the bars it was cut
    from - a loop made before that was kept has to be marked again before
    the other stems can be looped over it."""
    return bool(have.get("beat")) and beat_writer.read_span(have["beat"]) is not None


def library(limit: int = 20) -> list[dict]:
    """Recently downloaded songs, newest first, each with the link it came
    from and whatever files exist for it right now.

    Files are read from disk rather than remembered, so a stem deleted in
    Finder simply stops being listed - and a front end showing what's in
    the stash can't drift from what's actually there."""
    songs = []
    for entry in history.entries():
        if len(songs) >= limit:
            break
        song_path = entry["song"]
        # A song that isn't on disk any more has nothing to offer, and
        # listing it would cost one of the slots a real song wants. Filtered
        # here rather than trimmed first, so a run of dead entries can't
        # crowd out the songs behind them.
        if not os.path.exists(song_path):
            continue

        song_dir = instrument_isolator.song_output_dir(song_path)
        files = [song_path]
        try:
            names = sorted(os.listdir(song_dir)) if os.path.isdir(song_dir) else []
        except OSError:
            # macOS blocks apps outside their own sandbox from reading
            # ~/Downloads, ~/Desktop and ~/Documents without a permission
            # grant that a python3 subprocess can't reliably get (see
            # make_app.sh) - so a song the terminal front end downloaded to
            # ~/Downloads can exist and still not be listable from here.
            # One unreadable folder used to take the whole library down
            # with it; now it's treated the same as a song that isn't on
            # disk, since none of its stems could be reached from here
            # either.
            continue
        for name in names:
            # Skip the .source.json markers the isolators keep for their
            # own "is this still up to date" checks - not output anyone
            # asked for. The mp3 lives in here too now, so it would
            # otherwise be listed twice.
            path = os.path.join(song_dir, name)
            if name.startswith(".") or path == song_path:
                continue
            files.append(path)

        have = _what_a_song_has(song_path, files)
        songs.append(
            {
                "title": instrument_isolator.song_title(song_path),
                "url": entry.get("url", ""),
                "song": song_path,
                # Where everything for this song lives - the mp3, the stems
                # and the MIDI. What a front end wants to open, since the
                # stems get dragged out of it together.
                "dir": song_dir if os.path.isdir(song_dir) else "",
                "have": have,
                # The other stems can be looped over the newest drum loop's
                # bars without marking them again.
                "span": has_loop_span(have),
                "files": files,
            }
        )
    return songs


def _safe_folder_name(name: str) -> str:
    """A playlist's name, fit to be a folder. Slashes would make a path out
    of a name, and a leading dot would hide the folder from Finder."""
    cleaned = "".join(" " if c in "/\\:" else c for c in name).strip(" .")
    return cleaned or "Playlist"


def _resolve_one(song: dict, index, total, on_event, on_choose, cancelled) -> tuple[dict | None, str | None]:
    """The YouTube upload for one Spotify track, or (None, why not).

    The match comes back as {"url", "title", "channel", "guessed"}, where
    "title" and "channel" are the upload's own (what Sources.csv shows next
    to Spotify's name) and "guessed" says nobody confirmed it.

    Where the match is unambiguous this is silent. Where it isn't - a
    remaster at the same length, two uploads equally close - it's put to
    on_choose, because picking the wrong recording here quietly poisons
    every tempo, grid and MIDI file made from it further down.

    Without an on_choose (the CLI, and the app's Auto mode) the best
    candidate is taken, named in a warning and marked as a guess, so a
    5,000-song run can go overnight and still say what it assumed."""
    # The release itself, from YouTube Music, when it's plainly the one -
    # which it usually is. Only when it isn't does the broader search run.
    released = youtube_match.music_match(song["query"], song["duration_sec"], song["artist"])
    if released:
        return {"url": released["url"], "title": released.get("title", ""),
                "channel": released.get("channel", ""), "guessed": False}, None

    found = youtube_match.candidates(song["query"], song["duration_sec"])
    if not found:
        return None, (
            f"Couldn't find \"{song['query']}\" on YouTube, which is where the "
            "audio has to come from - Spotify only says what the song is."
        )

    def described(url, guessed):
        chosen = next((c for c in found if c.get("url") == url), {})
        return {"url": url, "title": chosen.get("title", ""),
                "channel": chosen.get("channel", ""), "guessed": guessed}

    best, needs_asking = youtube_match.pick(found)
    if not needs_asking:
        return described(best, False), None

    if on_choose is None:
        on_event({
            "stage": "warning",
            "message": (
                f"More than one YouTube version matched \"{song['query']}\" - "
                f"took \"{found[0]['title']}\"."
            ),
        })
        return described(best, True), None

    if cancelled():
        return None, None

    answer = on_choose({
        "query": song["query"],
        "title": song["title"],
        "artist": song["artist"],
        "want_sec": song["duration_sec"],
        "candidates": youtube_match.worth_offering(found),
        "index": index,
        "total": total,
    }) or {}
    picked = answer.get("url") or None
    if picked:
        return described(picked, False), None
    return None, f"Skipped {song['title']}."


def _spotify_queue(url, on_event, take_listed) -> dict:
    """What a Spotify link or a pasted copy of a playlist says the songs
    are - {"name", "songs", "is_playlist"} - without going near YouTube.
    Matching each song happens when it's that song's turn (see run).

    A pasted list carries only track links, so each song's title is read
    when its turn comes too; a playlist link already lists them.

    A playlist longer than Spotify's link will list (EMBED_ROW_LIMIT) is
    "too_long" rather than quietly cut short, unless take_listed says the
    listed songs are what's wanted - the way past it is pasting the whole
    list, and that's the page's to explain."""
    pasted = spotify.track_ids(url)
    if len(pasted) > 1:
        songs = [{"id": track_id, "number": n, "spotify": True}
                 for n, track_id in enumerate(pasted, start=1)]
        return {"name": "", "songs": songs, "is_playlist": True}

    on_event({"stage": "looking-up"})
    try:
        found = spotify.collection(url)
    except spotify.SpotifyUnavailable as e:
        on_event({"stage": "error", "message": str(e)})
        return {"name": "", "songs": [], "is_playlist": False, "error": str(e)}

    songs = found["songs"]
    for number, song in enumerate(songs, start=1):
        song["number"] = number
        song["spotify"] = True
    out = {"name": found["name"], "songs": songs, "is_playlist": len(songs) > 1}
    if len(songs) >= spotify.EMBED_ROW_LIMIT:
        really = spotify.song_count(url)
        if really is None or really > len(songs):
            out["too_long"] = {"listed": len(songs), "count": really, "name": found["name"]}
            if not take_listed:
                return out
            on_event({"stage": "warning", "message": (
                f"Spotify's link listed {len(songs)} songs"
                + (f" of the playlist's {really}" if really else "")
                + " - to get them all, copy them in Spotify (⌘A, ⌘C) and paste them here."
            )})
    return out


def _youtube_queue(url) -> dict:
    """A YouTube link as {"name", "songs", "is_playlist"}: a playlist is
    every video in it, one song each, so each can be taken all the way
    through - and picked up from - on its own. Anything else is one song,
    downloaded from the link as given."""
    if "list=" in url:
        info = _probe_info(url) or {}
        entries = [e for e in (info.get("entries") or []) if e]
        if len(entries) >= 2:
            songs = []
            for number, entry in enumerate(entries, start=1):
                video = entry.get("url") or entry.get("webpage_url") or (
                    f"https://www.youtube.com/watch?v={entry['id']}" if entry.get("id") else None)
                if not video:
                    continue
                songs.append({"youtube_url": video, "youtube_title": entry.get("title") or "",
                              "channel": entry.get("channel") or entry.get("uploader") or "",
                              "number": number})
            if songs:
                return {"name": info.get("title") or "", "songs": songs, "is_playlist": True}
            # Entries without links to them: hand yt-dlp the playlist whole.
            return {"name": info.get("title") or "", "is_playlist": True,
                    "songs": [{"youtube_url": url, "whole": True}]}
    return {"name": "", "songs": [{"youtube_url": url}], "is_playlist": False}


def _key(song: dict) -> str:
    """Which song this is, for sources.Sources - by its Spotify id when
    there is one, so a playlist link and a pasted copy of it agree."""
    if song.get("id"):
        return sources.spotify_key(song["id"])
    if song.get("spotify"):
        return "spotify-search:" + song.get("query", "")
    return sources.youtube_key(song.get("youtube_url") or "")


def _download_track(track_url, output_dir, on_event, position, total, known_title, name=None):
    """One track's download, including the stale-archive retry.

    The archive lists video ids, so it says "you already have this" about a
    song whose file has since been deleted, moved, or - as happened here -
    never got filed at all. yt-dlp then skips it and there is nothing to
    show for the link, forever. The file on disk is the authority: when it
    isn't there, the archive entry is simply wrong, and the honest thing is
    to go and get the song."""
    download = _Download(track_url, output_dir, on_event, queue_index=position,
                         queue_total=total if total > 1 else None,
                         known_title=known_title, name=name)
    status = download.run()
    if download.downloaded == 0 and download.failed == 0 and _nothing_on_disk_for(track_url, output_dir, name):
        download = _Download(track_url, output_dir, on_event, use_archive=False,
                             queue_index=position, queue_total=total if total > 1 else None,
                             known_title=known_title, name=name)
        status = download.run()
    return download, status



def _read_alignment(path: str, interactive) -> tuple[int, float] | None:
    """The song's beat-1 trim and tempo (instrument_isolator.song_alignment),
    or None if it can't be read. A song whose tempo can't be read is still a
    song - it just keeps the name it had."""
    try:
        return instrument_isolator.song_alignment(path, interactive=interactive)
    except Exception:
        return None


def _put_tempo_in_name(output_dir: str, filename: str, interactive) -> str:
    """Name a freshly sanitized song for its tempo - "Song - Artist (104.5
    BPM).mp3" - and write it to the BPM tag too, so Live, Finder and any
    other player can see it. Returns the song's filename afterwards.

    Done on the flat download, before filing, because the song's folder is
    named after the mp3."""
    path = os.path.join(output_dir, filename)
    if not os.path.isfile(path):
        return filename
    alignment = _read_alignment(path, interactive)
    if alignment is None:
        return filename
    tempo = alignment[1]
    renamed = instrument_isolator.with_song_tempo(os.path.splitext(filename)[0], tempo) + ".mp3"
    new_path = os.path.join(output_dir, renamed)
    if new_path != path:
        if os.path.exists(new_path):
            return filename
        try:
            os.rename(path, new_path)
        except OSError:
            return filename
    try:
        song_sanitizer.write_tempo_tag(new_path, instrument_isolator.bpm_text(tempo))
    except Exception:
        pass
    instrument_isolator.remember_alignment(new_path, alignment)
    return renamed


def _numbered(output_dir: str, filename: str, number: int | None) -> str:
    """Put a song's place in its playlist in front of its name - "004 -
    Song - Artist.mp3" - so a folder of them sorts in playlist order.
    Returns the filename afterwards; unchanged if there's no number or the
    name is taken."""
    if not number:
        return filename
    numbered = f"{number:03d} - {filename}"
    try:
        if os.path.exists(os.path.join(output_dir, numbered)):
            return filename
        os.rename(os.path.join(output_dir, filename), os.path.join(output_dir, numbered))
    except OSError:
        return filename
    return numbered


def _finish_track(track_url, fresh, output_dir, on_event, interactive, on_review,
                  sanitize, own_folder, number=None, bpm=True, name=None, tags=None,
                  steps=None) -> list[str]:
    """Tidy one track's download, put it where it belongs, and remember the
    link it came from. Returns the mp3 paths for that track.

    Done per track rather than for the whole queue at once so that what
    history remembers against each YouTube url is that url's own song -
    which is what makes coming back later for another stem work for every
    song in a playlist rather than only the first.

    number is the song's place in its playlist, when the songs are to be
    numbered (see _numbered). name is what the song was saved as when that
    was already known (Spotify's name) - the sanitizer leaves a name like
    that alone - and tags the (title, artist) to write into it. bpm puts
    the tempo in the name whether or not the song was sanitized. steps is
    which parts of sanitizing to do (song_sanitizer.ALL_STEPS by default)."""
    finished = []
    if fresh:
        if sanitize:
            try:
                finished = song_sanitizer.sanitize_new_downloads(
                    fresh, output_dir,
                    interactive=(interactive is not False),
                    review=on_review,
                    keep_name=bool(name),
                    steps=steps,
                )
            except Exception as e:
                on_event({"stage": "warning", "message": f"Sanitizing hit a snag, but your downloads are safe: {e}"})
                finished = [f for f in fresh if os.path.exists(os.path.join(output_dir, f))]
        else:
            # Nothing was tidied, so what's on disk is what yt-dlp wrote -
            # which is exactly what the rest of the run chains onto.
            finished = list(fresh)
        if tags:
            for filename in finished:
                try:
                    song_sanitizer.write_id3_tags(os.path.join(output_dir, filename), *tags)
                except Exception:
                    pass
        if bpm and finished:
            on_event({"stage": "tempo"})
            finished = [_put_tempo_in_name(output_dir, filename, interactive) for filename in finished]
        if number:
            finished = [_numbered(output_dir, filename, number) for filename in finished]

    # A song downloaded on an earlier run is skipped by yt-dlp's archive, so
    # no hook fires for it - but asking to isolate it is still a perfectly
    # ordinary request. Widen the scope to cover anything this url resolves
    # to that's already on disk, not just this run's fresh saves.
    #
    # Done whatever was armed, not only when an instrument was: pasting the
    # link of a song you already have has to put it back in front of you,
    # and "the stash forgot it" is exactly when somebody re-pastes a link.
    paths = [existing_song(output_dir, filename) or os.path.join(output_dir, filename)
             for filename in finished]
    if not fresh:
        for filename in requested_mp3_filenames(track_url, output_dir, name):
            path = existing_song(output_dir, filename)
            if path and path not in paths:
                paths.append(path)

    songs = []
    for path in paths:
        song = file_into_own_folder(path) if own_folder else path
        if song not in songs:
            songs.append(song)

    # Remember where these came from, so coming back later for another stem
    # doesn't mean going and finding the link again (see history.py).
    history.remember(track_url, songs)
    return songs


def _keep_what_finished(track_url, fresh, output_dir, own_folder, number=None) -> list[str]:
    """File and remember a song that finished downloading as a cancel came in.

    The song already down is a real song, so it's filed where the stash can
    see it. Nothing is sanitized here: that can stop to ask a question, and
    a cancel is someone saying they're done being asked. It isn't recorded
    as finished either, so the next run picks it up and tidies it."""
    fresh = [_numbered(output_dir, name, number) for name in fresh
             if os.path.exists(os.path.join(output_dir, name))]
    paths = [os.path.join(output_dir, name) for name in fresh]
    paths = [file_into_own_folder(path) if own_folder else path for path in paths]
    if paths:
        history.remember(track_url, paths)
    return paths


def _with_spotify_details(song: dict, record: dict) -> dict | None:
    """A pasted song's title, artist and length - from the record if an
    earlier run already read them, otherwise from Spotify. None if Spotify
    can't say, which skips the song this time and leaves it for the next."""
    if song.get("title") or not song.get("id"):
        return song
    if record.get("title") and record.get("query"):
        return dict(song, title=record["title"], artist=record.get("artist", ""),
                    duration_sec=record.get("duration_sec"), query=record["query"])
    try:
        found = spotify.track(spotify.track_url(song["id"]))
    except spotify.SpotifyUnavailable:
        return None
    return dict(song, **{k: found[k] for k in ("title", "artist", "duration_sec", "query")})


def run(
    url: str,
    output_dir: str = DEFAULT_OUTPUT,
    instruments=(),
    on_event=None,
    should_cancel=None,
    interactive: bool | None = None,
    on_review=None,
    on_choose=None,
    sanitize: bool = True,
    number: bool = False,
    bpm: bool = True,
    steps=None,
    folder_name: str = "",
    take_listed: bool = True,
) -> dict:
    """Download url into output_dir, sanitize it, and isolate the requested
    instruments. Returns a result dict describing what happened.

    url is a YouTube or Spotify link - a song or a playlist - or a pasted
    copy of a Spotify playlist: every track link in the text, however many.

    instruments is any iterable of "drums"/"bass"/"harmony"/"vocals"; empty
    means the song only. interactive=False suppresses every question the
    sanitizer and tempo detection would otherwise ask (see
    song_sanitizer.auto_resolve_flags and
    instrument_isolator.song_alignment) - what the GUI passes, since it has
    no way to answer them in a terminal.

    on_review is the exception to that: given one, an ambiguous intro or
    outro is put to it rather than decided alone (see
    song_sanitizer.review_flags), and the run waits on the answer. on_choose
    works the same way for a Spotify song with no obvious YouTube match (see
    _resolve_one). Without them the best answer is taken - the app's Auto.

    Every song is taken all the way through - matched, downloaded, tidied,
    named for its tempo, numbered, filed - before the next is started, and
    recorded in its folder's sources.Sources as it finishes. So a stop
    halfway through 5,000 songs loses only the one in hand, and running the
    same link (or the same pasted list) again picks up where it stopped
    without asking the network about any song already done.

    number=True puts each song of a playlist's place in it in front of its
    name ("004 - Song - Artist.mp3"). A single song is never numbered.

    sanitize=False leaves the download exactly as it came off YouTube: no
    trimming, no renaming, no duplicate check. steps chooses among those
    when it's on (song_sanitizer.ALL_STEPS). bpm puts the tempo in the name
    either way.

    A playlist lands in a folder of its own named after it - folder_name
    for a pasted list, which has no name; failing that, the folder already
    holding most of the list, so a paste carries on where it left off.
    Asked for songs alone, that folder is just the songs; asked for stems
    as well, each song gets a folder inside it.

    A Spotify playlist link lists at most spotify.EMBED_ROW_LIMIT songs.
    take_listed=False stops before downloading any when the playlist is
    longer and returns result["too_long"], for a page that can explain how
    to paste the whole thing; the default takes what was listed and warns.

    Cancelling: should_cancel is polled between songs and stages and during
    the slow demucs work. On cancel the run stops where it is and reports
    what it had already finished; nothing already written is removed."""
    if on_event is None:
        def on_event(_event):
            pass

    wanted = [name for name in INSTRUMENT_ORDER if name in set(instruments)]
    result = {
        "download_status": 0,
        "downloaded": 0,
        "skipped": 0,
        "failed": 0,
        "songs": [],
        "outputs": [],
        "cancelled": False,
        "output_dir": output_dir,
    }

    def cancelled() -> bool:
        return should_cancel is not None and should_cancel()

    # Checked up front rather than left to fail mid-run: without ffmpeg,
    # yt-dlp downloads the whole video happily and only then can't convert it
    # to mp3, so the symptom is a full progress bar, a stray .mp4 on disk and
    # nothing to show for it. Saying so before the download starts costs
    # nothing and names the actual problem.
    if shutil.which("ffmpeg") is None:
        message = (
            "ffmpeg isn't installed (or isn't on this app's PATH), so downloads "
            "can't be converted to mp3. Install it with: brew install ffmpeg"
        )
        on_event({"stage": "error", "message": message})
        result["error"] = message
        return result

    # A Spotify link can't be downloaded from, but it can say what the songs
    # are, and each is matched to a YouTube upload when its turn comes. One
    # song or five thousand, what comes back is a list, worked through one
    # ordinary url at a time.
    url = (url or "").strip()
    if spotify.spotify_id(url) or len(spotify.track_ids(url)) > 1:
        found = _spotify_queue(url, on_event, take_listed)
        if found.get("too_long") and not take_listed:
            result["too_long"] = found["too_long"]
            return result
        if not found["songs"]:
            result["error"] = found.get("error") or "Couldn't work out which song that Spotify link means."
            return result
    else:
        found = _youtube_queue(url)
    queue, playlist_name, is_playlist = found["songs"], found["name"], found["is_playlist"]

    # A playlist arrives as one thing and should land as one thing, so its
    # songs go in a folder named after it rather than scattered through the
    # downloads folder among everything else - whether it came from Spotify
    # or YouTube, and whether or not it has a name to go by.
    if is_playlist:
        folder = None
        name = (folder_name or "").strip() or playlist_name
        if name:
            folder = os.path.join(output_dir, _safe_folder_name(name))
        else:
            folder = sources.folder_holding(output_dir, [_key(song) for song in queue])
        if folder is None:
            folder = os.path.join(output_dir, _safe_folder_name(time.strftime("Playlist %Y-%m-%d %H.%M")))
        output_dir = folder
        result["output_dir"] = output_dir

    # Whether each song gets a folder of its own. Stems and MIDI need one -
    # they'd otherwise pile up unlabelled beside the mp3s - but a playlist
    # downloaded just for the songs is a folder of songs, and burying each
    # one in a folder of its own would only make it harder to use.
    own_folder = bool(wanted) or not is_playlist
    record = sources.Sources(output_dir)

    total = len(queue)
    already = sum(1 for song in queue if record.finished(_key(song)))
    if already and total > 1:
        on_event({"stage": "resuming", "done": already, "total": total})

    fatal = None
    songs = []
    for position, song in enumerate(queue, start=1):
        if cancelled():
            break
        key = _key(song)

        done = record.finished(key)
        if done:
            if done not in songs:
                songs.append(done)
            result["skipped"] += 1
            continue

        # Spotify: what the song is, then which upload is it.
        if song.get("spotify"):
            song = _with_spotify_details(song, record.get(key))
            if song is None:
                on_event({"stage": "warning", "message": f"Spotify wouldn't say what song {position} is - skipped it this time."})
                result["failed"] += 1
                continue
            record.record(key, title=song["title"], artist=song["artist"],
                          duration_sec=song.get("duration_sec"), query=song["query"],
                          number=song.get("number"),
                          spotify_url=spotify.track_url(song["id"]) if song.get("id") else None)
            if total > 1:
                on_event({"stage": "resolving", "index": position, "total": total,
                          "song": song["title"]})
            match, why_not = _resolve_one(song, position, total, on_event, on_choose, cancelled)
            if not match:
                if why_not:
                    on_event({"stage": "warning" if total > 1 else "error", "message": why_not})
                    record.record(key, status="skipped", note=why_not)
                    if total == 1:
                        result["error"] = why_not
                continue
            record.record(key, youtube_url=match["url"], youtube_title=match["title"],
                          channel=match["channel"], check=match["guessed"])
            track_url = match["url"]
            known_title = song["title"]
            name = file_name_for(song["title"], song["artist"])
            tags = (song["title"], song["artist"])
        else:
            track_url = song["youtube_url"]
            known_title = song.get("youtube_title") or None
            name = tags = None
            if song.get("youtube_title"):
                record.record(key, youtube_url=track_url, youtube_title=song["youtube_title"],
                              channel=song.get("channel"), number=song.get("number"))

        try:
            download, status = _download_track(track_url, output_dir, on_event,
                                               position, total, known_title, name)
        except yt_dlp.utils.DownloadError as e:
            if total == 1:
                # One link, one failure, nothing to carry on with.
                on_event({"stage": "error", "message": str(e)})
                result["error"] = str(e)
                return result
            on_event({"stage": "warning",
                      "message": f"Couldn't download {known_title or track_url}: {e}"})
            record.record(key, status="failed", note=str(e))
            result["failed"] += 1
            fatal = str(e)
            continue

        # yt-dlp's own return code: non-zero when some entry failed but
        # ignoreerrors let the rest through. Kept so the CLI can still exit
        # non-zero on a partial failure.
        result["download_status"] = result["download_status"] or status
        result["downloaded"] += download.downloaded
        result["failed"] += download.failed
        result["skipped"] += max((download.total or 0) - download.downloaded - download.failed, 0)
        fresh = list(download.filenames)
        place = song.get("number") if number and is_playlist else None

        if cancelled():
            songs.extend(p for p in _keep_what_finished(track_url, fresh, output_dir, own_folder, place)
                         if p not in songs)
            break

        if sanitize and fresh:
            on_event({"stage": "sanitizing"})
        finished = _finish_track(track_url, fresh, output_dir, on_event, interactive,
                                 on_review, sanitize, own_folder, place, bpm=bpm,
                                 name=name, tags=tags, steps=steps)
        for path in finished:
            if path not in songs:
                songs.append(path)
        if finished and not song.get("whole"):
            fields = {"status": "done", "file": finished[0], "number": song.get("number")}
            if not song.get("spotify"):
                title, artist, _ = song_sanitizer._derive_title_artist(os.path.basename(finished[0]))
                fields.update(title=instrument_isolator.song_title(title + ".mp3"), artist=artist)
            record.record(key, **fields)

    on_event(
        {
            "stage": "download-summary",
            "downloaded": result["downloaded"],
            "skipped": result["skipped"],
            "failed": result["failed"],
            "output_dir": output_dir,
        }
    )

    result["songs"] = songs
    result["outputs"] = list(songs)

    if cancelled():
        result["cancelled"] = True
        on_event({"stage": "cancelled"})
        return result

    if not songs and fatal and not result.get("error"):
        result["error"] = fatal

    _isolate_songs(result["songs"], wanted, on_event, cancelled, should_cancel, interactive, result)

    if result["cancelled"]:
        return result

    on_event({"stage": "done", "outputs": result["outputs"]})
    return result



def _songs_under(folder: str) -> list[tuple[str, str]]:
    """Every mp3 under folder, as (its folder, its filename), in the order
    Finder would list them. Duplicates/ is what sanitizing already set
    aside, and a hidden folder isn't anybody's music."""
    found = []
    for root, subdirs, files in os.walk(folder):
        subdirs[:] = sorted(d for d in subdirs
                            if not d.startswith(".") and d != song_sanitizer.DUPLICATES_DIR_NAME)
        for name in sorted(files):
            if name.lower().endswith(".mp3") and not name.startswith("."):
                found.append((root, name))
    return found


def _tidy_one(folder: str, filename: str, on_review, interactive) -> str | None:
    """Sanitize one song where it sits and put its tempo in its name.
    Returns where it ended up, or None if it didn't survive (set aside as a
    duplicate). A song in a folder of its own takes the folder along to its
    new name, stems and all, and the stash is told where it went."""
    old_path = os.path.join(folder, filename)
    # Its own folder is one named after it with no other song in it - a
    # playlist named after one of its songs is still the playlist's.
    own_folder = (os.path.basename(folder) == filename[:-len(".mp3")]
                  and [f for f in os.listdir(folder) if f.lower().endswith(".mp3")] == [filename])
    url = history.url_for(old_path)

    final = song_sanitizer.sanitize_new_downloads(
        [filename], folder, interactive=(interactive is not False), review=on_review)
    if not final:
        return None
    current = final[0]
    if instrument_isolator.song_title(current) == current[:-len(".mp3")]:
        current = _put_tempo_in_name(folder, current, interactive)
    new_path = os.path.join(folder, current)

    new_folder = os.path.join(os.path.dirname(folder), current[:-len(".mp3")])
    if own_folder and new_folder != folder and not os.path.exists(new_folder):
        try:
            os.rename(folder, new_folder)
            new_path = os.path.join(new_folder, current)
        except OSError:
            pass

    if url and new_path != old_path:
        history.remember(url, [new_path])
    return new_path


def sanitize_existing(folder: str, on_event=None, on_review=None, should_cancel=None,
                      interactive: bool | None = False) -> dict:
    """Sanitize songs already on disk: every mp3 in folder and the folders
    under it - flat, filed in a folder of their own, or in a playlist's.

    The same tidying a download gets (song_sanitizer.sanitize_new_downloads,
    one song at a time so a trim question comes up while that song is the
    one being talked about), then its tempo in its name. A song already
    sanitized is only given its tempo, and one that has that too is left
    alone - so running this over the whole stash again is quick.

    Returns {"songs", "outputs", "cancelled", "output_dir"} like run()."""
    if on_event is None:
        def on_event(_event):
            pass
    result = {"songs": [], "outputs": [], "cancelled": False, "output_dir": folder,
              "downloaded": 0}
    found = _songs_under(folder)
    total = len(found)
    for index, (song_dir, filename) in enumerate(found, start=1):
        if should_cancel is not None and should_cancel():
            result["cancelled"] = True
            on_event({"stage": "cancelled"})
            break
        on_event({"stage": "tidying", "index": index, "total": total,
                  "song": instrument_isolator.song_title(filename)})
        try:
            song = _tidy_one(song_dir, filename, on_review, interactive)
        except Exception as e:
            on_event({"stage": "warning", "message": f"Couldn't tidy {filename}: {e}"})
            continue
        if song:
            result["songs"].append(song)
    result["outputs"] = list(result["songs"])
    if not result["cancelled"]:
        on_event({"stage": "done", "outputs": result["outputs"]})
    return result


def _isolate_songs(song_paths, wanted, on_event, cancelled, should_cancel, interactive, result) -> None:
    """Run each wanted instrument over each song, filling result as it goes.

    Shared by run() and isolate() so there's one copy of the progress
    events, the cancel checks and the output collection - the two differ
    only in whether anything was downloaded first."""
    if not wanted:
        return

    context = instrument_isolator.RunContext(interactive=interactive, should_cancel=should_cancel)

    # The demucs pass each isolator needs is cached and shared, so asking for
    # all four instruments separates the song once rather than four times
    # (see instrument_isolator.separated_stems). Those passes are hundreds of
    # megabytes of temp files, so they're disposed of however this ends -
    # finished, cancelled or blown up.
    try:
        for mp3_path in song_paths:
            title = instrument_isolator.song_title(mp3_path)
            for name in wanted:
                if cancelled():
                    result["cancelled"] = True
                    on_event({"stage": "cancelled"})
                    return

                module, label = _INSTRUMENTS[name]
                index, total = wanted.index(name) + 1, len(wanted)
                on_event({"stage": "isolating", "instrument": name, "song": title,
                          "index": index, "total": total, "percent": None, "phase": None})

                def report(percent, _name=name, _title=title, _i=index, _n=total):
                    on_event({"stage": "isolating", "instrument": _name, "song": _title,
                              "index": _i, "total": _n, "percent": percent, "phase": None})

                def phase(message, _name=name, _title=title, _i=index, _n=total):
                    on_event({"stage": "isolating", "instrument": _name, "song": _title,
                              "index": _i, "total": _n, "percent": None, "phase": message})

                run_one = getattr(module, f"isolate_{name}_for_single_file")
                try:
                    run_one(mp3_path, context=context._replace(on_percent=report, on_phase=phase))
                except instrument_isolator.Cancelled:
                    result["cancelled"] = True
                    on_event({"stage": "cancelled"})
                    return

                produced = _instrument_outputs(mp3_path, label)
                result["outputs"].extend(produced)
                on_event({"stage": "isolated", "instrument": name, "song": title, "outputs": produced})
    finally:
        instrument_isolator.clear_stem_cache()


def isolate(song_paths, instruments=(), on_event=None, should_cancel=None, interactive: bool | None = None) -> dict:
    """Isolate instruments from songs already on disk, with no download and
    no link.

    Re-taking a stem from a song you already have shouldn't need the
    internet, and for a song whose link was never recorded there's no link
    to go back to. Same result dict as run(), with the download counters
    left at zero."""
    if on_event is None:
        def on_event(_event):
            pass

    song_paths = [path for path in song_paths if os.path.exists(path)]
    wanted = [name for name in INSTRUMENT_ORDER if name in set(instruments)]
    result = {
        "download_status": 0,
        "downloaded": 0,
        "skipped": 0,
        "failed": 0,
        "songs": list(song_paths),
        "outputs": list(song_paths),
        "cancelled": False,
        "output_dir": os.path.dirname(os.path.dirname(song_paths[0])) if song_paths else DEFAULT_OUTPUT,
    }

    def cancelled() -> bool:
        return bool(should_cancel and should_cancel())

    _isolate_songs(song_paths, wanted, on_event, cancelled, should_cancel, interactive, result)

    if result["cancelled"]:
        return result

    on_event({"stage": "done", "outputs": result["outputs"]})
    return result

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
import shutil

import yt_dlp

import bass_isolator
import drum_isolator
import harmony_isolator
import history
import instrument_isolator
import song_sanitizer
import vocals_isolator

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


class _SilentLogger:
    """Swallow yt-dlp's own chatter - progress is reported through on_event."""

    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass


def _base_ydl_opts(output_dir: str) -> dict:
    return {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "%(title)s - %(uploader)s.%(ext)s"),
        "noplaylist": True,
        "extractor_args": {"youtube": {"player_client": ["android"]}},
        "quiet": True,
        "no_warnings": True,
        "logger": _SilentLogger(),
    }


def count_entries(url: str) -> int | None:
    """How many songs this url resolves to, or None if that can't be
    determined quickly. Metadata only - nothing is downloaded."""
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
            info = ydl.extract_info(url, download=False)
    except Exception:
        return None
    entries = info.get("entries") if info else None
    return len(entries) if entries is not None else 1


def requested_mp3_filenames(url: str, output_dir: str) -> list[str]:
    """The mp3 filename(s) this url resolves to, whether they were just
    downloaded this run or already sat on disk from a previous one (skipped
    via the download archive).

    Used to scope isolation to what was actually asked for: the list of
    fresh downloads alone misses a request for a song you already have,
    since yt-dlp's archive skip means no download hook ever fires for it."""
    try:
        with yt_dlp.YoutubeDL(_base_ydl_opts(output_dir)) as ydl:
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

    def __init__(self, url: str, output_dir: str, on_event):
        self.url = url
        self.output_dir = output_dir
        self.on_event = on_event
        self.total = None
        self.active_title = None
        self.song_number = 0
        self.downloaded = 0
        self.failed = 0
        self.filenames: list[str] = []

    def _progress_hook(self, d):
        title = d.get("info_dict", {}).get("title", "Unknown")

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
                    "index": self.song_number,
                    "total": self.total,
                    "percent": percent,
                }
            )
        elif d["status"] == "finished":
            self.on_event({"stage": "downloading", "song": title, "index": self.song_number,
                           "total": self.total, "percent": 100.0})
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
            self.filenames.append(os.path.basename(filepath))
        self.on_event({"stage": "downloaded", "song": info.get("title", "Unknown")})
        self.active_title = None

    def run(self) -> int:
        os.makedirs(self.output_dir, exist_ok=True)

        self.on_event({"stage": "looking-up"})
        self.total = count_entries(self.url)
        self.on_event({"stage": "found", "total": self.total})

        opts = _base_ydl_opts(self.output_dir)
        opts.update(
            {
                "download_archive": os.path.join(self.output_dir, ".downloaded_archive.txt"),
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


def library(limit: int = 20) -> list[dict]:
    """Recently downloaded songs, newest first, each with the link it came
    from and whatever files exist for it right now.

    Files are read from disk rather than remembered, so a stem deleted in
    Finder simply stops being listed."""
    songs = []
    for entry in history.entries()[:limit]:
        song_path = entry["song"]
        files = [song_path]

        song_dir = instrument_isolator.song_output_dir(song_path)
        if os.path.isdir(song_dir):
            for name in sorted(os.listdir(song_dir)):
                # Skip the .source.json markers the isolators keep for their
                # own "is this still up to date" checks - not output anyone
                # asked for.
                if name.startswith("."):
                    continue
                files.append(os.path.join(song_dir, name))

        songs.append(
            {
                "title": os.path.splitext(os.path.basename(song_path))[0],
                "url": entry.get("url", ""),
                "song": song_path,
                # Where everything this song produced lives. What a front end
                # actually wants to open - the stems get dragged out of the
                # folder together, so the folder is the thing, not any one
                # wav in it. Empty until the song has been isolated at all.
                "dir": song_dir if os.path.isdir(song_dir) else "",
                "files": files,
            }
        )
    return songs


def run(
    url: str,
    output_dir: str = DEFAULT_OUTPUT,
    instruments=(),
    on_event=None,
    should_cancel=None,
    interactive: bool | None = None,
) -> dict:
    """Download url into output_dir, sanitize it, and isolate the requested
    instruments. Returns a result dict describing what happened.

    instruments is any iterable of "drums"/"bass"/"harmony"/"vocals"; empty
    means the song only. interactive=False suppresses every question the
    sanitizer and tempo detection would otherwise ask (see
    song_sanitizer.auto_resolve_flags and
    instrument_isolator.song_alignment) - what the GUI passes, since it has
    no way to answer them.

    Cancelling: should_cancel is polled between stages and during the slow
    demucs work. On cancel the run stops where it is and reports what it had
    already finished; nothing already written is removed, since a completed
    isolation is still perfectly good."""
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

    download = _Download(url, output_dir, on_event)
    try:
        # yt-dlp's own return code: non-zero when some entry failed but
        # ignoreerrors let the rest through. Kept so the CLI can still exit
        # non-zero on a partial failure.
        result["download_status"] = download.run()
    except yt_dlp.utils.DownloadError as e:
        on_event({"stage": "error", "message": str(e)})
        result["error"] = str(e)
        return result

    result["downloaded"] = download.downloaded
    result["failed"] = download.failed
    result["skipped"] = max((download.total or 0) - download.downloaded - download.failed, 0)
    on_event(
        {
            "stage": "download-summary",
            "downloaded": result["downloaded"],
            "skipped": result["skipped"],
            "failed": result["failed"],
            "output_dir": output_dir,
        }
    )

    if cancelled():
        result["cancelled"] = True
        on_event({"stage": "cancelled"})
        return result

    sanitized = []
    if download.filenames:
        on_event({"stage": "sanitizing"})
        try:
            sanitized = song_sanitizer.sanitize_new_downloads(
                download.filenames, output_dir, interactive=(interactive is not False)
            )
        except Exception as e:
            on_event({"stage": "warning", "message": f"Sanitizing hit a snag, but your downloads are safe: {e}"})

    # A song downloaded on an earlier run is skipped by yt-dlp's archive, so
    # no hook fires for it - but asking to isolate it is still a perfectly
    # ordinary request. Widen the scope to cover anything this url resolves
    # to that's already on disk, not just this run's fresh saves.
    filenames = list(sanitized)
    if wanted:
        seen = set(filenames)
        for filename in requested_mp3_filenames(url, output_dir):
            if filename not in seen and os.path.exists(os.path.join(output_dir, filename)):
                filenames.append(filename)
                seen.add(filename)
    result["songs"] = [os.path.join(output_dir, f) for f in filenames]
    result["outputs"] = list(result["songs"])

    # Remember where these came from, so coming back later for another stem
    # doesn't mean going and finding the link again (see history.py).
    history.remember(url, result["songs"])

    context = instrument_isolator.RunContext(interactive=interactive, should_cancel=should_cancel)

    # The demucs pass each isolator needs is cached and shared, so asking for
    # all four instruments separates the song once rather than four times
    # (see instrument_isolator.separated_stems). Those passes are hundreds of
    # megabytes of temp files, so they're disposed of however this ends -
    # finished, cancelled or blown up.
    try:
        for filename in filenames:
            mp3_path = os.path.join(output_dir, filename)
            title = os.path.splitext(filename)[0]
            for name in wanted:
                if cancelled():
                    result["cancelled"] = True
                    on_event({"stage": "cancelled"})
                    return result

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

                isolate = getattr(module, f"isolate_{name}_for_single_file")
                try:
                    isolate(mp3_path, context=context._replace(on_percent=report, on_phase=phase))
                except instrument_isolator.Cancelled:
                    result["cancelled"] = True
                    on_event({"stage": "cancelled"})
                    return result

                produced = _instrument_outputs(mp3_path, label)
                result["outputs"].extend(produced)
                on_event({"stage": "isolated", "instrument": name, "song": title, "outputs": produced})
    finally:
        instrument_isolator.clear_stem_cache()

    on_event({"stage": "done", "outputs": result["outputs"]})
    return result

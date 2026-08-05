#!/usr/bin/env python3
"""beat_thief: download all songs from a YouTube/YouTube Music playlist as
MP3s, sanitize them, and optionally isolate drums/bass to MIDI."""

from __future__ import annotations

import argparse
import os
import sys

import yt_dlp

import bass_isolator
import drum_isolator
import song_sanitizer

BAR_WIDTH = 30

_active_title = None
_song_number = 0
_total_songs = None
_downloaded_count = 0
_failed_count = 0
_downloaded_filenames = []


def _exit(code: int) -> None:
    """Flush output and force the process closed immediately.

    Plain sys.exit() can leave the shell prompt hanging if some dependency
    (yt-dlp/ffmpeg/urllib3) leaves a resource open in the background, so we
    force the OS-level exit once our own work is done.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def _short(title: str, max_len: int = 45) -> str:
    return title if len(title) <= max_len else title[: max_len - 1] + "…"


def _song_label() -> str:
    if _total_songs:
        return f"song {_song_number} of {_total_songs}"
    return f"song {_song_number}"


def _progress_hook(d):
    global _active_title, _song_number
    title = d.get("info_dict", {}).get("title", "Unknown")

    if d["status"] == "downloading":
        if title != _active_title:
            _active_title = title
            _song_number += 1
            print(f"Downloading {_song_label()}: {_short(title)}")

        total = d.get("total_bytes") or d.get("total_bytes_estimate")
        downloaded = d.get("downloaded_bytes", 0)
        if total:
            frac = min(downloaded / total, 1.0)
            filled = int(BAR_WIDTH * frac)
            bar = "#" * filled + "-" * (BAR_WIDTH - filled)
            print(f"\r  [{bar}] {frac * 100:5.1f}%", end="", flush=True)

    elif d["status"] == "finished":
        print(f"\r  [{'#' * BAR_WIDTH}] 100.0%")

    elif d["status"] == "error":
        global _failed_count
        _failed_count += 1
        print(f"  Could not download this one, skipping it: {_short(title)}")
        _active_title = None


def _postprocessor_hook(d):
    global _active_title, _downloaded_count
    if d["status"] == "finished" and d.get("postprocessor") == "ExtractAudio" and _active_title is not None:
        info = d.get("info_dict", {})
        title = info.get("title", "Unknown")
        _downloaded_count += 1
        filepath = info.get("filepath")
        if filepath:
            _downloaded_filenames.append(os.path.basename(filepath))
        print(f"Saved: {_short(title)}.mp3")
        _active_title = None


class _SilentLogger:
    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass


def _count_playlist_entries(url: str) -> int | None:
    """Return the number of songs in the playlist, or None if it can't be determined quickly."""
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


def _requested_mp3_filenames(url: str, output_dir: str) -> list[str]:
    """The mp3 filename(s) this url resolves to, whether they were just
    downloaded this run or already sat on disk from a previous one (skipped
    via the download archive). Used to scope drum/bass isolation to what
    was actually asked for this run - _downloaded_filenames alone misses a
    request for a song you already have, since yt-dlp's archive skip means
    no download/postprocessor hook ever fires for it."""
    probe_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "%(title)s - %(uploader)s.%(ext)s"),
        "noplaylist": True,
        "extractor_args": {"youtube": {"player_client": ["android"]}},
        "quiet": True,
        "no_warnings": True,
        "logger": _SilentLogger(),
    }
    try:
        with yt_dlp.YoutubeDL(probe_opts) as ydl:
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


def download_playlist(url: str, output_dir: str) -> int:
    global _total_songs
    os.makedirs(output_dir, exist_ok=True)
    archive_path = os.path.join(output_dir, ".downloaded_archive.txt")

    print("Looking up the playlist...")
    _total_songs = _count_playlist_entries(url)
    if _total_songs:
        word = "song" if _total_songs == 1 else "songs"
        print(f"Found {_total_songs} {word}. Songs already downloaded before will be skipped automatically.\n")
    else:
        print("Found the playlist. Songs already downloaded before will be skipped automatically.\n")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "%(title)s - %(uploader)s.%(ext)s"),
        "download_archive": archive_path,
        "ignoreerrors": True,
        "noplaylist": True,
        "extractor_args": {"youtube": {"player_client": ["android"]}},
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [_progress_hook],
        "postprocessor_hooks": [_postprocessor_hook],
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "320",
            }
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.download([url])

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a YouTube/YouTube Music playlist as MP3s."
    )
    parser.add_argument("url", help="YouTube Music (or YouTube) playlist URL")
    default_output = os.path.join(os.path.expanduser("~"), "Downloads", "Song Downloads")
    parser.add_argument(
        "-o",
        "--output",
        default=default_output,
        help=f"Output directory (default: {default_output})",
    )
    parser.add_argument(
        "--drums",
        action="store_true",
        help="Also isolate each song's drums to wav + MIDI. Slow, off by default. Can also be given as a bare 'drums' argument, before or after the URL.",
    )
    parser.add_argument(
        "--bass",
        action="store_true",
        help="Also isolate each song's bass to wav + MIDI. Slow, off by default. Can also be given as a bare 'bass' argument, before or after the URL.",
    )

    # "drums"/"bass" are accepted bare (no --) too, in any position, since
    # that's the more natural way to type them - pull those out of argv
    # before argparse ever sees them, so they don't collide with the url
    # positional no matter where they appear.
    raw_args = sys.argv[1:]
    bare_modes = {tok.lower() for tok in raw_args if tok.lower() in ("drums", "bass")}
    remaining_args = [tok for tok in raw_args if tok.lower() not in ("drums", "bass")]

    args = parser.parse_args(remaining_args)
    args.drums = args.drums or "drums" in bare_modes
    args.bass = args.bass or "bass" in bare_modes

    try:
        result = download_playlist(args.url, args.output)
    except KeyboardInterrupt:
        print("\nCancelled.")
        _exit(130)
    except yt_dlp.utils.DownloadError as e:
        message = str(e)
        if "ffprobe" in message.lower() or "ffmpeg" in message.lower():
            print(
                "Error: ffmpeg is required but wasn't found.\n"
                "Install it with: brew install ffmpeg\n"
                "See README.md for setup instructions.",
                file=sys.stderr,
            )
        else:
            print(f"Something went wrong: {message}", file=sys.stderr)
        _exit(1)

    print()
    skipped = max((_total_songs or 0) - _downloaded_count - _failed_count, 0)
    parts = [f"Downloaded {_downloaded_count} new song{'s' if _downloaded_count != 1 else ''}"]
    if skipped:
        parts.append(f"skipped {skipped} already downloaded")
    if _failed_count:
        parts.append(f"{_failed_count} failed")
    print(f"Downloading complete: {', '.join(parts)}.")
    print(f"Your music is in: {args.output}")

    sanitized_filenames = []
    if _downloaded_filenames:
        try:
            sanitized_filenames = song_sanitizer.sanitize_new_downloads(_downloaded_filenames, args.output)
        except KeyboardInterrupt:
            print("\nStopped reviewing. Whatever was already cleaned up is safe.")
            _exit(130)
        except Exception as e:
            print(f"Sanitizing hit a snag, but your downloads are safe: {e}")

    isolation_filenames = list(sanitized_filenames)
    if args.drums or args.bass:
        # A song that was already downloaded in a previous run (and thus
        # skipped this time, with no download/postprocessor hook firing for
        # it) is still something this run explicitly asked to isolate -
        # widen the scope to cover it too, not just this run's fresh saves.
        already_seen = set(isolation_filenames)
        for filename in _requested_mp3_filenames(args.url, args.output):
            if filename not in already_seen and os.path.exists(os.path.join(args.output, filename)):
                isolation_filenames.append(filename)
                already_seen.add(filename)

    if args.drums:
        for filename in isolation_filenames:
            try:
                drum_isolator.isolate_drums_for_single_file(os.path.join(args.output, filename))
            except KeyboardInterrupt:
                print("\nStopped isolating drums. Whatever was already produced is safe.")
                _exit(130)
            except Exception as e:
                print(f"Isolating drums hit a snag, but your downloads are safe: {e}")

    if args.bass:
        for filename in isolation_filenames:
            try:
                bass_isolator.isolate_bass_for_single_file(os.path.join(args.output, filename))
            except KeyboardInterrupt:
                print("\nStopped isolating bass. Whatever was already produced is safe.")
                _exit(130)
            except Exception as e:
                print(f"Isolating bass hit a snag, but your downloads are safe: {e}")

    _exit(0 if result == 0 else 1)


if __name__ == "__main__":
    main()

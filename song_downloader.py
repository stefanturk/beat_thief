#!/usr/bin/env python3
"""Download all songs from a YouTube/YouTube Music playlist as MP3s."""

from __future__ import annotations

import argparse
import os
import sys

import yt_dlp

BAR_WIDTH = 30

_active_title = None
_song_number = 0
_total_songs = None
_downloaded_count = 0
_failed_count = 0


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
        title = d.get("info_dict", {}).get("title", "Unknown")
        _downloaded_count += 1
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
        "noplaylist": False,
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
    args = parser.parse_args()

    try:
        result = download_playlist(args.url, args.output)
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(130)
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
        sys.exit(1)

    print()
    skipped = max((_total_songs or 0) - _downloaded_count - _failed_count, 0)
    parts = [f"downloaded {_downloaded_count} new song{'s' if _downloaded_count != 1 else ''}"]
    if skipped:
        parts.append(f"skipped {skipped} already downloaded")
    if _failed_count:
        parts.append(f"{_failed_count} failed")
    print(f"All done! {', '.join(parts)}.")
    print(f"Your music is in: {args.output}")

    sys.exit(0 if result == 0 else 1)


if __name__ == "__main__":
    main()

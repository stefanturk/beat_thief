#!/usr/bin/env python3
"""Download all songs from a YouTube/YouTube Music playlist as MP3s."""

import argparse
import os
import sys

import yt_dlp

BAR_WIDTH = 30
_active_title = None


def _short(title: str, max_len: int = 40) -> str:
    return title if len(title) <= max_len else title[: max_len - 1] + "…"


def _progress_hook(d):
    global _active_title
    title = d.get("info_dict", {}).get("title", "Unknown")

    if d["status"] == "downloading":
        if title != _active_title:
            _active_title = title
            print(f"Downloading: {_short(title)}")

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
        print(f"\nFailed: {_short(title)}")
        _active_title = None


def _postprocessor_hook(d):
    global _active_title
    if d["status"] == "finished" and d.get("postprocessor") == "ExtractAudio" and _active_title is not None:
        title = d.get("info_dict", {}).get("title", "Unknown")
        print(f"Finished: {_short(title)}")
        _active_title = None


def download_playlist(url: str, output_dir: str) -> int:
    os.makedirs(output_dir, exist_ok=True)
    archive_path = os.path.join(output_dir, ".downloaded_archive.txt")

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
            print(f"Error: {message}", file=sys.stderr)
        sys.exit(1)

    sys.exit(0 if result == 0 else 1)


if __name__ == "__main__":
    main()

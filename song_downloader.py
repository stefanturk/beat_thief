#!/usr/bin/env python3
"""Download all songs from a YouTube/YouTube Music playlist as MP3s."""

import argparse
import os
import sys

import yt_dlp


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
    default_output = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "downloads", "Song Downloads"
    )
    parser.add_argument(
        "-o",
        "--output",
        default=default_output,
        help=f"Output directory (default: {default_output})",
    )
    args = parser.parse_args()

    try:
        result = download_playlist(args.url, args.output)
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

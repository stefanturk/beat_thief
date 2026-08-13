#!/usr/bin/env python3
"""beat_thief: download all songs from a YouTube/YouTube Music playlist as
MP3s, sanitize them, and optionally isolate drums/bass/harmony/vocals to
wavs. Name no instrument and you just get the song; say "all" and you get
all four, which together are the song again.

This is the terminal front end. The work itself lives in pipeline.py, shared
with the GUI (gui.py) - everything here is argument parsing and turning the
pipeline's progress events into terminal output."""

from __future__ import annotations

import argparse
import os
import sys

import pipeline

BAR_WIDTH = 30

_BARE_FLAGS = ("all", "drums", "bass", "harmony", "vocals")

# Options whose next argument is a value rather than a flag of its own, so
# that "-o drums" means a folder called drums and not a request for them.
_VALUE_TAKING_OPTIONS = ("-o", "--output")


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


class _Printer:
    """Renders pipeline events as the terminal output this tool has always
    produced. Holds the little bit of state that needs carrying between
    events (which song's header line has already been printed)."""

    def __init__(self):
        self.announced = None

    def __call__(self, event):
        stage = event["stage"]

        if stage == "looking-up":
            print("Looking up the playlist...")

        elif stage == "found":
            total = event["total"]
            song = event.get("song")
            if song:
                print(f"Found {song}. Songs already downloaded before will be skipped automatically.\n")
            elif total:
                word = "song" if total == 1 else "songs"
                print(f"Found {total} {word}. Songs already downloaded before will be skipped automatically.\n")
            else:
                print("Found the playlist. Songs already downloaded before will be skipped automatically.\n")

        elif stage == "downloading":
            song, index, total = event["song"], event["index"], event["total"]
            if song != self.announced:
                self.announced = song
                label = f"song {index} of {total}" if total else f"song {index}"
                print(f"Downloading {label}: {_short(song)}")
            percent = event["percent"]
            if percent is None:
                return
            filled = int(BAR_WIDTH * percent / 100)
            bar = "#" * filled + "-" * (BAR_WIDTH - filled)
            end = "\n" if percent >= 100 else ""
            print(f"\r  [{bar}] {percent:5.1f}%", end=end, flush=True)

        elif stage == "downloaded":
            print(f"Saved: {_short(event['song'])}.mp3")

        elif stage == "download-failed":
            self.announced = None
            print(f"  Could not download this one, skipping it: {_short(event['song'])}")

        elif stage == "download-summary":
            print()
            parts = [f"Downloaded {event['downloaded']} new song{'s' if event['downloaded'] != 1 else ''}"]
            if event["skipped"]:
                parts.append(f"skipped {event['skipped']} already downloaded")
            if event["failed"]:
                parts.append(f"{event['failed']} failed")
            print(f"Downloading complete: {', '.join(parts)}.")
            print(f"Your music is in: {event['output_dir']}")

        elif stage == "warning":
            print(event["message"])

        elif stage == "cancelled":
            print("\nStopped. Whatever was already produced is safe.")

        elif stage == "error":
            message = event["message"]
            if "ffprobe" in message.lower() or "ffmpeg" in message.lower():
                print(
                    "Error: ffmpeg is required but wasn't found.\n"
                    "Install it with: brew install ffmpeg\n"
                    "See README.md for setup instructions.",
                    file=sys.stderr,
                )
            else:
                print(f"Something went wrong: {message}", file=sys.stderr)

        # "sanitizing", "isolating" and "isolated" need no output here - the
        # sanitizer and the isolators print their own per-song progress
        # (including demucs' bar) straight to the terminal already.


def _split_bare_flags(raw_args: list[str]) -> tuple[set[str], list[str]]:
    """Pull the bare (undashed) flags out of argv, and return them alongside
    everything argparse should still see.

    They're pulled out here rather than declared as positionals because they
    can appear anywhere, including before the url, and argparse has no way
    to tell "drums" from a url in that position.

    The subtlety is an option that takes a value: the "drums" in "-o drums"
    is a folder name, not a request for the drums, so it's left alone."""
    bare: set[str] = set()
    remaining: list[str] = []
    skip_next = False

    for token in raw_args:
        lowered = token.lower()
        if skip_next:
            # The value of an option that takes one, e.g. -o drums. It's a
            # path, not the drums.
            skip_next = False
            remaining.append(token)
            continue

        if token.startswith("-"):
            skip_next = token in _VALUE_TAKING_OPTIONS
            remaining.append(token)
        elif lowered in _BARE_FLAGS:
            bare.add(lowered)
        else:
            remaining.append(token)

    return bare, remaining


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a YouTube/YouTube Music playlist as MP3s."
    )
    parser.add_argument("url", help="YouTube Music (or YouTube) playlist URL")
    parser.add_argument(
        "-o",
        "--output",
        default=pipeline.DEFAULT_OUTPUT,
        help=f"Output directory (default: {pipeline.DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Isolate everything: drums, bass, harmony and vocals. Those four are the whole song between them - nothing belongs to two of them and nothing to none of them, so playing all four gives you the song back. Can also be given as a bare 'all' argument.",
    )
    parser.add_argument(
        "--drums",
        action="store_true",
        help="Also isolate each song's drums to a wav. Slow, off by default. Can also be given as a bare 'drums' argument, before or after the URL.",
    )
    parser.add_argument(
        "--bass",
        action="store_true",
        help="Also isolate each song's bass to a wav. Slow, off by default. Can also be given as a bare 'bass' argument, before or after the URL.",
    )
    parser.add_argument(
        "--harmony",
        action="store_true",
        help="Also isolate each song's harmony (guitars, keys, pads - everything but drums, bass and vocals) to a wav. Slow, off by default. Can also be given as a bare 'harmony' argument, before or after the URL.",
    )
    parser.add_argument(
        "--vocals",
        action="store_true",
        help="Also isolate each song's vocals to a wav. Slow, off by default. Can also be given as a bare 'vocals' argument, before or after the URL.",
    )

    # "all"/"drums"/"bass"/"harmony"/"vocals" are accepted bare (no --) too,
    # in any position, since that's the more natural way to type them.
    bare_modes, remaining_args = _split_bare_flags(sys.argv[1:])

    args = parser.parse_args(remaining_args)
    for flag in _BARE_FLAGS:
        setattr(args, flag, getattr(args, flag) or flag in bare_modes)

    # Expanded here rather than handled downstream, so "all" is purely a way
    # of typing the four names and everything after this point sees one kind
    # of request. Naming no instrument at all still means just the song.
    if args.all:
        for name in pipeline.INSTRUMENT_ORDER:
            setattr(args, name, True)

    instruments = [name for name in pipeline.INSTRUMENT_ORDER if getattr(args, name)]

    try:
        result = pipeline.run(
            args.url,
            output_dir=args.output,
            instruments=instruments,
            on_event=_Printer(),
        )
    except KeyboardInterrupt:
        print("\nStopped. Whatever was already produced is safe.")
        _exit(130)

    _exit(1 if (result.get("error") or result.get("download_status")) else 0)


if __name__ == "__main__":
    main()

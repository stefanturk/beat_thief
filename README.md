# song_downloader

Download all songs from a YouTube Music (or YouTube) playlist as MP3s.

## Setup

1. Install ffmpeg (required for MP3 conversion):
   ```
   brew install ffmpeg
   ```
2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

```
python3 song_downloader.py "https://music.youtube.com/playlist?list=YOUR_PLAYLIST_ID"
```

Songs are saved to `~/Downloads/Song Downloads` by default, named `Title - Artist.mp3`.

Use a custom output folder:

```
python3 song_downloader.py "https://music.youtube.com/playlist?list=YOUR_PLAYLIST_ID" -o "My Playlist"
```

Re-running the script on the same playlist skips songs you've already downloaded (tracked in `.downloaded_archive.txt` in the output folder), so it's safe to re-run as a playlist grows.

## Cleaning up your library

Every download run automatically cleans up the songs it just downloaded:
trimming dead air from the start/end, boosting volume on quiet tracks,
tidying up junky YouTube titles (removing things like "Official Video"),
writing proper Title/Artist tags, and removing duplicate downloads. The raw
download is replaced by the cleaned-up version — you won't end up with two
copies of the same song.

If a song's intro or outro is quiet but not silent (e.g. a lone instrument or
ambient sound), it's held for your review at the end of the run — it'll ask
you to press space when ready, then play exactly the 5 seconds where the
sanitized song would start or end, and you can choose to cut it, fade it
instead, keep it as-is, or adjust exactly where the cut happens.

You can also run the cleanup on its own, any time, against a folder or a
single MP3 file:

```
python3 song_sanitizer.py ["path/to/folder-or-file.mp3"]
```

The original is deleted once its cleaned-up replacement has been written
successfully — you never end up with two copies of the same song. If you run
it again on an already-cleaned file, it just notices there's nothing left to
do and leaves it alone — there's no hidden tracking file, it just looks at
what's in the folder.

Duplicate downloads that get found and removed are moved into a visible
`Duplicates` folder next to your songs, not deleted outright.

## Isolating drums (optional, slow)

Pass `drums` to also split each song's drums out into individual stems —
useful if you want to import them into a DAW (e.g. Ableton) to rebuild a
digitized version of the drum performance:

```
python3 song_downloader.py "https://music.youtube.com/playlist?list=YOUR_PLAYLIST_ID" drums
```

For each song this produces a `Drums/<Song Title>/` folder containing
`drums.wav` (the full drum mix), `kick.wav`, `snare.wav`, `toms.wav`, and
`cymbals_hihat.wav`. Hi-hat and cymbals come out bundled together in one
file for now — separating those two cleanly needs a heavier model that
isn't wired up yet.

This is off by default because it's slow (each song runs through two
machine-learning models) and downloads a one-time model file the first time
it's used. It also runs standalone against an existing folder or file:

```
python3 drum_isolator.py ["path/to/folder-or-file.mp3"]
```

Once you have the stems, drag them into Ableton and use Live's built-in
"Convert Drums to New MIDI Track" per stem — that's the most reliable way
to get from an isolated stem to editable MIDI right now, more reliable
than anything that could be fully scripted.

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
tidying up junky YouTube titles (removing things like "(Official Video)"),
writing proper Title/Artist tags, and removing duplicate downloads.

If a song's intro or outro is quiet but not silent (e.g. a lone instrument or
ambient sound), it's held for your review at the end of the run — you'll hear
the section played and can choose to cut it, fade it instead, keep it as-is,
or adjust exactly where the cut happens.

You can also run the cleanup on its own, any time, against any folder:

```
python3 song_sanitizer.py ["path/to/folder"]
```

If you don't finish reviewing ambiguous cuts (e.g. you Ctrl+C partway
through), resume later with:

```
python3 song_sanitizer.py --review
```

Original files are backed up untouched to a hidden `.originals/` folder
before anything is changed.

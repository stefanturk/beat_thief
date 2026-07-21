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

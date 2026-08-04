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
`drums.wav` (the full drum mix), `kick.wav`, `snare.wav`, `toms.wav`,
`cymbals.wav`, and `drums.mid` — a single combined MIDI file built by
detecting hits in each isolated stem and writing them all onto one drum
track, using the same note numbers as Ableton's default Drum Rack
(kick=36, snare=38, cymbals=40, toms=45). Drag `drums.mid` straight onto a
MIDI track with a drum rack loaded — no per-stem conversion or manual
combining needed. Any dead air or drum-less intro is trimmed off the front
first (reusing the same intro-cut detection as the sanitizer), so both the
stems and the MIDI start right on beat 1 instead of however many seconds
into the file the original intro happened to be.

The MIDI file's own tempo is detected, then precisely refined by fitting a
constant grid through every hit across the whole song (averaging over
however many hundred hits are in a full track cancels out individual
onset-detection jitter far better than a single windowed estimate can),
typically landing within a small fraction of a BPM of the song's real
tempo. Notes themselves are left at their raw, unquantized onset-detected
times — only the tempo is adjusted, so the grid lines up with the recording
instead of the recording being forced onto a grid.

Automatic tempo detection can still occasionally be off by an octave
(reading 95 BPM as 190, or vice versa) — refinement makes whichever octave
it picked very precise, but doesn't fix the octave itself. If the grid
looks twice as fast or half as slow as it should, that's the usual cause;
halving or doubling the tempo in Ableton fixes it without needing to touch
the notes.

Hi-hat and cymbals come out bundled together as one stem (and one MIDI
note) for now — separating those two cleanly needs a heavier model that
isn't wired up yet. Kick and snare transcribe well since they're clean,
punchy hits; toms and the cymbals/hi-hat stem are noisier and more likely
to need cleanup by hand.

This is off by default because it's slow (each song runs through two
machine-learning models) and downloads a one-time model file the first time
it's used. It also runs standalone against an existing folder or file:

```
python3 drum_isolator.py ["path/to/folder-or-file.mp3"]
```

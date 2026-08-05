# beat_thief

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
python3 beat_thief.py "https://music.youtube.com/playlist?list=YOUR_PLAYLIST_ID"
```

Quote the URL — playlist/video links usually contain `&`, which your shell
will otherwise treat as a command separator instead of part of the URL.

Songs are saved to `~/Downloads/Song Downloads` by default, named `Title - Artist.mp3`.

Use a custom output folder:

```
python3 beat_thief.py "https://music.youtube.com/playlist?list=YOUR_PLAYLIST_ID" -o "My Playlist"
```

Re-running the script on the same playlist skips songs you've already downloaded (tracked in `.downloaded_archive.txt` in the output folder), so it's safe to re-run as a playlist grows.

## Cleaning up your library

Every download run automatically cleans up the songs it just downloaded:
trimming dead air from the start/end, boosting volume on quiet tracks,
tidying up junky YouTube titles (removing things like "Official Video"),
writing proper Title/Artist tags, and removing duplicate downloads. The raw
download is replaced by the cleaned-up version — you won't end up with two
copies of the same song.

If a song's intro is quiet but not silent (e.g. a lone instrument or ambient
sound), it's held for your review at the end of the run — it'll ask you to
press space when ready, then play exactly the 5 seconds where the sanitized
song would start, and you can choose to cut it, fade it instead, keep it
as-is, or adjust exactly where the cut happens. Where the song starts
matters for beat-1/BPM accuracy, so it's worth a listen; a quiet-but-not-
silent outro is auto-faded instead, no review needed.

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

## Isolating instruments (optional, slow)

Pass `drums` and/or `bass` (with or without `--`, in either order, before or
after the URL) to also pull those parts out on their own — useful if you
want to import them into a DAW (e.g. Ableton) to rebuild a digitized
version of the performance, or just to understand how a song is built:

```
python3 beat_thief.py "https://music.youtube.com/playlist?list=YOUR_PLAYLIST_ID" drums bass
```

For each song this produces one `<Song Title> (Isolated)/` folder
containing whichever instruments you asked for, side by side — no separate
top-level `Drums`/`Bass` folders to dig through.

By default only the isolated `.wav` is written for each instrument. Add
`midi` (or `--midi`) to also get a matching `.mid` file, built by detecting
hits/notes directly in the wav:

```
python3 beat_thief.py "https://music.youtube.com/playlist?list=YOUR_PLAYLIST_ID" drums bass midi
```

Every instrument you isolate for a song shares the exact same beat-1 start
and the exact same tempo grid, so e.g. the drums and bass MIDI line up
exactly when dragged into a DAW together — and if a song's tempo turns out
to drift (see below) you're only asked about it once per song, even when
isolating both instruments. Both the trim point and the tempo are computed
once from the full song (reusing the sanitizer's own intro-cut detection,
and a tempo refined across the whole track), rather than each instrument
guessing its own — see below for what that grid computation looks like in
detail.

### Drums

`<Song Title> (Isolated Drums at N.NNN BPM).wav` is the isolated drum mix;
with `midi`, the matching `.mid` is a MIDI file built by detecting hits
directly in the wav and writing them all onto one drum track. Drag the
`.mid` file straight onto a MIDI track with a drum rack loaded; the exact
BPM in the filename is the same one baked into the MIDI file itself, so
it's there to read at a glance, not just to look up.

Each hit is guessed as kick, snare, or cymbal/hi-hat from its spectral
centroid (low-frequency hits are kicks, mid is snare, high is
cymbal/hi-hat) and written to Ableton's default Drum Rack note for that
piece (kick=36, snare=38, cymbal=42) — a cheap heuristic rather than a
second ML model, so expect it to be right most of the time on clean hits
and to need some manual cleanup on busier or more layered passages.

### Bass

`<Song Title> (Isolated Bass at N.NNN BPM).wav` is the isolated bass part;
with `midi`, the matching `.mid` is a MIDI file built by tracking the
bass's pitch directly (bass is monophonic, so this tracks one note at a
time rather than guessing a fixed drum-rack note per hit) and writing the
detected notes onto one track. Expect it to do well on a clear, single-note
bassline and to need cleanup on anything with slides, chords, or heavy
effects.

Before transcription, anything quieter than 5% of the isolated bass track's
own peak volume is gated out — imperfect stem separation tends to leave a
low-level noise floor behind that would otherwise get misread as extra,
spurious notes.

### The shared beat-1 / tempo grid

Any dead air or drum-less/bass-less intro is trimmed off the front of the
whole song first (reusing the same intro-cut detection as the sanitizer),
so every isolated instrument's wav and MIDI start right on beat 1 instead
of however many seconds into the file the original intro happened to be.

Each song's tempo is detected once from the full mix, then precisely
refined by fitting a constant grid through every onset across the whole
song (averaging over however many hundred onsets are in a full track
cancels out individual onset-detection jitter far better than a single
windowed estimate can), typically landing within a small fraction of a BPM
of the song's real tempo. Notes themselves are left at their raw,
unquantized detected times — only the tempo is adjusted, so the grid lines
up with the recording instead of the recording being forced onto a grid.

Automatic tempo detection can still occasionally be off by an octave
(reading 95 BPM as 190, or vice versa) — refinement makes whichever octave
it picked very precise, but doesn't fix the octave itself. If the grid
looks twice as fast or half as slow as it should, that's the usual cause;
halving or doubling the tempo in Ableton fixes it without needing to touch
the notes.

If a song's tempo isn't actually constant throughout (a DJ-style transition,
a tempo-ramped bridge, etc.), this is checked for directly: the song is
split into ~30 second windows, each with its own independently refined
tempo, and if any two windows disagree by 0.3 BPM or more you're asked
which one to use for the whole export (defaulting to the beginning of the
song on a bare Enter, since that's usually what you want for a loop). This
only happens interactively — a non-interactive run just uses the beginning
of the song automatically.

This is off by default because it's slow (each song runs through a
machine-learning model, once per instrument isolated). Each instrument also
runs standalone against an existing folder or file, with the same optional
`midi`/`--midi` flag:

```
python3 drum_isolator.py ["path/to/folder-or-file.mp3"] [midi]
python3 bass_isolator.py ["path/to/folder-or-file.mp3"] [midi]
```

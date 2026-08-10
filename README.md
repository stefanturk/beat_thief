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

## Beat Thief, the app

For the everyday job — one link, get its drums — there's a window:

```
./make_app.sh /Applications
```

That builds `Beat Thief.app` (leave off the path and it goes to
`~/Applications` instead). Double-click it, or drag it to your Dock, and you
get a real Mac window: paste a link, arm the parts you want, press Steal it.
Progress runs in the window, and each finished file has a Reveal button that
opens it in Finder ready to drag into Ableton. When a run finishes the big
readout shows the detected tempo — click it to copy.

Every song is listed under **Files** with the link it came from and a **Get
more** button, which puts that link back in the box — so coming back a week
later for the bass when you only took the drums is two clicks, not a hunt
through your browser history. The list survives quitting the app, reads which
files exist from disk (delete a stem in Finder and it stops being listed), and
covers songs downloaded from the command line too. Only the link and the
song's path are remembered, in `~/Library/Application Support/Beat Thief`.

Downloads go to `~/Music/Beat Thief`, not the `~/Downloads/Song Downloads`
the command line uses. macOS blocks apps from writing to Downloads (as it
does Desktop and Documents) without a permission grant that doesn't reliably
apply to a `python3` subprocess, so an app defaulting there would fail every
run. The window shows the destination at the bottom.

For the same reason the app carries a copy of the code inside its own bundle
rather than running the files in this folder — see the comments at the top of
`make_app.sh`. **Re-run `make_app.sh` after changing any Python or UI file**;
the app holds a snapshot. The terminal front end always runs the live code.

The window never asks questions. Where the terminal would stop and play you a
quiet intro or ask which tempo to use, the app just fades the intro and takes
the tempo from the start of the song. When you want that fine control, use the
command line below — it's the same pipeline either way (`pipeline.py`), so the
files come out identical.

If the app ever fails to start it says so in a dialog and writes the details
to `~/Library/Logs/beat_thief.log`.

The icon is `ui/logo.svg`, rendered to `.icns` at build time by
`render_logo.py`. Edit the SVG and rebuild to change both the Dock icon and
the mark in the window header.

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

Pass `drums`, `bass`, `harmony` and/or `vocals` (with or without `--`, in
either order, before or after the URL) to also pull those parts out on their
own — useful if you want to import them into a DAW (e.g. Ableton) to rebuild
a digitized version of the performance, or just to understand how a song is
built:

```
python3 beat_thief.py "https://music.youtube.com/playlist?list=YOUR_PLAYLIST_ID" drums bass harmony
```

Name no instrument at all and you just get the song. Say `all` and you get
every one of them:

```
python3 beat_thief.py "https://music.youtube.com/playlist?list=YOUR_PLAYLIST_ID" all
```

**Those four are the whole song.** Drums, bass, harmony and vocals are
disjoint and complete — nothing in the song belongs to two of them, and
nothing belongs to none of them. Drop all four onto four tracks at bar 1 at
unity gain and you're listening to the song. Mute one and you have the song
without it: the vocal alone, the instrumental, the band without the bass.

It is not a bit-exact rebuild of the master, though. Separation is a guess,
and measured on a real track the four summed back to within about 10% of the
original (roughly −20 dB of residual) — that's the model's error, not
missing content, and it's the same error you're already hearing in any one
stem on its own. Expect the usual faint separation artifacts, not a null
test.

Asking for all four costs about what asking for one costs. The separation
model produces all four parts in a single pass whatever you asked for, so
the pass is done once and shared.

For each song this produces one `<Song Title> (Isolated)/` folder
containing whichever instruments you asked for, side by side — no separate
top-level `Drums`/`Bass`/`Harmony` folders to dig through.

Each instrument is written as a `.wav`. There's no whole-song MIDI any
more — see [Beats, not songs](#beats-not-songs).

Every instrument you isolate for a song shares the exact same start and
the exact same tempo, so the stems line up
exactly when dragged into a DAW together — and if a song's tempo turns out
to drift (see below) you're only asked about it once per song, even when
isolating both instruments. Both the trim point and the tempo are computed
once from the full song (reusing the sanitizer's own intro-cut detection,
and a tempo refined across the whole track), rather than each instrument
guessing its own — see below for what that grid computation looks like in
detail.

### Drums

`<Song Title> (Isolated Drums at N.NNN BPM).wav` is the isolated drum mix.
The exact BPM is in the filename, there to read at a glance.

The drum transcriber that reads it — six pieces, per-hit velocity — is
still here in `drum_transcriber.py`, now aimed at a chosen section rather
than a whole song. It is [ADTOF](https://github.com/MZehren/ADTOF) — a
convolutional-recurrent network trained on 114 hours of real annotated
music — via [ADTOF-pytorch](https://github.com/xavriley/ADTOF-pytorch). The
model and its weights are vendored in `adtof/`; see `adtof/NOTICE.md` for
the attribution and the licence. It adds no dependencies Beat Thief didn't
already have, and it reproduces the published implementation note for note
on that project's own test file.

The model predicts five classes. Open vs closed hi-hat is Beat Thief's own
work on top: each hi-hat's high-frequency energy is followed to see whether
it's still ringing when the next hat lands. That part is a measurement, not
a trained behaviour, and it's deliberately biased toward calling a hat
closed — a closed hat written as open rings over the next one and sounds
wrong, while an open hat written as closed just sounds tight.

Velocity is measured per hit from the audio, in a frequency band belonging
to that piece — so a crash on beat 1 doesn't inflate the kick underneath
it — and scaled in decibels against the loudest hit **of the same piece**.
Hi-hats sit around 20dB under the kick in most mixes, so scaling everything
together would push every hat to the bottom of the range and flatten the
part.

### Bass

`<Song Title> (Isolated Bass at N.NNN BPM).wav` is the isolated bass part.

Anything quieter than 5% of the track's own peak volume is gated out —
imperfect stem separation tends to leave a low-level noise floor behind,
and this is what's left when the real bass content stops.

### Harmony

`<Song Title> (Isolated Harmony).wav` is everything left in the mix once
drums, bass and vocals are pulled out — guitars, keys, pads, whatever else
is holding the chords up — meant to drop straight into a DAW alongside the
other exports from the same song. No MIDI step for this one, just audio.

Harmony used to include the vocals as well, which meant there was no way to
get an instrumental. If you have a harmony file from before that change it
still has the vocals in it, so it will be rebuilt (not skipped) the next
time you ask for harmony on that song.

### Vocals

`<Song Title> (Isolated Vocals).wav` is the singing — lead and backing —
with the band taken out from under it. Audio only, no MIDI.

### Beats, not songs

Transcribing four minutes of a live drummer produced a faithful and
unusable wall of notes. What's actually wanted is a couple of bars with a
good beat in them, tight to a grid, that loop. So whole-song MIDI is gone —
both the drums one and the rough pitch-tracked bass one — and what replaces
it works on a section you choose.

That reverses this project's old "never quantize" rule, deliberately and
only here: a stolen loop wants to be tight, and there's no longer a
faithful full-song transcription for it to contradict.

`beat_writer.py` is the first piece of it — everything that ends up as a
`.mid` goes through it, so there's one place that knows what Ableton
accepts. Run it to write two reference files:

```
python3 beat_writer.py [output-folder]
```

- **Reference Groove** — a real two-bar beat. The "does it sound right" test.
- **Reference All Pads** — every piece, one per beat, in note order, at
  velocities 20/50/80/110. Not music: a diagnostic, where a mapping mistake
  or a velocity that didn't survive is obvious at a glance.

Fourteen pieces, all inside the sixteen pads of Ableton's default Drum
Rack, so a clip drops onto a stock rack with nothing to remap:

| | | | |
| --- | --- | --- | --- |
| 36 Kick | 37 Side stick | 38 Snare | 41 Low floor tom |
| 42 Closed hat | 43 High floor tom | 44 Pedal hat | 45 Low tom |
| 46 Open hat | 47 Low-mid tom | 48 Hi-mid tom | 49 Crash |
| 50 High tom | 51 Ride | | |

Splash (55) and china (52) are missing on purpose — they're outside the
rack's range, and a note with no pad is silent *and* invisible, which is the
worst way for a note to be wrong.

Note times are floats in steps, so swing and triplets don't need a
different grid. The one hard floor is the format itself: MIDI stores every
time as a whole number of ticks, so times land on a tick whatever we do.
`RESOLUTION` is set to 960 per quarter note, which puts that error under a
millisecond.

### The shared start / tempo grid

Any dead air or drum-less/bass-less intro is trimmed off the front of the
whole song first (reusing the same intro-cut detection as the sanitizer),
so every isolated instrument's wav starts where the music does
instead of however many seconds into the file the original intro happened
to be.

Nothing is snapped to a grid at any point — not the audio, not the notes.
Where the beats fall relative to that start is Ableton's business, and
nudging a clip is a drag.

**On dragging MIDI into Ableton.** Live ignores a MIDI file's tempo header
on import — it drops the notes onto your set's grid at whatever tempo the
set is. The notes are placed correctly in *beats*, so at a 120 BPM set a
157 BPM song's clip takes 1.3× as long in seconds and reads as everything
slowed down. Set the tempo to the BPM in the filename and it lines up. This
is Live's import behaviour, not something the file can carry.

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

This is off by default because it's slow — each song runs through a
machine-learning model. That happens once per song per run, though, not once
per instrument: asking for all four parts is roughly the cost of asking for
one. Each instrument also runs standalone against an existing folder or
file, none of them taking any flags:

```
python3 drum_isolator.py ["path/to/folder-or-file.mp3"]
python3 bass_isolator.py ["path/to/folder-or-file.mp3"]
python3 harmony_isolator.py ["path/to/folder-or-file.mp3"]
python3 vocals_isolator.py ["path/to/folder-or-file.mp3"]
```

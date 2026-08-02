# Song Sanitizer — Design

## Purpose

After songs are downloaded, some need cleanup: quiet/dead intros or outros carried
over from the YouTube source, tracks that are quieter than they should be, messy
YouTube-style titles, missing metadata, and occasional duplicates. The song
sanitizer analyzes and fixes these *technical* properties of an MP3 — never
creative edits like EQ or compression — either standalone or automatically after
a download.

## Invocation

- **Standalone**: `python3 song_sanitizer.py ["path"]` — defaults to
  `~/Downloads/Song Downloads` (same default as `song_downloader.py`) if no path
  is given. Runs the full pipeline over every MP3 in the folder.
- **Automatic**: `song_downloader.py` imports `sanitize_folder()` and calls it on
  the output directory at the end of every download run.
- **Resuming a review**: `python3 song_sanitizer.py --review` — skips
  processing and jumps straight to resolving any previously-flagged, unresolved
  ambiguous cases.

## Dependencies

Adds to `requirements.txt`:
- `pydub` — silence detection, trimming, fades, peak normalization (uses the
  `ffmpeg` binary already required by the downloader).
- `mutagen` — reading/writing ID3 tags.

No new dependency for duplicate detection — uses stdlib `difflib`. No new
dependency for playback — shells out to macOS's built-in `afplay`.

## State files (in the output directory)

- `.sanitized_archive.txt` — one filename per line; files already fully
  processed. Skipped on subsequent runs (mirrors `.downloaded_archive.txt`'s
  skip-existing pattern).
- `.sanitizer_flagged.json` — pending ambiguous cut decisions that haven't been
  resolved yet (survives Ctrl+C mid-review). Cleared per-entry as each is
  resolved.
- `.originals/` — hidden subfolder holding untouched copies of any file the
  sanitizer has modified or removed as a duplicate, made *before* the first
  modification. This is a safety net for the current, less-trusted version of
  the tool; may be dropped later once trusted.

## Per-file pipeline

Runs once per MP3 not already in `.sanitized_archive.txt`, in this order:

1. **Backup** — if this file has never been backed up before, copy it
   untouched into `.originals/` before any modification.

2. **Detect intro/outro cut candidates** — analyze the amplitude envelope from
   both ends of the track (via `pydub`) against the track's average loudness.
   Classify each end independently into one of three buckets:
   - **Clearly silent** (below a strict noise floor, e.g. -50dBFS) → auto-trim
     immediately, no review needed.
   - **Ambiguous** (quieter than the track average by a wide margin — e.g. more
     than ~25dB down — but not silent; typically a lone instrument or ambient
     intro/outro) → record a flag in `.sanitizer_flagged.json` with the
     candidate cut timestamp; do **not** modify the file yet.
   - **Normal volume** → no action.

3. **Peak-normalize volume** — if the track's peak amplitude is meaningfully
   below full scale (e.g. more than ~3dB of headroom unused), apply a uniform
   gain boost so the loudest sample reaches ~-1dBFS. Skipped if the track is
   already near full volume. This step is unambiguous and never needs review.

4. **Title cleanup** — strip a known list of common YouTube noise patterns from
   the filename (case-insensitive): `(Official Video)`, `[Official Audio]`,
   `(Official Music Video)`, `(Lyrics)`, `(Lyric Video)`, `HD`, `4K`,
   `(Visualizer)`, `(Audio)`, and similar bracket/paren-wrapped junk, reducing
   toward a clean `Song - Artist` filename. Renames the file if changed.

5. **Write ID3 tags** — using `mutagen`, write `Title` and `Artist` tags parsed
   from the cleaned `Song - Artist` filename (split on the last ` - `).

6. **Mark sanitized** — once steps 2-5 are complete for a file (regardless of
   whether it has a pending ambiguous flag from step 2), record its current
   filename in `.sanitized_archive.txt` so it isn't reprocessed. A file with a
   pending flag is still marked sanitized — the flag is tracked separately in
   `.sanitizer_flagged.json` and re-checked against the archive on `--review`.

Errors on any individual file (corrupt MP3, unreadable file, etc.) are caught
and skipped with a plain-English message; that file is left untouched and not
marked sanitized, so it's retried on the next run.

## Duplicate detection

Runs once per batch, after all per-file processing:

1. Build a normalized `"title - artist"` string for every MP3 currently in the
   folder (lowercased, punctuation stripped).
2. Fuzzy-compare all pairs using `difflib.SequenceMatcher`; pairs above a
   similarity threshold (e.g. 0.9) are treated as duplicates.
3. For each duplicate pair, keep the larger file (a simple proxy for higher
   source quality/bitrate) and move the other into `.originals/` (not deleted
   outright, consistent with the backup-first safety approach).

This is cheap (string comparison only) so it runs on the full folder every
batch, not just newly-downloaded files — it catches duplicates against your
existing library, not just within one run.

## Interactive review (ambiguous cut points)

Triggered automatically at the end of any batch — standalone run or the
automatic post-download pass — whenever `.sanitizer_flagged.json` has unresolved
entries (from this batch or a previous interrupted one). This intentionally
breaks the "walk away" background-friendliness of a plain download for this one
step, per explicit preference: cleanup decisions should be reviewed
immediately, not deferred to a separate later invocation, though `--review` is
still available if you Ctrl+C out partway through.

For each flagged song, in order:

1. **Play the snippet** — extract and play (via `afplay`) roughly 5 seconds
   before and after the candidate cut point (i.e. a ~10s window centered on the
   cut), so you can hear exactly what's being considered for removal.
2. **Prompt**: `(c)ut here / (f)ade instead / (k)eep as-is / (a)djust point`.
   - **Cut** — trim the file at the candidate point, apply, mark resolved.
   - **Fade** — instead of a hard cut, apply a fade-in (for intros) or
     fade-out (for outros) over a default duration (e.g. 2s) starting at the
     candidate point, rather than removing audio outright. Apply, mark
     resolved.
   - **Keep as-is** — no change to this section; mark resolved, no-op.
   - **Adjust** — prompt for a new offset in seconds (earlier/later), replay
     the new ~10s window centered on the adjusted point, and re-ask the same
     four-way prompt (loops until cut/fade/keep is chosen).
3. Remove the entry from `.sanitizer_flagged.json` once resolved and move to
   the next flagged song.

If the process is interrupted (Ctrl+C) mid-review, already-resolved entries
stay applied and resolved; unresolved ones remain in
`.sanitizer_flagged.json` for the next run (automatic or `--review`) to pick up.

## Explicitly out of scope (v1)

- Audio-fingerprint-based duplicate detection (filename-based only for now).
- Subjective/AI categorization (energy, mood, genre).
- Loudness (LUFS) normalization — peak normalization only.
- Any playback method other than macOS `afplay` (no cross-platform support
  needed at this time).

## Testing

- Manual testing against real downloaded MP3s covering: a track with an
  obviously silent intro (auto-trim), a track with a quiet-but-audible ambient
  intro (should flag and prompt), a track already at good volume (no-op), a
  quiet track (normalize triggers), a pair of near-duplicate titles (dedup
  triggers), and a title with common YouTube junk (cleanup triggers).
- Verify `--review` correctly resumes after a simulated Ctrl+C mid-review
  (interrupt the process, confirm `.sanitizer_flagged.json` still has the
  unresolved entries, re-run and confirm it picks up where it left off).
- Verify `song_downloader.py` still completes and exits cleanly (no shell hang,
  consistent with prior `os._exit()` hardening) when the automatic sanitize
  pass runs afterward, including when it drops into interactive review.

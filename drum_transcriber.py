#!/usr/bin/env python3
"""Turn an isolated drum wav into MIDI notes, using the ADTOF drum
transcription model (see adtof/NOTICE.md).

The model tells us what was hit and when. Everything a musician would
notice beyond that - how hard each hit was, whether a hi-hat was open or
closed, and which pad of a drum rack each note lands on - is worked out
here from the audio itself.

Used by drum_isolator.py, which owns the file naming, the tempo and the
actual MIDI file. This module only produces notes."""

from __future__ import annotations

import statistics
from functools import lru_cache

import numpy as np
import pretty_midi
import torch
from scipy.signal import butter, sosfilt

import adtof
import percussion_splitter

SAMPLE_RATE = 44100  # what adtof.AudioProcessor loads at; the bands below assume it

# The model's five classes, in the order it emits them.
_KICK, _SNARE, _TOM, _HIHAT, _CYMBAL = range(5)

# Where each class lands in a MIDI file. Every one of these sits inside the
# 36-51 range of Ableton's default Drum Rack, so the file drops onto a stock
# rack without remapping - which is why kick is 36 and not the 35 the model
# emits, since 35 is below the rack's first pad and would be invisible.
_NOTE_FOR_CLASS = (36, 38, 47, 42, 49)
_OPEN_HIHAT_NOTE = 46

# The snare class comes back out as three pads, because the model has one
# class for everything that hit like a snare and a real drummer had more than
# one thing that did.
#
#   39  percussion - a tambourine or a shaker, voted on by percussion_splitter
#       and confirmed or overturned by groove_reader once the loop is on a
#       grid and its rhythm can be read
#
# 39 is General MIDI's hand clap, borrowed: the rack has no tambourine pad
# inside 36-51, and GM's real one at 54 imports onto an empty pad and is
# silent (see beat_writer's docstring).
#
# The third pad the snare class ends up on - 37, for ghost notes - is not
# decided here. It's a judgement about how quiet a hit is relative to the
# loop's other snares, and by the time groove_reader has finished moving
# voices between pads, which hits those are has changed. So it happens there,
# once, at the end.
_PERCUSSION_NOTE = 39

# The hi-hat class gets the same treatment, and for the same reason: on the
# drum stem of Officially Missing You, the one section confirmed by ear to be
# solo tambourine has the model calling it hi-hat throughout, not snare - the
# tambourine there scores +7.1 on percussion_splitter.score, where real
# hi-hats elsewhere score about -18. So a hi-hat hit that scores as percussion
# is written provisionally to its own borrowed pad rather than 39: hi-hat and
# snare are different pools with different evidence (a real hi-hat is *also*
# almost always on a pulse, so groove_reader can't tell the two pools apart
# by rhythm the way it does for the snare class - see that module), and
# merging them here would erase the distinction before groove_reader ever
# gets to use it. 41 is unused by anything else this file writes.
_HAT_PERCUSSION_NOTE = 41

# calibrate_hat_threshold's margin above a song's own median hi-hat-class
# score, and how many hits it needs before trusting that median at all.
# Measured on Officially Missing You, where percussion_splitter's fixed line
# happens to sit close to the midpoint between real hi-hats elsewhere
# (~-18 median) and the confirmed tambourine section (~+7.1) - about 12-13dB
# either side. There's only the one song's worth of calibration behind this
# margin, same as the fixed threshold it's meant to improve on; it's a
# starting point, not a measured constant.
_CALIBRATION_MARGIN_DB = 10.0
_MIN_HITS_FOR_CALIBRATION = 20

# How long each class's notes are written as. Cosmetic in a rack full of
# one-shots, but a drum part whose notes are all 50ms slivers is hard to
# read in a MIDI editor. An open hi-hat gets its measured ring instead of a
# constant (see _hihat_ring_sec).
_NOTE_DURATION_SEC = (0.05, 0.05, 0.05, 0.05, 0.25)

# Band each class's velocity is measured in. Without this, a crash landing
# on beat 1 would inflate that beat's kick velocity, because the kick's
# measurement window contains the crash. (low, high) in Hz; None means open
# at that end.
_VELOCITY_BAND = (
    (None, 120.0),    # kick
    (150.0, 800.0),   # snare
    (80.0, 400.0),    # toms
    (5000.0, None),   # hi-hat
    (3000.0, None),   # cymbal
)

_VELOCITY_WINDOW_SEC = 0.03

# Velocity is scaled in decibels rather than in raw amplitude, across this
# much range below the loudest hit of the same class. Amplitude is the wrong
# scale for this: a ghost note is around a tenth the amplitude of a
# backbeat, which linearly is velocity 12 - somewhere between inaudible and
# absent. In dB it's velocity 40, which is what a drummer would actually
# program.
_VELOCITY_DYNAMIC_RANGE_DB = 30.0

# Velocity is normalized per class, not across the song. Hi-hats sit ~20dB
# under the kick in almost every mix, so one shared scale would put every
# hat near the bottom of its range and flatten the part. Each pad of a drum
# rack has its own gain anyway; what has to survive into the MIDI is the
# dynamics *within* each piece.

_OPEN_HIHAT_DECAY_SEC = 0.12   # a hat still ringing this long after the hit is an open one
_OPEN_HIHAT_DROP_DB = 12.0     # "still ringing" means it hasn't fallen this far below its own peak
_OPEN_HIHAT_MAX_RING_SEC = 0.5  # longest ring written as a note duration
_HIHAT_ATTACK_SEC = 0.025      # how far past the onset a hit's own peak is looked for
_HIHAT_CHOKE_MARGIN_SEC = 0.01  # measuring stops this far short of the next hat, which would otherwise be measured as part of this one
_HIHAT_MIN_MEASURABLE_SEC = 0.04  # hats closer together than this can't be told apart at all, so they're left closed

# Peak picking, ported from the model's own post-processing so that its
# published per-class thresholds still mean what they were measured to mean.
# All in seconds.
_PRE_AVG_SEC = 0.1
_POST_AVG_SEC = 0.01
_PRE_MAX_SEC = 0.02
_POST_MAX_SEC = 0.01
_COMBINE_SEC = 0.02

# The model is run in chunks rather than over a whole song at once. A four
# minute song is 24,000 frames, and the first convolution block alone holds
# 24000 x 84 x 32 floats - a quarter of a gigabyte per activation, several
# live at once, on a machine that has just finished running demucs. Measured
# on a real 4 minute file: 2.76 GB peak in one pass, 1.3 GB chunked.
_CHUNK_FRAMES = 1500  # 15 seconds kept per pass
# Context read on each side of a chunk and then thrown away. The recurrent
# layers are bidirectional, so a frame judged near a chunk edge is judged
# without the rest of the song on that side, and quiet hits near a seam get
# lost. These numbers aren't guesses: swept against a whole-song pass, 3s
# of context lost or invented 26 hits, 6s lost 9, 10s lost 2, 15s lost one,
# and 20s reproduces the whole-song pass exactly. Note it's the context that
# buys the accuracy, not the chunk size - a wide chunk with a narrow context
# is worse than the reverse at the same cost in memory.
_CONTEXT_FRAMES = 2000  # 20 seconds


_model = None


def _load_model():
    """Build the model and load its weights once per process. A folder run
    transcribes dozens of songs and must not rebuild a network each time."""
    global _model
    if _model is None:
        model = adtof.create_frame_rnn_model(adtof.calculate_n_bins())
        model.eval()
        # Raises rather than falling back to random weights - a transcriber
        # that runs and returns confident nonsense is the worst outcome here.
        _model = adtof.load_pytorch_weights(model, adtof.WEIGHTS_PATH, strict=False)
    return _model


def _activations(spectrogram: np.ndarray) -> np.ndarray:
    """Run the model over a whole song's spectrogram in overlapping chunks
    and stitch the results back into one [frames, 5] array of per-class
    activations. See _CHUNK_FRAMES for why this isn't one forward pass."""
    total_frames = spectrogram.shape[0]
    if total_frames == 0:
        return np.zeros((0, 5), dtype=np.float32)

    model = _load_model()

    out = np.zeros((total_frames, 5), dtype=np.float32)
    for start in range(0, total_frames, _CHUNK_FRAMES):
        end = min(start + _CHUNK_FRAMES, total_frames)
        padded_start = max(0, start - _CONTEXT_FRAMES)
        padded_end = min(total_frames, end + _CONTEXT_FRAMES)

        chunk = spectrogram[padded_start:padded_end]
        x = torch.from_numpy(np.ascontiguousarray(chunk)).float().unsqueeze(0)
        with torch.no_grad():
            predicted = model(x).numpy()[0]

        # Keep only the middle - the context frames on either side existed
        # to give the edges of this slice something to look at.
        out[start:end] = predicted[start - padded_start:end - padded_start]
    return out


def _moving_average(x: np.ndarray, left: int, right: int) -> np.ndarray:
    if left <= 0 and right <= 0:
        return x
    kernel = np.ones(left + 1 + right, dtype=np.float32) / float(left + 1 + right)
    return np.convolve(np.pad(x, (left, right), mode="edge"), kernel, mode="valid").astype(np.float32)


def _rolling_max(x: np.ndarray, left: int, right: int) -> np.ndarray:
    if left <= 0 and right <= 0:
        return x
    size = left + 1 + right
    padded = np.pad(x, (left, right), mode="edge")
    return np.stack([padded[k:k + x.size] for k in range(size)], axis=0).max(axis=0)


def _pick_peaks(activation: np.ndarray, threshold: float, fps: int) -> list[int]:
    """Frame indices of the hits in one class's activation curve.

    A port of the peak picker the model was published with (a local-maximum
    search over the activation minus its own running mean, then a
    de-duplication pass), kept faithful to its parameters because the
    model's measured accuracy is an accuracy *of this procedure* - a
    different picker is a different number."""
    activation = np.asarray(activation, dtype=np.float32).reshape(-1)
    if activation.size == 0:
        return []

    smoothed = _moving_average(activation, int(round(_PRE_AVG_SEC * fps)), int(round(_POST_AVG_SEC * fps)))
    detection = np.maximum(0.0, activation - smoothed)

    local_max = _rolling_max(detection, int(round(_PRE_MAX_SEC * fps)), int(round(_POST_MAX_SEC * fps)))
    candidates = np.where((detection >= local_max) & (detection >= threshold))[0]
    if candidates.size == 0:
        return []

    # A single hit can clear the threshold for several frames running. Keep
    # the strongest frame of each run.
    combine_frames = max(1, int(round(_COMBINE_SEC * fps)))
    picked: list[int] = []
    group = [int(candidates[0])]
    for index in candidates[1:]:
        if index - group[-1] <= combine_frames:
            group.append(int(index))
        else:
            picked.append(max(group, key=lambda i: detection[i]))
            group = [int(index)]
    picked.append(max(group, key=lambda i: detection[i]))
    return picked


def _band_filtered(audio: np.ndarray, low: float | None, high: float | None) -> np.ndarray:
    """audio with everything outside (low, high) taken out, so a hit's
    loudness can be measured without the rest of the kit leaking into it."""
    nyquist = SAMPLE_RATE / 2.0
    if low is not None and high is not None:
        sos = butter(4, [low / nyquist, min(high, nyquist * 0.99) / nyquist], btype="bandpass", output="sos")
    elif low is not None:
        sos = butter(4, low / nyquist, btype="highpass", output="sos")
    elif high is not None:
        sos = butter(4, high / nyquist, btype="lowpass", output="sos")
    else:
        return audio
    return sosfilt(sos, audio).astype(np.float32)


def _peak_amplitude(audio: np.ndarray, time_sec: float, window_sec: float) -> float:
    start = int(time_sec * SAMPLE_RATE)
    window = audio[start:start + max(1, int(window_sec * SAMPLE_RATE))]
    return float(np.abs(window).max()) if window.size else 0.0


def _velocities(amplitudes: list[float]) -> list[int]:
    """Scale one class's hit amplitudes into MIDI velocities, in decibels
    relative to that class's own loudest hit. See
    _VELOCITY_DYNAMIC_RANGE_DB."""
    loudest = max(amplitudes, default=0.0)
    if loudest <= 0:
        return [1 for _ in amplitudes]

    velocities = []
    for amplitude in amplitudes:
        if amplitude <= 0:
            velocities.append(1)
            continue
        below_db = -20.0 * np.log10(loudest / amplitude)
        fraction = 1.0 + below_db / _VELOCITY_DYNAMIC_RANGE_DB
        velocities.append(int(np.clip(round(fraction * 127), 1, 127)))
    return velocities


def _hihat_ring_sec(hihat_band: np.ndarray, time_sec: float, window_sec: float) -> float:
    """How long a hi-hat hit keeps ringing: the time its own high-frequency
    energy takes to fall _OPEN_HIHAT_DROP_DB below the peak of its attack.
    Never looks further than window_sec, and returns window_sec when it's
    still ringing at the end of it.

    The peak is taken from the first _HIHAT_ATTACK_SEC and the search runs
    forward from there, rather than from the loudest point of the whole
    window. That distinction is the whole measurement: a window this long
    usually contains a kick or a snare louder than the hi-hat, and taken
    against that every hat looks like it stopped instantly."""
    start = int(time_sec * SAMPLE_RATE)
    window = np.abs(hihat_band[start:start + max(1, int(window_sec * SAMPLE_RATE))])
    if window.size == 0:
        return 0.0

    # Smoothed over a millisecond, so a single sample crossing zero in the
    # middle of a loud ring doesn't read as the sound having stopped.
    smoothing = max(1, int(0.001 * SAMPLE_RATE))
    envelope = np.convolve(window, np.ones(smoothing, dtype=np.float32) / smoothing, mode="same")

    attack = envelope[:max(1, int(_HIHAT_ATTACK_SEC * SAMPLE_RATE))]
    peak_index = int(np.argmax(attack))
    peak = float(envelope[peak_index])
    if peak <= 0:
        return 0.0

    quiet = np.where(envelope[peak_index:] < peak * (10.0 ** (-_OPEN_HIHAT_DROP_DB / 20.0)))[0]
    if quiet.size == 0:
        return window_sec
    return float(peak_index + quiet[0]) / SAMPLE_RATE


def _split_hihats(
    hit_times: dict[int, list[float]],
    hihat_band: np.ndarray,
) -> dict[float, float]:
    """Which hi-hat hits were played open, and how long each one rang.

    The model has one hi-hat class, so this is Beat Thief's guess rather
    than something it was trained to do, and it is deliberately biased
    toward calling a hat closed: a closed hat written as open is a note
    that rings over the next one and sounds wrong, while an open hat
    written as closed just sounds tight.

    Measuring stops just short of the next hi-hat, because that hat ends
    this one whether it was open or not - so what's actually being asked is
    "was this hat still ringing when the next one arrived, or by
    _OPEN_HIHAT_DECAY_SEC, whichever comes first". A closed hat is quiet
    within about 25ms, so there's plenty of room to tell them apart even in
    a fast pattern.

    Two things make the answer unavailable rather than long, and both mean
    "leave it closed":

      - the next hat is so close there's no room to measure at all
      - a cymbal lands inside the window, which occupies the same part of
        the spectrum and rings for seconds

    Returns {onset time: ring length} for the hits judged open."""
    hihat_times = sorted(hit_times.get(_HIHAT, []))
    cymbal_times = sorted(hit_times.get(_CYMBAL, []))

    open_hits: dict[float, float] = {}
    for index, time_sec in enumerate(hihat_times):
        window_sec = _OPEN_HIHAT_MAX_RING_SEC
        if index + 1 < len(hihat_times):
            window_sec = min(window_sec, hihat_times[index + 1] - time_sec - _HIHAT_CHOKE_MARGIN_SEC)
        if window_sec < _HIHAT_MIN_MEASURABLE_SEC:
            continue
        if any(time_sec <= cymbal < time_sec + window_sec for cymbal in cymbal_times):
            continue

        ring = _hihat_ring_sec(hihat_band, time_sec, window_sec)
        if ring >= min(_OPEN_HIHAT_DECAY_SEC, window_sec):
            open_hits[time_sec] = ring
    return open_hits


@lru_cache(maxsize=None)
def calibrate_hat_threshold(wav_path: str) -> float:
    """How bright a hi-hat-class hit has to score, on this recording's own
    hi-hat/ride, before it's called percussion rather than the real thing.

    percussion_splitter._PERCUSSION_SCORE_DB is one fixed number, calibrated
    against one song. What "real" sounds like varies by kit, mic'ing and
    genre - measured on a jazz kit ("Three", Nicholas Payton), the fixed
    line read as close to a coin flip on hits that, filtered to the actual
    rhythmic pulse groove_reader uses, were confidently real. So instead of
    a fixed line, this reads the whole file's own hi-hat class and sets the
    line relative to what's typical *here*: most songs' hi-hat parts are
    mostly the real instrument (full-song tambourine-instead-of-hi-hat is
    rare), so the median of the whole population is a reasonable stand-in
    for "what a real hit here scores," and a hit has to clear that by a real
    margin (_CALIBRATION_MARGIN_DB) before it's treated as percussion.

    Falls back to the fixed threshold when there's too little of the class
    to calibrate against - a handful of hits don't make a trustworthy
    median. Cached per file for the life of the process: this is a whole-
    song pass, done once, not per marked section - beat_loop.build calls it
    before every transcribe() so multiple loops stolen from the same song
    share one calibration."""
    processor = adtof.create_adtof_processor()
    audio = processor.load_audio(wav_path)
    if audio.size == 0:
        return percussion_splitter._PERCUSSION_SCORE_DB

    spectrogram = processor.apply_filterbank(processor.compute_stft(audio)).T[:, :, np.newaxis]
    activations = _activations(np.asarray(spectrogram, dtype=np.float32))
    frames = _pick_peaks(activations[:, _HIHAT], adtof.THRESHOLDS_5[_HIHAT], adtof.FPS)
    times = [frame / float(adtof.FPS) for frame in frames]
    if len(times) < _MIN_HITS_FOR_CALIBRATION:
        return percussion_splitter._PERCUSSION_SCORE_DB

    scores = [s for s in percussion_splitter.score(audio, times, SAMPLE_RATE) if s != float("-inf")]
    if len(scores) < _MIN_HITS_FOR_CALIBRATION:
        return percussion_splitter._PERCUSSION_SCORE_DB

    return statistics.median(scores) + _CALIBRATION_MARGIN_DB


def transcribe(wav_path: str, hat_threshold: float | None = None) -> list[pretty_midi.Note]:
    """Detect every drum hit in wav_path and return them as MIDI notes,
    sorted by time, with times in seconds from the start of the file.

    Seven pieces out of five classes: kick, snare, percussion, toms, closed
    hi-hat, open hi-hat and cymbal. The five classes come from the model;
    open vs closed hi-hat is measured here (see _split_hihats), and the snare
    class is split in two by percussion_splitter.

    The percussion note is provisional. It's one weak vote on one hit, and
    groove_reader is what decides - over a whole voice, on a grid, once
    there's a rhythm to read.

    `hat_threshold` overrides percussion_splitter's default line for the
    hi-hat class's vote only (the snare class's is untouched). None uses the
    default - see calibrate_hat_threshold for where a caller gets a better
    number, one fitted to this recording rather than to Officially Missing
    You."""
    processor = adtof.create_adtof_processor()
    audio = processor.load_audio(wav_path)
    if audio.size == 0:
        return []

    spectrogram = processor.apply_filterbank(processor.compute_stft(audio)).T[:, :, np.newaxis]
    activations = _activations(np.asarray(spectrogram, dtype=np.float32))

    fps = adtof.FPS
    hit_times: dict[int, list[float]] = {}
    for class_index, threshold in enumerate(adtof.THRESHOLDS_5):
        frames = _pick_peaks(activations[:, class_index], threshold, fps)
        hit_times[class_index] = [frame / float(fps) for frame in frames]

    hihat_band = _band_filtered(audio, *_VELOCITY_BAND[_HIHAT])
    open_hihats = _split_hihats(hit_times, hihat_band)

    # Which of the snare class's hits weren't drums. Weak per hit on purpose -
    # groove_reader aggregates it over a whole voice once the loop is on a
    # grid. Until then it travels as the note the hit was written to.
    snares = hit_times.get(_SNARE, [])
    votes = dict(zip(snares, percussion_splitter.split(audio, snares, SAMPLE_RATE)))

    # Same vote, same audio, same bands - taken on the hi-hat class's hits
    # instead. See _HAT_PERCUSSION_NOTE for why this is a second pool rather
    # than folded into the one above.
    hats = hit_times.get(_HIHAT, [])
    hat_split_kwargs = {} if hat_threshold is None else {"threshold": hat_threshold}
    hat_votes = dict(zip(hats, percussion_splitter.split(audio, hats, SAMPLE_RATE, **hat_split_kwargs)))

    notes: list[pretty_midi.Note] = []
    for class_index, times in hit_times.items():
        if not times:
            continue
        band = hihat_band if class_index == _HIHAT else _band_filtered(audio, *_VELOCITY_BAND[class_index])
        amplitudes = [_peak_amplitude(band, time_sec, _VELOCITY_WINDOW_SEC) for time_sec in times]

        for time_sec, velocity in zip(times, _velocities(amplitudes)):
            ring = open_hihats.get(time_sec) if class_index == _HIHAT else None
            if class_index == _HIHAT and hat_votes.get(time_sec) == percussion_splitter.PERCUSSION:
                # A vote against being a drum at all overrides open-vs-closed -
                # that question doesn't apply to whatever this turns out to be.
                pitch, duration = _HAT_PERCUSSION_NOTE, _NOTE_DURATION_SEC[_HIHAT]
            elif ring is not None:
                pitch, duration = _OPEN_HIHAT_NOTE, min(ring, _OPEN_HIHAT_MAX_RING_SEC)
            else:
                pitch, duration = _NOTE_FOR_CLASS[class_index], _NOTE_DURATION_SEC[class_index]
            if class_index == _SNARE and votes.get(time_sec) == percussion_splitter.PERCUSSION:
                pitch = _PERCUSSION_NOTE
            notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=time_sec, end=time_sec + duration))

    notes.sort(key=lambda note: (note.start, note.pitch))
    return notes

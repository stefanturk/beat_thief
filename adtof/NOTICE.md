# ADTOF, vendored

`model.py`, `audio.py` and `weights.pth` in this folder are not Beat Thief's
work. They are the drum transcription model that `drum_transcriber.py` runs.

## Where they came from

**ADTOF-pytorch** — <https://github.com/xavriley/ADTOF-pytorch>, by Xavier
Riley. A PyTorch port of the original Keras/TensorFlow implementation, which
also drops its madmom dependency. `weights.pth` here is that repo's
`src/adtof_pytorch/data/adtof_frame_rnn_pytorch_weights.pth`, converted from
the officially released Keras weights.

**ADTOF** — <https://github.com/MZehren/ADTOF>, by Mickaël Zehren, Marco
Alunno and Paolo Bientinesi. The dataset, the model and the trained weights.

- M. Zehren, M. Alunno, and P. Bientinesi, "ADTOF: A large dataset of
  non-synthetic music for automatic drum transcription," *Proceedings of the
  22nd International Society for Music Information Retrieval Conference*,
  2021, pp. 818–824. <https://arxiv.org/abs/2111.11737>
- M. Zehren, M. Alunno, and P. Bientinesi, "High-Quality and Reproducible
  Automatic Drum Transcription from Crowdsourced Data," *Signals* 4(4), 2023,
  pp. 768–787. <https://doi.org/10.3390/signals4040042>

## Licence

ADTOF is released under **Creative Commons Attribution-NonCommercial-ShareAlike
4.0 International** (CC BY-NC-SA 4.0). <https://creativecommons.org/licenses/by-nc-sa/4.0/>

The NonCommercial term is the one to know about: it doesn't affect using Beat
Thief, it affects selling it.

## What was changed

`audio.py` is byte-for-byte upstream. `model.py` has two edits, both in
`load_pytorch_weights` and both commented in place:

1. `torch.load(...)` is called with `weights_only=True`. The upstream call
   unpickles arbitrary objects, which torch 2.5 warns about.
2. Missing weights now raise instead of printing a warning and continuing with
   randomly initialized layers. A model that loads, runs and returns confident
   nonsense is the worst way for this to fail.

`post_processing.py` and `__init__.py` from upstream are deliberately **not**
here. Upstream's `post_processing.py` doesn't import on Python 3.9 (it
annotates a function signature with `Sequence[float] | float` without
`from __future__ import annotations`, so the annotation is evaluated at import
time), and it is also the part Beat Thief replaces: it writes a flat velocity
of 100 on every note, a 120 BPM tempo header on every song, and a kick on note
35, which sits below the first pad of Ableton's default Drum Rack. Beat Thief's
own peak picking, velocity, note map and MIDI writing are in
`drum_transcriber.py`.

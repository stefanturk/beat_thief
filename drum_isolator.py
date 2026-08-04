#!/usr/bin/env python3
"""Isolate drum stems from downloaded MP3s: a full drums.wav, then a first
pass at splitting it into individual kit pieces (kick/snare/toms), with
cymbals and hi-hat still bundled together as one file for now — the free
local model this uses doesn't separate those two yet."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

DRUMS_DIR_NAME = "Drums"
DEFAULT_OUTPUT = os.path.join(os.path.expanduser("~"), "Downloads", "Song Downloads")

_HTDEMUCS_MODEL = "htdemucs"

# drumsep (github.com/inagoy/drumsep) is just the standard `demucs` CLI
# pointed at a custom Hybrid-Demucs checkpoint. Its install script downloads
# that checkpoint from Google Drive into a local "model" repo directory,
# which we mirror here rather than any separate inference code.
_DRUMSEP_MODEL_NAME = "49469ca8"
_DRUMSEP_GDRIVE_FILE_ID = "1-Dm666ScPkg8Gt2-lK3Ua0xOudWHZBGC"
_DRUMSEP_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drumsep_model")

_STEM_NAMES = ("drums.wav", "kick.wav", "snare.wav", "toms.wav", "cymbals_hihat.wav")


def _map_drumsep_stem_name(filename: str) -> str | None:
    """The drumsep checkpoint's own stem filenames aren't documented in
    English (its README describes them as Bombo/Redoblante/Platillos/Toms),
    so match loosely on either language instead of assuming one exact name."""
    lowered = filename.lower()
    if "kick" in lowered or "bombo" in lowered:
        return "kick"
    if "snare" in lowered or "redoblante" in lowered or "caja" in lowered:
        return "snare"
    if "tom" in lowered:
        return "toms"
    if "cymbal" in lowered or "platillo" in lowered or "hihat" in lowered or "hi-hat" in lowered or "hh" in lowered:
        return "cymbals_hihat"
    return None


def _ensure_drumsep_model() -> None:
    if os.path.isdir(_DRUMSEP_MODEL_DIR) and os.listdir(_DRUMSEP_MODEL_DIR):
        return
    print("  Downloading the kit-piece separation model (one-time)...")
    os.makedirs(_DRUMSEP_MODEL_DIR, exist_ok=True)
    try:
        subprocess.run(
            [sys.executable, "-m", "gdown", _DRUMSEP_GDRIVE_FILE_ID, "-O", _DRUMSEP_MODEL_DIR + os.sep],
            check=True,
        )
    except Exception:
        shutil.rmtree(_DRUMSEP_MODEL_DIR, ignore_errors=True)
        raise


def _run_demucs(input_path: str, out_dir: str, model_name: str, repo: str | None = None, two_stems: str | None = None) -> str:
    """Run demucs and return the directory containing the separated stems."""
    cmd = [sys.executable, "-m", "demucs", "-n", model_name, "-o", out_dir]
    if repo:
        cmd += ["--repo", repo]
    if two_stems:
        cmd += ["--two-stems", two_stems]
    cmd.append(input_path)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    track_name = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(out_dir, model_name, track_name)


def isolate_drums(mp3_path: str, drums_root: str) -> bool:
    """Produce drums.wav, kick.wav, snare.wav, toms.wav, and cymbals_hihat.wav
    for a single song under drums_root/<title>/. Returns False (skipped)
    if all of those already exist."""
    title = os.path.splitext(os.path.basename(mp3_path))[0]
    song_dir = os.path.join(drums_root, title)

    if os.path.isdir(song_dir) and all(os.path.exists(os.path.join(song_dir, f)) for f in _STEM_NAMES):
        print(f"{title}: drum stems already exist, nothing to do.")
        return False

    print(f"{title}: isolating drums (this can take a few minutes)...")
    tmp_dir = tempfile.mkdtemp()
    try:
        drums_stem_dir = _run_demucs(mp3_path, tmp_dir, _HTDEMUCS_MODEL, two_stems="drums")
        drums_wav = os.path.join(drums_stem_dir, "drums.wav")

        os.makedirs(song_dir, exist_ok=True)
        shutil.copy(drums_wav, os.path.join(song_dir, "drums.wav"))

        print(f"{title}: splitting into kick/snare/toms/cymbals...")
        _ensure_drumsep_model()
        kit_stem_dir = _run_demucs(drums_wav, tmp_dir, _DRUMSEP_MODEL_NAME, repo=_DRUMSEP_MODEL_DIR)

        for filename in os.listdir(kit_stem_dir):
            our_name = _map_drumsep_stem_name(filename)
            if our_name:
                shutil.copy(os.path.join(kit_stem_dir, filename), os.path.join(song_dir, our_name + ".wav"))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"{title}: drum stems saved to {song_dir}")
    return True


def isolate_drums_for_folder(output_dir: str) -> None:
    mp3_files = sorted(f for f in os.listdir(output_dir) if f.lower().endswith(".mp3"))
    if not mp3_files:
        print("No MP3s found to isolate drums from.")
        return

    drums_root = os.path.join(output_dir, DRUMS_DIR_NAME)
    for filename in mp3_files:
        path = os.path.join(output_dir, filename)
        try:
            isolate_drums(path, drums_root)
        except Exception as e:
            print(f"  Could not isolate drums for {filename}, skipping: {e}")


def isolate_drums_for_single_file(path: str) -> None:
    output_dir = os.path.dirname(os.path.abspath(path)) or "."
    drums_root = os.path.join(output_dir, DRUMS_DIR_NAME)
    try:
        isolate_drums(path, drums_root)
    except Exception as e:
        print(f"  Could not isolate drums for {os.path.basename(path)}, skipping: {e}")


def isolate_drums_for_path(path: str) -> None:
    if os.path.isfile(path):
        isolate_drums_for_single_file(path)
    else:
        isolate_drums_for_folder(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isolate drum stems (kick/snare/toms/cymbals) from downloaded MP3s."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help=f"Folder of MP3s (or a single MP3 file) to isolate drums from (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    try:
        isolate_drums_for_path(args.path)
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(130)


if __name__ == "__main__":
    main()

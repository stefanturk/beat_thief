#!/bin/bash
# Set up Beat Thief on a Mac that has never run it, and build the app.
#
#   Double-click this file, or run ./setup.sh in Terminal.
#   ./setup.sh --songs-only      skip the question, no instrument splitter
#   ./setup.sh --with-splitter   skip the question, install it
#
# Safe to run again: every step checks whether it's already been done, so a
# second run after a failure picks up where the first one stopped rather
# than redoing the downloading. Running it again is also how the instrument
# splitter is added to a songs-only install.
#
# What this installs, and why "Beat Thief.app" on its own isn't enough:
# the bundle is a small launcher. Python, yt-dlp, the window and ffmpeg
# (which makes the mp3s) all live in ~/Library/Application Support/Beat
# Thief, which this fills. None of it travels inside the app, so copying
# the app to another Mac gets you an icon that says it couldn't start.
#
# Why its own Python and ffmpeg rather than the Mac's or Homebrew's:
#   - Homebrew stopped supporting Intel Macs in September 2026, and it was
#     only ever here for ffmpeg. A ready-made ffmpeg download does the same
#     job on every Mac, with no password.
#   - The Mac's own /usr/bin/python3 is 3.9 at best (3.8 before macOS 12),
#     and yt-dlp - which has to keep up with YouTube - has left both behind.
#     A private Python 3.12 can always take the latest yt-dlp.
# Both are pinned by version and checked against a known SHA-256, so what
# runs is exactly what was tested.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Not inside the repo: the repo can live on the Desktop, and macOS blocks
# an app from reading the Desktop (see make_app.sh). Application Support
# isn't protected, and it's per-person rather than per-copy-of-the-code.
RUNTIME="$HOME/Library/Application Support/Beat Thief"
PY="$RUNTIME/python/bin/python3"
BIN="$RUNTIME/bin"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
note() { printf '   %s\n' "$*"; }

splitter=""
for arg in "$@"; do
    case "$arg" in
        --songs-only) splitter=no ;;
        --with-splitter) splitter=yes ;;
    esac
done

# Anything that arrived over Google Drive, AirDrop or a download is tagged
# by macOS as quarantined, and a quarantined script or app is refused with
# "unidentified developer" rather than run. The tag applies to the whole
# folder that came across, so clear it once here - on the copy the user
# themselves chose to open, which is the point at which the warning has
# served its purpose.
xattr -dr com.apple.quarantine "$REPO" 2>/dev/null || true

say "Beat Thief setup"
note "This takes a few minutes, most of it downloading. No password needed."

# ------------------------------------------------------------- which Mac
# Asked of the hardware, not `uname -m`: a Terminal running under Rosetta
# says x86_64 on an Apple Silicon Mac, and would fetch the Intel builds.
if [ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" = "1" ]; then
    MACHINE=arm64
    NEEDS_MACOS=12      # the Apple Silicon ffmpeg build's minimum
else
    MACHINE=x86_64
    NEEDS_MACOS=10.15   # the Intel Python build's minimum
fi
MACOS="$(sw_vers -productVersion)"
older() {   # older A B: is version A below version B?
    [ "$(printf '%s\n%s\n' "$1" "$2" | sort -t. -k1,1n -k2,2n -k3,3n | head -n1)" = "$1" ] && [ "$1" != "$2" ]
}
if older "$MACOS" "$NEEDS_MACOS"; then
    say "This Mac's macOS ($MACOS) is too old."
    note "Beat Thief needs macOS $NEEDS_MACOS or later on this Mac."
    exit 1
fi
note "macOS $MACOS on $([ "$MACHINE" = arm64 ] && echo "Apple Silicon" || echo "Intel") - fine."

# Downloads url to path and refuses it unless it's the exact file expected.
fetch() {
    local url="$1" sha="$2" path="$3"
    curl -fL --retry 3 --progress-bar -o "$path" "$url"
    if [ "$(shasum -a 256 "$path" | cut -d' ' -f1)" != "$sha" ]; then
        rm -f "$path"
        note "That download wasn't the file it should have been."
        note "Run setup again; if it keeps happening, tell Stefan."
        exit 1
    fi
}

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$RUNTIME" "$BIN"

# ---------------------------------------------------------------- python
say "1. Python"
PBS="https://github.com/astral-sh/python-build-standalone/releases/download/20261003"
if [ "$MACHINE" = arm64 ]; then
    PY_FILE="cpython-3.12.15+20261003-aarch64-apple-darwin-install_only.tar.gz"
    PY_SHA="316a463172740e71d8dca1f2730784e325f3f720941137b5d674d5801a632213"
else
    PY_FILE="cpython-3.12.15+20261003-x86_64-apple-darwin-install_only.tar.gz"
    PY_SHA="a8fd7a91852f19b6d959793ef41fad048631ccb2a334a9ecdf573255298f7978"
fi
if "$PY" -c 'import sys; assert sys.version_info >= (3, 12)' >/dev/null 2>&1; then
    note "$("$PY" --version) - already here."
else
    note "Downloading Python 3.12 (25MB)..."
    fetch "$PBS/${PY_FILE//+/%2B}" "$PY_SHA" "$WORK/python.tar.gz"
    rm -rf "$RUNTIME/python"
    # The archive holds a single python/ folder.
    tar -xzf "$WORK/python.tar.gz" -C "$RUNTIME"
    note "$("$PY" --version) - ok."
fi

# ---------------------------------------------------------------- ffmpeg
# Without ffmpeg yt-dlp downloads the video and then can't turn it into an
# mp3, which fails at the very end of a long download with nothing useful
# to say. ffprobe comes too: pydub reads every song's format with it.
say "2. ffmpeg (the mp3 converter)"
if [ "$MACHINE" = arm64 ]; then
    # Martin Riedl's builds: signed, and native on Apple Silicon.
    FF_URL="https://ffmpeg.martin-riedl.de/download/macos/arm64/1789931890_9.0.2"
    FFMPEG_SHA="c8ed4c4e6978a03c485edbfe4e0a5dc2380f8a30bba5150531b31b094492d924"
    FFPROBE_SHA="fcbe839537485eaee7a7a8bc5cbc0f90d53617e80943e8a5b2e31cb851197ea6"
    ff_zip() { echo "$FF_URL/$1.zip"; }
else
    # evermeet.cx's: the Intel build that still runs on macOS 10.13 up
    # (Riedl's Intel build wants macOS 12).
    FFMPEG_SHA="4acc0be580f9b2788029eb7bd4d645ff87968911b0a62aeeb3940d42d54558d5"
    FFPROBE_SHA="24a9c968cd4da72d99c7245e914b921815835eb6dff01d99868031aebaf1d439"
    ff_zip() { echo "https://evermeet.cx/ffmpeg/$1-9.0.2.zip"; }
fi
for tool in ffmpeg ffprobe; do
    if "$BIN/$tool" -version >/dev/null 2>&1; then
        note "$tool - already here."
        continue
    fi
    if command -v "$tool" >/dev/null 2>&1; then
        note "$tool - using the one at $(command -v "$tool")."
        continue
    fi
    note "Downloading $tool..."
    sha_var="$(echo "$tool" | tr a-z A-Z)_SHA"
    fetch "$(ff_zip "$tool")" "${!sha_var}" "$WORK/$tool.zip"
    unzip -oq "$WORK/$tool.zip" -d "$WORK/$tool"
    mv -f "$WORK/$tool/$tool" "$BIN/$tool"
    chmod +x "$BIN/$tool"
    # A binary macOS won't run unsigned (any arm64 one) gets the same ad-hoc
    # signature every locally built program has.
    "$BIN/$tool" -version >/dev/null 2>&1 || codesign --force -s - "$BIN/$tool" 2>/dev/null || true
    if ! "$BIN/$tool" -version >/dev/null 2>&1; then
        note "$tool won't start on this Mac. Send this to Stefan."
        exit 1
    fi
    note "$tool - ok."
done

# ------------------------------------------------------------- packages
say "3. Installing the Python packages"
"$PY" -m pip install --quiet --upgrade pip
# --prefer-binary: the newest release of a package sometimes ships without a
# ready-made Intel Mac build (llvmlite, under librosa, did in 2026), and pip
# would then try to compile it - which needs tools nobody has, and fails.
# This takes the newest version that does have one.
"$PY" -m pip install --prefer-binary --upgrade -r "$REPO/requirements.txt"

# The instrument splitter (torch and demucs) is most of the download and
# only matters for stems, beat loops and MIDI. Somebody who only wants the
# songs shouldn't wait for 2GB they'll never use - so it's asked, unless
# it's already here (keep it) or the command line said.
if [ -z "$splitter" ] && "$PY" -c "import torch, demucs" >/dev/null 2>&1; then
    note "The instrument splitter is already installed - keeping it."
    splitter=yes
fi
if [ -z "$splitter" ]; then
    if [ -t 0 ]; then
        say "Also install the instrument splitter?"
        note "It pulls songs apart into drums, bass, harmony and vocals, and"
        note "makes beat loops. It's another 2GB. Without it, Beat Thief still"
        note "downloads, tidies and names songs with their tempo. You can add"
        note "it later by running this setup again."
        printf '   Install it? [y/N] '
        read -r answer
        case "$answer" in [Yy]*) splitter=yes ;; *) splitter=no ;; esac
    else
        splitter=no
    fi
fi
if [ "$splitter" = yes ]; then
    say "3b. Installing the instrument splitter"
    note "torch is about 2GB, so this is the long part."
    "$PY" -m pip install --prefer-binary -r "$REPO/requirements-splitter.txt"
fi

# The window, the downloader and the rest, each imported on its own so a
# failure names the piece that's missing rather than "something didn't
# install".
say "4. Checking they work"
modules="webview yt_dlp pydub librosa"
[ "$splitter" = yes ] && modules="$modules torch demucs"
for module in $modules; do
    if "$PY" -c "import $module" >/dev/null 2>&1; then
        note "$module - ok"
    else
        note "$module - FAILED to import."
        note "Send the output above to Stefan; the app won't run without it."
        exit 1
    fi
done

# --------------------------------------------------------------- the app
say "5. Building the app"
PYTHON="$PY" "$REPO/make_app.sh"

say "Done."
note "Beat Thief is in your Applications folder (the one in your home"
note "folder, which Finder shows under Go > Home > Applications)."
note "Open it once from there, then drag it to your Dock."
note
note "Songs land in ~/Music/Beat Thief."
note "If it ever won't start, the reason is in ~/Library/Logs/beat_thief.log."
note
note "Keep this folder. The app is a snapshot of the code in it, so when"
note "there's a new version, run ./update.sh here and open the app again."

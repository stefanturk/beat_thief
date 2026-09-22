#!/bin/bash
# Set up Beat Thief on a Mac that has never run it, and build the app.
#
#   Double-click this file, or run ./setup.sh in Terminal.
#
# Safe to run again: every step checks whether it's already been done, so a
# second run after a failure picks up where the first one stopped rather
# than redoing the hour of downloading.
#
# What this installs, and why "Beat Thief.app" on its own isn't enough:
# the bundle is a 5MB launcher. The heavy parts - yt-dlp for downloading,
# demucs and torch for pulling instruments apart, pywebview for the window -
# live in your Python's packages folder, and ffmpeg (which makes the mp3s)
# is a separate program entirely. None of that travels inside the app, so
# copying the app to another Mac gets you an icon that opens a dialog
# saying it couldn't start. This puts the missing pieces in place first,
# then builds the app here where it can find them.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
note() { printf '   %s\n' "$*"; }

# Anything that arrived over Google Drive, AirDrop or a download is tagged
# by macOS as quarantined, and a quarantined script or app is refused with
# "unidentified developer" rather than run. The tag applies to the whole
# folder that came across, so clear it once here - on the copy the user
# themselves chose to open, which is the point at which the warning has
# served its purpose.
xattr -dr com.apple.quarantine "$REPO" 2>/dev/null || true

say "Beat Thief setup"
note "This takes about ten minutes, most of it downloading."
note "It will ask for your Mac password if Homebrew needs installing."

# ---------------------------------------------------------------- python3
# /usr/bin/python3 exists on a fresh Mac as a stub that does nothing but
# offer to install the Xcode command line tools. Asking it for its version
# is what tells the two apart: the stub can't answer.
say "1. Checking Python"
if ! /usr/bin/python3 --version >/dev/null 2>&1; then
    note "macOS needs its command line tools before Python works."
    note "A window will open now - click Install, wait for it to finish,"
    note "then run this setup again."
    xcode-select --install 2>/dev/null || true
    exit 1
fi
note "$(/usr/bin/python3 --version) - fine."

# ---------------------------------------------------------------- ffmpeg
# Without ffmpeg yt-dlp downloads the video and then can't turn it into an
# mp3, which fails at the very end of a long download with nothing useful
# to say. So it's checked for before anything slow happens.
say "2. Checking ffmpeg (the mp3 converter)"
if command -v ffmpeg >/dev/null 2>&1; then
    note "Already installed at $(command -v ffmpeg)."
else
    if ! command -v brew >/dev/null 2>&1; then
        # Homebrew's installer needs an admin password and a keypress, so
        # it is never run without saying so first.
        note "ffmpeg is installed with Homebrew, which isn't here yet."
        note "Installing Homebrew asks for your Mac password."
        printf '   Install it now? [y/N] '
        read -r answer
        case "$answer" in
            [Yy]*) ;;
            *)
                note "Stopped. Install Homebrew yourself from https://brew.sh"
                note "then run this setup again."
                exit 1
                ;;
        esac
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    # A just-installed Homebrew isn't on this shell's PATH yet - its
    # installer only writes the line into ~/.zprofile, which the next
    # terminal reads and this one never will.
    for brew_path in /opt/homebrew/bin/brew /usr/local/bin/brew; do
        [ -x "$brew_path" ] && eval "$("$brew_path" shellenv)" && break
    done
    brew install ffmpeg
fi

# ------------------------------------------------------------- packages
say "3. Installing the Python packages"
note "torch is about 2GB, so this is the long part."
/usr/bin/python3 -m pip install --user --upgrade pip >/dev/null
/usr/bin/python3 -m pip install --user -r "$REPO/requirements.txt"

# The window, the downloader and the instrument separator, each imported
# on its own so a failure names the piece that's missing rather than
# "something didn't install".
say "4. Checking they work"
for module in webview yt_dlp torch demucs; do
    if /usr/bin/python3 -c "import $module" >/dev/null 2>&1; then
        note "$module - ok"
    else
        note "$module - FAILED to import."
        note "Send the output above to Stefan; the app won't run without it."
        exit 1
    fi
done

# --------------------------------------------------------------- the app
say "5. Building the app"
# Pinned to the same Python the packages just went into: installing ffmpeg
# can bring a Homebrew python3 along with it, and that one would win
# `command -v` while having none of them.
PYTHON=/usr/bin/python3 "$REPO/make_app.sh"

say "Done."
note "Beat Thief is in your Applications folder (the one in your home"
note "folder, which Finder shows under Go > Home > Applications)."
note "Open it once from there, then drag it to your Dock."
note
note "Songs land in ~/Music/Beat Thief."
note "If it ever won't start, the reason is in ~/Library/Logs/beat_thief.log."
note
note "Keep this folder. The app is a snapshot of the code in it, so a newer"
note "version means replacing this folder and running setup.sh again."

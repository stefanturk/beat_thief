#!/bin/bash
# Get the latest Beat Thief and rebuild the app.
#
#   Double-click this file, or run ./update.sh in Terminal.
#
# Takes a few seconds. Nothing is downloaded again except the code itself -
# the Python packages and ffmpeg that setup.sh installed stay where they are,
# which is the whole reason this exists instead of a new zip every time.
#
# Run setup.sh, not this, on a Mac that has never had Beat Thief on it.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
note() { printf '   %s\n' "$*"; }

if [ ! -d "$REPO/.git" ]; then
    say "This copy didn't come from git."
    note "It was unzipped, so there's nothing to pull from. To get updates"
    note "this way, clone it instead:"
    note
    note "  git clone https://github.com/stefanturk/beat_thief.git"
    note
    note "then run setup.sh in the folder that makes."
    exit 1
fi

say "1. Getting the latest code"
# --ff-only rather than a plain pull: if this copy has been edited, a merge
# would either stop halfway with a conflict or quietly commit something
# nobody meant to write. Better to say so and leave the working code alone.
if ! git pull --ff-only; then
    say "Couldn't update."
    note "This folder has changes of its own, so there's nothing to fast-"
    note "forward onto. The app you already have still works. If you didn't"
    note "mean to change anything, 'git status' says what's different."
    exit 1
fi

say "2. Rebuilding the app"
# Same Python setup.sh installed the packages into - see the note there.
PYTHON=/usr/bin/python3 "$REPO/make_app.sh"

say "Up to date."
note "Beat Thief is in your Applications folder, same as before. If it was"
note "open, quit it and open it again to get the new version."

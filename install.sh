#!/bin/bash
# The one line that sets up Beat Thief on a Mac that has never seen it:
#
#   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/stefanturk/beat_thief/main/install.sh)"
#
# Run the same line again any time to update - it pulls and rebuilds, which
# takes seconds, because the slow part (torch, ffmpeg) is already installed.
#
# Why this exists on top of setup.sh: setup.sh lives inside the repo, so
# something has to fetch the repo first, and on a Mac that has never built
# anything `git` isn't installed either. This is the part that can run
# before any of that is true.

set -euo pipefail

REPO_URL="https://github.com/stefanturk/beat_thief.git"
DEST="$HOME/beat_thief"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
note() { printf '   %s\n' "$*"; }

say "Beat Thief"

# git ships with the Xcode command line tools and nowhere else. Asking for
# its version on a Mac without them pops Apple's installer and fails, which
# is why this is checked before anything is cloned: the install is a GUI
# thing that has to finish on its own before we can carry on.
if ! git --version >/dev/null 2>&1; then
    note "macOS needs its command line tools first (this is where git comes"
    note "from). A window is opening now - click Install, wait for it to"
    note "finish, then run this same line again."
    xcode-select --install 2>/dev/null || true
    exit 1
fi

if [ -d "$DEST/.git" ]; then
    say "Already here - updating instead"
    exec "$DEST/update.sh"
fi

if [ -e "$DEST" ]; then
    note "There's already something at $DEST that isn't a Beat Thief clone."
    note "Move it out of the way and run this again."
    exit 1
fi

say "Getting Beat Thief"
git clone --depth 1 "$REPO_URL" "$DEST"

# From here on setup.sh is in charge: Python, ffmpeg, the packages, the app.
exec "$DEST/setup.sh"

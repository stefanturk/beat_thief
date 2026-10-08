"""Guards on setup.sh, the script a new Mac runs once.

Nobody here runs it - it downloads a Python, ffmpeg and (if asked) two
gigabytes of torch. What can be checked cheaply is that it doesn't drift
away from the things it depends on: the requirements files it installs, and
the build script it finishes with. Both have moved before while something
pointing at them didn't.
"""

import hashlib
import os
import re
import unittest

REPO = os.path.dirname(os.path.abspath(__file__))
SETUP = os.path.join(REPO, "setup.sh")
UPDATE = os.path.join(REPO, "update.sh")
INSTALL = os.path.join(REPO, "install.sh")


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


def _setup() -> str:
    return _read(SETUP)


REQUIREMENTS = ("requirements.txt", "requirements-splitter.txt")


def _required_packages() -> set[str]:
    """The distribution names in both requirements files, version pins and
    extras dropped."""
    names = set()
    for name in REQUIREMENTS:
        with open(os.path.join(REPO, name)) as f:
            lines = [line.strip() for line in f
                     if line.strip() and not line.strip().startswith("#")]
        names |= {re.split(r"[<>=!\[;]", line)[0].strip() for line in lines}
    return names


# What the import name in setup.sh's check is called on PyPI, for the ones
# where those differ.
DISTRIBUTION_OF = {"webview": "pywebview", "yt_dlp": "yt-dlp"}


class TestSetupInstallsWhatItChecksFor(unittest.TestCase):
    def test_every_module_it_verifies_is_one_it_installs(self):
        # The check loop exists to name the piece that failed to install. A
        # module that isn't in requirements.txt at all would fail that check
        # on every machine, for good, and read as a broken install.
        modules = " ".join(re.findall(r'modules="(?:\$modules )?([^"]+)"', _setup())).split()
        self.assertIn("yt_dlp", modules)
        self.assertIn("torch", modules)
        installed = _required_packages()

        missing = {m for m in modules
                   if DISTRIBUTION_OF.get(m, m) not in installed}

        self.assertEqual(missing, set(),
                         "setup.sh checks for these but never installs them")

    def test_it_installs_from_both_requirements_files(self):
        for name in REQUIREMENTS:
            with self.subTest(name=name):
                self.assertIn(f'-r "$REPO/{name}"', _setup())

    def test_the_splitter_is_not_in_the_songs_only_list(self):
        # The whole point of asking: torch and demucs are the 2GB.
        with open(os.path.join(REPO, "requirements.txt")) as f:
            songs = f.read()
        for heavy in ("torch", "demucs"):
            self.assertNotIn(heavy, songs)


class TestSetupNeedsNoHomebrew(unittest.TestCase):
    """Homebrew stopped supporting Intel Macs, and on one it stopped the
    whole install at ffmpeg. Setup fetches its own instead."""

    def test_it_never_installs_homebrew_or_anything_with_it(self):
        self.assertNotIn("brew install", _setup())
        self.assertNotIn("Homebrew/install", _setup())

    def test_every_download_is_checked_against_a_known_sha(self):
        # A fetch without a hash would run whatever the server sent.
        calls = re.findall(r"^\s*fetch (.+)$", _setup(), re.M)
        self.assertGreaterEqual(len(calls), 2)
        for call in calls:
            with self.subTest(call=call):
                self.assertRegex(call, r"(_SHA|sha_var)", "fetch takes url, sha, path")

    def test_it_picks_downloads_by_the_hardware_not_the_terminal(self):
        # A Rosetta Terminal says x86_64 on Apple Silicon.
        self.assertIn("hw.optional.arm64", _setup())
        for build in ("aarch64-apple-darwin", "x86_64-apple-darwin"):
            self.assertIn(build, _setup())


RUNTIME_LINE = 'RUNTIME="$HOME/Library/Application Support/Beat Thief"'


class TestSetupAndTheBuildAgreeOnPython(unittest.TestCase):
    """setup.sh puts the packages in its own Python and builds the app
    against that same one by name. If the scripts disagreed about where it
    is, the app would be built against whichever python3 happened to be on
    the PATH - which has none of the packages, and fails at launch rather
    than here."""

    def test_all_three_agree_where_the_runtime_lives(self):
        for script in (SETUP, UPDATE, os.path.join(REPO, "make_app.sh")):
            with self.subTest(script=os.path.basename(script)):
                self.assertIn(RUNTIME_LINE, _read(script))

    def test_setup_names_the_python_it_builds_against(self):
        self.assertIn('PY="$RUNTIME/python/bin/python3"', _setup())
        self.assertIn('PYTHON="$PY" "$REPO/make_app.sh"', _setup())

    def test_make_app_lets_it_be_overridden(self):
        with open(os.path.join(REPO, "make_app.sh")) as f:
            self.assertIn('PYTHON="${PYTHON:-', f.read())


class TestSetupIsRunnable(unittest.TestCase):
    def test_all_three_scripts_are_executable(self):
        # Double-clicking a script that isn't executable opens it in a text
        # editor, which is a confusing first thing to happen. install.sh is
        # fetched over https and exec'd, where a lost bit is worse still: it
        # runs as somebody's one-line introduction to the whole thing.
        for script in (SETUP, UPDATE, INSTALL):
            with self.subTest(script=os.path.basename(script)):
                self.assertTrue(os.access(script, os.X_OK))

    def test_they_all_stop_on_the_first_failure(self):
        # Without this, a failed pip install would be followed by a build
        # and a cheerful "Done."
        for script in (SETUP, UPDATE, INSTALL):
            with self.subTest(script=os.path.basename(script)):
                self.assertIn("set -euo pipefail", _read(script))


class TestTheOneLineInstallerPointsSomewhereReal(unittest.TestCase):
    """install.sh is the only file anyone is asked to run by URL, and it is
    fetched from this repo's own main branch. So its idea of where the repo
    is has to match where the repo actually is - a stale URL here is a
    clone that 404s in front of somebody who has done nothing wrong yet."""

    def _origin(self) -> str:
        import subprocess
        return subprocess.run(
            ["git", "-C", REPO, "remote", "get-url", "origin"],
            capture_output=True, text=True, check=True).stdout.strip()

    def test_it_clones_the_repo_it_ships_in(self):
        url = re.search(r'REPO_URL="([^"]+)"', _read(INSTALL)).group(1)

        self.assertEqual(url.removesuffix(".git"),
                         self._origin().removesuffix(".git"))

    def test_the_line_in_its_comment_fetches_itself(self):
        # The curl line is documentation that is also the product. If the
        # path in it drifts from the filename, the instructions Stefan
        # copies out of here stop working.
        install = _read(INSTALL)
        curl = re.search(r"https://raw\.githubusercontent\.com/[^\s\"')]+",
                         install).group(0)

        self.assertTrue(curl.endswith("/main/install.sh"), curl)
        owner_repo = re.search(
            r"github\.com/([^/]+/[^/.]+)", self._origin()).group(1)
        self.assertIn(owner_repo, curl)

    def test_it_hands_over_to_the_scripts_in_the_clone(self):
        # install.sh deliberately does nothing itself beyond getting the
        # repo down; both halves of the work live in the repo where they
        # can be tested. If either handover is renamed away, the one-liner
        # ends after the clone with nothing built.
        install = _read(INSTALL)

        self.assertIn('exec "$DEST/setup.sh"', install)
        self.assertIn('exec "$DEST/update.sh"', install)


class TestUpdateRebuildsTheSameWaySetupDid(unittest.TestCase):
    def test_it_builds_against_the_python_setup_installed_into(self):
        # Same trap as setup.sh's: any other python3 on the PATH would win
        # `command -v` and have none of the packages.
        self.assertIn('PYTHON="$PY" "$REPO/make_app.sh"', _read(UPDATE))

    def test_it_takes_the_newest_yt_dlp_every_time(self):
        self.assertIn('pip install --quiet --prefer-binary --upgrade -r "$REPO/requirements.txt"', _read(UPDATE))

    def test_the_part_an_old_copy_is_still_running_never_changes(self):
        # update.sh rewrites itself with `git pull` while bash is reading it,
        # and bash carries on from the same byte offset in the new file. If
        # anything above the marker moves, every install that updates from
        # here on starts executing mid-line.
        script = _read(UPDATE)
        prefix = script[:script.index("# Everything above this line")]
        self.assertEqual(hashlib.sha256(prefix.encode()).hexdigest(),
                         "96e41b02f5b24e56e2d6d1f7b1efddfa023299447549fe42c4db841714989c1b")

    def test_it_refuses_to_merge(self):
        # A plain `git pull` on a copy somebody has edited either stops in a
        # conflict or writes a merge commit - both in front of a person who
        # only wanted the new version. --ff-only turns that into a sentence.
        self.assertIn("git pull --ff-only", _read(UPDATE))


if __name__ == "__main__":
    unittest.main()

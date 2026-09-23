"""Guards on setup.sh, the script a new Mac runs once.

Nobody here runs it - it installs Homebrew and two gigabytes of torch. What
can be checked cheaply is that it doesn't drift away from the things it
depends on: the requirements file it installs, and the build script it
finishes with. Both have moved before while something pointing at them
didn't.
"""

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


def _required_packages() -> set[str]:
    """The distribution names in requirements.txt, version pins dropped."""
    with open(os.path.join(REPO, "requirements.txt")) as f:
        lines = [line.strip() for line in f if line.strip()]
    return {re.split(r"[<>=!]", line)[0].strip() for line in lines}


# What the import name in setup.sh's check is called on PyPI, for the ones
# where those differ.
DISTRIBUTION_OF = {"webview": "pywebview", "yt_dlp": "yt-dlp"}


class TestSetupInstallsWhatItChecksFor(unittest.TestCase):
    def test_every_module_it_verifies_is_one_it_installs(self):
        # The check loop exists to name the piece that failed to install. A
        # module that isn't in requirements.txt at all would fail that check
        # on every machine, for good, and read as a broken install.
        modules = re.search(r"for module in (.+?); do", _setup()).group(1).split()
        installed = _required_packages()

        missing = {m for m in modules
                   if DISTRIBUTION_OF.get(m, m) not in installed}

        self.assertEqual(missing, set(),
                         "setup.sh checks for these but never installs them")

    def test_it_installs_from_the_requirements_file(self):
        self.assertIn("requirements.txt", _setup())


class TestSetupAndTheBuildAgreeOnPython(unittest.TestCase):
    """setup.sh puts the packages in /usr/bin/python3 and builds the app
    against that same one by name. If make_app.sh stopped honouring the
    override, the app would be built against whichever python3 Homebrew
    happened to leave on the PATH - which has none of the packages, and
    fails at launch rather than here."""

    def test_setup_names_the_python_it_builds_against(self):
        self.assertIn("PYTHON=/usr/bin/python3", _setup())

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
        # Same trap as setup.sh's: a Homebrew python3 that arrived with
        # ffmpeg would win `command -v` and have none of the packages.
        self.assertIn("PYTHON=/usr/bin/python3", _read(UPDATE))

    def test_it_refuses_to_merge(self):
        # A plain `git pull` on a copy somebody has edited either stops in a
        # conflict or writes a merge commit - both in front of a person who
        # only wanted the new version. --ff-only turns that into a sentence.
        self.assertIn("git pull --ff-only", _read(UPDATE))


if __name__ == "__main__":
    unittest.main()

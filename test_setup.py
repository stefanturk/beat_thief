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


def _setup() -> str:
    with open(SETUP) as f:
        return f.read()


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
    def test_it_is_executable(self):
        # Double-clicking a script that isn't executable opens it in a text
        # editor, which is a confusing first thing to happen.
        self.assertTrue(os.access(SETUP, os.X_OK))

    def test_it_stops_on_the_first_failure(self):
        # Without this, a failed pip install would be followed by a build
        # and a cheerful "Done."
        self.assertIn("set -euo pipefail", _setup())


if __name__ == "__main__":
    unittest.main()

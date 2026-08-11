import ast
import os
import re
import unittest

REPO = os.path.dirname(os.path.abspath(__file__))


def _sources_in_make_app() -> set[str]:
    """The SOURCES=( ... ) list make_app.sh copies into the bundle."""
    with open(os.path.join(REPO, "make_app.sh")) as f:
        script = f.read()
    body = re.search(r"^SOURCES=\((.*?)^\)", script, re.S | re.M).group(1)
    return {line.strip() for line in body.splitlines() if line.strip()}


def _local_modules() -> set[str]:
    return {name[:-3] for name in os.listdir(REPO)
            if name.endswith(".py") and not name.startswith("test_")}


def _imports_of(module: str) -> set[str]:
    with open(os.path.join(REPO, module + ".py")) as f:
        tree = ast.parse(f.read())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


def _reachable_from(entry: str) -> set[str]:
    """Every module in this repo that `entry` imports, directly or not."""
    local = _local_modules()
    seen, queue = set(), [entry]
    while queue:
        module = queue.pop()
        if module in seen:
            continue
        seen.add(module)
        queue.extend(_imports_of(module) & local)
    return seen


class TestTheAppBundleCarriesEverythingItNeeds(unittest.TestCase):
    """SOURCES in make_app.sh is a hand-written list, and a module missing
    from it doesn't fail the build, or the tests, or anything else - it fails
    when the app is double-clicked, with a ModuleNotFoundError and nothing
    else to go on. That has happened. This is the check that would have
    caught it."""

    def test_every_module_the_gui_imports_is_copied_into_the_bundle(self):
        needed = {module + ".py" for module in _reachable_from("gui")}
        missing = needed - _sources_in_make_app()

        self.assertEqual(missing, set(), "make_app.sh SOURCES is missing these")

    def test_it_does_not_carry_modules_nothing_reaches(self):
        # The other direction, so the list doesn't quietly accumulate files
        # that were deleted or split up.
        reachable = {module + ".py" for module in _reachable_from("gui")}
        stale = _sources_in_make_app() - reachable

        self.assertEqual(stale, set(), "make_app.sh SOURCES lists these for no reason")

    def test_the_list_is_actually_found(self):
        # If the SOURCES regex ever stops matching, both tests above would
        # pass vacuously.
        self.assertIn("gui.py", _sources_in_make_app())


if __name__ == "__main__":
    unittest.main()

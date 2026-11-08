"""Windows-only code paths, exercised on any OS by faking sys.platform."""
import os
import sys
import unittest
from unittest import mock

import gui
import history


class WindowsSupport(unittest.TestCase):
    def test_open_path_on_mac_is_unchanged(self):
        with mock.patch.object(sys, "platform", "darwin"), mock.patch("gui.subprocess.run") as run:
            gui._open_path("/x/y")
            run.assert_called_once_with(["open", "/x/y"], check=False)
            run.reset_mock()
            gui._open_path("/x/y", reveal=True)
            run.assert_called_once_with(["open", "-R", "/x/y"], check=False)

    def test_open_path_on_windows(self):
        with mock.patch.object(sys, "platform", "win32"), mock.patch("gui.subprocess.run") as run, \
                mock.patch("os.startfile", create=True) as start:
            gui._open_path("C:/songs")
            start.assert_called_once_with("C:/songs")
            gui._open_path("C:/songs/a.wav", reveal=True)
            self.assertEqual(run.call_args[0][0][:2], ["explorer", "/select,"])

    def test_support_dir_on_windows_uses_appdata(self):
        with mock.patch.object(sys, "platform", "win32"), mock.patch.dict(os.environ, {"APPDATA": "C:/AD"}):
            self.assertEqual(history._support_dir(), os.path.join("C:/AD", "Beat Thief"))

    def test_support_dir_on_mac(self):
        with mock.patch.object(sys, "platform", "darwin"):
            self.assertTrue(history._support_dir().endswith("Library/Application Support/Beat Thief"))

    def test_engine_bin_goes_on_path_only_on_windows(self):
        import tempfile
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "bin"))
        with mock.patch.dict(os.environ, {"STEEZY_ENGINE_DIR": d, "PATH": "/usr/bin"}):
            with mock.patch.object(sys, "platform", "darwin"):
                gui._put_engine_on_path()
                self.assertEqual(os.environ["PATH"], "/usr/bin")
            with mock.patch.object(sys, "platform", "win32"):
                gui._put_engine_on_path()
                self.assertTrue(os.environ["PATH"].startswith(os.path.join(d, "bin")))


if __name__ == "__main__":
    unittest.main()


def test_gui_imports_without_unix_only_modules():
    """Windows has no termios/tty; Beat Thief crashed on start there because
    song_sanitizer imported them at the top (found by Steezy Studio's Windows CI)."""
    import subprocess, sys
    code = ("import sys\n"
            "for m in ('termios', 'tty', 'fcntl', 'pty', 'resource', 'pwd', 'grp'): sys.modules[m] = None\n"
            "import gui\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=os.path.dirname(os.path.abspath(__file__)),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]


class TestNoConsoleWindows(unittest.TestCase):
    def setUp(self):
        import subprocess
        self.original = subprocess.Popen.__init__
        self.addCleanup(self._restore)

    def _restore(self):
        import subprocess
        subprocess.Popen.__init__ = self.original
        if hasattr(subprocess.Popen, "_steezy_quiet"):
            del subprocess.Popen._steezy_quiet

    def test_nothing_changes_on_a_mac(self):
        import winquiet
        self.assertFalse(winquiet.install("darwin"))

    def test_every_new_process_is_hidden_on_windows(self):
        import subprocess
        import winquiet
        seen = []
        subprocess.Popen.__init__ = lambda self, *a, **kw: seen.append(kw)
        self.assertTrue(winquiet.install("win32"))
        subprocess.Popen(["ffmpeg"])
        subprocess.Popen(["ffmpeg"], creationflags=0x10)
        self.assertEqual([kw["creationflags"] for kw in seen], [winquiet.CREATE_NO_WINDOW, 0x10])

    def test_demucs_runs_with_the_patch_in_place_on_windows(self):
        import winquiet
        with mock.patch.object(sys, "platform", "win32"):
            cmd = winquiet.python_module_cmd("pythonw.exe", "demucs")
        self.assertEqual(cmd[:2], ["pythonw.exe", "-c"])
        self.assertIn("winquiet.install()", cmd[2])
        self.assertIn("run_module('demucs'", cmd[2])
        self.assertEqual(winquiet.python_module_cmd("python3", "demucs"), ["python3", "-m", "demucs"])

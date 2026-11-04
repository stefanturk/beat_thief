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

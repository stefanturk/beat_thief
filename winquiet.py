"""No console windows on Windows.

Beat Thief runs under pythonw, which has no console. Every console program
it starts - ffmpeg and ffprobe, from here and from inside demucs and
audition - then opens a console window of its own, which flashes up and
vanishes several times a song. Imported first, this gives every new
process CREATE_NO_WINDOW unless its caller chose its own flags.

Does nothing anywhere but Windows."""
from __future__ import annotations

import os
import subprocess
import sys

CREATE_NO_WINDOW = 0x08000000


def install(platform: str | None = None) -> bool:
    """Patch subprocess.Popen. True if it was patched (now or before)."""
    if (platform or sys.platform) != "win32":
        return False
    if getattr(subprocess.Popen, "_steezy_quiet", False):
        return True
    original = subprocess.Popen.__init__

    def __init__(self, *args, **kwargs):
        if not kwargs.get("creationflags"):
            kwargs["creationflags"] = CREATE_NO_WINDOW
        original(self, *args, **kwargs)

    subprocess.Popen.__init__ = __init__
    subprocess.Popen._steezy_quiet = True
    return True


def python_module_cmd(python: str, module: str) -> list:
    """[python, ...] that runs `python -m module` with this patch in place
    first, so the processes that module starts stay hidden too. A plain
    `-m module` everywhere but Windows."""
    if sys.platform != "win32":
        return [python, "-m", module]
    here = os.path.dirname(os.path.abspath(__file__))
    code = ("import sys, runpy; sys.path.insert(0, %r); import winquiet; winquiet.install(); "
            "sys.argv[0] = %r; runpy.run_module(%r, run_name='__main__', alter_sys=True)") % (here, module, module)
    return [python, "-c", code]


install()


# ---- the app's own icon on Windows ------------------------------------------
#
# Started with pythonw, the window and its taskbar button would wear Python's
# icon and be grouped with every other Python program. Giving the process an
# app id of its own splits it into its own taskbar group, and that group
# shows the window's icon, which is set from the repo's icon.ico.

def windows_app_identity(app_id: str) -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Steezy." + app_id)
    except Exception:
        pass


def set_window_icon(title: str, ico_path: str) -> bool:
    """Put ico_path on the top-level window called title (big and small
    icon). Safe from any thread; False if anything is missing."""
    if sys.platform != "win32" or not os.path.exists(ico_path):
        return False
    try:
        import ctypes
        user32 = ctypes.windll.user32
        user32.FindWindowW.restype = ctypes.c_void_p
        user32.LoadImageW.restype = ctypes.c_void_p
        user32.SendMessageW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
        hwnd = user32.FindWindowW(None, title)
        if not hwnd:
            return False
        for which, size in ((1, 0), (0, 16)):        # ICON_BIG at default size, ICON_SMALL at 16
            icon = user32.LoadImageW(None, ico_path, 1, size, size, 0x10 | (0x40 if size == 0 else 0))
            if icon:
                user32.SendMessageW(hwnd, 0x80, which, icon)   # WM_SETICON
        return True
    except Exception:
        return False

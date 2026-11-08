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

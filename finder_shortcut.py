"""Put ~/Music/Beat Thief in Finder's sidebar, next to Desktop and Downloads.

Run by setup.sh after the app is built (so on every update too), never by the
app itself. macOS has no supported way to do this from a script. The
LSSharedFileList calls are the ones `mysides` uses: deprecated, but present,
and reachable with ctypes, so the install needs no extra tool. If a future
macOS drops them, install carries on and says how to do it by hand.

Safe to run any number of times: if the sidebar already has an entry pointing
at the folder (one made by hand counts), nothing is added.
"""

import os
import sys
from typing import Callable, Iterable, Optional

FOLDER = os.path.join(os.path.expanduser("~"), "Music", "Beat Thief")
NAME = "Beat Thief"


def _same(a: str, b: str) -> bool:
    return os.path.normpath(a).rstrip("/") == os.path.normpath(b).rstrip("/")


def ensure(folder: str, existing: Iterable[str], add: Callable[[str], None]) -> bool:
    """Add `folder` unless one of the `existing` paths already is it.
    Returns whether it added."""
    if any(_same(path, folder) for path in existing if path):
        return False
    add(folder)
    return True


def _coreservices():
    import ctypes
    import ctypes.util
    cs = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreServices"))
    cf = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation"))
    vp = ctypes.c_void_p
    cs.LSSharedFileListCreate.restype = vp
    cs.LSSharedFileListCreate.argtypes = [vp, vp, vp]
    cs.LSSharedFileListCopySnapshot.restype = vp
    cs.LSSharedFileListCopySnapshot.argtypes = [vp, vp]
    cs.LSSharedFileListItemCopyResolvedURL.restype = vp
    cs.LSSharedFileListItemCopyResolvedURL.argtypes = [vp, ctypes.c_uint32, vp]
    cs.LSSharedFileListInsertItemURL.restype = vp
    cs.LSSharedFileListInsertItemURL.argtypes = [vp, vp, vp, vp, vp, vp, vp]
    cf.CFArrayGetCount.restype = ctypes.c_long
    cf.CFArrayGetCount.argtypes = [vp]
    cf.CFArrayGetValueAtIndex.restype = vp
    cf.CFArrayGetValueAtIndex.argtypes = [vp, ctypes.c_long]
    cf.CFURLCopyFileSystemPath.restype = vp
    cf.CFURLCopyFileSystemPath.argtypes = [vp, ctypes.c_long]
    cf.CFStringGetCString.restype = ctypes.c_bool
    cf.CFStringGetCString.argtypes = [vp, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
    cf.CFStringCreateWithCString.restype = vp
    cf.CFStringCreateWithCString.argtypes = [vp, ctypes.c_char_p, ctypes.c_uint32]
    cf.CFURLCreateWithFileSystemPath.restype = vp
    cf.CFURLCreateWithFileSystemPath.argtypes = [vp, vp, ctypes.c_long, ctypes.c_bool]
    cf.CFRelease.argtypes = [vp]
    return cs, cf


def _text(cf, string) -> str:
    import ctypes
    buf = ctypes.create_string_buffer(4096)
    return buf.value.decode("utf-8") if cf.CFStringGetCString(string, buf, 4096, 0x08000100) else ""


def _sidebar():
    import ctypes
    cs, cf = _coreservices()
    favorites = ctypes.c_void_p.in_dll(cs, "kLSSharedFileListFavoriteItems")
    return cs, cf, cs.LSSharedFileListCreate(None, favorites, None)


def sidebar_paths() -> list:
    """The folders the Finder sidebar's Favorites already point at."""
    cs, cf, lst = _sidebar()
    paths = []
    items = cs.LSSharedFileListCopySnapshot(lst, None)
    for i in range(cf.CFArrayGetCount(items)):
        url = cs.LSSharedFileListItemCopyResolvedURL(cf.CFArrayGetValueAtIndex(items, i), 0, None)
        if url:
            string = cf.CFURLCopyFileSystemPath(url, 0)
            if string:
                paths.append(_text(cf, string))
    return paths


def add_to_sidebar(folder: str) -> None:
    import ctypes
    cs, cf, lst = _sidebar()
    last = ctypes.c_void_p.in_dll(cs, "kLSSharedFileListItemLast")
    name = cf.CFStringCreateWithCString(None, NAME.encode(), 0x08000100)
    path = cf.CFStringCreateWithCString(None, folder.encode(), 0x08000100)
    url = cf.CFURLCreateWithFileSystemPath(None, path, 0, True)
    if not cs.LSSharedFileListInsertItemURL(lst, last, name, None, url, None, None):
        raise RuntimeError("macOS refused the sidebar entry")


def main(folder: str = FOLDER) -> int:
    if sys.platform != "darwin":
        return 0
    try:
        os.makedirs(folder, exist_ok=True)
        added = ensure(folder, sidebar_paths(), add_to_sidebar)
        print("   Added Beat Thief to the Finder sidebar." if added
              else "   Beat Thief is already in the Finder sidebar.")
    except Exception as e:  # never fail an install over a shortcut
        print(f"   Couldn't add the Finder shortcut ({e}); drag ~/Music/Beat Thief "
              "to the Finder sidebar yourself.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

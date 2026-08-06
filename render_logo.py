#!/usr/bin/env python3
"""Rasterize ui/logo.svg to a transparent PNG, for building the app icon.

Used by make_app.sh. Exists because Quick Look's renderer (qlmanage), the
obvious built-in way to turn an SVG into a PNG, flattens transparency onto
white - which gave the Dock icon a white square behind the record. AppKit
loads SVG directly and keeps the alpha channel, so the disc stays a disc.

    python3 render_logo.py <output.png> [size]
"""

from __future__ import annotations

import os
import sys

from AppKit import (
    NSAlphaFirstBitmapFormat,
    NSBitmapImageRep,
    NSCalibratedRGBColorSpace,
    NSCompositingOperationCopy,
    NSGraphicsContext,
    NSImage,
    NSMakeRect,
    NSPNGFileType,
)

LOGO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "logo.svg")


def render(out_path: str, size: int = 1024, logo_path: str = LOGO) -> None:
    image = NSImage.alloc().initWithContentsOfFile_(logo_path)
    if image is None:
        raise SystemExit(f"Could not read {logo_path}")

    rep = NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bitmapFormat_bytesPerRow_bitsPerPixel_(
        None, size, size, 8, 4, True, False, NSCalibratedRGBColorSpace, NSAlphaFirstBitmapFormat, 0, 0
    )

    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.setCurrentContext_(
        NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep)
    )
    image.drawInRect_fromRect_operation_fraction_(
        NSMakeRect(0, 0, size, size), NSMakeRect(0, 0, 0, 0), NSCompositingOperationCopy, 1.0
    )
    NSGraphicsContext.restoreGraphicsState()

    data = rep.representationUsingType_properties_(NSPNGFileType, {})
    if not data.writeToFile_atomically_(out_path, True):
        raise SystemExit(f"Could not write {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    render(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1024)

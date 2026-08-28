"""Compose the desktop hero from the generated grand-piano plate.

The instrument is used exactly as generated. Nothing here redraws, replaces,
masks or otherwise touches the piano: no keyboard is reconstructed, no decal is
laid on, no part of the anatomy is altered. An earlier version of this script
did rebuild the keyboard; that is removed.

What remains are the three things the page needs and that do not touch the
instrument:

  1. the studio backdrop is forced to exactly #f6f5f3, the colour of the
     piano-video section, so the hero shares one ground with it;
  2. the plate is scaled and positioned for a full-bleed hero, letting the
     oversized piano crop through the viewport edges;
  3. the bottom of the frame is dissolved into that same backdrop, starting
     below the keyboard's lowest point so the keys are never faded.
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SRC = "images/backgorund-images/hero-piano-raw.jpg"
DST = "images/backgorund-images/hero-piano.jpg"
BONE = np.array([0xF6, 0xF5, 0xF3], dtype=np.float64)   # == the piano-video section

RIM_Y = 1700                       # top of the case; above it, only lid and backdrop
KB_LOW = 2711                      # the keyboard's lowest point on the plate

# Design frame is 16:9; built at 2x and delivered at 3840x2160.
SS = 2
DW, DH = 1600 * SS, 900 * SS
KB_BOTTOM = 800 * SS               # where the keyboard's lowest point lands
PIANO_TOP = -150 * SS              # design y of the plate's top row
PIANO_DX = -33 * SS                # slide so the lid clears the nav's left links
FADE_FROM, FADE_TO = 815 * SS, 892 * SS   # starts below the keys, never over them


def snap_backdrop(a):
    """Force the studio backdrop onto exactly BONE, leaving the piano alone.

    Above the rim the frame holds only lid, prop and backdrop, so a threshold is
    safe there and reaches the void the lid encloses, which a border flood fill
    cannot. Below the rim the fill is seeded from every edge -- the instrument
    reaches both sides, so pockets beside it are not all reachable from one
    corner -- and it never touches the naturals, which the case encloses.
    """
    h, w, _ = a.shape
    lum = 0.2126*a[:, :, 0] + 0.7152*a[:, :, 1] + 0.0722*a[:, :, 2]
    m = np.zeros((h, w), bool)
    m[:RIM_Y] = lum[:RIM_Y] > 185

    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    small = im.convert("L").resize((w // 4, h // 4), Image.BILINEAR)
    light = small.point(lambda v: 255 if v > 196 else 0)
    sw, sh = light.size
    for sx, sy in [(0, 0), (sw-1, 0), (0, sh-1), (sw-1, sh-1),
                   (sw//2, 0), (0, sh//2), (sw-1, sh//2), (sw//2, sh-1)]:
        if light.getpixel((sx, sy)) == 255:
            ImageDraw.floodfill(light, (sx, sy), 128)
    below = np.asarray(light.point(lambda v: 255 if v == 128 else 0)
                       .resize((w, h), Image.NEAREST), np.uint8) > 0
    m[RIM_Y:] = below[RIM_Y:]

    soft = np.asarray(Image.fromarray((m * 255).astype(np.uint8))
                      .filter(ImageFilter.GaussianBlur(2.0)), np.float64) / 255.0
    return a * (1 - soft[:, :, None]) + BONE[None, None, :] * soft[:, :, None]


a = np.asarray(Image.open(SRC).convert("RGB"), dtype=np.float64)
im = Image.fromarray(np.clip(snap_backdrop(a), 0, 255).astype(np.uint8))

K = (KB_BOTTOM - PIANO_TOP) / KB_LOW
scaled = im.resize((round(im.width * K), round(im.height * K)), Image.LANCZOS)
canvas = Image.new("RGB", (DW, DH), tuple(BONE.astype(int)))
canvas.paste(scaled, (DW // 2 - round(im.width / 2 * K) + PIANO_DX, PIANO_TOP))

out = np.asarray(canvas, dtype=np.float64)
y = np.arange(DH, dtype=np.float64)
t = np.clip((y - FADE_FROM) / (FADE_TO - FADE_FROM), 0.0, 1.0)
al = (t * t * (3 - 2 * t))[:, None, None]
out = out * (1 - al) + BONE[None, None, :] * al
out[FADE_TO:] = BONE[None, None, :]

final = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).resize((3840, 2160), Image.LANCZOS)
final.save(DST, quality=86, optimize=True, progressive=True, subsampling=0)
print("wrote %s %s  %.0f KB   piano %.0f design px (%.0f%% of 1440)"
      % (DST, final.size, os.path.getsize(DST) / 1024,
         im.width * K / SS, 100 * im.width * K / SS / 1440))

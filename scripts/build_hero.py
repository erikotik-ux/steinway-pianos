"""Compose the desktop hero from the generated grand-piano plate.

The plate is a near-front elevation turned a few degrees to the left, so the
instrument reads with depth while the keyboard still faces the viewer. The
triangle framed by the raised lid, its prop and the top of the case is the
hero's negative space, and the headline is seated inside it.

Two things are rebuilt rather than used as generated. The keyboard came back
with uneven naturals and broken sharp groups, so it is redrawn from real
dimensions (scripts/keyboard.py) and laid into the picture through a projective
transform (scripts/warp.py) so it sits on the receding key plane instead of
being pasted square. The fallboard carries the real Steinway wordmark from
S&S_logo, laid in the same way.

The backdrop is forced to exactly #f6f5f3 -- the colour of the piano-video
section below -- so the hero and that section share one continuous ground.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from reportlab.graphics import renderPM
from svglib.svglib import svg2rlg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import keyboard as kbmod
import warp

SRC = "images/backgorund-images/hero-piano-raw.jpg"
LOGO = "S&S_logo/steinway-and-sons.svg"
DST = "images/backgorund-images/hero-piano.jpg"
BONE = np.array([0xF6, 0xF5, 0xF3], dtype=np.float64)   # == the piano-video section

# Landmarks measured on the plate.
RIM_Y = 1700                       # top of the case; lower edge of the void
KB_QUAD = [(2065, 2438), (5408, 2184), (5408, 2382), (2065, 2709)]   # tl tr br bl
KB_LOW = 2711                      # the keyboard's lowest point
FELT_M, FELT_C = -0.0832, 2699.8   # the felt line, y = m x + c

# Design frame is 16:9; built at 2x and delivered at 3840x2160.
SS = 2
DW, DH = 1600 * SS, 900 * SS
KB_BOTTOM = 800 * SS               # where the keyboard's lowest point lands
PIANO_TOP = -150 * SS              # design y of the plate's top row
PIANO_DX = -33 * SS               # slide so the lid clears the nav's left links
FADE_FROM, FADE_TO = 815 * SS, 892 * SS   # starts below the keys, never over them
LOGO_W, LOGO_CX = 200, 3400        # wordmark width and centre on the fallboard


def snap_backdrop(a):
    """Force the studio backdrop onto exactly BONE.

    Above the rim the frame holds only lid, prop and backdrop, so a threshold is
    safe there and reaches the void the lid encloses, which a border flood fill
    cannot. Below the rim the fill is seeded from every edge -- the instrument
    reaches both sides, so pockets beside it are not all reachable from one
    corner -- and never touches the naturals, which the case encloses.
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


def rebuild_keyboard(im):
    """Lay a geometrically exact keyboard onto the receding key plane."""
    a = np.asarray(im, dtype=np.float64)
    (x0, _), (x1, _) = KB_QUAD[0], KB_QUAD[1]
    width = x1 - x0
    # The keyboard runs off the right edge, so sample what is in frame and
    # stretch that lighting profile across the full key run.
    xs = min(x1, a.shape[1])
    face = a[2620:2680, x0:xs].mean(axis=(0, 2))
    prof = np.convolve(face, np.ones(301)/301, mode="same")
    prof = np.clip(prof / np.median(prof[150:-150]), 0.93, 1.07)
    prof = np.interp(np.linspace(0, len(prof) - 1, width), np.arange(len(prof)), prof)
    kb = kbmod.draw(width, 0, 51, 163, 18, shade=prof, dark_h=90)
    return warp.place(im, kb, KB_QUAD)


def lay_decal(im):
    """Inlay the real wordmark into the fallboard, on the fallboard's plane."""
    d = svg2rlg(LOGO)
    sc = (LOGO_W * 4) / d.width
    d.width *= sc; d.height *= sc; d.scale(sc, sc)
    tmp = os.path.join(os.environ.get("TEMP", "."), "_logo_plate.png")
    renderPM.drawToFile(d, tmp, fmt="PNG", bg=0xFFFFFF)
    g = np.asarray(Image.open(tmp).convert("L"), dtype=np.float64)
    alpha = np.clip((255.0 - g) / 255.0, 0, 1)
    lh, lw = alpha.shape
    art = np.zeros((lh, lw, 4), np.uint8)
    art[:, :, :3] = np.array([201, 163, 94], np.uint8)
    art[:, :, 3] = (alpha * 235).astype(np.uint8)

    felt = lambda x: FELT_M * x + FELT_C
    keyh = lambda x: 271 - 0.01466 * (x - KB_QUAD[0][0])      # local key-plane scale
    hx = LOGO_W * lh / lw
    quad = []
    for x, up in ((LOGO_CX - LOGO_W/2, 0), (LOGO_CX + LOGO_W/2, 0),
                  (LOGO_CX + LOGO_W/2, 1), (LOGO_CX - LOGO_W/2, 1)):
        base = felt(x) - 0.62 * keyh(x)
        quad.append((x, base + up * hx * keyh(x) / 271))
    return warp.place(im, Image.fromarray(art, "RGBA"), quad)


a = np.asarray(Image.open(SRC).convert("RGB"), dtype=np.float64)
im = Image.fromarray(np.clip(snap_backdrop(a), 0, 255).astype(np.uint8))
im = lay_decal(rebuild_keyboard(im))

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
final.save(DST, quality=84, optimize=True, progressive=True, subsampling=0)
print("wrote %s %s  %.0f KB   piano %.0f design px (%.0f%% of 1440)"
      % (DST, final.size, os.path.getsize(DST) / 1024,
         im.width * K / SS, 100 * im.width * K / SS / 1440))

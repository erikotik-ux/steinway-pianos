"""Compose the desktop hero from the generated grand-piano plate.

The plate is a front elevation with the lid hinged to the rim on its prop. The
triangle framed by the lid underside, the prop and the rim is the hero's
negative space, and the headline is seated inside it.

Two things about that triangle drive everything here. Its usable size scales
with how large the piano is drawn, and the piano's scale is capped by height:
the nav sits above it and the keyboard has to stay above the fold. So the way to
a bigger headline is not a steeper lid -- steepening raises the instrument's
overall height, forcing the piano smaller and the void with it. It is to shorten
the instrument. The fallboard between the rim and the keys is a uniform black
expanse (measured standard deviation 5-8 across its width), so it can be
squeezed vertically without leaving a trace, which buys back scale for the whole
piano and therefore a much larger headline.

The model's keyboard came out with uneven key widths and broken sharp groups, so
it is redrawn from real dimensions; see scripts/keyboard.py. The fallboard also
carries the real Steinway wordmark from S&S_logo, laid in in brass.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from reportlab.graphics import renderPM
from svglib.svglib import svg2rlg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import keyboard as kbmod

SRC = "images/backgorund-images/hero-piano-raw.jpg"
LOGO = "S&S_logo/steinway-and-sons.svg"
DST = "images/backgorund-images/hero-piano.jpg"
BONE = np.array([0xF6, 0xF5, 0xF3], dtype=np.float64)

# Landmarks measured on the plate.
PIANO_L, PIANO_R = 165, 5181
LID_TIP = 5
RIM_Y = 1560
KB_X0, KB_X1 = 578, 4805          # run of naturals
KB_FELT, KB_TOP, KB_SHARP, KB_BOT = 2606, 2624, 2740, 2844
KB_END = 2870                     # below the naturals, where the slip begins
SQ_TOP, SQ_BOT, SQ_KEEP = 1880, 2570, 120   # fallboard squeeze
HINGE_Y = 1400                    # below this the lid's real hinge starts

# Design frame is 1440x900 (the hero at the reference desktop width), built at 2x.
SS = 2
DW, DH = 1440 * SS, 900 * SS
PIANO_TOP = 68 * SS                # lid tip, clear of the nav links at every viewport
KB_BOTTOM = 888 * SS              # keyboard floor
FADE_FROM, FADE_TO = 878 * SS, 899 * SS


def remove_duplicate_lid(a):
    """Delete the second lid panel the model drew alongside the real one.

    Row by row above the rim the plate reads as: a ~157px slab, a gap, the lid
    proper (300-450px, carrying the brass rim), then the prop far right. The
    slab is the duplicate, and it is always the run left of the widest one, so
    everything left of the lid can simply be returned to backdrop.
    """
    h, w, _ = a.shape
    lum = 0.2126*a[:, :, 0] + 0.7152*a[:, :, 1] + 0.0722*a[:, :, 2]
    kill = np.zeros((h, w), bool)
    for y in range(RIM_Y):
        row = lum[y] < 185
        runs, st = [], None
        for x in range(w):
            if row[x] and st is None:
                st = x
            elif not row[x] and st is not None:
                if x - st > 8:
                    runs.append((st, x - 1))
                st = None
        if len(runs) < 2:
            continue
        main = max(runs, key=lambda r: r[1] - r[0])
        if y < HINGE_Y:
            # Clear the whole span rather than just the dark runs -- the
            # duplicate carries its own bright brass edging, which a darkness
            # test would leave behind as floating speckles. Cut at the midpoint
            # of the gap, so the real lid keeps its own brass rim.
            before = [r for r in runs if r[1] < main[0]] + [main]
            if len(before) < 2:
                continue
            # Split at the widest gap: that is what separates the duplicate from
            # the lid, whereas the narrow gaps are the lid's own edge and brass.
            gaps = [(before[i + 1][0] - before[i][1], i) for i in range(len(before) - 1)]
            _, i = max(gaps)
            kill[y, :(before[i][1] + before[i + 1][0]) // 2] = True
        else:
            for r in runs:               # near the rim, keep the real hinge
                if r[1] < main[0]:
                    kill[y, r[0]:r[1] + 1] = True
    soft = np.asarray(Image.fromarray((kill * 255).astype(np.uint8))
                      .filter(ImageFilter.MaxFilter(5))
                      .filter(ImageFilter.GaussianBlur(2.0)), np.float64) / 255.0
    return a * (1 - soft[:, :, None]) + BONE[None, None, :] * soft[:, :, None]


def close_lid_nicks(a):
    """Fill the short backdrop gaps the slab removal bit out of the lid's edge.

    Any run of backdrop above the rim that is narrower than a lid is a nick, not
    a real opening, so it can be closed from the lacquer either side of it.
    """
    h, w, _ = a.shape
    lum = 0.2126*a[:, :, 0] + 0.7152*a[:, :, 1] + 0.0722*a[:, :, 2]
    for y in range(RIM_Y):
        light = lum[y] > 185
        st = None
        for x in range(w):
            if light[x] and st is None:
                st = x
            elif not light[x] and st is not None:
                if 0 < st and x - st <= 70:          # flanked both sides, short
                    a[y, st:x] = (a[y, st - 1] + a[y, x]) / 2.0
                st = None
    return a


def snap_backdrop(a):
    """Force the studio backdrop onto exactly BONE.

    Above the rim the only things in frame are the lid, the prop and backdrop,
    so a threshold is safe there and reaches the void the lid encloses, which a
    border flood fill cannot. Below the rim the fill is seeded from the border
    instead, so the ivory naturals the case encloses are never touched.
    """
    h, w, _ = a.shape
    lum = 0.2126*a[:, :, 0] + 0.7152*a[:, :, 1] + 0.0722*a[:, :, 2]
    m = np.zeros((h, w), bool)
    m[:RIM_Y] = lum[:RIM_Y] > 185

    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    small = im.convert("L").resize((w // 4, h // 4), Image.BILINEAR)
    light = small.point(lambda v: 255 if v > 196 else 0)
    ImageDraw.floodfill(light, (0, 0), 128)
    below = np.asarray(light.point(lambda v: 255 if v == 128 else 0)
                       .resize((w, h), Image.NEAREST), np.uint8) > 0
    m[RIM_Y:] = below[RIM_Y:]

    soft = np.asarray(Image.fromarray((m * 255).astype(np.uint8))
                      .filter(ImageFilter.GaussianBlur(2.0)), np.float64) / 255.0
    return a * (1 - soft[:, :, None]) + BONE[None, None, :] * soft[:, :, None]


def rebuild_keyboard(a):
    """Swap the model's warped keyboard for a geometrically exact one."""
    face = a[KB_BOT - 50:KB_BOT - 12, KB_X0:KB_X1].mean(axis=(0, 2))
    prof = np.convolve(face, np.ones(401) / 401, mode="same")
    prof = np.clip(prof / np.median(prof[200:-200]), 0.93, 1.07)
    kb = kbmod.draw(KB_X1 - KB_X0, KB_TOP, KB_SHARP, KB_BOT, KB_TOP - KB_FELT, shade=prof)
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    im.paste(kb, (KB_X0, KB_FELT))
    return np.asarray(im, dtype=np.float64)


def squeeze_fallboard(a):
    """Compress the uniform fallboard so the instrument fits taller in frame."""
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    w = im.width
    top = im.crop((0, 0, w, SQ_TOP))
    mid = im.crop((0, SQ_TOP, w, SQ_BOT)).resize((w, SQ_KEEP), Image.LANCZOS)
    bot = im.crop((0, SQ_BOT, w, im.height))
    out = Image.new("RGB", (w, top.height + SQ_KEEP + bot.height))
    out.paste(top, (0, 0))
    out.paste(mid, (0, top.height))
    out.paste(bot, (0, top.height + SQ_KEEP))
    return np.asarray(out, dtype=np.float64), (SQ_BOT - SQ_TOP) - SQ_KEEP


def lay_decal(a, fb_top, fb_bot):
    """Inlay the real wordmark into the fallboard in brass."""
    d = svg2rlg(LOGO)
    w = 460
    sc = w / d.width
    d.width *= sc; d.height *= sc; d.scale(sc, sc)
    tmp = os.path.join(os.environ.get("TEMP", "."), "_logo_plate.png")
    renderPM.drawToFile(d, tmp, fmt="PNG", bg=0xFFFFFF)
    logo = np.asarray(Image.open(tmp).convert("L"), dtype=np.float64)
    alpha = np.clip((255.0 - logo) / 255.0, 0, 1)
    lh, lw = alpha.shape

    x0 = (PIANO_L + PIANO_R) // 2 - lw // 2
    y0 = int((fb_top + fb_bot) / 2 - lh / 2)
    patch = a[y0:y0 + lh, x0:x0 + lw]
    lumin = 0.2126*patch[:, :, 0] + 0.7152*patch[:, :, 1] + 0.0722*patch[:, :, 2]
    shade = np.clip(0.72 + lumin / 90.0, 0.6, 1.25)[:, :, None]
    brass = np.array([201, 163, 94], dtype=np.float64)[None, None, :] * shade
    al = (alpha * 0.93)[:, :, None]
    a[y0:y0 + lh, x0:x0 + lw] = patch * (1 - al) + brass * al
    return a


a = np.asarray(Image.open(SRC).convert("RGB"), dtype=np.float64)
a = snap_backdrop(close_lid_nicks(remove_duplicate_lid(a)))
a = rebuild_keyboard(a)
a, removed = squeeze_fallboard(a)
a = lay_decal(a, SQ_TOP, SQ_TOP + SQ_KEEP)

kb_bottom_src = KB_END - removed
K = (KB_BOTTOM - PIANO_TOP) / (kb_bottom_src - LID_TIP)

plate = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
scaled = plate.resize((round(plate.width * K), round(plate.height * K)), Image.LANCZOS)
canvas = Image.new("RGB", (DW, DH), tuple(BONE.astype(int)))
ox = DW // 2 - round((PIANO_L + PIANO_R) / 2 * K)
oy = PIANO_TOP - round(LID_TIP * K)
canvas.paste(scaled, (ox, oy))

out = np.asarray(canvas, dtype=np.float64)
y = np.arange(DH, dtype=np.float64)
t = np.clip((y - FADE_FROM) / (FADE_TO - FADE_FROM), 0.0, 1.0)
al = (t * t * (3 - 2 * t))[:, None, None]
out = out * (1 - al) + BONE[None, None, :] * al
out[FADE_TO:] = BONE[None, None, :]

final = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).resize((2400, 1500), Image.LANCZOS)
final.save(DST, quality=84, optimize=True, progressive=True, subsampling=0)
print("wrote %s %s  %.0f KB   piano %.0f design px (%.0f%% of 1440)"
      % (DST, final.size, os.path.getsize(DST) / 1024,
         (PIANO_R - PIANO_L) * K / SS, 100 * (PIANO_R - PIANO_L) * K / SS / 1440))

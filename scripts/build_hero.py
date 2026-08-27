"""Compose the desktop hero from the generated grand-piano plate.

The plate is a physically correct front elevation: the lid is hinged to the rim
and held at ~40 degrees on its prop, so lid, hinge, body, keyboard and support
read as one instrument. The triangle framed by the lid underside, the prop and
the rim is the hero's negative space -- the headline is seated inside it.

That triangle has a useful property: its width is 2.2x its depth below the apex,
in whatever units it is drawn at. So the type fits or does not fit independently
of how large the piano is rendered; what the piano's scale buys is depth. The
instrument's own height is what caps that scale here, since the keyboard has to
stay above the fold.

The model also stamped a garbled imitation of the Steinway fallboard decal. That
is painted out and replaced with the real wordmark from S&S_logo, laid into the
fallboard in brass and shaded by the surface underneath it.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

SRC = "images/backgorund-images/hero-piano-raw.jpg"
LOGO = "S&S_logo/steinway-and-sons.svg"
DST = "images/backgorund-images/hero-piano.jpg"
BONE = np.array([0xF6, 0xF5, 0xF3], dtype=np.float64)

# Landmarks on the plate.
PIANO_L, PIANO_R, LID_TIP = 1621, 3824, 319
DECAL_X0, DECAL_X1, DECAL_Y0, DECAL_Y1 = 2470, 2970, 1655, 1790

# Design frame is 1440x900 (the hero at the reference desktop width), built at 2x.
SS = 2
DW, DH = 1440 * SS, 900 * SS
PIANO_W = 1128 * SS                # instrument width in the design frame (78% of 1440)
PIANO_TOP = 58 * SS                # lid tip, clear of the nav links above it
FADE_FROM, FADE_TO = 868 * SS, 899 * SS


def flat_field(im, band_bot_f):
    a = np.asarray(im, dtype=np.float64)
    h, w, _ = a.shape
    bt, bb = int(h * 0.02), int(h * band_bot_f)
    plate = a.copy()
    plate[bb:, :, :] = a[bb - 1: bb, :, :]
    plate[:bt, :, :] = a[bt: bt + 1, :, :]
    field = np.asarray(Image.fromarray(plate.astype(np.uint8))
                       .filter(ImageFilter.GaussianBlur(radius=w / 18.0)), dtype=np.float64)
    gain = np.clip(BONE[None, None, :] / np.clip(field, 1.0, None), 0.78, 1.34)
    y = np.arange(h, dtype=np.float64)[:, None, None]
    ra, rb = h * (band_bot_f + 0.02), h * (band_bot_f + 0.14)
    wgt = 0.42 + 0.58 * np.clip((rb - y) / (rb - ra), 0.0, 1.0)
    return Image.fromarray(np.clip(a * (1.0 + (gain - 1.0) * wgt), 0, 255).astype(np.uint8))


def snap_backdrop(im):
    """Force the studio backdrop to exactly BONE.

    Flat-fielding alone leaves patches once the instrument covers most of the
    clean rows the field is estimated from. Instead, flood-fill the light region
    inward from the border: that selects the backdrop only -- it can reach the
    gap framed by the raised lid, because that gap opens to the edge, but never
    the ivory naturals, which the case encloses. The mask is then feathered so
    the instrument's edges keep their anti-aliasing.
    """
    a = np.asarray(im, dtype=np.float64)
    h, w, _ = a.shape
    small = im.convert("L").resize((w // 4, h // 4), Image.BILINEAR)
    light = small.point(lambda v: 255 if v > 196 else 0)
    ImageDraw.floodfill(light, (0, 0), 128)
    bg = light.point(lambda v: 255 if v == 128 else 0)
    bg = bg.resize((w, h), Image.BILINEAR).filter(ImageFilter.GaussianBlur(radius=2.5))
    m = (np.asarray(bg, dtype=np.float64) / 255.0)[:, :, None]
    return np.clip(a * (1 - m) + BONE[None, None, :] * m, 0, 255)


def clear_decal(a):
    """Paint out the model's garbled fallboard lettering.

    The fallboard is near-uniform along each row, so interpolating across the
    stamp from clean lacquer either side of it rebuilds the surface invisibly.
    """
    left = a[DECAL_Y0:DECAL_Y1, DECAL_X0 - 40:DECAL_X0 - 10].mean(axis=1)
    right = a[DECAL_Y0:DECAL_Y1, DECAL_X1 + 10:DECAL_X1 + 40].mean(axis=1)
    span = DECAL_X1 - DECAL_X0
    t = (np.arange(span) / (span - 1))[None, :, None]
    a[DECAL_Y0:DECAL_Y1, DECAL_X0:DECAL_X1] = left[:, None, :] * (1 - t) + right[:, None, :] * t
    return a


def lay_decal(a):
    """Inlay the real wordmark into the fallboard in brass."""
    d = svg2rlg(LOGO)
    w = 470
    sc = w / d.width
    d.width *= sc; d.height *= sc; d.scale(sc, sc)
    tmp = "C:/Users/eriko/AppData/Local/Temp/_logo_plate.png"
    renderPM.drawToFile(d, tmp, fmt="PNG", bg=0xFFFFFF)
    logo = np.asarray(Image.open(tmp).convert("L"), dtype=np.float64)
    alpha = np.clip((255.0 - logo) / 255.0, 0, 1)

    h, wl = alpha.shape
    x0 = (DECAL_X0 + DECAL_X1) // 2 - wl // 2
    y0 = 1655
    patch = a[y0:y0 + h, x0:x0 + wl]
    # Modulate the brass by the lacquer underneath so the decal picks up the
    # surface's own falloff instead of reading as a flat sticker.
    lumin = (0.2126 * patch[:, :, 0] + 0.7152 * patch[:, :, 1] + 0.0722 * patch[:, :, 2])
    shade = np.clip(0.72 + lumin / 90.0, 0.6, 1.25)[:, :, None]
    brass = np.array([201, 163, 94], dtype=np.float64)[None, None, :] * shade
    al = (alpha * 0.93)[:, :, None]
    a[y0:y0 + h, x0:x0 + wl] = patch * (1 - al) + brass * al
    return a


src = Image.open(SRC).convert("RGB")
a = snap_backdrop(flat_field(src, 0.10))
a = lay_decal(clear_decal(a))
plate = Image.fromarray(a.astype(np.uint8))

K = PIANO_W / (PIANO_R - PIANO_L)
scaled = plate.resize((round(plate.width * K), round(plate.height * K)), Image.LANCZOS)

canvas = Image.new("RGB", (DW, DH), tuple(BONE.astype(int)))
ox = DW // 2 - round((PIANO_L + PIANO_R) / 2 * K)
oy = PIANO_TOP - round(LID_TIP * K)
canvas.paste(scaled, (ox, oy))

out = np.asarray(canvas, dtype=np.float64)
y = np.arange(DH, dtype=np.float64)
t = np.clip((y - FADE_FROM) / (FADE_TO - FADE_FROM), 0.0, 1.0)
alpha = (t * t * (3 - 2 * t))[:, None, None]
out = out * (1 - alpha) + BONE[None, None, :] * alpha
out[FADE_TO:] = BONE[None, None, :]

final = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).resize((2400, 1500), Image.LANCZOS)
final.save(DST, quality=84, optimize=True, progressive=True, subsampling=0)
print("wrote %s %s  %.0f KB" % (DST, final.size, os.path.getsize(DST) / 1024))
c = np.asarray(final, dtype=float)
print("corner", c[20, 20].round().astype(int).tolist(),
      " above-piano", c[int(1500 * 300 / 900), 1200].round().astype(int).tolist(),
      " bottom", c[-1].mean(axis=0).round().astype(int).tolist())

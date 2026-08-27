"""Compose the desktop hero from the generated piano plate.

The raw frame has the right elements (floating canopy lid, bare fallboard,
keyboard) but the bone gap between lid and case is far too shallow to seat the
headline, and an asymmetric lid prop crosses it. Both are fixed by rebuilding
the frame: the lid and the body are kept at a single shared scale so the
instrument stays consistent, and the gap between them is replaced outright with
clean backdrop at the height the typography actually needs. The prop lived
entirely inside that discarded band, so it disappears with it.

Finally the lower body is dissolved into the page background in-pixel, so the
photograph has no rectangular edge at any size the CSS ever renders it at.
"""
import numpy as np
from PIL import Image, ImageFilter

SRC = "images/backgorund-images/hero-piano-raw.jpg"
DST = "images/backgorund-images/hero-piano.jpg"
BONE = np.array([0xF6, 0xF5, 0xF3], dtype=np.float64)

# Landmarks measured off the source plate.
LID_BOT, BAND_TOP, CASE_TOP = 1593, 1597, 2127

# Design frame is 1440x900 (the hero at the reference desktop width); built at 2x.
SS = 2
DW, DH = 1440 * SS, 900 * SS
K = 0.29 * SS                    # source px -> design px
LID_BOTTOM_Y = 215 * SS          # lid underside: upper edge of the opening
CASE_TOP_Y = 615 * SS            # case: lower edge of the opening  (400px gap)
FADE_FROM, FADE_TO = 715 * SS, 840 * SS


def flat_field(im, band_top_f, band_bot_f):
    """Neutralise the warm, uneven studio backdrop onto BONE."""
    a = np.asarray(im, dtype=np.float64)
    h, w, _ = a.shape
    bt, bb = int(h * band_top_f), int(h * band_bot_f)
    plate = a.copy()
    plate[bb:, :, :] = a[bb - 1: bb, :, :]
    plate[:bt, :, :] = a[bt: bt + 1, :, :]
    field = np.asarray(
        Image.fromarray(plate.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=w / 18.0)),
        dtype=np.float64)
    gain = np.clip(BONE[None, None, :] / np.clip(field, 1.0, None), 0.78, 1.34)
    y = np.arange(h, dtype=np.float64)[:, None, None]
    ra, rb = h * (band_bot_f + 0.02), h * (band_bot_f + 0.14)
    wgt = 0.42 + 0.58 * np.clip((rb - y) / (rb - ra), 0.0, 1.0)
    return Image.fromarray(np.clip(a * (1.0 + (gain - 1.0) * wgt), 0, 255).astype(np.uint8))


def snap_backdrop(im):
    """Pull every near-backdrop pixel onto exactly BONE.

    Flat-fielding lands the backdrop within a couple of levels of target, but a
    couple of levels is enough to draw a visible seam where a pasted slab meets
    the canvas. Snapping only pixels already close to BONE removes the seams
    while leaving the instrument -- even the ivory naturals, which sit ~60
    levels darker -- completely untouched.
    """
    a = np.asarray(im, dtype=np.float64)
    d = np.sqrt(((a - BONE[None, None, :]) ** 2).sum(axis=2))
    t = np.clip((34.0 - d) / 22.0, 0.0, 1.0)          # 1 under 12, 0 over 34
    w = (t * t * (3 - 2 * t))[:, :, None]
    return Image.fromarray(np.clip(a * (1 - w) + BONE[None, None, :] * w, 0, 255).astype(np.uint8))


src = snap_backdrop(flat_field(Image.open(SRC).convert("RGB"), 0.02, 0.40))
sw, sh = src.size
scaled = src.resize((round(sw * K), round(sh * K)), Image.LANCZOS)
ox = (DW - scaled.width) // 2                       # centred; bone gutters crop away

canvas = Image.new("RGB", (DW, DH), tuple(BONE.astype(int)))

# Lid slab, hung so its underside lands on the top edge of the opening.
top = scaled.crop((0, 0, scaled.width, round(LID_BOT * K)))
canvas.paste(top, (ox, LID_BOTTOM_Y - top.height))

# Body slab, seated so the case top lands on the bottom edge of the opening.
# The fallboard between the case top and the keys is a uniform black expanse;
# left at full height it makes the lower half far heavier than the headline it
# is supposed to frame. Squeeze that stretch vertically (a smooth scale, not a
# cut, so the lit cheek blocks either side keep continuous gradients) to bring
# the keyboard up under the case and lighten the base of the composition.
FB_TOP, FB_BOT, FB_KEEP = 2200, 2650, 210
top_pad = scaled.crop((0, round(CASE_TOP * K), scaled.width, round(FB_TOP * K)))
squeeze = scaled.crop((0, round(FB_TOP * K), scaled.width, round(FB_BOT * K)))     .resize((scaled.width, round(FB_KEEP * K)), Image.LANCZOS)
tail = scaled.crop((0, round(FB_BOT * K), scaled.width, scaled.height))

bot = Image.new("RGB", (scaled.width, top_pad.height + squeeze.height + tail.height))
bot.paste(top_pad, (0, 0))
bot.paste(squeeze, (0, top_pad.height))
bot.paste(tail, (0, top_pad.height + squeeze.height))
# The body is shorter than the dissolve is long, so on its own the photograph
# would run out mid-fade and leave a hard step where content meets backdrop.
# Carry the keybed down past the end of the ramp so the dissolve always has
# something to dissolve, and the image reaches pure backdrop with no edge.
need = FADE_TO + 24 - CASE_TOP_Y
if bot.height < need:
    ext = Image.new("RGB", (bot.width, need))
    ext.paste(bot, (0, 0))
    ext.paste(bot.crop((0, bot.height - 1, bot.width, bot.height))
                 .resize((bot.width, need - bot.height), Image.NEAREST),
              (0, bot.height))
    bot = ext
canvas.paste(bot, (ox, CASE_TOP_Y))

# Dissolve the lower body into the page background (smoothstep, no hard edge).
a = np.asarray(canvas, dtype=np.float64)
y = np.arange(DH, dtype=np.float64)
t = np.clip((y - FADE_FROM) / (FADE_TO - FADE_FROM), 0.0, 1.0)
alpha = (t * t * (3 - 2 * t))[:, None, None]                 # 0 -> 1 eased
out = a * (1 - alpha) + BONE[None, None, :] * alpha
out[FADE_TO:] = BONE[None, None, :]

final = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
final = final.resize((2400, 1500), Image.LANCZOS)
final.save(DST, quality=84, optimize=True, progressive=True, subsampling=0)

import os
print("wrote %s  %s  %.0f KB" % (DST, final.size, os.path.getsize(DST) / 1024))
chk = np.asarray(final.convert("RGB"), dtype=float)
print("corner bone:", chk[20, 20].round().astype(int).tolist(),
      " opening centre:", chk[int(1500 * 415 / 900), 1200].round().astype(int).tolist(),
      " bottom row:", chk[-1].mean(axis=0).round().astype(int).tolist())

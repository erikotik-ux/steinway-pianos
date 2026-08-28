"""Draw a geometrically exact piano keyboard to replace the generated one.

The plate's keyboard came out of the model with uneven key widths and broken
sharp groups. A front elevation makes the keyboard a straight run, so it can be
drawn from real dimensions instead: 52 naturals, 36 sharps, white 23.5mm wide,
sharp 13.7mm, and the sharps placed so the natural fronts inside each group come
out equal -- which is what gives a real keyboard its slightly uneven-looking
sharp spacing.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W_MM, B_MM = 23.5, 13.7
# Sharp centre offsets from the natural boundary, in white-key widths, derived
# from equal natural fronts within the 3-key and 4-key groups.
OFF = {"C": -0.097, "D": +0.097, "F": -0.146, "G": 0.0, "A": +0.145}
NOTES = "ABCDEFG"


def draw(width, top, sharp_bottom, bottom, felt_h, shade=None, ss=3, dark_h=0):
    """Return an RGB image of the keyboard, `width` px wide."""
    h = bottom - top + felt_h + dark_h
    W, H = width * ss, h * ss
    img = Image.new("RGB", (W, H), (12, 11, 11))
    d = ImageDraw.Draw(img)
    w = W / 52.0
    dark = dark_h * ss
    felt = dark + felt_h * ss
    kt, sb, kb = felt, felt + (sharp_bottom - top) * ss, felt + (bottom - top) * ss

    # lacquer above the felt, so the replacement covers the plate's own key tops
    d.rectangle([0, 0, W, dark], fill=(26, 23, 18))
    d.rectangle([0, dark, W, felt], fill=(58, 12, 10))                 # felt
    d.rectangle([0, kt, W, kb], fill=(250, 244, 226))                  # naturals

    for i in range(1, 52):                                             # separations
        x = i * w
        d.rectangle([x - 1.1 * ss, kt, x + 0.6 * ss, kb], fill=(196, 188, 170))

    # sharps: walk the naturals from A0 and place one after every A C D F G
    bw = W / 52.0 * (B_MM / W_MM)
    for i in range(51):
        note = NOTES[(i + 0) % 7]
        if note not in OFF:
            continue
        cx = (i + 1) * w + OFF[note] * w
        x0, x1 = cx - bw / 2, cx + bw / 2
        d.rectangle([x0 + 2 * ss, kt, x1 + 2 * ss, sb + 3 * ss], fill=(120, 112, 104))   # cast shadow
        d.rectangle([x0, kt, x1, sb], fill=(26, 23, 20))
        d.rectangle([x0, kt, x1, kt + 3 * ss], fill=(52, 47, 42))                        # top facet
        d.rectangle([x0, sb - 5 * ss, x1, sb], fill=(46, 41, 36))                        # front lip
        d.rectangle([x0, kt, x0 + 2 * ss, sb], fill=(64, 58, 52))                        # left highlight

    img = img.resize((width, h), Image.LANCZOS)
    a = np.asarray(img, dtype=np.float64)

    # Naturals sit in shadow where the sharps enclose them, opening out below.
    y = np.arange(h, dtype=np.float64)[:, None, None]
    t = np.clip((y - felt_h - dark_h) / max(1, (sharp_bottom - top) + 26), 0, 1)
    lit = 0.58 + 0.42 * (t * t * (3 - 2 * t))
    natural = (a.mean(axis=2) > 150)[:, :, None]
    # Warm the shadow slightly rather than just darkening it, so the enclosed
    # part of each natural reads as ivory in shade, not grey.
    warm = np.array([1.0, 0.975, 0.94])[None, None, :]
    tint = 1.0 + (1.0 - lit) * (warm - 1.0) * 6.0
    a = np.where(natural, a * lit * tint, a)

    if shade is not None:                    # inherit the scene's lateral falloff
        a *= shade[None, :, None]

    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert("RGBA")
    if dark_h:
        # Fade the lacquer band in from nothing so the patch has no top edge
        # against the fallboard it is covering.
        alpha = np.full((h, width), 255, np.uint8)
        ramp = np.clip(np.arange(dark_h) / max(1, dark_h * 0.7), 0, 1)
        alpha[:dark_h] = (ramp[:, None] * 255).astype(np.uint8)
        img.putalpha(Image.fromarray(alpha))
    return img

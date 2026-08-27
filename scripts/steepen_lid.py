"""Rotate the generated lid up to a steeper opening angle.

The image models would not honour a 50-55 degree lid -- every attempt came back
near 27 degrees, or with the music desk raised into the void. Since the lid is a
slim panel seen almost edge-on against a plain backdrop, it can simply be
rotated about its hinge instead. That both gives the lid the angle a fully
raised grand lid actually has, and deepens the triangular void beneath it, which
is what lets the headline inside grow.

The prop cannot be rotated with the lid (its foot stays on the rim), so it is
removed and redrawn to meet the lid in its new position.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

BONE = np.array([0xF6, 0xF5, 0xF3], dtype=np.float64)
RIM_Y = 1560                       # everything above this is lid or prop
PIVOT = (1475.0, 1540.0)           # where the lid's underside meets the rim
PROP_A, PROP_B = (3550, 490), (4200, 1560)


def _seg_dist(pts, a, b):
    p = pts - np.array(a, dtype=np.float64)
    d = np.array(b, dtype=np.float64) - np.array(a, dtype=np.float64)
    t = np.clip((p @ d) / (d @ d), 0, 1)
    return np.linalg.norm(p - t[:, None] * d[None, :], axis=1)


def steepen(im, delta_deg):
    a = np.asarray(im, dtype=np.float64)
    h, w, _ = a.shape
    lum = 0.2126*a[:,:,0] + 0.7152*a[:,:,1] + 0.0722*a[:,:,2]

    band = np.zeros((h, w), bool)
    band[:RIM_Y] = lum[:RIM_Y] < 170
    ys, xs = np.nonzero(band)
    keep = _seg_dist(np.stack([xs, ys], 1).astype(float), PROP_A, PROP_B) > 85
    lid = np.zeros((h, w), bool)
    lid[ys[keep], xs[keep]] = True

    # soften the matte so the rotated lid keeps clean anti-aliased edges
    m = Image.fromarray((lid * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5))
    m = m.filter(ImageFilter.GaussianBlur(1.6))

    layer = Image.merge("RGBA", (*im.split(), m))
    layer = layer.rotate(delta_deg, resample=Image.BICUBIC, center=PIVOT)

    base = a.copy()
    erase = np.asarray(Image.fromarray((band * 255).astype(np.uint8))
                       .filter(ImageFilter.MaxFilter(7))
                       .filter(ImageFilter.GaussianBlur(2.0)), dtype=np.float64) / 255.0
    base = base * (1 - erase[:, :, None]) + BONE[None, None, :] * erase[:, :, None]

    out = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    out.paste(layer, (0, 0), layer)
    return out


def underside(delta_deg, base_slope_dx_dy=1.999):
    """Return (x0,y0,dx,dy) of the lid's underside after rotation."""
    th0 = np.arctan2(1.0, base_slope_dx_dy)          # original rise angle
    th = th0 + np.radians(delta_deg)
    return PIVOT[0], PIVOT[1], np.cos(th), -np.sin(th)


def draw_prop(im, delta_deg, foot_x, length=1252, width=34):
    """Redraw the lid prop from the rim up to the lid's new underside."""
    x0, y0, dx, dy = underside(delta_deg)
    foot = np.array([foot_x, RIM_Y + 4], dtype=np.float64)
    best = None
    for t in np.arange(200, 4000, 2.0):
        p = np.array([x0 + dx*t, y0 + dy*t])
        if abs(np.linalg.norm(p - foot) - length) < 3.0:
            best = p
    if best is None:
        return im
    d = ImageDraw.Draw(im)
    n = np.array([-(best - foot)[1], (best - foot)[0]])
    n = n / np.linalg.norm(n)
    top_w, bot_w = width * 0.55, width * 0.75
    quad = [tuple(best + n*top_w), tuple(best - n*top_w),
            tuple(foot - n*bot_w), tuple(foot + n*bot_w)]
    d.polygon(quad, fill=(20, 18, 16))
    d.line([tuple(best + n*top_w*0.45), tuple(foot + n*bot_w*0.45)], fill=(74, 68, 62), width=5)
    return im

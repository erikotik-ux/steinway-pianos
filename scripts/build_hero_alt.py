"""Cut the two editorial hero elements out of their studio backgrounds.

The alternate hero frames an empty centre with photography on the left and the
right. That centre has to be the page's own #f6f5f3 -- not a colour the model
approximated, and not an image boundary -- so neither element may carry any
background with it.

So each plate is reduced to a true alpha cutout: the studio sweep is flood
filled from the borders and discarded, leaving only the instrument and the
figure. What lands on the page is a transparent PNG over the hero's own CSS
background, which means there is no rectangle, no gradient, no seam and nothing
to colour match. The centre is literally untouched page.

The subjects themselves are used exactly as generated: nothing is redrawn,
retouched or recoloured. What happens here is framing and separation only --
cropping, one horizontal flip, and dissolves on the edges the camera's own
frame cut through.
"""
import os

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage as nd

SRC_DIR = "images/backgorund-images"

# A region enclosed by the subject counts as background below this much local
# texture; the sweep sits near 1.2, the instrument's interior at 5 and above.
SMOOTH_MAX = 2.5

# (source, output, side, flood threshold, source crop, mirror, inner drop)
# The source crop runs before anything else and is pure framing: it drops
# columns at the outer edge, which the element is cut off by anyway. It is used
# to lose a fragment of gold lettering on the left plate's fallboard, since the
# brief rules out branding in this photography.
JOBS = [
    ("kb-left-a.jpg",   "hero-alt-piano.webp",   "left",  120, (210, 0, None, None), False, 0.0),
    # Mirrored: in the source the keyboard runs off to the RIGHT of the player,
    # so anchored to the right edge it would turn the pianist's back to the
    # headline with the instrument behind them. Flipped, the figure sits at the
    # edge and the keys run inward, which is what lets the two sides read as one
    # instrument continuing behind the type. A flip, not a retouch: nothing in
    # the photograph is altered.
    # `inner drop` sheds the far end of the keyboard once mirrored. Without it the
    # element is mostly keys, and scaling it to leave a large centre would push
    # the figure almost entirely off the page.
    ("pi-right-f.jpg",  "hero-alt-pianist.webp", "right", 150, None, True, 0.32),
]


def cutout(src, thr, crop):
    """Background -> transparent.

    A plain threshold would punch holes in the subject wherever it is bright --
    the naturals, a cuff, a highlight on the ebony -- so brightness alone is not
    enough. Instead the light pixels are labelled into connected regions at full
    resolution and only those touching the frame edge are treated as background.
    That keeps enclosed bright detail (keys, gold plate, soundboard) and still
    reaches the pockets between the legs, which connect to the sweep through
    gaps too thin to survive downsampling.

    A region fully enclosed by the subject is background only if it is also
    smooth: the studio sweep has almost no local texture, while the instrument's
    interior does. That distinction is what separates a glimpse of the backdrop
    from the plate behind the strings.
    """
    im = Image.open(os.path.join(SRC_DIR, src)).convert("RGB")
    if crop:
        l, t, r, b = crop
        im = im.crop((l or 0, t or 0, r or im.width, b or im.height))
    lum = np.asarray(im.convert("L"), dtype=np.float64)
    light = lum > thr
    lbl, n = nd.label(light)

    edge = np.concatenate([lbl[0], lbl[-1], lbl[:, 0], lbl[:, -1]])
    bg_labels = set(np.unique(edge)) - {0}

    mean = nd.uniform_filter(lum, 11)
    var = nd.uniform_filter(lum * lum, 11) - mean * mean
    std = np.sqrt(np.clip(var, 0, None))
    idx = np.arange(1, n + 1)
    smooth = nd.mean(std, lbl, idx)
    for lab, sd in zip(idx, smooth):
        if sd < SMOOTH_MAX:
            bg_labels.add(lab)

    bg = np.isin(lbl, list(bg_labels))
    alpha = Image.fromarray(np.where(bg, 0, 255).astype(np.uint8))
    # a touch of feathering so the silhouette does not alias against the page
    alpha = alpha.filter(ImageFilter.GaussianBlur(1.4))
    out = im.convert("RGBA")
    out.putalpha(alpha)
    return out


def trim(img, half, pad=24):
    """Crop to the subject.

    The bounding box comes from the largest connected opaque region, not from
    every opaque pixel: both plates carry a faint duplicate of the subject near
    the far edge, and a naive box would stretch to include it. Taking the
    largest component ignores those without assuming which half the subject
    happens to occupy -- an earlier version restricted the box to the outer half
    and sliced straight through the pianist's head.

    The edge the subject runs off is kept flush so it still bleeds off the page.
    """
    a = np.asarray(img)[:, :, 3]
    h, w = a.shape
    lbl, n = nd.label(a > 40)
    if n:
        sizes = nd.sum(np.ones_like(lbl), lbl, range(1, n + 1))
        keep = int(np.argmax(sizes)) + 1
        solid = lbl == keep
    else:
        solid = a > 40
    cols = np.where(solid.any(axis=0))[0]
    rows = np.where(solid.any(axis=1))[0]
    x0, x1 = max(cols.min() - pad, 0), min(cols.max() + pad + 1, w)
    y0, y1 = max(rows.min() - pad, 0), min(rows.max() + pad + 1, h)
    if half == "left":
        x0 = 0
    else:
        x1 = w
    return img.crop((x0, y0, x1, y1))


def fade_bottom(img, frac=0.14):
    """Dissolve the element's lower edge into the page.

    The left fragment sits clear of the hero's bottom so its keyboard lines up
    with the pianist's, which leaves the crop its own frame made across the keys
    hanging in the middle of the page. Ramping it out avoids that hard line.
    """
    a = np.asarray(img).copy()
    h = a.shape[0]
    n = max(int(h * frac), 1)
    t = np.linspace(1.0, 0.0, n)
    ramp = t * t * (3 - 2 * t)
    a[h - n:, :, 3] = (a[h - n:, :, 3] * ramp[:, None]).astype(np.uint8)
    return Image.fromarray(a)


def fade_inner_edge(img, side, frac=0.10):
    """Dissolve the edge where the instrument was cut by its own frame.

    Mirroring puts the pianist's keyboard end -- which ran off the source's
    frame -- facing the headline. Left as-is that is a hard vertical slice
    through the keys. Ramping the alpha out over the last stretch lets it fade
    into the page instead, the same dissolve the main hero uses at its bottom.
    """
    a = np.asarray(img).copy()
    w = a.shape[1]
    n = max(int(w * frac), 1)
    t = np.linspace(0.0, 1.0, n)
    ramp = (t * t * (3 - 2 * t))
    if side == "right":            # element sits at the right; inner edge is its left
        a[:, :n, 3] = (a[:, :n, 3] * ramp[None, :]).astype(np.uint8)
    else:
        a[:, w - n:, 3] = (a[:, w - n:, 3] * ramp[::-1][None, :]).astype(np.uint8)
    return Image.fromarray(a)


for src, dst, half, thr, crop, mirror, inner_drop in JOBS:
    img = trim(cutout(src, thr, crop), half)
    if mirror:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if inner_drop:
        n = int(img.width * inner_drop)
        img = img.crop((n, 0, img.width, img.height)) if half == "right"             else img.crop((0, 0, img.width - n, img.height))
    if mirror or inner_drop:
        img = fade_inner_edge(img, half)
    if half == "left":
        img = fade_bottom(img)
    path = os.path.join(SRC_DIR, dst)
    img.save(path, quality=88, method=6)   # WebP keeps the alpha at a tenth of PNG
    a = np.asarray(img)[:, :, 3]
    print("%-22s %sx%s  %.0f KB   %.0f%% opaque"
          % (dst, img.width, img.height, os.path.getsize(path) / 1024,
             100 * (a > 200).mean()))

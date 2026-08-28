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
retouched or recoloured, only separated from the ground they were shot on.
"""
import os

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage as nd

SRC_DIR = "images/backgorund-images"

# A region enclosed by the subject counts as background below this much local
# texture; the sweep sits near 1.2, the instrument's interior at 5 and above.
SMOOTH_MAX = 2.5

# (source, output, which half the subject occupies, flood threshold)
JOBS = [
    ("alt-left-a.jpg",  "hero-alt-piano.webp",   "left",  120),
    ("alt-right-b.jpg", "hero-alt-pianist.webp", "right", 150),
]


def cutout(src, thr):
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
    """Crop to the subject, ignoring anything in the opposite half.

    Both plates carry a faint duplicate of the subject near the far edge; the
    half restriction drops it instead of letting it stretch the bounding box.
    """
    a = np.asarray(img)[:, :, 3]
    h, w = a.shape
    solid = a > 40
    if half == "left":
        solid[:, w // 2:] = False
    else:
        solid[:, :w // 2] = False
    cols = np.where(solid.any(axis=0))[0]
    rows = np.where(solid.any(axis=1))[0]
    x0, x1 = max(cols.min() - pad, 0), min(cols.max() + pad + 1, w)
    y0, y1 = max(rows.min() - pad, 0), min(rows.max() + pad + 1, h)
    # keep the edge the subject runs off flush, so it still bleeds off the page
    if half == "left":
        x0 = 0
    else:
        x1 = w
    return img.crop((x0, y0, x1, y1))


for src, dst, half, thr in JOBS:
    img = trim(cutout(src, thr), half)
    path = os.path.join(SRC_DIR, dst)
    img.save(path, quality=88, method=6)   # WebP keeps the alpha at a tenth of PNG
    a = np.asarray(img)[:, :, 3]
    print("%-22s %sx%s  %.0f KB   %.0f%% opaque"
          % (dst, img.width, img.height, os.path.getsize(path) / 1024,
             100 * (a > 200).mean()))

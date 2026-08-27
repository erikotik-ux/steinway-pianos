"""Flat-field correct the hero image so its studio backdrop reads exactly #f6f5f3.

The generated backdrop is warm cream and uneven left-to-right. We model the
backdrop as a smooth low-frequency field sampled from the clean band above the
keys, extrapolate it downward, and divide it out so the backdrop lands on the
target while the keys keep most of their own tone.
"""
import sys
import numpy as np
from PIL import Image, ImageFilter

SRC, DST = sys.argv[1], sys.argv[2]
TARGET = np.array([0xF6, 0xF5, 0xF3], dtype=np.float64)

im = Image.open(SRC).convert("RGB")
w, h = im.size
a = np.asarray(im, dtype=np.float64)

# Clean backdrop band (above the fallboard / keys).
band_top, band_bot = int(h * 0.02), int(h * 0.50)

# Build a backdrop-only plate first: hold the clean band's edge rows over the
# keys and the piano body, so blurring can never drag dark pixels into the
# illumination estimate.
plate = a.copy()
plate[band_bot:, :, :] = a[band_bot - 1: band_bot, :, :]
plate[:band_top, :, :] = a[band_top: band_top + 1, :, :]

# Heavy blur -> low-frequency illumination + colour field.
field = np.asarray(
    Image.fromarray(plate.astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius=w / 18.0)
    ),
    dtype=np.float64,
)

gain = TARGET[None, None, :] / np.clip(field, 1.0, None)
gain = np.clip(gain, 0.78, 1.34)

# Full correction across the backdrop, easing to a gentler global white balance
# over the keys so the ivories don't go flat grey.
y = np.arange(h, dtype=np.float64)[:, None, None]
ramp_a, ramp_b = h * 0.52, h * 0.64
wgt = np.clip((ramp_b - y) / (ramp_b - ramp_a), 0.0, 1.0)
wgt = 0.42 + 0.58 * wgt

out = np.clip(a * (1.0 + (gain - 1.0) * wgt), 0, 255).astype(np.uint8)
Image.fromarray(out).save(DST, quality=92, optimize=True, progressive=True)

chk = np.asarray(Image.open(DST).convert("RGB"), dtype=np.float64)
for name, (fx, fy) in {
    "top-left": (0.08, 0.10), "top-center": (0.50, 0.12), "top-right": (0.92, 0.10),
    "mid-left": (0.10, 0.40), "mid-right": (0.90, 0.40), "above-keys": (0.50, 0.50),
}.items():
    print(name, chk[int(h * fy), int(w * fx)].round().astype(int).tolist())

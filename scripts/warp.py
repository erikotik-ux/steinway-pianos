"""Projective transform helpers for placing flat artwork onto an angled surface.

The perspective plate shows the keyboard and fallboard receding, so the exact
keyboard drawn by scripts/keyboard.py has to be laid into the picture on the
same plane rather than pasted square.
"""
import numpy as np
from PIL import Image


def coeffs(dst_quad, src_size):
    """PIL transform coefficients mapping destination pixels back into a
    src_size rectangle, given where that rectangle's corners land."""
    w, h = src_size
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    A, B = [], []
    for (dx, dy), (sx, sy) in zip(dst_quad, src):
        A.append([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy]); B.append(sx)
        A.append([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy]); B.append(sy)
    return np.linalg.solve(np.asarray(A, float), np.asarray(B, float))


def place(base, art, dst_quad, feather=2.0):
    """Warp `art` onto `base` so its corners land on dst_quad (tl, tr, br, bl)."""
    c = coeffs(dst_quad, art.size)
    layer = art.convert("RGBA").transform(base.size, Image.PERSPECTIVE, c, Image.BICUBIC)
    out = base.convert("RGBA")
    out.alpha_composite(layer)
    return out.convert("RGB")

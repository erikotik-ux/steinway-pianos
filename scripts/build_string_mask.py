"""Build the hero's bottom mask: an irregular ending line with a few thin
string-like fragments running on below it. Run from the repo root."""
import numpy as np
from PIL import Image

MW, MH = 1920, 1080          # mask resolution; stretched to the hero by mask-size
FADE_TOP = 0.125             # dissolve occupies the bottom 12.5%
BAND = 0.030                 # thickness of the photograph's own ending ramp

OUT = 'images/backgorund-images/hero-string-mask.png'

def smooth(t):
    t = np.clip(t, 0, 1)
    return t*t*(3-2*t)

def build():
    ys, xs = np.mgrid[0:MH, 0:MW]
    up = (MH-1-ys)/MH                       # 0 at the bottom edge, 1 at the top
    xf = xs/MW

    # --- irregular base edge: the photograph's own ending line -------------
    wob = (0.016*np.sin(xf*np.pi*2.1 + 0.6)
         + 0.010*np.sin(xf*np.pi*3.7 + 2.1)
         + 0.006*np.sin(xf*np.pi*6.3 + 4.0))
    tilt = 0.018*smooth((xf-0.15)/0.75)      # pianist side ends higher than the piano side
    edge = FADE_TOP*0.60 - tilt + wob        # top of the ending ramp
    # a narrow ramp AT the line, so everything below it is genuinely cleared
    # and the fragments have empty ground to reach down into
    base = smooth((up - (edge - BAND))/BAND)

    # --- string fragments -------------------------------------------------
    # few, thin, weighted towards the piano/keyboard side
    rng = np.random.default_rng(11)
    cols = []
    for x0, x1, n in ((0.14, 0.42, 3), (0.47, 0.71, 4), (0.72, 0.98, 7)):
        for i in range(n):
            cols.append(x0 + (x1-x0)*((i+0.5)/n) + rng.uniform(-0.014, 0.014))
    edge_x = edge[0]
    strings = np.zeros_like(base)
    for cxf in cols:
        w = rng.uniform(2.4, 5.0)                     # px at mask scale
        reach = rng.uniform(0.030, 0.062)             # how far below the line it runs
        peak = rng.uniform(0.80, 1.0)
        e = float(np.interp(cxf*MW, np.arange(MW), edge_x))
        top = e                                       # starts inside the ramp
        bot = max(e - BAND - reach, 0.010)            # always ends above the bottom row
        d = np.abs(xs - cxf*MW)
        prof = np.clip(1 - d/w, 0, 1)                 # narrow line, soft edges
        span = np.clip((top - up)/(top - bot), 0, 1)  # 0 at its top, 1 at its end
        taper = (1-span)**1.4                         # thins and fades downward
        strings = np.maximum(strings, peak*prof*taper*(up < top)*(up > bot))

    m = np.clip(np.maximum(base, strings), 0, 1)
    m[up < 0.006] = 0.0                               # resolve fully at the very edge
    return m

if __name__ == '__main__':
    m = build()
    print('bottom row      : max %.4f  (0 = resolves fully into #f6f5f3)'%m[-1].max())
    for pct in (10,12,14,16,20):
        r = MH-1-int(pct/100*MH)
        print('  %2d%% up: min %.3f (1.0 = photo fully intact)'%(pct, m[r].min()))
    edge=[]
    for x in range(0, MW, MW//16):
        col=m[:,x]; idx=np.where(col>=0.5)[0]
        edge.append(round(100*(MH-1-idx.max())/MH,1) if len(idx) else 0)
    print('  50%% edge across width:', edge)
    a = (m*255).round().astype(np.uint8)
    la = np.dstack([np.full(a.shape, 255, np.uint8), a])   # white, alpha = mask
    Image.fromarray(la, 'LA').save(OUT, optimize=True)
    print('wrote', OUT)

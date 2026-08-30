"""Build the hero's bottom mask: the photograph fragmenting into a fine
soundwave before it dissolves into #f6f5f3. Run from the repo root.

The mask is the photograph's own aspect (16:9), so `mask-size: cover` crops it
exactly the way `background-size: cover` crops the picture -- the waveform stays
welded to the subjects instead of sliding across them as the viewport changes.

Density is not decorative: it is read out of the photograph. Columns are weighted
by how much ink the picture actually has just above the break, so the fragments
come out of the pianist's jacket on the left and the piano case on the right and
leave the bright middle -- which is already close to page bone -- clean.
"""
import numpy as np
from PIL import Image

SRC = 'images/backgorund-images/hero-alt-scene.webp'
OUT = 'images/backgorund-images/hero-wave-mask.png'

MW, MH = 1920, 1080          # 16:9, matching the photograph
BAND   = 0.022               # ramp at the break, as a fraction of height
HMAX   = 0.105               # deepest a fragment normally runs below the ramp
PITCH  = 3.4                 # average spacing of the fragments, px at mask scale

def smooth(t):
    t = np.clip(t, 0, 1)
    return t*t*(3-2*t)

def ink_profile():
    """Per-column ink of the photograph in the band the fragments grow from."""
    im = Image.open(SRC).convert('RGB').resize((MW, MH), Image.LANCZOS)
    a = np.asarray(im, float)
    lum = 0.2126*a[..., 0] + 0.7152*a[..., 1] + 0.0722*a[..., 2]
    r0, r1 = int(MH*(1-0.24)), int(MH*(1-0.09))
    col = 1.0 - lum[r0:r1].mean(axis=0)/255.0
    k = np.hanning(121); k /= k.sum()                 # broad, so it reads as regions
    col = np.convolve(np.pad(col, 60, mode='edge'), k, 'same')[60:-60]
    return np.clip((col - 0.06)/0.80, 0, 1)           # bone -> 0, case/jacket -> 1

def build():
    ys, xs = np.mgrid[0:MH, 0:MW]
    up = (MH-1-ys)/MH                                 # 0 at the bottom edge, 1 at the top
    xf = xs/MW
    ink = ink_profile()

    # --- where the solid photograph ends ----------------------------------
    # a gentle wander plus a tilt: the jacket side breaks a little higher than
    # the piano side, whose keys run deeper into the frame.
    wob = (0.0090*np.sin(xf*np.pi*2.3 + 0.7)
         + 0.0055*np.sin(xf*np.pi*4.1 + 2.4)
         + 0.0030*np.sin(xf*np.pi*7.9 + 4.6))
    tilt = 0.026*smooth((xf-0.10)/0.85)
    topline = 0.158 - tilt + wob
    base = smooth((up - (topline - BAND))/BAND)       # 1 above the break, 0 below it

    rng = np.random.default_rng(23)
    top_x = topline[0]

    # --- the waveform envelope --------------------------------------------
    # low-frequency noise so the fragments cluster into louder and quieter
    # passages instead of reading as an even comb.
    xg = np.arange(MW)/MW
    env = np.zeros(MW)
    for f, amp in ((2.0, 0.55), (3.7, 0.30), (6.1, 0.22), (11.3, 0.13), (19.7, 0.08)):
        env += amp*np.sin(2*np.pi*f*xg + rng.uniform(0, 2*np.pi))
    env = 0.5 + 0.5*np.tanh(env)                      # 0..1, smooth

    frags = np.zeros_like(base)
    x = PITCH*rng.uniform(0.3, 1.0)
    while x < MW:
        i = int(min(x, MW-1))
        ik = ink[i]
        if ik > 0.05:
            w = rng.uniform(1.1, 2.8)                             # hair-thin
            # height: ink sets the ceiling, the envelope and per-line noise
            # roughen it. A few run deeper, but nothing spikes.
            hn = rng.random()**1.55
            h = HMAX*(0.30 + 0.70*ik**0.75)*(0.28 + 0.72*env[i])*(0.22 + 1.05*hn)
            peak = (0.62 + 0.38*rng.random())*(0.45 + 0.55*ik)
            # a few stragglers carry on much further at very low opacity, so the
            # waveform thins out into the page instead of stopping on a line
            if rng.random() < 0.13:
                h *= rng.uniform(1.5, 2.3)
                peak *= rng.uniform(0.30, 0.55)
            top = float(np.interp(x, np.arange(MW), top_x))
            bot = max(top - BAND - h, 0.008)
            d = np.abs(xs - x)
            prof = np.clip(1 - d/w, 0, 1)
            span = np.clip((top - up)/(top - bot), 0, 1)
            taper = (1-span)**1.5                                 # thins and fades downward
            frags = np.maximum(frags, peak*prof*taper*(up < top)*(up > bot))
        x += PITCH*rng.uniform(0.55, 1.6)                         # jittered pitch

    m = np.clip(np.maximum(base, frags), 0, 1)
    m[up < 0.004] = 0.0                               # resolve fully at the very edge
    return m

if __name__ == '__main__':
    m = build()
    print('bottom row      : max %.4f  (0 = resolves fully into #f6f5f3)'%m[-1].max())
    for pct in (17, 19, 21, 24, 30):
        r = MH-1-int(pct/100*MH)
        print('  %2d%% up: min %.3f (1.0 = photo fully intact)'%(pct, m[r].min()))
    ink = ink_profile()
    print('  fragment reach / ink by region:')
    for x0, x1, lab in ((0.00,0.22,'jacket   '), (0.24,0.47,'bright   '),
                        (0.48,0.76,'keys+case'), (0.77,1.00,'case     ')):
        s = slice(int(x0*MW), int(x1*MW))
        sub = m[:, s]
        reach = [ (MH-1-np.where(sub[:,c]>0.06)[0].max())/MH for c in range(0, sub.shape[1], 7)
                  if (sub[:,c]>0.06).any() ]
        print('    %s ink %.2f   deepest fragment %.1f%% up'%(lab, ink[s].mean(),
              100*min(reach) if reach else 0))
    a = (m*255).round().astype(np.uint8)
    Image.fromarray(np.dstack([np.full(a.shape, 255, np.uint8), a]), 'LA').save(OUT, optimize=True)
    import os; print('wrote %s  %.1f KB'%(OUT, os.path.getsize(OUT)/1024))

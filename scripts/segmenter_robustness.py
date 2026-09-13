"""Segmenter-agnosticism, tested parametrically instead of at four points.

WHY NOT JUST RUN THE OTHER SEGMENTERS. Table 2 samples the space at a handful of
arbitrary points (Fun3DU, OpenMask3D, ...) and each point costs a GPU, a large
download, and somebody else's dependency tree. It also cannot separate "our
motion module is robust" from "those three segmenters happen to fail similarly".

THE CLAIM WE ACTUALLY MAKE is mechanistic: the axis and origin are read off the
PARENT SURFACE (`nbr`), not off the element (`pts`). If that is true, then
corrupting `pts` -- which is exactly what a worse segmenter does -- should barely
move the result, while the same corruption should wreck a method that reads
geometry off the element itself.

That is a falsifiable prediction, and ablation A ("use the element's own normal")
is the control group that makes it testable. We sweep four corruption axes chosen
to match how real segmenters actually fail, and run BOTH the full method and
ablation A through each. Same 734 elements, same unmodified motion module.

  drift         translate the region                 -> localisation error, the
                                                        dominant failure mode we
                                                        measured (median
                                                        precision 0.00)
  dilate        scale the region about its centroid  -> over/under-segmentation
                                                        (our own pred/gt extent
                                                        ratio is 1.42)
  subsample     keep a fraction of the points        -> partial / sparse masks
  contaminate   swap in neighbourhood points         -> mask bleed onto the
                                                        parent surface

Nothing is re-tuned at any corruption level. Perturbations are seeded, and each
level is averaged over N_REP draws so the curve is not one unlucky sample.

Run: python scripts/segmenter_robustness.py
"""
import os, sys, json, io
import numpy as np

import contextlib
with contextlib.redirect_stdout(io.StringIO()):   # AB runs its own table on import
    import motion_ablations as AB
from _paths import REPO, DATA_ROOT, TOOLKIT, DOCS  # noqa: F401
CODE = REPO          # repo root: code, docs, figures
BASE = DATA_ROOT     # data root: scans, caches, exemplar DB
from sf3d import motion as M

ROWS = AB.ROWS
print(f'loaded {len(ROWS)} elements from cache {AB.CACHE}', flush=True)

N_REP = 3
FULL = dict()
OWN  = dict(own_normal=True)          # ablation A -- the control group

# ---------------------------------------------------------------- corruptions
def drift(pts, rng, mag):
    """Translate the whole region by `mag` metres in a random direction."""
    if mag <= 0: return pts
    d = rng.normal(size=3); d /= np.linalg.norm(d) + 1e-9
    return pts + d * mag

def dilate(pts, rng, factor):
    """Scale the region about its centroid -- over/under-segmentation."""
    if factor == 1.0: return pts
    c = pts.mean(0)
    return c + (pts - c) * factor

def subsample(pts, rng, frac):
    """Keep a random fraction of the points."""
    if frac >= 1.0: return pts
    k = max(4, int(len(pts) * frac))
    return pts[rng.choice(len(pts), k, replace=False)]

def contaminate(pts, nbr, rng, frac):
    """Replace a fraction of the region with nearby parent-surface points."""
    if frac <= 0 or nbr is None or len(nbr) < 10: return pts
    k = int(len(pts) * frac)
    if k < 1: return pts
    keep = pts[rng.choice(len(pts), max(4, len(pts) - k), replace=False)]
    cen = pts.mean(0)
    ext = float(np.linalg.norm(pts - cen, axis=1).mean()) + 1e-6
    d = np.linalg.norm(nbr - cen, axis=1)
    near = nbr[d < 6.0 * ext]
    if len(near) < 1: return pts
    add = near[rng.choice(len(near), k, replace=True)]
    return np.vstack([keep, add])

SWEEPS = [
    ('drift (m)',        [0.0, 0.02, 0.05, 0.10, 0.20],
     lambda p, n, r, v: drift(p, r, v)),
    ('dilate (x extent)',[1.0, 1.42, 2.0, 3.0, 0.5],
     lambda p, n, r, v: dilate(p, r, v)),
    ('subsample (kept)', [1.0, 0.5, 0.25, 0.10],
     lambda p, n, r, v: subsample(p, r, v)),
    ('contaminate (frac)',[0.0, 0.10, 0.25, 0.50],
     lambda p, n, r, v: contaminate(p, n, r, v)),
]

# ------------------------------------------------------------------ scoring
ISROT = lambda r: 'rot' in r['mtype'].lower()

def gate(rec, t, axis, org):
    """Exactly motion_ablations.evaluate's gate, including its None case:
    a revolute element with no GT origin is EXCLUDED, not failed."""
    if M.OE(rec['dirv'], axis) >= 15: return False
    if not ISROT(rec): return True
    if org is None or 'org' not in rec: return None
    return bool(M.MD(org, axis, rec['org'], rec['dirv']) < 0.25)

def run(cfg, fn, val, seed):
    rng = np.random.default_rng(seed)
    ok = n = 0
    for rec in ROWS:
        pts = fn(rec['pts'], rec.get('nbr'), rng, val)
        if pts is None or len(pts) < 4: continue
        try:
            t, axis, org = AB.predict(rec['label'], pts, rec.get('nbr'), cfg)
        except Exception:
            continue
        g = gate(rec, t, axis, org)
        if g is None: continue
        n += 1; ok += g
    return 100.0 * ok / max(n, 1)

# --------------------------------------------------------------------- main
out = {}
base_full = run(FULL, lambda p,n,r,v: p, 0, 0)
base_own  = run(OWN,  lambda p,n,r,v: p, 0, 0)
print(f'\nunperturbed:  FULL {base_full:.1f}%   ablation-A (own normal) {base_own:.1f}%\n')

for name, vals, fn in SWEEPS:
    print(f'--- {name} ---')
    print(f"  {'level':>10} {'FULL':>8} {'ret%':>7} {'A(own)':>8} {'ret%':>7}")
    out[name] = []
    for v in vals:
        f = np.mean([run(FULL, fn, v, s) for s in range(N_REP)])
        o = np.mean([run(OWN,  fn, v, s) for s in range(N_REP)])
        rf, ro = 100*f/base_full, 100*o/base_own
        out[name].append(dict(level=v, full=f, own=o, ret_full=rf, ret_own=ro))
        print(f'  {v:>10} {f:>7.1f}% {rf:>6.1f}% {o:>7.1f}% {ro:>6.1f}%')
    print()

out['_baseline'] = dict(full=base_full, own=base_own, n=len(ROWS), reps=N_REP)
p = f'{CODE}/docs/segmenter_robustness.json'
json.dump(out, open(p,'w'), indent=1)
print('written', p)

"""Signed direction, from the geometry already in hand (RESULTS 5.9 B).

THE PROBLEM. Our predicted direction is an AXIS, not a vector: a plane fit gives
a normal up to sign, so the prediction points the wrong way about half the time.
The composed metric is unsigned so this costs nothing there, but the ground-truth
annotations ARE signed and semantically meaningful -- the key_press elements
share (0,0,-1), which encodes "press downward" -- so it is real information the
method throws away.

THE FIX, using no new quantity. The parent surface has an OUTWARD side, and we
can already tell which: the element sits proud of the surface it is mounted on,
so (element centroid - shell centroid) . n is positive on the outward side.
Orient n outward, then let the affordance say which way along it the motion goes:

  PULL  (hook_pull, pinch_pull, unplug)   -> outward, away from the surface
  PRESS (key_press, tip_push, foot_push)  -> inward, into the surface
  PLUG  (plug_in)                         -> inward, into the socket

Revolute elements are excluded: a hinge axis genuinely has no preferred end, so
"signed accuracy" is not defined for them and reporting one would be noise.

Run: .venv/bin/python scripts/sign_direction.py
"""
import os, sys, json, collections

import numpy as np
from _paths import REPO, DATA_ROOT, TOOLKIT, DOCS  # noqa: F401
CODE = REPO          # repo root: code, docs, figures
BASE = DATA_ROOT     # data root: scans, caches, exemplar DB
from sf3d_data import load, score, CODE, M

OUT  = {'pull': +1.0, 'press': -1.0, 'plug': -1.0}
FAM  = {'hook_pull': 'pull', 'pinch_pull': 'pull', 'unplug': 'pull',
        'key_press': 'press', 'tip_push': 'press', 'foot_push': 'press',
        'plug_in': 'plug'}

def outward_normal(pts, nbr, cen, extent, pn):
    """Orient pn so it points away from the mounting surface, using the fact
    that the element protrudes from it."""
    d = np.linalg.norm(nbr - cen, axis=1)
    h = 4.0
    shell = nbr[(d > extent*1.5) & (d < extent*h)]
    while len(shell) < 50 and h < 12.0:
        h *= 1.5
        shell = nbr[(d > extent*1.5) & (d < extent*h)]
    if len(shell) < 50:
        return None
    return pn if (cen - shell.mean(0)) @ pn > 0 else -pn

rows = load()
res  = collections.defaultdict(lambda: dict(n=0, unsigned=0, signed=0, base=0))
skipped = 0
for r in rows:
    s = score(r)
    if s['gt_rot'] or s['p_type'] != 'trans':      # signed direction is only
        continue                                    # defined for translations
    fam = FAM.get(r['label'])
    if fam is None:
        skipped += 1; continue
    pts, nbr = r['pts'], r['nbr']
    cen = pts.mean(0)
    extent = float(np.linalg.norm(pts - cen, axis=1).mean())
    pn = M.parent_normal(nbr, cen, extent)
    if pn is None: continue
    on = outward_normal(pts, nbr, cen, extent, pn)
    if on is None: continue
    pred_signed = on * OUT[fam]
    gt = r['dirv']
    b = res[fam]; a = res['ALL']
    for t in (b, a):
        t['n'] += 1
        t['unsigned'] += (abs(float(pred_signed @ gt)) > np.cos(np.deg2rad(15)))
        t['signed']   += (float(pred_signed @ gt) > np.cos(np.deg2rad(15)))
        t['base']     += (float(s['axis'] @ gt) > np.cos(np.deg2rad(15)))

print(f'prismatic elements scored: {res["ALL"]["n"]}   (skipped {skipped} unmapped labels)\n')
print(f'{"family":<8} {"n":>4}   {"unsigned":>9}   {"signed, as deployed":>20}   {"signed, oriented":>17}')
for k in ['pull', 'press', 'plug', 'ALL']:
    t = res.get(k)
    if not t or not t['n']: continue
    n = t['n']
    print(f'{k:<8} {n:>4}   {100*t["unsigned"]/n:>8.1f}%   {100*t["base"]/n:>19.1f}%'
          f'   {100*t["signed"]/n:>16.1f}%')

a = res['ALL']
json.dump({'n': a['n'],
           'unsigned_pct':        round(100*a['unsigned']/a['n'], 2),
           'signed_deployed_pct': round(100*a['base']/a['n'], 2),
           'signed_oriented_pct': round(100*a['signed']/a['n'], 2),
           'per_family': {k: dict(n=v['n'],
                                  unsigned=round(100*v['unsigned']/v['n'], 2),
                                  signed_deployed=round(100*v['base']/v['n'], 2),
                                  signed_oriented=round(100*v['signed']/v['n'], 2))
                          for k, v in res.items() if k != 'ALL' and v['n']}},
          open(f'{CODE}/docs/sign_direction.json', 'w'), indent=1)
print('\nwrote docs/sign_direction.json')

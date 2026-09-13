"""Stress-test the Articulate3D result before believing it.

Checks:
  1. handle vs no-handle -- the "farther" rule needs the interactable element,
     and USDNet's interactable checkpoint is NOT released, so the hybrid may not
     have one. Score both, and the best handle-free rule.
  2. scene-level bootstrap CI, matching the convention used for every other
     adopted change in this project.
  3. mask degradation -- erode/jitter the GT part mask to simulate a predicted
     mask, since 88.5% is an ORACLE number and the hybrid will use USDNet's masks.
"""
import sys, os, glob, json
import numpy as np, h5py
from scipy.spatial import cKDTree

from _paths import REPO, DATA_ROOT, TOOLKIT, DOCS, A3D_VAL  # noqa: F401
CODE = REPO          # repo root: code, docs, figures
BASE = DATA_ROOT     # data root: scans, caches, exemplar DB
from sf3d import motion as M

VAL = A3D_VAL   # $ARTICULATE3D_ROOT; see README "Data layout"
rng = np.random.default_rng(0)

def degrade(part, frac, rs):
    """Simulate a predicted mask: keep a random contiguous-ish subset."""
    if frac >= 1.0: return part
    k = max(30, int(len(part)*frac))
    c = part[rs.integers(len(part))]
    d = np.linalg.norm(part - c, axis=1)
    return part[np.argsort(d)[:k]]           # a blob, like a partial detection

def origin_from_panel(elem, part, nbr, pn, rule):
    cen = elem.mean(0)
    ax = np.cross(M.G, pn); n = np.linalg.norm(ax)
    if n < 1e-6: return cen
    ax /= n
    proj = (part - cen) @ ax
    far, near = cen + ax*proj.max(), cen + ax*proj.min()
    if rule == 'farther':
        return far if abs(proj.max()) >= abs(proj.min()) else near
    sf = M.corner_score(nbr, far, pn); sn = M.corner_score(nbr, near, pn)
    return far if sf >= sn else near

recs = []
for f in sorted(glob.glob(f'{VAL}/*.npy')):
    sid = os.path.basename(f)[:-4]
    a = np.load(f); xyz = a[:, :3].astype(np.float64); inst = a[:, 10].astype(np.int64)
    tree = cKDTree(xyz)
    with h5py.File(f'{VAL}/{sid}_articulation.h5','r') as h:
        for key in h:
            g = h[key]
            sel = np.flatnonzero(inst == int(key))
            if len(sel) < 20: continue
            part = xyz[sel]
            im = np.array(g['inter_mask'], bool)
            has_h = im.shape[0] == len(sel) and im.sum() >= 20
            elem = part[im] if has_h else part
            ga = np.array(g['axis'], float)
            if np.linalg.norm(ga) < 1e-6: continue
            ga /= np.linalg.norm(ga); go = np.array(g['origin'], float)
            kind = 'rot' if int(np.array(g['sem_id'])) == 1 else 'trans'
            cen = elem.mean(0)
            nbr = xyz[tree.query_ball_point(cen, 1.1)]
            if len(nbr) < 50: continue
            r = dict(scene=sid, kind=kind, has_h=bool(has_h))
            for frac in (1.0, 0.7, 0.5):
                rs = np.random.default_rng(abs(hash((sid,key)))%2**31)
                p = degrade(part, frac, rs)
                pn = M.plane_normal_robust(p)
                oe = M.OE(ga, pn if kind=='trans' else M.G)
                r[f'oe{frac}'] = oe
                if kind == 'rot':
                    for rule in ('farther','corner'):
                        org = origin_from_panel(elem, p, nbr, pn, rule)
                        r[f'md{frac}_{rule}'] = M.MD(org, M.G, go, ga)
            recs.append(r)

def gate(R, frac, rule):
    ok = []
    for r in R:
        if r['kind'] == 'trans': ok.append(r[f'oe{frac}'] < 15)
        else: ok.append(r[f'oe{frac}'] < 15 and r[f'md{frac}_{rule}'] < 0.25)
    return 100*np.mean(ok) if ok else float('nan')

print(f'n={len(recs)}  with handle: {sum(r["has_h"] for r in recs)}  '
      f'without: {sum(not r["has_h"] for r in recs)}\n')

print('1. ORIGIN RULE  x  HANDLE AVAILABILITY   (GT masks)')
print(f'   {"subset":<18} {"farther":>9} {"corner":>9}')
for name, R in (('with handle', [r for r in recs if r['has_h']]),
                ('without handle', [r for r in recs if not r['has_h']]),
                ('all', recs)):
    print(f'   {name:<18} {gate(R,1.0,"farther"):8.1f}% {gate(R,1.0,"corner"):8.1f}%')

print('\n2. SCENE BOOTSTRAP  (with-handle subset, farther rule)')
H = [r for r in recs if r['has_h']]
scenes = sorted({r['scene'] for r in H})
bs = []
for _ in range(2000):
    s = set(rng.choice(scenes, len(scenes), replace=True))
    sub = [r for r in H if r['scene'] in s]
    if sub: bs.append(gate(sub, 1.0, 'farther'))
print(f'   gate {np.mean(bs):.1f}%   95% CI [{np.percentile(bs,2.5):.1f}, {np.percentile(bs,97.5):.1f}]')

print('\n3. MASK DEGRADATION  (how much does an imperfect part mask cost?)')
print(f'   {"kept":<8} {"with handle":>13} {"all":>10}')
for frac in (1.0, 0.7, 0.5):
    print(f'   {int(frac*100):>3}%     {gate(H,frac,"farther"):12.1f}% {gate(recs,frac,"farther"):9.1f}%')

json.dump(recs, open(f'{CODE}/docs/articulate3d_stress.json','w'), indent=1, default=float)
print('\nwritten docs/articulate3d_stress.json')

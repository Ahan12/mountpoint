"""Held-out validation of the combined edge rule, at the full motion gate.

The rule: trust the ergonomic prior (hinge = panel edge farther from the element)
only when the element is decisively off-centre within the panel; otherwise the
prior carries no information and the corner score decides.

    asym = | |proj_max| - |proj_min| | / span      in [0,1], unit-free
    edge = ergonomic if asym >= t else corner

t is SELECTED ON DEV SCENES and reported on held-out TEST scenes, over 20 random
scene splits -- the protocol used for every other adopted change in this project.
Everything upstream of the origin (type, axis) is byte-identical to the deployed
pipeline; only the edge decision changes.

Run: python scripts/sf3d_edge_heldout.py
"""
import sys, io, contextlib, json
import numpy as np

sys.path.insert(0, f'{CODE}/scripts'); sys.path.insert(0, f'{CODE}/scenefun3d 2')
with contextlib.redirect_stdout(io.StringIO()):
    import motion_ablations as AB
from _paths import REPO, DATA_ROOT, TOOLKIT, DOCS  # noqa: F401
CODE = REPO          # repo root: code, docs, figures
BASE = DATA_ROOT     # data root: scans, caches, exemplar DB
from sf3d import motion as M

GRID = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01]   # 1.01 == pure corner
ISROT = lambda r: 'rot' in r['mtype'].lower()

# ---- precompute, per element, the gate outcome for every threshold ----
cache = []
for r in AB.ROWS:
    t, axis, org_dep = AB.predict(r['label'], r['pts'], r['nbr'], dict())
    base_ok = (t == r['mtype']) and M.OE(r['dirv'], axis) < 15
    e = dict(visit=r['visit'], gt_rot=ISROT(r), base_ok=bool(base_ok))
    if not ISROT(r):
        e['gate'] = {th: bool(base_ok) for th in GRID}
        cache.append(e); continue
    if 'org' not in r or not base_ok:
        e['gate'] = {th: False for th in GRID}
        cache.append(e); continue
    # knobs: origin is the element itself, untouched by the edge rule
    if r['label'] in M.KNOB:
        ok = M.MD(r['pts'].mean(0), axis, r['org'], r['dirv']) < 0.25
        e['gate'] = {th: bool(ok) for th in GRID}
        cache.append(e); continue
    pts, nbr = r['pts'], r['nbr']
    cen = pts.mean(0); extent = float(np.linalg.norm(pts - cen, axis=1).mean())
    pn = M.parent_normal(nbr, cen, extent)
    src = None
    if pn is not None:
        src = M.panel_component(nbr, cen, extent, hi=40)
        if src is None:
            src = M.cop_points(nbr, cen, extent, hi=40)
    if pn is None or src is None:
        ok = M.MD(cen, axis, r['org'], r['dirv']) < 0.25
        e['gate'] = {th: bool(ok) for th in GRID}
        cache.append(e); continue
    ax = np.cross(axis, pn); n = np.linalg.norm(ax)
    if n < 1e-6:
        ok = M.MD(cen, axis, r['org'], r['dirv']) < 0.25
        e['gate'] = {th: bool(ok) for th in GRID}
        cache.append(e); continue
    ax /= n
    proj = (src - cen) @ ax
    span = float(proj.max() - proj.min())
    far, near = cen + ax*proj.max(), cen + ax*proj.min()
    ok_f = M.MD(far, axis, r['org'], r['dirv']) < 0.25
    ok_n = M.MD(near, axis, r['org'], r['dirv']) < 0.25
    sf = M.corner_score(nbr, far, pn); sn = M.corner_score(nbr, near, pn)
    corner_ok = ok_f if sf >= sn else ok_n
    ergo_ok   = ok_f if abs(proj.max()) >= abs(proj.min()) else ok_n
    asym = abs(abs(proj.max()) - abs(proj.min())) / span if span > 1e-6 else 0.0
    e['gate'] = {th: bool(ergo_ok if asym >= th else corner_ok) for th in GRID}
    cache.append(e)

scenes = sorted({e['visit'] for e in cache})
def gate(sub, th): return 100*np.mean([e['gate'][th] for e in sub]) if sub else np.nan

print(f'{len(cache)} elements, {len(scenes)} scenes\n')
print('FULL-SET gate by threshold (in-sample, for orientation only)')
for th in GRID:
    tag = '  = pure corner (deployed)' if th > 1 else ('  = pure ergonomic' if th == 0 else '')
    print(f'   t={th:<5} {gate(cache, th):5.2f}%{tag}')

rng = np.random.default_rng(0)
picks, deltas = [], []
for s in range(20):
    perm = list(rng.permutation(scenes))
    dev, test = set(perm[:len(perm)//2]), set(perm[len(perm)//2:])
    D = [e for e in cache if e['visit'] in dev]
    T = [e for e in cache if e['visit'] in test]
    best = max([th for th in GRID if th <= 1], key=lambda th: gate(D, th))
    picks.append(best)
    deltas.append(gate(T, best) - gate(T, 1.01))
deltas = np.array(deltas)
print(f'\nHELD-OUT, 20 random 50/50 scene splits')
print(f'   threshold chosen on DEV: { {t: picks.count(t) for t in sorted(set(picks))} }')
print(f'   TEST delta vs deployed:  {deltas.mean():+.2f} pts   '
      f'[{deltas.min():+.2f}, {deltas.max():+.2f}]')
print(f'   improves on TEST in {100*np.mean(deltas>0):.0f}% of splits')

MODE = max(set(picks), key=picks.count)
print(f'\nPAIRED SCENE BOOTSTRAP at the modal threshold t={MODE}')
bs = []
for _ in range(2000):
    s = set(rng.choice(scenes, len(scenes), replace=True))
    sub = [e for e in cache if e['visit'] in s]
    if sub: bs.append(gate(sub, MODE) - gate(sub, 1.01))
bs = np.array(bs)
print(f'   delta {bs.mean():+.2f} pts   95% CI [{np.percentile(bs,2.5):+.2f}, '
      f'{np.percentile(bs,97.5):+.2f}]   positive in {100*np.mean(bs>0):.0f}% of resamples')
print(f'\n   deployed gate {gate(cache,1.01):.2f}%  ->  combined gate {gate(cache,MODE):.2f}%')
json.dump(dict(grid={str(k): gate(cache,k) for k in GRID}, picks=picks,
               heldout_delta=float(deltas.mean()), modal=float(MODE),
               ci=[float(np.percentile(bs,2.5)), float(np.percentile(bs,97.5))]),
          open(f'{CODE}/docs/sf3d_edge_heldout.json','w'), indent=1)
print('\nwritten docs/sf3d_edge_heldout.json')

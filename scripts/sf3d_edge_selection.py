"""Close the +22.6 pt edge-selection gap on SceneFun3D hinge origin.

Deployed corner score picks the worse of the two panel edges 42.5% of the time,
and perfect selection would take hinge origin 55.9% -> 78.5%. Two hypotheses,
neither introducing a dataset-specific constant:

  PANEL   the recovered panel is the whole cabinet FRONT, not the one drawer, so
          its edges are not the element's edges. Its median span is 0.957 m.
          Fix: make the flood-fill link distance RELATIVE TO THE SCAN's own point
          spacing instead of a fixed 2 cm, so a drawer seam breaks connectivity
          rather than being bridged. Same adaptive principle already used for the
          shell radius.

  SELECT  neither signal works alone (corner 55.9%, ergonomic 50.0%). Combine
          them: use the ergonomic prior only when the element is decisively
          off-centre, where that prior actually carries information, and fall
          back to the corner score when it does not.

Run: python scripts/sf3d_edge_selection.py
"""
import sys, io, contextlib, json
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, f'{CODE}/scripts'); sys.path.insert(0, f'{CODE}/scenefun3d 2')
with contextlib.redirect_stdout(io.StringIO()):
    import motion_ablations as AB
from _paths import REPO, DATA_ROOT, TOOLKIT, DOCS  # noqa: F401
CODE = REPO          # repo root: code, docs, figures
BASE = DATA_ROOT     # data root: scans, caches, exemplar DB
from sf3d import motion as M

ROWS = [r for r in AB.ROWS if 'rot' in r['mtype'].lower() and 'org' in r
        and r.get('nbr') is not None and r['label'] not in M.KNOB]
print(f'{len(ROWS)} hinge elements\n', flush=True)

def spacing(P, k=2000):
    """Median nearest-neighbour distance -- the scan's own resolution."""
    if len(P) < 8: return None
    idx = np.random.default_rng(0).choice(len(P), min(k, len(P)), replace=False)
    d, _ = cKDTree(P).query(P[idx], k=2)
    return float(np.median(d[:, 1]))

def grow(cop, cen, link, min_pts=150):
    if len(cop) < min_pts: return None
    tree = cKDTree(cop)
    seed = int(np.argmin(np.linalg.norm(cop - cen, axis=1)))
    seen = np.zeros(len(cop), bool); seen[seed] = True
    frontier = [seed]
    while frontier:
        nxt = []
        for i in frontier:
            for j in tree.query_ball_point(cop[i], link):
                if not seen[j]: seen[j] = True; nxt.append(j)
        frontier = nxt
    return cop[seen] if seen.sum() >= min_pts else None

def panel_variants(nbr, cen, extent, pn):
    d = np.linalg.norm(nbr - cen, axis=1)
    near = nbr[(d > extent*1.2) & (d < extent*40)]
    if len(near) < 30: return {}
    cop = near[np.abs((near - cen) @ pn) < 0.05]
    if len(cop) < 30: return {}
    sp = spacing(cop)
    out = {'fixed_2cm': grow(cop, cen, 0.02)}
    if sp:
        out['adaptive_2.5x'] = grow(cop, cen, 2.5*sp)
        out['adaptive_1.5x'] = grow(cop, cen, 1.5*sp)
    out = {k: (v if v is not None else cop) for k, v in out.items()}   # cop fallback
    out['_spacing'] = sp
    return out

res = {}
for r in ROWS:
    pts, nbr = r['pts'], r['nbr']
    cen = pts.mean(0); extent = float(np.linalg.norm(pts - cen, axis=1).mean())
    pn = M.parent_normal(nbr, cen, extent)
    if pn is None: continue
    axis = M.G if (r['label'] in M.LEVER or r['label'] in M.MIXED) else pn
    ax = np.cross(axis, pn); n = np.linalg.norm(ax)
    if n < 1e-6: continue
    ax /= n
    pv = panel_variants(nbr, cen, extent, pn)
    for pname, src in pv.items():
        if pname.startswith('_') or src is None or len(src) < 30: continue
        proj = (src - cen) @ ax
        span = float(proj.max() - proj.min())
        if span < 1e-6: continue
        far, near_ = cen + ax*proj.max(), cen + ax*proj.min()
        e_f = M.MD(far, axis, r['org'], r['dirv']); e_n = M.MD(near_, axis, r['org'], r['dirv'])
        sf = M.corner_score(nbr, far, pn); sn = M.corner_score(nbr, near_, pn)
        # how decisively off-centre is the element within this panel?
        asym = abs(abs(proj.max()) - abs(proj.min())) / span
        picks = {
            'corner':    e_f if sf >= sn else e_n,
            'ergonomic': e_f if abs(proj.max()) >= abs(proj.min()) else e_n,
            'oracle':    min(e_f, e_n),
        }
        # combined: trust the ergonomic prior only when the element is clearly
        # off-centre; otherwise it carries no information, so use the corner score
        for thr in (0.2, 0.35, 0.5):
            picks[f'combined@{thr}'] = (picks['ergonomic'] if asym >= thr
                                        else picks['corner'])
        res.setdefault(pname, {'span': [], **{k: [] for k in picks}})
        res[pname]['span'].append(span)
        for k, v in picks.items(): res[pname][k].append(v)

for pname, d in res.items():
    sp = np.array(d['span'])
    print(f'{"="*62}\nPANEL: {pname}   n={len(sp)}   median span {np.median(sp):.3f} m')
    print(f'{"="*62}')
    for rule in ('corner', 'ergonomic', 'combined@0.2', 'combined@0.35',
                 'combined@0.5', 'oracle'):
        if rule not in d: continue
        e = np.array(d[rule])
        tag = '  <-- deployed' if (rule=='corner' and pname=='fixed_2cm') else ''
        print(f'  {rule:<15} median {np.median(e):6.3f} m   <0.25m {100*np.mean(e<0.25):5.1f}%{tag}')

json.dump({k: {kk: list(map(float, vv)) for kk, vv in v.items()}
           for k, v in res.items()},
          open(f'{CODE}/docs/sf3d_edge_selection.json','w'), indent=1)
print('\nwritten docs/sf3d_edge_selection.json')

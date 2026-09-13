"""Bootstrap confidence intervals for every headline number in the paper.

WHY CLUSTERED. Elements in one scene share a scanner pose, a room, and often a
single piece of furniture -- three handles on one cabinet fail or succeed
together. Resampling elements i.i.d. would treat those as independent evidence
and report an interval roughly sqrt(k) too narrow. We resample SCENES with
replacement and recompute the statistic on the induced element set, which is the
standard cluster bootstrap and the honest unit of independence here.

WHAT WE CANNOT DO. USDNet's 82.8 / 75.1 / 59.8 are point estimates recovered
from its published AP table; no per-element results are released. So there is no
two-sample test available, and we do not fake one. We report OUR interval and
state whether their point estimate falls inside it -- which is a weaker claim
than a difference test and is the only one the data supports.

Run: .venv/bin/python scripts/confidence_intervals.py
"""
import os, sys, json, glob, collections
import numpy as np
from _paths import REPO, DATA_ROOT, TOOLKIT, DOCS  # noqa: F401
CODE = REPO          # repo root: code, docs, figures
BASE = DATA_ROOT     # data root: scans, caches, exemplar DB

B    = 10000
RNG  = np.random.default_rng(0)

def cluster_ci(groups, stat, B=B, alpha=0.05):
    """groups: {scene: [records]}.  stat: list[record] -> float or None."""
    keys = list(groups)
    point = stat([r for k in keys for r in groups[k]])
    draws = []
    for _ in range(B):
        pick = RNG.integers(0, len(keys), len(keys))
        s = stat([r for i in pick for r in groups[keys[i]]])
        if s is not None: draws.append(s)
    lo, hi = np.percentile(draws, [100*alpha/2, 100*(1-alpha/2)])
    return point, lo, hi, len(draws)

OUT = {}
def show(name, point, lo, hi, n, extra='', key=None):
    print(f'  {name:<34} {point:6.1f}   95% CI [{lo:5.1f}, {hi:5.1f}]   '
          f'+/-{(hi-lo)/2:4.1f}  {extra}')
    OUT[key or name] = dict(point=round(point,2), lo=round(lo,2), hi=round(hi,2),
                            half_width=round((hi-lo)/2,2), note=extra.strip('| '))

# ============================================================ SceneFun3D
print('=' * 78)
print('SceneFun3D -- motion on ground-truth regions, 48 scenes')
print('=' * 78)
import motion_ablations as AB          # builds ROWS and runs the deployed code
res = AB.evaluate({})                  # {} == unablated == sf3d/motion.py as deployed

by = collections.defaultdict(list)
for x in res: by[x['visit']].append(x)

gated = lambda R: (100*np.mean([x['gate'] for x in R if x['gate'] is not None])
                   if any(x['gate'] is not None for x in R) else None)
typed = lambda R: (100*np.mean([('rot' if x['gt_rot'] else 'trans') == x['p_type'] for x in R])
                   if R else None)

show('motion gate (all)',    *cluster_ci(by, gated))
show('type accuracy',        *cluster_ci(by, typed))

for nm, f in [('gate | prismatic', lambda x: not x['gt_rot']),
              ('gate | revolute',  lambda x: x['gt_rot'])]:
    sub = collections.defaultdict(list)
    for v, R in by.items():
        s = [x for x in R if f(x)]
        if s: sub[v] = s
    show(nm, *cluster_ci(sub, gated))

# ============================================================ Articulate3D
print()
print('=' * 78)
print('Articulate3D -- our motion on GROUND-TRUTH part masks, USDNet evaluator')
print('=' * 78)
rows = json.load(open(f'{CODE}/docs/articulate3d_official.json'))
ba = collections.defaultdict(list)
for r in rows: ba[r['scene']].append(r)

def frac(key, need_rot=False):
    def f(R):
        S = [r for r in R if (not need_rot or r.get('kind') == 'rot')]
        S = [r for r in S if r.get(key) is not None]
        return 100*np.mean([bool(r[key]) for r in S]) if S else None
    return f

USDNET = {'axis_ok': 82.8, 'org_official': 75.1, 'mao_official': 59.8}
for key, label in [('axis_ok','axis (<=15 deg)'),
                   ('org_official','origin (two-sided 0.25 m)'),
                   ('mao_official','composed gate (axis+origin)')]:
    need = (key == 'org_official')
    p, lo, hi, nb = cluster_ci(ba, frac(key, need_rot=need))
    u = USDNET[key]
    inside = lo <= u <= hi
    verdict = 'USDNet INSIDE our CI' if inside else 'USDNet OUTSIDE our CI'
    show(label, p, lo, hi, nb, f'| USDNet {u:.1f} -> {verdict}', key='a3d_'+key)
    OUT['a3d_'+key]['usdnet'] = u
    OUT['a3d_'+key]['separated'] = not inside

print()
print(f'n parts = {len(rows)}   scenes = {len(ba)}   bootstrap B = {B}, scene-clustered')
print('NOTE: our rows are on ground-truth part masks; USDNet\'s figures are')
print('      retention on its own predicted masks. Not like-for-like -- the')
print('      hybrid run is what settles it. See RESULTS_SECTION 5.9 A.')

OUT['_meta'] = dict(B=B, method='scene-clustered bootstrap', alpha=0.05,
                    n_sf3d=len(res), scenes_sf3d=len(by),
                    n_a3d=len(rows), scenes_a3d=len(ba))
json.dump(OUT, open(f'{CODE}/docs/confidence_intervals.json','w'), indent=1)
print('\nwrote docs/confidence_intervals.json')

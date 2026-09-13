"""Rank every element as a figure candidate, and cache the ranking.

Figures should show the method working, but "working" has to be earned rather
than eyeballed: a candidate qualifies only if it PASSES the gate under the
deployed code. Among those we prefer ones that are legible on the page -- enough
points to see, a neighbourhood large enough to show the mounting surface, and a
ground-truth origin present so the comparison is visible.
"""
import os, sys, json

import numpy as np
from _paths import REPO, DATA_ROOT, TOOLKIT, DOCS  # noqa: F401
CODE = REPO          # repo root: code, docs, figures
BASE = DATA_ROOT     # data root: scans, caches, exemplar DB
from sf3d_data import load, score, CODE

rows = load()
print(f'loaded {len(rows)} elements, {len({r["visit"] for r in rows})} scenes', flush=True)

cand = []
for i, r in enumerate(rows):
    s = score(r)
    cand.append(dict(i=i, visit=r['visit'], annot=r['annot'], label=r['label'],
                     mtype=r['mtype'], gt_rot=s['gt_rot'], gate=s['gate'],
                     oe=round(float(s['oe']), 2),
                     md=(round(float(s['md']), 4) if s['md'] is not None else None),
                     n_pts=len(r['pts']),
                     n_nbr=(0 if r['nbr'] is None else len(r['nbr'])),
                     has_org=('org' in r)))
    if (i+1) % 200 == 0: print(f'  {i+1}/{len(rows)}', flush=True)

passed = [c for c in cand if c['gate'] is True]
print(f'\ngate passes: {len(passed)}/{len([c for c in cand if c["gate"] is not None])}')
for lab in sorted({c['label'] for c in cand}):
    p = [c for c in passed if c['label'] == lab]
    a = [c for c in cand if c['label'] == lab and c['gate'] is not None]
    best = sorted(p, key=lambda c: (c['oe'], c['md'] if c['md'] is not None else 0))[:3]
    print(f'  {lab:<12} pass {len(p):>3}/{len(a):<3}  best: '
          + ', '.join(f'{c["visit"]}/{c["annot"][:8]} oe={c["oe"]:.1f}'
                      + (f' md={c["md"]:.3f}' if c['md'] is not None else '')
                      + f' n={c["n_pts"]}' for c in best))

json.dump(cand, open(f'{CODE}/docs/figure_candidates.json', 'w'), indent=1)
print('\nwrote docs/figure_candidates.json')

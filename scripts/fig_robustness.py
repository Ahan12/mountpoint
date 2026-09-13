"""Figure 7 -- segmenter-agnosticism, measured parametrically.

Reads docs/segmenter_robustness.json (4 corruption axes x levels, n=734, 3 reps).
Two series per panel:
  OURS  -- the deployed method, which reads axis/origin off the PARENT surface
  CTRL  -- ablation A, which reads them off the ELEMENT's own points

The panels are ordered by how much a real segmenter actually does each thing.
Dilation and subsampling are what a worse mask looks like in practice; 20 cm of
drift is a gross failure, not a degradation, and is included only to find where
the method finally breaks.
"""
import os, sys, json

import numpy as np, matplotlib.pyplot as plt
from _paths import REPO, DATA_ROOT, TOOLKIT, DOCS  # noqa: F401
CODE = REPO          # repo root: code, docs, figures
BASE = DATA_ROOT     # data root: scans, caches, exemplar DB
from figstyle import setup, COL2, OURS, CTRL, MUTED

setup()
D = json.load(open(f'{CODE}/docs/segmenter_robustness.json'))
base = D['_baseline']

# title = what the corruption is; xlabel = its unit. Splitting them this way
# keeps four panels legible at 7.16in without the labels colliding.
PANELS = [
    ('dilate (x extent)',  'Dilation',     r'$\times$ element extent'),
    ('subsample (kept)',   'Subsampling',  'fraction of points kept'),
    ('contaminate (frac)', 'Contamination','fraction from surroundings'),
    ('drift (m)',          'Drift',        'centroid offset (m)'),
]

fig, axes = plt.subplots(1, 4, figsize=(COL2, 2.15), sharey=True)
for ax, (key, title, xlabel) in zip(axes, PANELS):
    rows = sorted(D[key], key=lambda r: r['level'])
    x  = [r['level'] for r in rows]
    ax.plot(x, [r['full'] for r in rows], 'o-',  color=OURS, ms=3.2,
            label='ours (parent surface)', zorder=3)
    ax.plot(x, [r['own']  for r in rows], 's--', color=CTRL, ms=3.0,
            label='control (element normal)', zorder=3)
    ax.set_title(title, pad=4)
    ax.set_xlabel(xlabel, labelpad=2)
    ax.axhline(base['full'], color=MUTED, lw=0.5, ls=':', zorder=0)
    ax.margins(x=0.08)

axes[0].set_ylabel('motion gate (%)')
axes[0].set_ylim(25, 72)
axes[0].set_yticks([30, 40, 50, 60, 70])
# Reserve the strip first, then anchor the legend inside it. A negative
# bbox_to_anchor puts the legend outside the figure, where savefig's tight
# bbox drags it back on top of the x-labels.
fig.subplots_adjust(wspace=0.16, bottom=0.30)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=2, handlelength=1.9,
           bbox_to_anchor=(0.5, 0.0), columnspacing=2.5)
out = f'{CODE}/docs/figures/fig7_robustness.pdf'
fig.savefig(out, bbox_inches=None); fig.savefig(out.replace('.pdf', '.png'), bbox_inches=None)
print('wrote', out)

# ---- the quantitative claims the figure is asked to support ----
print('\nclaims checkable from this figure:')
for key, label in [('dilate (x extent)', '2x dilation'),
                   ('subsample (kept)',  '50% of points dropped'),
                   ('contaminate (frac)','10% contamination'),
                   ('drift (m)',         '2 cm drift')]:
    rows = {r['level']: r for r in D[key]}
    lvl = {'dilate (x extent)': 2.0, 'subsample (kept)': 0.5,
           'contaminate (frac)': 0.1, 'drift (m)': 0.02}[key]
    r = rows[lvl]
    print(f'  {label:<24} ours {r["full"]:.1f} ({r["full"]-base["full"]:+.1f})   '
          f'control {r["own"]:.1f} ({r["own"]-base["own"]:+.1f})')
c = {r['level']: r for r in D['contaminate (frac)']}
print(f'\n  MECHANISM CHECK -- contaminating the element with its surroundings')
print(f'  moves the control {c[0.0]["own"]:.1f} -> {c[0.5]["own"]:.1f} '
      f'({c[0.5]["own"]-c[0.0]["own"]:+.1f}), i.e. accidentally handing it the')
print(f'  parent surface is the single thing that helps it most.')

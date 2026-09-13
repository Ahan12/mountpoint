"""Figure 6 -- where the motion module fails, and why.

Of 734 elements, 267 fail the gate. The split is lopsided and the two halves
have different causes, so the figure reports both rather than a single number:

  81.6%  the AXIS is wrong (OE >= 15 deg)
  18.4%  the axis is right and the ORIGIN is wrong -- the panel was found and
         the wrong edge of it was chosen

The second is the honest weak point of the ergonomic edge rule: it assumes the
handle is mounted opposite the hinge, which fails on double doors and on panels
whose neighbourhood region-grows into an adjoining surface.

The 89.5 deg axis errors are a distinct and more interesting mode: TYPE
confusion. Called prismatic, a revolute element gets the parent NORMAL as its
direction, which is perpendicular to the hinge it actually turns about -- hence
errors clustered at 90 deg rather than spread. This is not rare: 33.5% of the
218 axis failures also have the type wrong, and 55 of those are revolute
elements called prismatic. Type classification, not axis geometry, is the lever
on the larger half of the failures.
"""
import os, sys, json, collections

import numpy as np, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import pcrender as R
from _paths import REPO, DATA_ROOT, TOOLKIT, DOCS  # noqa: F401
CODE = REPO          # repo root: code, docs, figures
BASE = DATA_ROOT     # data root: scans, caches, exemplar DB
from figstyle import setup, COL2, OURS, CTRL, GT, MUTED
from sf3d_data import load, score, CODE, M

setup()
cand  = json.load(open(f'{CODE}/docs/figure_candidates.json'))
rows  = {(r['visit'], r['annot'][:8]): r for r in load()}

scored = [x for x in cand if x['gate'] is not None]
fail   = [x for x in scored if x['gate'] is False]
n_axis = sum(1 for x in fail if x['oe'] >= 15)
n_org  = len(fail) - n_axis
print(f'{len(fail)}/{len(scored)} fail: axis {n_axis} ({100*n_axis/len(fail):.1f}%), '
      f'origin {n_org} ({100*n_org/len(fail):.1f}%)')
# reported in the caption; recomputed here so the claim cannot drift from the data
_rows = load()
_key  = {(q['visit'], q['annot'][:8]): q for q in _rows}
_tw = 0
for x in [y for y in fail if y['oe'] >= 15]:
    _s = score(_key[(x['visit'], x['annot'][:8])])
    if ('rot' if _s['gt_rot'] else 'trans') != _s['p_type']: _tw += 1
print(f'  of the {n_axis} axis failures, {_tw} ({100*_tw/n_axis:.1f}%) also got TYPE wrong')

EX = [('422195', '011e6fea', 'wrong panel edge chosen',
       'axis correct, origin 0.83 m off'),
      ('421069', '5faf8b11', 'type confusion',
       'called prismatic; axis $89.5^\\circ$ off')]

fig = plt.figure(figsize=(COL2, 2.25))
gs  = GridSpec(1, 3, figure=fig, width_ratios=[1.05, 1, 1], wspace=0.10)

# ---- (a) the breakdown -------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
labels = ['axis wrong\n(OE $\\geq 15^\\circ$)', 'axis right,\norigin wrong']
vals   = [100*n_axis/len(scored), 100*n_org/len(scored)]
b = ax.barh([1, 0], vals, height=0.52, color=[CTRL, OURS])
ax.set_yticks([1, 0]); ax.set_yticklabels(labels, fontsize=6.8)
ax.set_xlabel('% of all 734 elements', labelpad=2)
ax.set_xlim(0, 36); ax.grid(axis='y', visible=False)
for r_, v, n in zip(b, vals, [n_axis, n_org]):
    ax.text(v + 0.8, r_.get_y() + r_.get_height()/2, f'{v:.1f}%  (n={n})',
            va='center', fontsize=6.5)
ax.set_title('(a) failure modes, $n=267$', fontsize=7, pad=4)

# ---- (b),(c) one example of each --------------------------------------
for k, (vis, ann, title, sub) in enumerate(EX):
    ax = fig.add_subplot(gs[0, k+1])
    r  = rows[(vis, ann)]
    pts, nbr = r['pts'], r['nbr']
    cen = pts.mean(0)
    ext = float(np.linalg.norm(pts - cen, axis=1).mean())
    s   = score(r)
    pn  = M.parent_normal(nbr, cen, ext)
    if pn is None: pn = M.local_frame(pts)[0]
    basis = R.readable_view(pn, r['dirv'])
    feet = [R.axis_through(r.get('org', cen), r['dirv'], cen)]
    if s['org'] is not None:
        feet.append(R.axis_through(s['org'], s['axis'], cen))
    hw = max(0.30, max(float(np.linalg.norm(f - cen)) for f in feet) * 1.25, ext*4)
    hy = hw * 0.88
    ctx = R.crop_view(nbr[np.linalg.norm(nbr - cen, axis=1) < hw*2.2], basis, cen, hw, hy)
    R.scatter(ax, ctx, basis=basis, centre=cen, color=MUTED, s=0.8, alpha=0.30,
              zorder=1, max_pts=15000)
    R.scatter(ax, pts, basis=basis, centre=cen, color=OURS, s=5.0, alpha=0.95, zorder=6)
    L = hw*0.92
    R.axis_line(ax, feet[0], r['dirv'], basis, cen, L, color=GT, lw=2.6,
                alpha=0.80, zorder=4, solid_capstyle='round')
    o_pr = feet[-1] if s['org'] is not None else cen
    R.axis_line(ax, R.axis_through(o_pr, s['axis'], cen), s['axis'], basis, cen, L,
                color=CTRL, lw=1.6, ls=(0, (3.5, 2.0)), zorder=5)
    if s['org'] is not None:
        R.marker(ax, s['org'], basis, cen, marker='o', ms=3.8, mfc='white',
                 mec=CTRL, mew=1.1, zorder=7)
    ax.set_xlim(-hw, hw); ax.set_ylim(-hy, hy)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f'({chr(98+k)}) {title}\n{sub}', fontsize=6.8, pad=3, linespacing=1.35)
    print(f'  {vis}/{ann}: OE={s["oe"]:.1f} MD={s["md"]}')

leg = [Line2D([], [], color=OURS, marker='o', ls='', ms=3.2, label='functional element'),
       Line2D([], [], color=GT,   lw=2.4, label='ground truth'),
       Line2D([], [], color=CTRL, lw=1.5, ls=(0,(3.5,2.0)), label='prediction (failed)')]
fig.subplots_adjust(bottom=0.17, top=0.84, left=0.135, right=0.995)
fig.legend(handles=leg, loc='lower center', ncol=3, handlelength=1.8,
           columnspacing=1.8, bbox_to_anchor=(0.58, 0.0))
out = f'{CODE}/docs/figures/fig6_failures.pdf'
fig.savefig(out, bbox_inches=None); fig.savefig(out.replace('.pdf','.png'), bbox_inches=None)
print('wrote', out)

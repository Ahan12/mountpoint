"""Figure 3 -- where the motion information actually lives.

The paper's central claim is that a functional element carries almost no motion
information itself; the surface it is MOUNTED ON carries all of it. This figure
walks that claim through one real element, using the deployed code's own
intermediate quantities rather than a schematic redrawing of them.

(a) the region a segmenter returns -- a handle, 127 points, no usable geometry
(b) the shell the method reads instead: 1.5-4.0x the region's extent, with the
    region's own points deliberately excluded (sf3d.motion.parent_normal)
(c) the robust plane fit to that shell -- the parent normal
(d) the recovered hinge axis and origin, against ground truth
"""
import os, sys, json

import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pcrender as R
from _paths import REPO, DATA_ROOT, TOOLKIT, DOCS  # noqa: F401
CODE = REPO          # repo root: code, docs, figures
BASE = DATA_ROOT     # data root: scans, caches, exemplar DB
from figstyle import setup, COL2, OURS, CTRL, GT, MUTED
PANEL = '#9db8cc'   # the isolated mounting panel: a desaturated tint of OURS,
                    # so it reads as "same object family" without competing
from sf3d_data import load, score, CODE, M

setup()
VISIT, ANNOT = '421647', '43f442f2'          # hook_pull, oe=0.00, md=0.012

rows = load()
r = next(x for x in rows if x['visit'] == VISIT and x['annot'].startswith(ANNOT))
pts, nbr = r['pts'], r['nbr']
cen = pts.mean(0)
extent = float(np.linalg.norm(pts - cen, axis=1).mean())
s = score(r)
print(f'{VISIT}/{ANNOT}  label={r["label"]}  n_pts={len(pts)}  extent={extent:.3f} m')
print(f'  OE={s["oe"]:.2f} deg   MD={s["md"]:.3f} m   gate={s["gate"]}')

# ---- the deployed code's own intermediates, recomputed identically ----
d      = np.linalg.norm(nbr - cen, axis=1)
shell  = nbr[(d > extent*1.5) & (d < extent*4.0)]
pn     = M.parent_normal(nbr, cen, extent)
panel  = M.panel_component(nbr, cen, extent, hi=40)
local  = nbr[d < 1.1]                                   # what we draw as context
print(f'  shell={len(shell)}  panel={len(panel) if panel is not None else None}'
      f'  local_context={len(local)}')

basis  = R.camera(R.orbit(pn, az_deg=46, el_deg=13))
VIEW   = dict(basis=basis, centre=cen)
HX, HY = 0.46, 0.40
local  = R.crop_view(local, basis, cen, HX, HY)         # rectangular framing
panelc = (R.crop_view(panel, basis, cen, HX, HY) if panel is not None else None)

fig, axs = plt.subplots(1, 4, figsize=(COL2, 2.05))
CAPS = ['(a) the region a segmenter returns',
        '(b) shell of the mounting surface',
        '(c) robust plane fit $\\rightarrow$ parent normal',
        '(d) recovered axis and origin']

def context(ax, alpha=0.30, s=0.7):
    R.scatter(ax, local, **VIEW, color=MUTED, s=s, alpha=alpha, zorder=1,
              max_pts=14000)

# (a) ---------------------------------------------------------------
R.scatter(axs[0], pts, **VIEW, color=OURS, s=5.0, alpha=0.95, zorder=4)
ex, ey, _ = R.project(pts, basis, cen)
_cx, _cy = (ex.min()+ex.max())/2, (ey.min()+ey.max())/2
_h = max(np.ptp(ex)/(HX/HY), np.ptp(ey)) * 0.72
axs[0].set_xlim(_cx-_h*(HX/HY), _cx+_h*(HX/HY)); axs[0].set_ylim(_cy-_h, _cy+_h)

# (b) ---------------------------------------------------------------
context(axs[1])
R.scatter(axs[1], shell, **VIEW, color=CTRL, s=1.9, alpha=0.85, zorder=3, max_pts=9000)
R.scatter(axs[1], pts,   **VIEW, color=OURS, s=4.0, alpha=0.95, zorder=4)

# (c) ---------------------------------------------------------------
context(axs[2])
R.scatter(axs[2], shell, **VIEW, color=CTRL, s=1.3, alpha=0.35, zorder=3, max_pts=9000)
# plane patch: a square in the fitted plane, centred on the element
e1 = R._unit(np.cross(pn, R.UP)); e2 = np.cross(pn, e1)
h  = 0.34
quad = np.stack([cen + e1*h + e2*h, cen - e1*h + e2*h,
                 cen - e1*h - e2*h, cen + e1*h - e2*h])
qx, qy, _ = R.project(quad, basis, cen)
axs[2].add_patch(Polygon(np.c_[qx, qy], closed=True, facecolor=OURS,
                         alpha=0.16, edgecolor=OURS, lw=0.9, zorder=5))
R.axis_line(axs[2], cen, pn, basis, cen, 0.0, color='none')
R.segment(axs[2], cen, cen + pn*0.30, basis, cen, color=OURS, lw=1.4, zorder=6)
R.marker(axs[2], cen + pn*0.30, basis, cen, marker='^', ms=3.4, color=OURS, zorder=6)
R.scatter(axs[2], pts, **VIEW, color=OURS, s=4.0, alpha=0.95, zorder=7)

# (d) ---------------------------------------------------------------
context(axs[3], alpha=0.16, s=0.6)
if panel is not None:
    R.scatter(axs[3], panelc, **VIEW, color=PANEL, s=1.2, alpha=0.75, zorder=2,
              max_pts=12000)
R.scatter(axs[3], pts, **VIEW, color=OURS, s=4.0, alpha=0.95, zorder=6)
R.axis_line(axs[3], r['org'], r['dirv'], basis, cen, 0.42,
            color=GT, lw=2.4, alpha=0.85, zorder=4, solid_capstyle='round')
R.axis_line(axs[3], s['org'], s['axis'], basis, cen, 0.42,
            color=OURS, lw=1.5, ls=(0, (3.5, 2.0)), zorder=5)
R.marker(axs[3], s['org'], basis, cen, marker='o', ms=3.6, mfc='white',
         mec=OURS, mew=1.1, zorder=7)

for ax in axs[1:]:
    ax.set_xlim(-HX, HX); ax.set_ylim(-HY, HY)

for ax, c in zip(axs, CAPS):
    ax.set_title(c, fontsize=7, pad=3)
    ax.set_aspect('equal'); ax.axis('off')

from matplotlib.lines import Line2D
leg = [Line2D([], [], color=OURS, marker='o', ls='', ms=3, label='functional element'),
       Line2D([], [], color=CTRL, marker='o', ls='', ms=3, label='shell (1.5--4.0$\\times$ extent)'),
       Line2D([], [], color=PANEL, marker='o', ls='', ms=3, label='isolated panel'),
       Line2D([], [], color=GT,   lw=2.2, label='ground-truth axis'),
       Line2D([], [], color=OURS, lw=1.4, ls=(0,(3.5,2.0)), label='predicted axis + origin')]
fig.subplots_adjust(wspace=0.05, bottom=0.14)
fig.legend(handles=leg, loc='lower center', ncol=5, handlelength=1.7,
           columnspacing=1.3, bbox_to_anchor=(0.5, 0.0))
out = f'{CODE}/docs/figures/fig3_insight.pdf'
fig.savefig(out, bbox_inches=None); fig.savefig(out.replace('.pdf','.png'), bbox_inches=None)
print('wrote', out)

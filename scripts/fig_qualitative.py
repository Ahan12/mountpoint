"""Figure 5 -- qualitative results across the affordance taxonomy.

One successful element per class, chosen by the ranking in
docs/figure_candidates.json (gate PASS under the deployed code, then lowest
orientation error, then enough points to be legible). Green is ground truth,
blue dashed is ours; where the class is revolute the open circle is the
predicted origin.

REGIONS ARE GROUND TRUTH. Our own predicted masks live on the Colab run and are
not local, so this figure isolates the motion module, which is what it is for.
The caption must say so.
"""
import os, sys, json

import numpy as np, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pcrender as R
from _paths import REPO, DATA_ROOT, TOOLKIT, DOCS  # noqa: F401
CODE = REPO          # repo root: code, docs, figures
BASE = DATA_ROOT     # data root: scans, caches, exemplar DB
from figstyle import setup, COL2, OURS, GT, MUTED
from sf3d_data import load, score, CODE, M

setup()
PANEL = '#9db8cc'

# class -> (visit, annot prefix, half-width of the view box in metres)
PICKS = [
    ('hook_pull',  '421647', '43f442f2', 0.42, 'hinged door'),
    ('hook_turn',  '421602', 'c88035ea', 0.42, 'lever handle'),
    ('pinch_pull', '420683', '77ce36a2', 0.30, 'drawer'),
    ('rotate',     '421069', '7f4cbf10', 0.22, 'knob'),
    ('key_press',  '422521', 'c6ae101c', 0.26, 'switch'),
    ('tip_push',   '422200', '047e8c22', 0.22, 'button'),
    ('plug_in',    '422516', '8178ff9e', 0.22, 'socket'),
    ('unplug',     '422523', 'a36ea03c', 0.30, 'plug'),
]

rows = {(r['visit'], r['annot'][:8]): r for r in load()}

def draw(ax, r, hw, title):
    pts, nbr = r['pts'], r['nbr']
    cen = pts.mean(0)
    extent = float(np.linalg.norm(pts - cen, axis=1).mean())
    s = score(r)
    pn = M.parent_normal(nbr, cen, extent)
    if pn is None: pn = M.local_frame(pts)[0]
    basis = R.readable_view(pn, r['dirv'])
    # The crop must contain the axis, not just the element. A hinge line sits
    # up to 0.7 m from its handle (a wide door), and SceneFun3D stores the
    # origin at an arbitrary point ALONG the axis -- some with |z| > 100 m,
    # which is harmless for a line but fatal for a hand-tuned view box.
    feet = [R.axis_through(r.get('org', cen), r['dirv'], cen)]
    if s['org'] is not None:
        feet.append(R.axis_through(s['org'], s['axis'], cen))
    reach = max(float(np.linalg.norm(f - cen)) for f in feet)
    hw = max(hw, reach * 1.30, extent * 4.0)
    hy = hw * 0.88
    ctx = R.crop_view(nbr[np.linalg.norm(nbr - cen, axis=1) < hw*2.2],
                      basis, cen, hw, hy)
    R.scatter(ax, ctx, basis=basis, centre=cen, color=MUTED, s=0.8,
              alpha=0.30, zorder=1, max_pts=16000)
    R.scatter(ax, pts, basis=basis, centre=cen, color=OURS, s=6.0,
              alpha=0.95, zorder=6)
    L = hw * 0.92
    o_gt = R.axis_through(r.get('org', cen), r['dirv'], cen)
    R.axis_line(ax, o_gt, r['dirv'], basis, cen, L,
                color=GT, lw=2.6, alpha=0.80, zorder=4, solid_capstyle='round')
    o_pr = R.axis_through(s['org'] if s['org'] is not None else cen,
                          s['axis'], cen)
    R.axis_line(ax, o_pr, s['axis'], basis, cen, L,
                color=OURS, lw=1.5, ls=(0,(3.5,2.0)), zorder=5)
    if s['org'] is not None:
        R.marker(ax, s['org'], basis, cen, marker='o', ms=3.8, mfc='white',
                 mec=OURS, mew=1.1, zorder=7)
    ax.set_xlim(-hw, hw); ax.set_ylim(-hy, hy)
    ax.set_aspect('equal'); ax.axis('off')
    kind = 'revolute' if s['gt_rot'] else 'prismatic'
    err = f'OE {s["oe"]:.1f}$^\\circ$' + (f', MD {s["md"]*100:.0f} cm'
                                          if s['md'] is not None else '')
    ax.set_title(f'{title} -- {kind}\n{err}', fontsize=6.5, pad=2.5,
                 linespacing=1.35)
    return s

fig, axs = plt.subplots(2, 4, figsize=(COL2, 3.85))
for ax, (lab, vis, ann, hw, nice) in zip(axs.ravel(), PICKS):
    r = rows.get((vis, ann))
    if r is None:
        ax.axis('off'); print('MISSING', lab, vis, ann); continue
    s = draw(ax, r, hw, f'{nice} ({lab})')
    print(f'{lab:<11} {vis}/{ann}  OE={s["oe"]:5.2f}  '
          f'MD={"-" if s["md"] is None else f"{s[chr(39)+chr(39)] if False else s['md']:.3f}"}  gate={s["gate"]}')

leg = [Line2D([], [], color=OURS, marker='o', ls='', ms=3.2, label='functional element (ground-truth region)'),
       Line2D([], [], color=GT,   lw=2.4, label='ground-truth axis'),
       Line2D([], [], color=OURS, lw=1.4, ls=(0,(3.5,2.0)), label='predicted axis'),
       Line2D([], [], color=OURS, marker='o', ls='', ms=4, mfc='white', mew=1.1,
              label='predicted origin (revolute only)')]
fig.subplots_adjust(wspace=0.04, hspace=0.16, bottom=0.085)
fig.legend(handles=leg, loc='lower center', ncol=2, handlelength=1.8,
           columnspacing=2.0, bbox_to_anchor=(0.5, 0.0))
out = f'{CODE}/docs/figures/fig5_qualitative.pdf'
fig.savefig(out, bbox_inches=None); fig.savefig(out.replace('.pdf','.png'), bbox_inches=None)
print('wrote', out)

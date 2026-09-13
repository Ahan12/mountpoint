"""E2 -- do our geometric motion rules transfer to Articulate3D, unchanged?

Data: USDNet's released preprocessed pack, which the challenge README states is
the exact point cloud the server scores against. No ScanNet++ access needed.

  <scene>.npy               (N,13) float32  xyz | rgb | normal | sem | inst | 0 | inst
  <scene>_articulation.h5   per movable-part id:
        axis (3,)  origin (3,)  sem_id {1=rotation, 2=translation}
        inter_mask (n_part_points,) bool  -- which of the part's points are the handle

Because Articulate3D has part nouns rather than SceneFun3D's affordance verbs, our
DETERMINED_TYPE lookup cannot transfer (E0 measured that at -13.2 gate points). So
motion TYPE is taken as given -- exactly as the hybrid will take it from USDNet's
pred_classes -- and we predict only the two things we actually claim:

    axis    = gravity if rotation, else the parent-surface normal
    origin  = corner-score edge cascade (rotations only)

Nothing in sf3d/motion.py is modified. Two framings are scored:

    handle : element = the interactable points  (faithful to our method)
    part   : element = the whole movable part   (what the benchmark segments)

Thresholds are Articulate3D's, which are identical to SceneFun3D's: 15 deg, 0.25 m.

Run: python scripts/articulate3d_transfer.py
"""
import sys, os, glob, json, collections
import numpy as np, h5py
from scipy.spatial import cKDTree

from _paths import REPO, DATA_ROOT, TOOLKIT, DOCS, A3D_VAL  # noqa: F401
CODE = REPO          # repo root: code, docs, figures
BASE = DATA_ROOT     # data root: scans, caches, exemplar DB
from sf3d import motion as M

VAL = A3D_VAL   # $ARTICULATE3D_ROOT; see README "Data layout"
NBR_R = 1.1          # same radius as our SceneFun3D caches
INST_COL = 10

def predict(kind, pts, nbr):
    """kind: 'rot' | 'trans'. Type is GIVEN; we predict axis and origin only."""
    cen = pts.mean(0)
    extent = float(np.linalg.norm(pts - cen, axis=1).mean())
    own = M.local_frame(pts)[0]
    pn = M.parent_normal(nbr, cen, extent)
    if kind == 'trans':
        return (pn if pn is not None else own), None
    axis = M.G
    if pn is None:
        return axis, cen
    panel = M.panel_component(nbr, cen, extent, hi=40)
    src = panel if panel is not None else M.cop_points(nbr, cen, extent, hi=40)
    if src is None:
        return axis, cen
    ax = np.cross(axis, pn); n = np.linalg.norm(ax)
    if n < 1e-6:
        return axis, cen
    ax = ax / n
    proj = (src - cen) @ ax
    far, near = cen + ax*proj.max(), cen + ax*proj.min()
    org = far if M.corner_score(nbr, far, pn) >= M.corner_score(nbr, near, pn) else near
    return axis, org

rows = []
scenes = sorted(glob.glob(f'{VAL}/*.npy'))
print(f'{len(scenes)} validation scenes\n', flush=True)

for si, f in enumerate(scenes):
    sid = os.path.basename(f)[:-4]
    a = np.load(f)
    xyz = a[:, :3].astype(np.float64)
    inst = a[:, INST_COL].astype(np.int64)
    tree = cKDTree(xyz)
    with h5py.File(f'{VAL}/{sid}_articulation.h5', 'r') as h:
        for key in h:
            g = h[key]
            pid = int(key)
            sel = np.flatnonzero(inst == pid)
            if len(sel) < 20:
                continue
            part = xyz[sel]
            im = np.array(g['inter_mask'], bool)
            gt_axis = np.array(g['axis'], float)
            gt_org  = np.array(g['origin'], float)
            sem     = int(np.array(g['sem_id']))
            if np.linalg.norm(gt_axis) < 1e-6:
                continue
            gt_axis = gt_axis / np.linalg.norm(gt_axis)
            kind = 'rot' if sem == 1 else 'trans'

            variants = {'part': part}
            if im.shape[0] == len(sel) and im.sum() >= 20:
                variants['handle'] = part[im]

            for name, pts in variants.items():
                cen = pts.mean(0)
                nbr = xyz[tree.query_ball_point(cen, NBR_R)]
                if len(nbr) < 50:
                    continue
                ax, org = predict(kind, pts, nbr)
                oe = M.OE(gt_axis, ax)
                md = (M.MD(org, ax, gt_org, gt_axis)
                      if (kind == 'rot' and org is not None) else None)
                rows.append(dict(scene=sid, pid=pid, kind=kind, variant=name,
                                 n=len(pts), oe=oe, md=md))
    if (si+1) % 10 == 0:
        print(f'  {si+1}/{len(scenes)} scenes, {len(rows)} rows', flush=True)

# ------------------------------------------------------------------ report
def block(variant):
    R = [r for r in rows if r['variant'] == variant]
    if not R: return
    print(f'\n{"="*62}\nELEMENT = {variant.upper()}   (n={len(R)})\n{"="*62}')
    for kind in ('rot', 'trans'):
        S = [r for r in R if r['kind'] == kind]
        if not S: continue
        oe = np.array([r['oe'] for r in S])
        print(f'  {kind:<6} n={len(S):4d}   axis median {np.median(oe):6.2f}deg   '
              f'mean {oe.mean():6.2f}   within 15deg {100*np.mean(oe<15):5.1f}%')
        if kind == 'rot':
            md = np.array([r['md'] for r in S if r['md'] is not None])
            if len(md):
                print(f'         {"":4}   origin median {np.median(md):6.3f} m    '
                      f'within 0.25 m {100*np.mean(md<0.25):5.1f}%')
                both = np.array([(r['oe'] < 15 and r['md'] is not None and r['md'] < 0.25)
                                 for r in S])
                print(f'         {"":4}   axis AND origin {100*both.mean():5.1f}%')
    # combined gate: trans needs axis only, rot needs axis+origin
    ok = [(r['oe'] < 15) if r['kind'] == 'trans'
          else (r['oe'] < 15 and r['md'] is not None and r['md'] < 0.25) for r in R]
    print(f'  {"OVERALL":<6}        motion gate {100*np.mean(ok):5.1f}%   '
          f'(SceneFun3D deployed: 62.4%)')

for v in ('handle', 'part'):
    block(v)

json.dump(rows, open(f'{CODE}/docs/articulate3d_transfer.json', 'w'), indent=1)
print(f'\nwritten docs/articulate3d_transfer.json  ({len(rows)} rows)')

"""Re-score our Articulate3D predictions with the OFFICIAL Articulate3D criteria.

Our own implementation and theirs agree on the axis test but NOT on the origin
test, and theirs is stricter. Transcribed faithfully from USDNet
benchmark/evaluate_semantic_instance.py (master):

  axis, match_criteria_MA_pred (L137-149)
      dot = <a_ref, a_pred> / (|a_ref| |a_pred|)
      dot = np.abs(dot)                      <-- UNSIGNED, same as our OE
      pass if arccos(dot) < 15 deg

  origin, match_criteria_MAO_pred_standard (L160-186)   <-- the leaderboard's MAO-ST
      translations skip the origin test entirely (same as ours)
      d = pred_origin - gt_origin
      pass if  |d - (d.gt_axis)   gt_axis / |gt_axis| |   < 0.25   AND
               |d - (d.pred_axis) pred_axis/|pred_axis||  < 0.25
      i.e. BOTH the distance from the predicted origin to the GT axis line AND
      the distance from the GT origin to the PREDICTED axis line.

  Ours used the minimum distance between the two lines, which is <= both of
  those, so every number we reported for origin was measured under a MORE
  PERMISSIVE rule than the benchmark applies.

Evaluated on ground-truth part masks, so the IoU gate passes by construction and
what remains is exactly the motion criteria.

Run: python scripts/articulate3d_official_metric.py
"""
import sys, os, glob, json
import numpy as np, h5py
from scipy.spatial import cKDTree

from _paths import REPO, DATA_ROOT, TOOLKIT, DOCS, A3D_VAL  # noqa: F401
CODE = REPO          # repo root: code, docs, figures
BASE = DATA_ROOT     # data root: scans, caches, exemplar DB
from sf3d import motion as M

VAL = A3D_VAL   # $ARTICULATE3D_ROOT; see README "Data layout"

def official_axis_ok(gt_axis, pred_axis, axis_th=15):
    d = np.dot(gt_axis, pred_axis) / (np.linalg.norm(gt_axis)*np.linalg.norm(pred_axis))
    return bool(np.arccos(np.clip(np.abs(d), -1, 1)) < np.deg2rad(axis_th))

def official_origin_ok(gt_axis, gt_origin, pred_axis, pred_origin, th=0.25):
    d = np.asarray(pred_origin, float) - np.asarray(gt_origin, float)
    a = np.asarray(gt_axis, float); b = np.asarray(pred_axis, float)
    p_gt   = d - np.dot(d, a) * a / np.linalg.norm(a)
    p_pred = d - np.dot(d, b) * b / np.linalg.norm(b)
    return bool(np.linalg.norm(p_gt) < th and np.linalg.norm(p_pred) < th)

rows = []
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
            ga = ga/np.linalg.norm(ga); go = np.array(g['origin'], float)
            kind = 'rot' if int(np.array(g['sem_id'])) == 1 else 'trans'
            cen = elem.mean(0)
            nbr = xyz[tree.query_ball_point(cen, 1.1)]
            if len(nbr) < 50: continue
            p = M.predict_motion_given_panel(elem, part, kind, nbr=nbr)
            pa, po = p['motion_dir'], p['motion_origin']
            ax_ok = official_axis_ok(ga, pa)
            if kind == 'trans':
                off_official = True                       # translations skip origin
                off_ours     = True
            else:
                off_official = official_origin_ok(ga, go, pa, po) if po is not None else False
                off_ours     = (M.MD(po, pa, go, ga) < 0.25) if po is not None else False
            rows.append(dict(scene=sid, kind=kind, has_h=bool(has_h), axis_ok=ax_ok,
                             mao_official=bool(ax_ok and off_official),
                             mao_ours=bool(ax_ok and off_ours),
                             org_official=bool(off_official), org_ours=bool(off_ours)))

def block(R, name):
    if not R: return
    rot = [r for r in R if r['kind'] == 'rot']
    print(f'  {name:<20} n={len(R):4d}   axis {100*np.mean([r["axis_ok"] for r in R]):5.1f}%'
          f'   origin(rot) ours {100*np.mean([r["org_ours"] for r in rot]):5.1f}%'
          f' / official {100*np.mean([r["org_official"] for r in rot]):5.1f}%'
          f'   MAO ours {100*np.mean([r["mao_ours"] for r in R]):5.1f}%'
          f' / official {100*np.mean([r["mao_official"] for r in R]):5.1f}%')

print(f'Articulate3D validation, ground-truth part masks -- {len(rows)} parts\n')
print('OURS, scored two ways (axis test is identical; only origin differs)')
block([r for r in rows if r['has_h']], 'with handle')
block([r for r in rows if not r['has_h']], 'without handle')
block(rows, 'all parts')
print(f'\nUSDNet published (its own predicted masks, AP50 = 41.8):'
      f'   axis 82.8%   origin 75.1%   MAO 59.8%')
json.dump(rows, open(f'{CODE}/docs/articulate3d_official.json','w'), indent=1, default=bool)
print('\nwritten docs/articulate3d_official.json')

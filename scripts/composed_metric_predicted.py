"""SceneFun3D's composed motion metric on OUR PREDICTED regions.

Everything reported so far under the composed metric (AP25 -> +M -> +MA -> +MAO)
used GROUND-TRUTH regions, where AP25 is 100 by construction. That is the motion
module's ceiling, not a system number. This computes the same nested metric on
the regions our own pipeline produced, which is what SceneFun3D's motion task
actually asks for and what a deployed system would achieve.

Two chains are produced so the cost of each stage is separable:

  GT-region chain    AP25=100 -> +M -> +MA -> +MAO   (motion error only)
  PREDICTED chain    AP25=our -> +M -> +MA -> +MAO   (segmentation AND motion)

Motion is computed by the unmodified `sf3d.motion.predict_motion`. Neighbourhood
context comes from the laser scan around the region centroid (the parent-surface
rule needs >=1.1 m, which a small predicted blob cannot supply), so the
comparison isolates "noisy region" from "noisy scene".

Prediction ids are globally unique: evaluate_matches keeps ONE pred_visited dict
across scenes and colliding ids silently destroy recall.

Run: python composed_metric_predicted.py --cache <drive>/liftcache_split0.pkl
"""
import os, sys, json, pickle, argparse, types, collections
from copy import deepcopy

import numpy as np

if 'open3d' not in sys.modules:
    def _nope(*a, **k): raise RuntimeError('open3d stubbed but called')
    class _S:
        def __getattr__(self, n): return _nope
    _m = types.ModuleType('open3d'); _m.io = _S(); _m.geometry = _S(); _m.utility = _S()
    sys.modules['open3d'] = _m
if not hasattr(np, 'in1d'):
    np.in1d = np.isin

import eval.functionality_segmentation.eval_utils.eval_script as ES
from _paths import REPO, DATA_ROOT, TOOLKIT, DOCS  # noqa: F401
CODE = REPO          # repo root: code, docs, figures
BASE = DATA_ROOT     # data root: scans, caches, exemplar DB
from sf3d import motion as MOT
from sf3d import lifting as L
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument('--cache', required=True)
ap.add_argument('--data-root', default=f'{DATA_ROOT}/data_merged')
ap.add_argument('--gt-dir', default=f'{DATA_ROOT}/comp_pred_gt')
ap.add_argument('--out', default=f'{DOCS}/composed_predicted.json')
ap.add_argument('--conf', default='combined', choices=['frames', 'agree', 'extent', 'combined'])
a = ap.parse_args()
os.makedirs(a.gt_dir, exist_ok=True)

PRIOR, NBR_R, NBR_CAP = 0.0170, 1.10, 120000
DEPLOY = dict(at=0.5, mv=1, mf=2, link=0.05)

_ID = 0
def mk(pred):
    return {_ID + i: dict(label_id=pred['pred_classes'][i], conf=pred['pred_scores'][i],
                          mask=pred['pred_masks'][:, i])
            for i in range(len(pred['pred_classes']))}
ES.make_pred_info = mk

def largest_cluster(points, link=0.05, min_pts=3):
    if len(points) < min_pts:
        return np.arange(len(points))
    tree = cKDTree(points); seen = np.zeros(len(points), bool); best = []
    for s in range(len(points)):
        if seen[s]: continue
        comp, fr = [s], [s]; seen[s] = True
        while fr:
            nxt = []
            for i in fr:
                for j in tree.query_ball_point(points[i], link):
                    if not seen[j]:
                        seen[j] = True; comp.append(j); nxt.append(j)
            fr = nxt
        if len(comp) > len(best): best = comp
    return np.array(best)

def predict(r):
    if 'idx' not in r or len(r['idx']) == 0:
        return np.array([], np.int32), None, 0.0
    acc, seen = r['acc'], r['seen']
    with np.errstate(divide='ignore', invalid='ignore'):
        agree = np.where(seen > 0, acc / seen, 0)
    keep = (agree >= DEPLOY['at']) & (seen >= DEPLOY['mv']) & (acc >= DEPLOY['mf'])
    idx, pts, ag = r['idx'][keep], r['xyz'][keep], agree[keep]
    if len(idx) >= 3:
        sel = largest_cluster(pts, link=DEPLOY['link'])
        idx, pts, ag = idx[sel], pts[sel], ag[sel]
    return idx, pts, (float(ag.mean()) if len(ag) else 0.0)

def motion_correct(gt_m, pm, gate):
    """Paper Sec 6.3 nesting: type, then axis <15 deg, then origin MD<0.25 m."""
    if gt_m is None or pm is None:
        return False
    if gt_m['motion_type'] != pm['motion_type']:
        return False
    if gate == 'M':
        return True
    if MOT.OE(gt_m['motion_dir'], pm['motion_dir']) >= 15:
        return False
    if gate == 'MA':
        return True
    if gt_m['motion_type'] != 'rot':
        return True
    if gt_m['motion_origin'] is None or pm['motion_origin'] is None:
        return False
    return MOT.MD(pm['motion_origin'], pm['motion_dir'],
                  gt_m['motion_origin'], gt_m['motion_dir']) < 0.25

cache = pickle.load(open(a.cache, 'rb'))
scenes = sorted({r['visit'] for r in cache})
by_scene = collections.defaultdict(list)
for r in cache:
    by_scene[r['visit']].append(r)
print(f'{len(cache)} elements, {len(scenes)} scenes', flush=True)

CONF = {'frames': lambda r: r.get('used_frames', 0) / 6.0,
        'agree':  lambda r: r['_agree'],
        'extent': lambda r: r['_extent'],
        'combined': lambda r: r['_agree'] * r['_extent']}[a.conf]

chains = {'gt': {g: {} for g in ('raw', 'M', 'MA', 'MAO')},
          'pred': {g: {} for g in ('raw', 'M', 'MA', 'MAO')}}
_NEXT = 0
stats = collections.Counter()

for vi, v in enumerate(scenes, 1):
    d = f'{a.data_root}/{v}'
    A = json.load(open(f'{d}/{v}_annotations.json'))
    A = A.get('annotations', A) if isinstance(A, dict) else A
    items = [x for x in A if x.get('label') in ES.LABEL_TO_ID and x.get('indices')]
    if not items:
        continue
    mp = f'{d}/{v}_motions.json'
    mots = {}
    if os.path.exists(mp):
        M = json.load(open(mp)); M = M.get('motions', M) if isinstance(M, dict) else M
        mots = {x['annot_id']: x for x in M}

    xyz = L.load_laser_scan_xyz(a.data_root, v)
    n_points = len(xyz)
    tree = cKDTree(xyz)

    gt_ids = np.zeros(n_points, np.int64)
    inst_of = {}
    for i, x in enumerate(items):
        ix = np.asarray(x['indices'], np.int64); ix = ix[(ix >= 0) & (ix < n_points)]
        iid = int(ES.LABEL_TO_ID[x['label']]) * 1000 + i
        gt_ids[ix] = iid
        inst_of[x['annot_id']] = iid
    gt_file = f'{a.gt_dir}/{v}.txt'; np.savetxt(gt_file, gt_ids, fmt='%d')

    def motion_for(idx, label):
        pts = xyz[idx]
        cen = pts.mean(0)
        nbr = xyz[tree.query_ball_point(cen, NBR_R)]
        if len(nbr) > NBR_CAP:
            nbr = nbr[np.random.default_rng(0).choice(len(nbr), NBR_CAP, replace=False)]
        return MOT.predict_motion(label, pts, nbr)

    # GT motion per instance
    gt_motion = {}
    for x in items:
        m = mots.get(x['annot_id'])
        if m is None: continue
        dv = np.asarray(m.get('motion_dir', []), float)
        if dv.shape != (3,) or np.linalg.norm(dv) < 1e-6: continue
        oi = m.get('motion_origin_idx')
        gt_motion[inst_of[x['annot_id']]] = dict(
            motion_type=str(m.get('motion_type')),
            motion_dir=dv / np.linalg.norm(dv),
            motion_origin=(xyz[int(oi)] if oi is not None and 0 <= int(oi) < n_points else None))

    for side in ('gt', 'pred'):
        rows, masks, classes, scores, pmotion = [], [], [], [], {}
        for r in by_scene[v]:
            if side == 'gt':
                idx = r['gt_idx']
                if len(idx) < 10: continue
            else:
                idx, pts, agm = predict(r)
                r['_agree'] = agm
                if pts is not None and len(pts) >= 2:
                    ext = float(np.linalg.norm(pts - pts.mean(0), axis=1).mean())
                    r['_extent'] = float(np.exp(-abs(np.log(max(ext, 1e-6) / PRIOR))))
                else:
                    r['_extent'] = 0.0
                if len(idx) < 10:
                    stats['pred_too_small'] += 1
                    continue
            m = np.zeros(n_points, bool); m[idx[idx < n_points]] = True
            j = len(rows)
            masks.append(m); classes.append(int(ES.LABEL_TO_ID[r['label']]))
            scores.append(1.0 if side == 'gt' else float(CONF(r)))
            pmotion[_NEXT + j] = motion_for(idx, r['label'])
            rows.append(r)
        if not rows:
            continue
        pred = dict(pred_masks=np.stack(masks, 1),
                    pred_classes=np.array(classes), pred_scores=np.array(scores))
        globals()['_ID'] = _NEXT
        g2p, p2g = ES.assign_instances_for_scan(pred, gt_file)
        key = os.path.abspath(gt_file)
        for gate in ('raw', 'M', 'MA', 'MAO'):
            gg, pp = deepcopy(g2p), deepcopy(p2g)
            if gate != 'raw':
                for ln in gg:
                    for gt in gg[ln]:
                        gm = gt_motion.get(gt['instance_id'])
                        gt['matched_pred'] = [p for p in gt['matched_pred']
                                              if motion_correct(gm, pmotion.get(p['uuid']), gate)]
                for ln in pp:
                    for p in pp[ln]:
                        pm = pmotion.get(p['uuid'])
                        p['matched_gt'] = [g for g in p['matched_gt']
                                           if motion_correct(gt_motion.get(g['instance_id']), pm, gate)]
            chains[side][gate][key] = dict(gt=gg, pred=pp)
        _NEXT += len(rows)
    print(f'  [{vi}/{len(scenes)}] {v}', flush=True)

print('\n' + '=' * 78)
print('COMPOSED MOTION METRIC -- GT regions vs OUR PREDICTED regions')
print('=' * 78)
out = {}
print(f'{"chain":<10} {"AP25":>8} {"+M":>8} {"+MA":>8} {"+MAO":>8}')
for side, lbl in (('gt', 'GT region'), ('pred', 'PREDICTED')):
    vals = []
    for gate in ('raw', 'M', 'MA', 'MAO'):
        if not chains[side][gate]:
            vals.append(float('nan')); continue
        av = ES.compute_averages(ES.evaluate_matches(chains[side][gate]))
        vals.append(av['all_ap_25%'])
    out[side] = vals
    print(f'{lbl:<10} ' + ' '.join(f'{v:8.2f}' for v in vals))

print('\nMask3D-FM (supervised, hidden test set, NOT a shared column):')
print('           AP25 26.6   +M 23.8   +MA 9.8   +MAO 7.9')
print('\nRetention (fraction of AP25 surviving each gate):')
for side, lbl in (('gt', 'GT region'), ('pred', 'PREDICTED')):
    v = out[side]
    if v[0] and not np.isnan(v[0]):
        print(f'  {lbl:<10} +M {v[1]/v[0]:6.1%}  +MA {v[2]/v[0]:6.1%}  +MAO {v[3]/v[0]:6.1%}')
print('  Mask3D-FM  +M  89.8%  +MA  36.8%  +MAO  29.7%')
print(f'\nnotes: {dict(stats)}')
json.dump(out, open(a.out, 'w'), indent=1)
print(f'written to {a.out}')

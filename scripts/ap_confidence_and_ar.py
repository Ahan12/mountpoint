"""Raise AP by fixing the CONFIDENCE SCORE, and report AR alongside it.

Diagnosis this is built on. Against Fun3DU's baseline table:

    method       mAP   AP50   AP25   mIoU
    Fun3DU       7.6   16.9   33.3   15.2
    ours         0.53  1.49    9.51  ~11.7

Our mIoU is close to Fun3DU's, but our AP25 is 3.5x lower. Fun3DU's AP25 is
2.2x its mIoU; ours is 0.8x. Mask quality alone cannot explain that gap --
AP is a *ranked* metric, and ours is ranked by

    score = used_frames / n_frames

which is nearly constant (most elements use 5-6 of 6 frames). A near-constant
score gives AP an arbitrary ordering, so correct detections are interleaved with
wrong ones and precision collapses at every recall level. mIoU and AR do not
depend on ranking, which is exactly why they look fine while AP does not.

THIS IS NOT A SWEEP. No threshold is fitted. We replace a meaningless score with
physically meaningful ones already computed by the pipeline, and measure:

  frames    used_frames / n_frames            (current -- the baseline)
  agree     mean multi-view agreement acc/seen over the predicted points
            -- a point flagged by 4 of 5 views that saw it is better evidence
               than one flagged by 1 of 5. This is the fusion's own confidence.
  extent    metric plausibility: how close the region's real physical extent is
            to the fixed 1.70 cm prior measured over 757 GT elements. The same
            geometric test rescore_candidates_by_metric already applies per
            frame, reused as a per-instance score.
  combined  agree * extent -- both in [0,1], no weight to tune
  ORACLE    true IoU3D. NOT deployable; it measures how much AP is recoverable
            by ranking alone, and therefore whether this diagnosis is right.

AR is also computed because Fun3DU reports it and the official evaluator does
not: recall is ranking-independent, so it isolates detection quality from the
scoring problem above.

Run: python ap_confidence_and_ar.py --cache <drive>/liftcache_split0.pkl
"""
import os, sys, json, pickle, argparse, types, collections

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
from scipy.spatial import cKDTree
from _paths import REPO, DATA_ROOT, TOOLKIT, DOCS  # noqa: F401
CODE = REPO          # repo root: code, docs, figures
BASE = DATA_ROOT     # data root: scans, caches, exemplar DB

ap = argparse.ArgumentParser()
ap.add_argument('--cache', required=True)
ap.add_argument('--data-root', default=f'{DATA_ROOT}/data_merged')
ap.add_argument('--gt-dir', default=f'{DATA_ROOT}/apconf_gt')
ap.add_argument('--out', default=f'{DOCS}/ap_confidence.json')
a = ap.parse_args()
os.makedirs(a.gt_dir, exist_ok=True)

PRIOR = 0.0170
DEPLOY = dict(at=0.5, mv=1, mf=2, link=0.05)

_ID = 0
def mk(pred):
    return {_ID + i: dict(label_id=pred['pred_classes'][i], conf=pred['pred_scores'][i],
                          mask=pred['pred_masks'][:, i])
            for i in range(len(pred['pred_classes']))}
ES.make_pred_info = mk

cache = pickle.load(open(a.cache, 'rb'))
scenes = sorted({r['visit'] for r in cache})
print(f'{len(cache)} elements, {len(scenes)} scenes', flush=True)

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
    """Deployed thresholds. Returns (indices, xyz_of_those, agreement_mean)."""
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

def iou(p, g):
    if len(p) == 0: return 0.0
    A, B = set(p.tolist()), set(g.tolist())
    return len(A & B) / max(len(A | B), 1)

# ---------------- per-element quantities ----------------
for r in cache:
    idx, pts, agmean = predict(r)
    r['_pred'] = idx
    r['_iou'] = iou(idx, r['gt_idx'])
    r['_agree'] = agmean
    if pts is not None and len(pts) >= 2:
        ext = float(np.linalg.norm(pts - pts.mean(0), axis=1).mean())
        # plausibility in log space: 1.0 at the prior, decaying either side.
        # Symmetric in ratio, so 10x too big and 10x too small score equally.
        r['_extent'] = float(np.exp(-abs(np.log(max(ext, 1e-6) / PRIOR))))
    else:
        r['_extent'] = 0.0
    nfr = max(int(r.get('used_frames', 0)), 0)
    r['_frames'] = nfr / 6.0

CONFS = {
    'frames (current)': lambda r: r['_frames'],
    'agree':            lambda r: r['_agree'],
    'extent':           lambda r: r['_extent'],
    'combined':         lambda r: r['_agree'] * r['_extent'],
    'ORACLE (ceiling)': lambda r: r['_iou'],
}

# ---------------- build per-scene GT once ----------------
scene_items = collections.defaultdict(list)
for r in cache:
    scene_items[r['visit']].append(r)

gt_files, gt_counts = {}, {}
for v in scenes:
    d = f'{a.data_root}/{v}'
    A = json.load(open(f'{d}/{v}_annotations.json'))
    A = A.get('annotations', A) if isinstance(A, dict) else A
    items = [x for x in A if x.get('label') in ES.LABEL_TO_ID and x.get('indices')]
    n_points = max(int(max(x['indices'])) for x in items) + 1
    n_points = max(n_points, max((int(r['n_points']) for r in scene_items[v]), default=0))
    gt_ids = np.zeros(n_points, np.int64)
    for i, x in enumerate(items):
        ix = np.asarray(x['indices'], np.int64); ix = ix[(ix >= 0) & (ix < n_points)]
        gt_ids[ix] = int(ES.LABEL_TO_ID[x['label']]) * 1000 + i
    f = f'{a.gt_dir}/{v}.txt'; np.savetxt(f, gt_ids, fmt='%d')
    gt_files[v] = (f, n_points)
    gt_counts[v] = len(items)
print(f'GT instances total: {sum(gt_counts.values())}', flush=True)

def run_eval(conf_fn):
    global _ID
    _ID = 0
    matches = {}
    for v in scenes:
        f, n_points = gt_files[v]
        rows = [r for r in scene_items[v] if len(r['_pred']) > 0]
        if not rows:
            continue
        masks = np.zeros((n_points, len(rows)), bool)
        for j, r in enumerate(rows):
            ix = r['_pred']; ix = ix[ix < n_points]
            masks[ix, j] = True
        pred = dict(pred_masks=masks,
                    pred_classes=np.array([int(ES.LABEL_TO_ID[r['label']]) for r in rows]),
                    pred_scores=np.array([float(conf_fn(r)) for r in rows]))
        g2p, p2g = ES.assign_instances_for_scan(pred, f)
        matches[os.path.abspath(f)] = dict(gt=g2p, pred=p2g)
        _ID += len(rows)
    if not matches:
        return None, None
    avgs = ES.compute_averages(ES.evaluate_matches(matches))
    return avgs, matches

# ---------------- AR: ranking-independent, computed from the matcher ----------------
def compute_ar(matches):
    """Recall per class at each overlap threshold, averaged over classes --
    mirrors how AP is averaged, so the two are comparable."""
    ovs = ES.opt['overlaps']
    rec = np.full((len(ES.CLASS_LABELS), len(ovs)), np.nan)
    for li, lab in enumerate(ES.CLASS_LABELS):
        for oi, ov in enumerate(ovs):
            n_gt = n_hit = 0
            for m in matches.values():
                for gt in m['gt'].get(lab, []):
                    if gt['instance_id'] < 1000:
                        continue
                    n_gt += 1
                    best = 0.0
                    for p in gt['matched_pred']:
                        u = gt['vert_count'] + p['vert_count'] - p['intersection']
                        if u > 0:
                            best = max(best, p['intersection'] / u)
                    n_hit += (best >= ov)
            if n_gt:
                rec[li, oi] = n_hit / n_gt
    o50 = np.where(np.isclose(ovs, 0.5))[0]
    o25 = np.where(np.isclose(ovs, 0.25))[0]
    oall = np.where(~np.isclose(ovs, 0.25))[0]
    with np.errstate(invalid='ignore'):
        return (np.nanmean(rec[:, oall]) * 100,
                np.nanmean(rec[:, o50]) * 100,
                np.nanmean(rec[:, o25]) * 100)

results = {}
print(f'\n{"confidence":<20} {"mAP":>7} {"AP50":>7} {"AP25":>7}', flush=True)
base_matches = None
for name, fn in CONFS.items():
    avgs, matches = run_eval(fn)
    if avgs is None:
        print(f'{name:<20} (no predictions)'); continue
    results[name] = dict(mAP=avgs['all_ap'], AP50=avgs['all_ap_50%'], AP25=avgs['all_ap_25%'])
    if base_matches is None:
        base_matches = matches
    print(f'{name:<20} {avgs["all_ap"]:7.2f} {avgs["all_ap_50%"]:7.2f} {avgs["all_ap_25%"]:7.2f}',
          flush=True)

mAR, AR50, AR25 = compute_ar(base_matches)
ious = np.array([r['_iou'] for r in cache])
print(f'\n{"mAR":>7} {"AR50":>7} {"AR25":>7}   (ranking-independent)')
print(f'{mAR:7.2f} {AR50:7.2f} {AR25:7.2f}')
print(f'\nmIoU (mean IoU3D over all attempted): {ious.mean()*100:.2f}')
print(f'  zero rate {np.mean(ious < 0.02):.1%} | mean|non-zero {ious[ious>=0.02].mean()*100 if (ious>=0.02).any() else 0:.2f}')
print(f'  elements {len(ious)}, produced a region {sum(1 for r in cache if len(r["_pred"])>0)}')

print('\nFun3DU baseline table for reference:')
print('  OpenMask3D  mAP 0.2  AP50 0.2  AP25 0.4  | mAR 20.3 AR50 24.5 AR25 27.0 | mIoU 0.2')
print('  OpenIns3D   mAP 0.0  AP50 0.0  AP25 0.0  | mAR 40.5 AR50 46.7 AR25 51.5 | mIoU 0.1')
print('  LERF        mAP 0.0  AP50 0.0  AP25 0.0  | mAR 34.2 AR50 35.1 AR25 36.0 | mIoU 0.0')
print('  Fun3DU      mAP 7.6  AP50 16.9 AP25 33.3 | mAR 27.4 AR50 38.2 AR25 46.7 | mIoU 15.2')

results['AR'] = dict(mAR=mAR, AR50=AR50, AR25=AR25)
results['mIoU'] = float(ious.mean() * 100)
json.dump(results, open(a.out, 'w'), indent=1)
print(f'\nwritten to {a.out}')

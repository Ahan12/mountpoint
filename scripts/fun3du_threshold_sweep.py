"""Find Fun3DU's best readout threshold before judging Fun3DU.

WHAT WENT WRONG THE FIRST TIME. motion_on_fun3du.py thresholded the lifted score
field the way Fun3DU's evaluate.py writes it:

    np_normalize(acc_f / n_views) > 0.7

np_normalize rescales each file to [0,1], so "> 0.7" keeps the top 30% of the
score RANGE, not the top 30% of the points. Where scores are concentrated --
the normal case -- that admits almost the whole parent object. Measured on the
7-visit run: Fun3DU regions were 14.2x the median ground-truth element, recall
40.8% but precision 0.003, so IoU collapsed and only 9 of 116 elements cleared
IoU 0.25. Recall of 40.8% is consistent with Fun3DU's published AP25 of 33.3,
so the model found these elements; our readout threw the localisation away.

WHAT THIS DOES. Sweeps two readout families and reports Fun3DU's detection rate
and mIoU at each, so Table 2 is built at the operating point BEST FOR FUN3DU
rather than the one that happens to flatter us:

  fixed t  : keep points whose normalised score exceeds t
  top-k    : keep the k highest-scoring points, k a multiple of the median
             ground-truth element size -- a size prior, not an oracle

One threshold is chosen for the whole set. The per-element best is reported
separately and only as a ceiling; it is an oracle and is not a usable operating
point.

INPUTS (all from the Fun3DU Colab run, none currently local):
  $PCDS/<visit>_<desc_id>.npz        stage-4 lifting output
  $SFDATA/<visit>/<visit>_laser_scan.ply
  $SFDATA/<visit>/<visit>_crop_mask.npy
  $SFDATA/<visit>/<visit>_{annotations,motions,descriptions}.json

Run:  PCDS=... SFDATA=... .venv/bin/python scripts/fun3du_threshold_sweep.py
"""
import os, sys, json, glob

import numpy as np
from _paths import REPO, DATA_ROOT, TOOLKIT, DOCS  # noqa: F401
CODE = REPO          # repo root: code, docs, figures
BASE = DATA_ROOT     # data root: scans, caches, exemplar DB

PCDS   = os.environ.get('PCDS',   f'{CODE}/fun3du_pcds')
SFDATA = os.environ.get('SFDATA', f'{CODE}/scenefun3d 2/data')
FIXED  = [0.70, 0.80, 0.90, 0.95, 0.98, 0.99]
TOPK   = [1.0, 2.0, 4.0, 8.0]
DET    = 0.25

def np_normalize(a):
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)

def readout(score, mode, param, k_base):
    if mode == 'fixed':
        return np.flatnonzero(score > param)
    k = max(1, int(round(k_base * param)))
    return (np.flatnonzero(score > 0) if k >= len(score)
            else np.argpartition(-score, k)[:k])

def load_scan(visit):
    from plyfile import PlyData
    v = PlyData.read(f'{SFDATA}/{visit}/{visit}_laser_scan.ply')['vertex']
    return np.stack([v['x'], v['y'], v['z']], 1).astype(np.float64)

def unwrap(path, key):
    d = json.load(open(path))
    return d[key] if isinstance(d, dict) else d

def preflight():
    """Name every missing input rather than failing on the first one."""
    files = sorted(glob.glob(f'{PCDS}/*.npz'))
    missing = []
    if not files:
        missing.append(f'{PCDS}/*.npz  (stage-4 lifting output)')
    visits = sorted({os.path.basename(f).split('_')[0] for f in files})
    for v in visits:
        for suf in ('_laser_scan.ply', '_crop_mask.npy', '_annotations.json',
                    '_motions.json', '_descriptions.json'):
            p = f'{SFDATA}/{v}/{v}{suf}'
            if not os.path.exists(p): missing.append(p)
    return files, visits, missing

def main():
    files, visits, missing = preflight()
    if missing:
        print('Cannot run yet -- these inputs are on Drive, not local:\n')
        for m in missing[:14]: print('   ', m)
        if len(missing) > 14: print(f'    ... and {len(missing)-14} more')
        print('\nCopy them locally (no GPU needed, a CPU Colab runtime can do it),'
              '\nthen re-run with PCDS=... SFDATA=... set.')
        return 1

    from sf3d_data import load
    k_base = int(np.median([len(r['pts']) for r in load()]))
    print(f'{len(files)} pcds over {len(visits)} visits; '
          f'median GT element = {k_base} points\n')

    # ---- gather (score field, gt index set) pairs once ----
    pairs = []
    for visit in visits:
        full = load_scan(visit)
        crop_idx = np.where(np.load(f'{SFDATA}/{visit}/{visit}_crop_mask.npy'))[0]
        ann  = {a['annot_id']: a for a in
                unwrap(f'{SFDATA}/{visit}/{visit}_annotations.json', 'annotations')}
        desc = unwrap(f'{SFDATA}/{visit}/{visit}_descriptions.json', 'descriptions')
        for d in desc:
            f = f'{PCDS}/{visit}_{d["desc_id"]}.npz'
            if not os.path.exists(f): continue
            z = np.load(f)
            if 'acc_f' not in z or float(z['acc_f'].max()) <= 0: continue
            s = np_normalize(z['acc_f'] / np.maximum(z['n_views'], 1))
            n = min(len(s), len(crop_idx))
            for aid in d['annot_id']:
                a = ann.get(aid)
                if a is None or a.get('label') == 'exclude': continue
                gi = a['indices']
                if isinstance(gi, str): gi = json.loads(gi)
                gi = set(np.asarray(gi, int).tolist())
                if len(gi) < 10: continue
                pairs.append((s[:n], crop_idx[:n], gi))
    print(f'{len(pairs)} (region, ground-truth element) pairs\n')

    def evaluate(mode, p):
        ious, sizes = [], []
        for s, ci, gi in pairs:
            sel = readout(s, mode, p, k_base)
            fi  = set(ci[sel].tolist())
            inter = len(fi & gi)
            ious.append(inter / max(len(fi | gi), 1))
            sizes.append(len(fi))
        ious = np.array(ious)
        return dict(median=float(np.median(ious)), mean=float(ious.mean()),
                    det=float((ious >= DET).mean()), size=int(np.median(sizes)))

    print(f'{"readout":<14}{"median IoU":>11}{"mean IoU":>10}'
          f'{"det@0.25":>10}{"med size":>10}')
    best, rows = None, []
    for mode, params in (('fixed', FIXED), ('topk', TOPK)):
        for p in params:
            r = evaluate(mode, p); r.update(mode=mode, param=p); rows.append(r)
            print(f'{mode+" "+str(p):<14}{r["median"]:>11.3f}{r["mean"]:>10.3f}'
                  f'{100*r["det"]:>9.1f}%{r["size"]:>10}')
            if best is None or r['det'] > best['det']: best = r

    print(f'\nBEST SINGLE OPERATING POINT: {best["mode"]} {best["param"]}  '
          f'det@0.25 = {100*best["det"]:.1f}%  mean IoU = {best["mean"]:.3f}')
    print(f'(deployed readout was "fixed 0.7": det 7.8%, mean IoU 0.065, '
          f'median size 702 vs GT {k_base})')
    json.dump(dict(rows=rows, best=best, k_base=k_base, n_pairs=len(pairs)),
              open(f'{CODE}/docs/fun3du_threshold_sweep.json', 'w'), indent=1)
    print('wrote docs/fun3du_threshold_sweep.json')
    return 0

if __name__ == '__main__':
    sys.exit(main())

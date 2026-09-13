"""Table 2: the same unmodified motion module, fed regions from two very
different segmenters, on exactly the same scenes.

Rows
  GT      -- ground-truth annotation indices (the ceiling)
  Fun3DU  -- Fun3DU's own lifted masks, thresholded exactly as their
             evaluate.py does: np_normalize(acc_f / n_views) > 0.7

Nothing in sf3d/motion.py is modified or re-tuned between rows. The claim under
test is that RETENTION (the fraction of detections surviving each motion gate)
stays comparable even though the two mask sources differ enormously in quality.

Run on Colab, where the laser scans live:
    SF3D_ROOT=... FUN3DU_PCDS=... python scripts/motion_on_fun3du.py
"""
import os, sys, json, glob
import numpy as np
from scipy.spatial import cKDTree

DATA = os.environ.get('SF3D_SPLIT_DIR', f'{DATA_ROOT}/data')
PCDS = os.environ.get('FUN3DU_PCDS', f'{DATA_ROOT}/fun3du/pcds')

from _paths import REPO, DATA_ROOT, TOOLKIT, DOCS  # noqa: F401
CODE = REPO          # repo root: code, docs, figures
BASE = DATA_ROOT     # data root: scans, caches, exemplar DB
from sf3d import motion as M

THRESH = 0.7
NBR_R = 1.1          # same radius our annot_points_r11 caches use

def np_normalize(a):
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)

def load_scan(visit):
    from plyfile import PlyData
    v = PlyData.read(f'{DATA}/{visit}/{visit}_laser_scan.ply')['vertex'].data
    return np.stack([v['x'], v['y'], v['z']], 1).astype(np.float64)

def gate(rec, t, axis, org):
    """Deployed gate, identical to motion_ablations.evaluate."""
    if t != rec['mtype']:              return ('type', False)
    if M.OE(rec['dirv'], axis) >= 15:  return ('axis', False)
    if rec['mtype'] != 'rot':          return ('pass', True)
    if org is None or rec.get('org') is None: return ('origin', None)
    return ('pass', bool(M.MD(org, axis, rec['org'], rec['dirv']) < 0.25))

rows = []
visits = sorted(os.path.basename(os.path.dirname(p))
                for p in glob.glob(f'{DATA}/*/[0-9]*_annotations.json'))
visits = [v for v in visits if glob.glob(f'{PCDS}/{v}_*.npz')]
print(f'visits with Fun3DU output: {len(visits)}  {visits}', flush=True)

# GUARD: run_lifting skips files that already exist, so a failed stage 3 leaves
# zero-filled acc_f behind. That reads as a catastrophic Fun3DU result rather
# than as our bug, and it has very nearly happened once. Refuse to score it.
_pcds = sorted(glob.glob(f'{PCDS}/*.npz'))
_nz = sum(1 for f in _pcds if float(np.load(f)['acc_f'].max()) > 0)
print(f'pcds files: {len(_pcds)}   with non-zero acc_f: {_nz}', flush=True)
assert _pcds, 'no pcds written -- stage 4 did not run'
assert _nz >= 0.5 * len(_pcds), (
    f'only {_nz}/{len(_pcds)} pcds have non-zero acc_f -- stage 3 output was '
    f'missing when lifting ran. Clear exps/table2/pcds and re-run stage 4.')

for visit in visits:
    full = load_scan(visit)
    crop = np.load(f'{DATA}/{visit}/{visit}_crop_mask.npy')
    crop_idx = np.where(crop)[0]
    cropped = full[crop_idx]
    tree_full = cKDTree(full)

    # these files are dicts wrapping a list, not bare lists -- same unwrapping
    # the other scripts in this repo do (see composed_metric_gt.py)
    def _load(kind, key):
        d = json.load(open(f'{DATA}/{visit}/{visit}_{kind}.json'))
        return d[key] if isinstance(d, dict) else d
    ann = {a['annot_id']: a for a in _load('annotations', 'annotations')}
    mot = {m['annot_id']: m for m in _load('motions', 'motions')}
    desc = _load('descriptions', 'descriptions')

    for d in desc:
        f = f'{PCDS}/{visit}_{d["desc_id"]}.npz'
        if not os.path.exists(f):
            continue
        for aid in d['annot_id']:
            if aid not in ann or aid not in mot:
                continue
            a, m = ann[aid], mot[aid]
            if a.get('label') == 'exclude':
                continue
            dirv = np.asarray(m.get('motion_dir', []), float)
            if dirv.shape != (3,) or np.linalg.norm(dirv) < 1e-6:
                continue
            idx = a['indices']
            if isinstance(idx, str):
                idx = json.loads(idx)
            idx = np.asarray(idx, int)
            gt_pts = full[idx]
            if len(gt_pts) < 10:
                continue
            rec = dict(visit=visit, annot=aid, label=a['label'],
                       mtype=str(m['motion_type']), dirv=dirv/np.linalg.norm(dirv))
            oi = m.get('motion_origin_idx')
            if oi is not None:
                rec['org'] = full[int(oi)]

            # ---- GT row ----
            cen = gt_pts.mean(0)
            nbr = full[tree_full.query_ball_point(cen, NBR_R)]
            t, ax, og = M.predict_motion(rec['label'], gt_pts, nbr).values()
            rec['gt'] = gate(rec, t, ax, og)

            # ---- Fun3DU row ----
            z = np.load(f)
            p_f = np_normalize(z['acc_f'] / np.maximum(z['n_views'], 1))
            # acc_f is defined over the cropped scan; guard a length mismatch
            n = min(len(p_f), len(cropped))
            sel, base = p_f[:n] > THRESH, cropped[:n]
            if sel.sum() < 10:
                rec['f3'] = ('empty', False)
            else:
                fp = base[sel]
                # IoU3D of Fun3DU's region against the GT element, in full-scan
                # index space. Retention is defined on DETECTED elements, so a
                # region that simply missed the element is a detection failure
                # and must not be scored as a motion failure.
                f_idx = set(crop_idx[:n][sel].tolist())
                g_idx = set(idx.tolist())
                inter = len(f_idx & g_idx)
                rec['iou3d'] = inter / max(len(f_idx | g_idx), 1)
                cen2 = fp.mean(0)
                nbr2 = full[tree_full.query_ball_point(cen2, NBR_R)]
                if len(nbr2) < 50:
                    rec['f3'] = ('no_nbr', False)
                else:
                    t2, ax2, og2 = M.predict_motion(rec['label'], fp, nbr2).values()
                    rec['f3'] = gate(rec, t2, ax2, og2)
                rec['n_pred'] = int(sel.sum()); rec['n_gt'] = len(gt_pts)
            rows.append(rec)
    print(f'  {visit}: cumulative elements {len(rows)}', flush=True)

# ------------------------------------------------------------------ report
DET_IOU = 0.25   # SceneFun3D's own detection threshold

def summarize(key, subset=None):
    src = subset if subset is not None else rows
    ok = [r for r in src if r[key][1] is not None]
    passed = sum(r[key][1] for r in ok)
    fails = {}
    for r in ok:
        if not r[key][1]:
            fails[r[key][0]] = fails.get(r[key][0], 0) + 1
    return len(ok), passed, 100*passed/max(len(ok),1), fails

det = [r for r in rows if r.get('iou3d', 0) >= DET_IOU]
print(f'\n{"="*70}')
print('ALL ATTEMPTED ELEMENTS -- conflates detection with motion, shown for context')
print(f'{"="*70}')
print(f'{"source":<10} {"n":>5} {"passed":>7} {"gate %":>8}   failure breakdown')
for key, name in (('gt','GT'), ('f3','Fun3DU')):
    n, p, pct, fails = summarize(key)
    print(f'{name:<10} {n:>5} {p:>7} {pct:>7.1f}%   {fails}')

print(f'\n{"="*70}')
print(f'TABLE 2 -- RETENTION: only elements Fun3DU actually DETECTED (IoU3D >= {DET_IOU})')
print(f'{"="*70}')
print(f'Fun3DU detected {len(det)}/{len(rows)} elements '
      f'({100*len(det)/max(len(rows),1):.1f}%) at IoU3D >= {DET_IOU}')
print(f'{"source":<10} {"n":>5} {"passed":>7} {"gate %":>8}   failure breakdown')
for key, name in (('gt','GT'), ('f3','Fun3DU')):
    n, p, pct, fails = summarize(key, det)
    print(f'{name:<10} {n:>5} {p:>7} {pct:>7.1f}%   {fails}')
import numpy as _np
_i = _np.array([r.get('iou3d', 0.0) for r in rows])
print(f'\nFun3DU IoU3D against GT elements: median {_np.median(_i):.3f}  '
      f'mean {_i.mean():.3f}  zero {100*_np.mean(_i < 0.02):.1f}%')

json.dump([{k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in r.items()} for r in rows],
          open(f'{DOCS}/fun3du_table2_rows.json','w'), indent=1)
print(f'\nwritten {DOCS}/fun3du_table2_rows.json')

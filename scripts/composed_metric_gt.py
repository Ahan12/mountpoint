"""SceneFun3D's own composed motion metric (AP25 -> +M -> +MA -> +MAO),
computed on GROUND-TRUTH regions.

This is the ORACLE-SEGMENTATION setting: predictions ARE the GT masks, so
AP25 is 100 by construction and every subsequent drop is caused purely by
motion error. It is the motion module's ceiling under this benchmark's own
metric -- NOT comparable like-for-like with full-scene detection baselines
such as Mask3D-FM, which must also find the elements. Reported as a ceiling.

Uses the official evaluator unmodified: `assign_instances_for_scan`,
`evaluate_matches` and `compute_averages` are called as-is. The one patch is
the same correctness-neutral one notebook 03 uses -- `make_pred_info` assigns
each prediction a random uuid4, discarding which element it came from, so it
is swapped for a deterministic index. A uuid is only ever an opaque dict key,
so this changes zero AP math.
"""
import os, sys, json, glob, pickle
from copy import deepcopy
import numpy as np

TOOLKIT = '/tmp/sf3d_toolkit'
sys.path.insert(0, TOOLKIT)

if not hasattr(np, 'in1d'):
    np.in1d = np.isin

import eval.functionality_segmentation.eval_utils.eval_script as ES
from _paths import REPO, DATA_ROOT, TOOLKIT, DOCS  # noqa: F401
CODE = REPO          # repo root: code, docs, figures
BASE = DATA_ROOT     # data root: scans, caches, exemplar DB
from sf3d import motion as MOT

# --- deterministic prediction ids (see docstring) ---
# MUST be globally unique across scenes, not per-scene 0..N: evaluate_matches
# keeps ONE `pred_visited` dict keyed by uuid across every scene, so colliding
# ids make scene B's prediction look already-matched and silently destroy
# recall. That is why the original uses uuid4(). We keep determinism AND
# uniqueness with a running offset.
_ID_BASE = 0

def make_pred_info_det(pred):
    info_all = {}
    assert pred['pred_classes'].shape[0] == pred['pred_scores'].shape[0] == pred['pred_masks'].shape[1]
    for i in range(len(pred['pred_classes'])):
        info_all[_ID_BASE + i] = dict(label_id=pred['pred_classes'][i],
                                      conf=pred['pred_scores'][i],
                                      mask=pred['pred_masks'][:, i])
    return info_all
ES.make_pred_info = make_pred_info_det

def ply_n_points(path):
    """Read only the header -- the scans are 40-70 MB each."""
    with open(path, 'rb') as f:
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError(f'no vertex count in {path}')
            s = line.decode('ascii', 'ignore').strip()
            if s.startswith('element vertex'):
                return int(s.split()[-1])
            if s == 'end_header':
                raise RuntimeError(f'no vertex count in {path}')

def find_scan(visit):
    for d in ['scenefun3d', 'scenefun3d 2', 'scenefun3d 3']:
        p = f'{CODE}/{d}/data/{visit}/{visit}_laser_scan.ply'
        if os.path.exists(p):
            return p
    return None

# ---------------- motion predicate (paper Sec 6.3 nesting) ----------------
def motion_correct(gt_m, pred_m, gate):
    if gt_m is None or pred_m is None:
        return False
    if gt_m['motion_type'] != pred_m['motion_type']:
        return False
    if gate == 'M':
        return True
    if MOT.OE(gt_m['motion_dir'], pred_m['motion_dir']) >= 15:
        return False
    if gate == 'MA':
        return True
    if gt_m['motion_type'] != 'rot':
        return True            # prismatic is origin-invariant
    if gt_m['motion_origin'] is None or pred_m['motion_origin'] is None:
        return False
    return MOT.MD(pred_m['motion_origin'], pred_m['motion_dir'],
                  gt_m['motion_origin'], gt_m['motion_dir']) < 0.25

# ---------------- load cached per-element motion predictions ----------------
rows = pickle.load(open(f'{CODE}/scripts/repro_rows.pkl', 'rb'))
PRED = {(r['visit'], r['annot']): r for r in rows}
print(f'cached motion predictions: {len(PRED)} elements')

ann = {os.path.basename(os.path.dirname(f)): f
       for f in glob.glob(f'{BASE}/data/*/*_annotations.json')}
mot = {os.path.basename(os.path.dirname(f)): f
       for f in glob.glob(f'{BASE}/data/*/*_motions.json')}

NPOINTS = json.load(open('/private/tmp/claude-501/-Users-ahanb-Desktop-Research-sf3dSegMotion/cbd4f47d-18cf-4a81-b63c-89073b623f82/scratchpad/npoints.json'))
print(f'scan vertex counts available for {len(NPOINTS)} visits')

GT_DIR = '/tmp/composed_gt'
os.makedirs(GT_DIR, exist_ok=True)
all_matches = {g: {} for g in ('raw', 'M', 'MA', 'MAO')}
n_inst = n_gt_mot = n_pred_mot = 0
skipped = []
_NEXT_ID = 0

LIMIT = int(os.environ.get('LIMIT', '0'))
_visits = sorted(set(ann) & set(mot))
if LIMIT:
    _visits = _visits[:LIMIT]
for visit in _visits:
    global_ok = True
    scan = find_scan(visit)
    if scan is not None:
        n_points = ply_n_points(scan)
    elif visit in NPOINTS:
        # Only 4 of 48 laser scans are held locally, but the evaluator needs
        # nothing from a scan except its vertex count. Those were fetched from
        # the SceneFun3D host with 400-byte HTTP range requests over the PLY
        # header, so the metric runs at full scale without 2.5 GB of downloads.
        n_points = NPOINTS[visit]
    else:
        skipped.append(visit); continue

    A = json.load(open(ann[visit])); A = A['annotations'] if isinstance(A, dict) else A
    Mo = json.load(open(mot[visit])); Mo = Mo['motions'] if isinstance(Mo, dict) else Mo
    gt_motion_raw = {m['annot_id']: m for m in Mo}
    orgf = f'{BASE}/annot_origins/{visit}.npz'
    oz = np.load(orgf) if os.path.exists(orgf) else None

    items = [a for a in A if a.get('label') in ES.LABEL_TO_ID and len(a.get('indices', [])) > 0]
    if not items:
        continue

    _ID_BASE = _NEXT_ID
    _NEXT_ID += len(items)
    gt_ids = np.zeros(n_points, dtype=np.int64)
    masks = np.zeros((n_points, len(items)), dtype=bool)
    classes, scores = [], []
    gt_motion, pred_motion = {}, {}

    for i, a in enumerate(items):
        lid = int(ES.LABEL_TO_ID[a['label']])
        idx = np.asarray(a['indices'], dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < n_points)]
        inst_id = lid * 1000 + i
        gt_ids[idx] = inst_id
        masks[idx, i] = True
        classes.append(lid); scores.append(1.0)
        n_inst += 1

        # GT motion for this instance
        gm = gt_motion_raw.get(a['annot_id'])
        if gm is not None:
            d = np.asarray(gm.get('motion_dir', []), float)
            org = (oz[a['annot_id']].astype(np.float64)
                   if (oz is not None and a['annot_id'] in oz.files) else None)
            if d.shape == (3,) and np.linalg.norm(d) > 1e-6:
                gt_motion[inst_id] = dict(motion_type=str(gm.get('motion_type')),
                                          motion_dir=d / np.linalg.norm(d),
                                          motion_origin=org)
                n_gt_mot += 1

        # our predicted motion for the same region
        pr = PRED.get((visit, a['annot_id']))
        if pr is not None:
            pred_motion[_ID_BASE + i] = dict(motion_type=pr['p_type'],
                                  motion_dir=pr['p_dir'],
                                  motion_origin=pr['p_org'])
            n_pred_mot += 1

    gt_file = f'{GT_DIR}/{visit}.txt'
    np.savetxt(gt_file, gt_ids, fmt='%d')
    pred = dict(pred_masks=masks, pred_classes=np.array(classes), pred_scores=np.array(scores))
    gt2pred, pred2gt = ES.assign_instances_for_scan(pred, gt_file)

    key = os.path.abspath(gt_file)
    for gate in ('raw', 'M', 'MA', 'MAO'):
        g2p, p2g = deepcopy(gt2pred), deepcopy(pred2gt)
        if gate != 'raw':
            for ln in g2p:
                for gt in g2p[ln]:
                    gm = gt_motion.get(gt['instance_id'])
                    gt['matched_pred'] = [p for p in gt['matched_pred']
                                          if motion_correct(gm, pred_motion.get(p['uuid']), gate)]
            for ln in p2g:
                for p in p2g[ln]:
                    pm = pred_motion.get(p['uuid'])
                    p['matched_gt'] = [g for g in p['matched_gt']
                                       if motion_correct(gt_motion.get(g['instance_id']), pm, gate)]
        all_matches[gate][key] = dict(gt=g2p, pred=p2g)

print(f'scenes: {len(all_matches["raw"])}   instances: {n_inst}')
print(f'GT motion resolvable: {n_gt_mot}   our predictions available: {n_pred_mot}')
if skipped:
    print(f'skipped (no local laser scan): {len(skipped)} {skipped}')

print(f"\n{'gate':<6} {'AP25':>7} {'AP50':>7} {'mAP':>7}   meaning")
out = {}
for gate, meaning in [('raw', 'segmentation only (must be 100 -- GT as prediction)'),
                      ('M',   '+ correct motion type'),
                      ('MA',  '+ axis within 15 deg'),
                      ('MAO', '+ origin MD < 0.25 m (revolute only)')]:
    avgs = ES.compute_averages(ES.evaluate_matches(all_matches[gate]))
    out[gate] = avgs
    print(f"{gate:<6} {avgs['all_ap_25%']:>6.1f}% {avgs['all_ap_50%']:>6.1f}% "
          f"{avgs['all_ap']:>6.1f}%   {meaning}")

print('\nper-class AP25 by gate:')
print(f"  {'class':<12} {'raw':>7} {'M':>7} {'MA':>7} {'MAO':>7}")
for cl in ES.CLASS_LABELS:
    vals = []
    for g in ('raw', 'M', 'MA', 'MAO'):
        v = out[g]['classes'][cl]['ap25%']
        vals.append(f'{v*100:6.1f}%' if v <= 1.0 else f'{v:6.1f}%')
    print(f"  {cl:<12} " + ' '.join(vals))

pickle.dump(out, open('/tmp/composed_gt_results.pkl', 'wb'))
print('\nNOTE: oracle segmentation. AP25=100 by construction; every drop below')
print('is motion error alone. Not like-for-like with full-scene detection.')

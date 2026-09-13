"""Load the cached SceneFun3D elements once, for figures and analysis.

Deliberately NOT importing motion_ablations: that module runs the whole ablation
table at import time (minutes), and it is a validated script whose published
numbers we do not want to perturb by refactoring it. The loader below is the
same twenty lines, lifted verbatim in behaviour.
"""
import os, sys, json, glob
import numpy as np

from _paths import REPO, DATA_ROOT, TOOLKIT, DOCS  # noqa: F401
CODE = REPO          # repo root: code, docs, figures
BASE = DATA_ROOT     # data root: scans, caches, exemplar DB
from sf3d import motion as M

def load(cache='annot_points_r11'):
    """Load every cached element. Fails immediately and by name if the data
    root is not set -- an unset SF3D_ROOT used to surface as a ZeroDivisionError
    in whichever script happened to divide by the element count first."""
    if not os.path.isdir(f'{BASE}/data'):
        raise SystemExit(
            f'No data under SF3D_ROOT={BASE}\n'
            f'  expected {BASE}/data/<visit>/<visit>_annotations.json\n\n'
            f'Set SF3D_ROOT to your SceneFun3D data root.\n'
            f'See README.md, "Data layout".')
    if not os.path.isdir(f'{BASE}/{cache}'):
        raise SystemExit(
            f'Missing cache {BASE}/{cache}\n\n'
            f'Build it with:\n'
            f'  python scripts/build_annot_caches.py --data $SF3D_ROOT/data \\\n'
            f'      --pts-dir $SF3D_ROOT/{cache} \\\n'
            f'      --org-dir $SF3D_ROOT/annot_origins --radius 1.10 --cap 120000')
    ann = {os.path.basename(os.path.dirname(f)): f
           for f in glob.glob(f'{BASE}/data/*/*_annotations.json')}
    mot = {os.path.basename(os.path.dirname(f)): f
           for f in glob.glob(f'{BASE}/data/*/*_motions.json')}
    rows = []
    for v in sorted(set(ann) & set(mot)):
        pf = f'{BASE}/{cache}/{v}.npz'
        if not os.path.exists(pf): continue
        pz = np.load(pf)
        of = f'{BASE}/annot_origins/{v}.npz'
        oz = np.load(of) if os.path.exists(of) else None
        A = json.load(open(ann[v])); A = A['annotations'] if isinstance(A, dict) else A
        Mo = json.load(open(mot[v])); Mo = Mo['motions'] if isinstance(Mo, dict) else Mo
        lab = {a['annot_id']: a.get('label') for a in A}
        for m in Mo:
            aid = m.get('annot_id')
            if aid not in pz.files or lab.get(aid) == 'exclude': continue
            pts = pz[aid].astype(np.float64)
            if len(pts) < 10: continue
            d = np.asarray(m.get('motion_dir', []), float)
            if d.shape != (3,) or np.linalg.norm(d) < 1e-6: continue
            r = dict(visit=v, annot=aid, label=lab.get(aid),
                     mtype=str(m.get('motion_type')), dirv=d/np.linalg.norm(d),
                     pts=pts,
                     nbr=pz['NBR_'+aid].astype(np.float64) if 'NBR_'+aid in pz.files else None)
            if oz is not None and aid in oz.files:
                r['org'] = oz[aid].astype(np.float64)
            rows.append(r)
    return rows

def score(r):
    """Run the DEPLOYED motion module on one row and grade it."""
    p = M.predict_motion(r['label'], r['pts'], r['nbr'])
    t, axis, org = p['motion_type'], p['motion_dir'], p['motion_origin']
    oe = M.OE(r['dirv'], axis)
    md = (M.MD(org, axis, r['org'], r['dirv'])
          if (org is not None and 'org' in r) else None)
    gt_rot = 'rot' in r['mtype'].lower()
    gate = (False if oe >= 15 else True if not gt_rot
            else None if md is None else bool(md < 0.25))
    return dict(p_type=t, axis=axis, org=org, oe=oe, md=md,
                gt_rot=gt_rot, gate=gate)

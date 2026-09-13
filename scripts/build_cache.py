"""Build annot_points / annot_origins caches for new visits (split0).

Extracted verbatim in behaviour from 02_motion_parameters.ipynb cell 8, but
parameterised by radius so BOTH caches the pipeline actually uses can be
produced:

  annot_points      NBR_RADIUS=0.60, cap  40000   -- better for DIRECTION
  annot_points_r11  NBR_RADIUS=1.10, cap 120000   -- better for ORIGIN

Both are needed: `predict_motion` takes a single `nbr`, and the two caches trade
off (see docs/motion-reproduction.md). Keeping both lets us report either
without regenerating.

Run on Colab:
  python scripts/build_annot_caches.py --data $SF3D_ROOT/data \
      --pts-dir $SF3D_ROOT/annot_points \
      --org-dir $SF3D_ROOT/annot_origins --radius 0.60 --cap 40000
  python scripts/build_annot_caches.py --data $SF3D_ROOT/data \
      --pts-dir $SF3D_ROOT/annot_points_r11 \
      --org-dir $SF3D_ROOT/annot_origins --radius 1.10 --cap 120000

  The r11 cache (radius 1.10 m) is the one every motion result uses; the 0.60 m
  cache exists only to reproduce the neighbourhood-radius study.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mountpoint import config as C, motion as M
import os, glob, json, argparse
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

def rec(o, *ks):
    if isinstance(o, list):
        return o
    for k in ks:
        if isinstance(o, dict) and k in o:
            return o[k]
    return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='dir containing <visit>/ subdirs')
    ap.add_argument('--pts-dir', required=True)
    ap.add_argument('--org-dir', required=True)
    ap.add_argument('--radius', type=float, default=0.60)
    ap.add_argument('--cap', type=int, default=40000)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()

    os.makedirs(a.pts_dir, exist_ok=True)
    os.makedirs(a.org_dir, exist_ok=True)

    visits = sorted(v for v in os.listdir(a.data) if os.path.isdir(f'{a.data}/{v}'))
    todo = [v for v in visits
            if a.force or not os.path.exists(f'{a.pts_dir}/{v}.npz')]
    print(f'{len(visits)} visits, {len(todo)} to build '
          f'(radius={a.radius}, cap={a.cap})', flush=True)

    for i, v in enumerate(todo, 1):
        vdir = f'{a.data}/{v}'
        ply = glob.glob(f'{vdir}/{v}_laser_scan.ply')
        if not ply:
            print(f'  [{i}/{len(todo)}] {v}: no scan', flush=True); continue
        annp = glob.glob(f'{vdir}/{v}_annotations.json')
        motp = glob.glob(f'{vdir}/{v}_motions.json')
        if not annp:
            print(f'  [{i}/{len(todo)}] {v}: no annotations', flush=True); continue

        vx = PlyData.read(ply[0])['vertex']
        pts = np.stack([vx['x'], vx['y'], vx['z']], 1).astype(np.float32)
        ann = json.load(open(annp[0]))
        tree = cKDTree(pts)

        store, bad = {}, 0
        for an in rec(ann, 'annotations', 'annotation'):
            if an.get('label') == 'exclude':
                continue
            idx = np.asarray(an['indices'], np.int64)
            if len(idx) == 0 or idx.max() >= len(pts):
                bad += 1; continue
            reg = pts[idx]
            store[an['annot_id']] = reg
            cen = reg.mean(0)
            nbr = pts[tree.query_ball_point(cen, a.radius)]
            if len(nbr) > a.cap:
                nbr = nbr[np.random.default_rng(0).choice(len(nbr), a.cap, replace=False)]
            store['NBR_' + an['annot_id']] = nbr.astype(np.float32)
        np.savez_compressed(f'{a.pts_dir}/{v}.npz', **store)

        n_org = 0
        if motp:
            mot = json.load(open(motp[0]))
            origins = {}
            for m in rec(mot, 'motions', 'motion'):
                oi = m.get('motion_origin_idx')
                aid = m.get('annot_id')
                if oi is None or aid is None:
                    continue
                oi = int(oi)
                if 0 <= oi < len(pts):
                    origins[aid] = pts[oi].astype(np.float32)
            if origins:
                np.savez_compressed(f'{a.org_dir}/{v}.npz', **origins)
                n_org = len(origins)

        n_el = len([k for k in store if not k.startswith('NBR_')])
        print(f'  [{i}/{len(todo)}] {v}: {n_el} elements, {n_org} origins'
              f'{f", {bad} bad-index" if bad else ""}', flush=True)

    print('done')

if __name__ == '__main__':
    main()

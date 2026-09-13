"""Motion parameters on SceneFun3D ground-truth regions.

Reports the per-class table and the overall motion gate (paper Sec. 7.1), plus
signed direction accuracy (Sec. 4.2). Scene-clustered bootstrap intervals,
because elements in one scene share a room and a piece of furniture and do not
fail independently.

    python scripts/eval_motion.py
    python scripts/eval_motion.py --bootstrap 0     # skip intervals, ~10x faster
"""
import argparse, collections, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mountpoint import config as C, motion as M

def load(cache_name='cache', origins_name='origins'):
    """Cached per-element points and neighbourhoods, built by build_cache.py."""
    cache = os.path.join(C.ROOT, cache_name)
    C.require(C.DATA, cache)
    ann = {os.path.basename(os.path.dirname(f)): f
           for f in glob.glob(f'{C.DATA}/*/*_annotations.json')}
    mot = {os.path.basename(os.path.dirname(f)): f
           for f in glob.glob(f'{C.DATA}/*/*_motions.json')}
    rows = []
    for v in sorted(set(ann) & set(mot)):
        pf = f'{cache}/{v}.npz'
        if not os.path.exists(pf):
            continue
        pz = np.load(pf)
        of = f'{C.ROOT}/{origins_name}/{v}.npz'
        oz = np.load(of) if os.path.exists(of) else None
        unwrap = lambda p, k: (lambda d: d[k] if isinstance(d, dict) else d)(json.load(open(p)))
        lab = {a['annot_id']: a.get('label') for a in unwrap(ann[v], 'annotations')}
        for m in unwrap(mot[v], 'motions'):
            aid = m.get('annot_id')
            if aid not in pz.files or lab.get(aid) == 'exclude':
                continue
            pts = pz[aid].astype(np.float64)
            d = np.asarray(m.get('motion_dir', []), float)
            if len(pts) < 10 or d.shape != (3,) or np.linalg.norm(d) < 1e-6:
                continue
            r = dict(visit=v, annot=aid, label=lab[aid], mtype=str(m.get('motion_type')),
                     dirv=d / np.linalg.norm(d), pts=pts,
                     nbr=pz['NBR_' + aid].astype(np.float64) if 'NBR_' + aid in pz.files else None)
            if oz is not None and aid in oz.files:
                r['org'] = oz[aid].astype(np.float64)
            rows.append(r)
    return rows

def score(r):
    p = M.predict_motion(r['label'], r['pts'], r['nbr'])
    oe = M.OE(r['dirv'], p['motion_dir'])
    md = (M.MD(p['motion_origin'], p['motion_dir'], r['org'], r['dirv'])
          if p['motion_origin'] is not None and 'org' in r else None)
    gt_rot = 'rot' in r['mtype'].lower()
    gate = (False if oe >= 15 else True if not gt_rot
            else None if md is None else bool(md < 0.25))
    return dict(oe=oe, md=md, gt_rot=gt_rot, p_type=p['motion_type'], gate=gate)

def ci(groups, stat, B, rng):
    keys = list(groups)
    point = stat([r for k in keys for r in groups[k]])
    if not B:
        return point, None, None
    draws = [s for _ in range(B)
             if (s := stat([r for i in rng.integers(0, len(keys), len(keys))
                            for r in groups[keys[i]]])) is not None]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, lo, hi

def show(name, point, lo, hi):
    tail = f'  95% CI [{lo:5.1f}, {hi:5.1f}]' if lo is not None else ''
    print(f'  {name:<22} {point:6.1f}{tail}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bootstrap', type=int, default=10000)
    ap.add_argument('--cache', default='cache')
    ap.add_argument('--origins', default='origins')
    a = ap.parse_args()

    rows = load(a.cache, a.origins)
    res = [dict(visit=r['visit'], label=r['label'], **score(r)) for r in rows]
    print(f'{len(rows)} elements, {len({r["visit"] for r in rows})} scenes\n')

    print('median error by class')
    by_label = collections.defaultdict(list)
    for x in res:
        by_label[x['label']].append(x)
    for lab in sorted(by_label):
        S = by_label[lab]
        oes = [x['oe'] for x in S]
        mds = [x['md'] for x in S if x['md'] is not None]
        g = [x['gate'] for x in S if x['gate'] is not None]
        print(f'  {lab:<12} n={len(S):>3}  OE {np.median(oes):6.2f}deg'
              f'  MD {("%.3f m" % np.median(mds)) if mds else "     n/a":>9}'
              f'  gate {100*np.mean(g) if g else float("nan"):5.1f}%')

    by_scene = collections.defaultdict(list)
    for x in res:
        by_scene[x['visit']].append(x)
    rng = np.random.default_rng(0)
    gated = lambda R: (100*np.mean([x['gate'] for x in R if x['gate'] is not None])
                       if any(x['gate'] is not None for x in R) else None)
    typed = lambda R: 100*np.mean([('rot' if x['gt_rot'] else 'trans') == x['p_type']
                                   for x in R]) if R else None
    print('\noverall')
    show('motion gate', *ci(by_scene, gated, a.bootstrap, rng))
    show('type accuracy', *ci(by_scene, typed, a.bootstrap, rng))
    for nm, keep in [('gate | prismatic', lambda x: not x['gt_rot']),
                     ('gate | revolute', lambda x: x['gt_rot'])]:
        sub = {v: s for v, R in by_scene.items() if (s := [x for x in R if keep(x)])}
        show(nm, *ci(sub, gated, a.bootstrap, rng))

    # ---- signed direction (Sec. 4.2) ----
    n = unsigned = deployed = signed = 0
    for r in rows:
        if 'rot' in r['mtype'].lower():
            continue
        sd = M.signed_dir(r['label'], r['pts'], r['nbr'])
        if sd is None:
            continue
        p = M.predict_motion(r['label'], r['pts'], r['nbr'])
        t = np.cos(np.deg2rad(15))
        n += 1
        unsigned += abs(float(sd @ r['dirv'])) > t
        deployed += float(p['motion_dir'] @ r['dirv']) > t
        signed   += float(sd @ r['dirv']) > t
    if n:
        print(f'\nsigned direction, n={n} prismatic')
        print(f'  {"unsigned (ceiling)":<22} {100*unsigned/n:6.1f}')
        print(f'  {"signed, axis only":<22} {100*deployed/n:6.1f}')
        print(f'  {"signed, oriented":<22} {100*signed/n:6.1f}')

if __name__ == '__main__':
    main()

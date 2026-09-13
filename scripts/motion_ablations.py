"""Motion ablations: remove each component, measure what it was worth.

Every row is the FULL method minus one design decision, scored end-to-end on the
motion gate (OE<15 deg for all; AND MD<0.25 m for revolute). Nothing is re-tuned
when a component is removed -- the point is what that component contributes, not
what the method could achieve without it after refitting.

Components ablated, and the claim each one supports:

  A  parent surface normal      "motion is determined by what the element is
                                 attached to, not by its own shape"
  B  gravity prior for hinges   "hinge axes are vertical; use the scan's up axis
                                 rather than fitting anything"
  C  combined type rule         "panel height OR offset disambiguates the two
                                 mixed affordance classes"
  D  corner-score edge choice   "a hinge is structurally anchored, so the panel
                                 edge with more non-coplanar neighbours is the
                                 hinge side"
  E  cascading origin fallback  "prefer a weak answer to no answer"
  F  robust trimmed plane fit   "the shell contains off-panel points that least
                                 squares cannot down-weight"
  G  adaptive shell growth      "a shell too thin to fit a plane is ill-posed;
                                 grow it rather than fall back to the region's
                                 own normal"

Run: python scripts/motion_ablations.py            (wide cache, deployed config)
     CACHE=annot_points python scripts/motion_ablations.py
"""
import os, sys, json, glob, collections
import numpy as np

from _paths import REPO, DATA_ROOT, TOOLKIT, DOCS  # noqa: F401
CODE = REPO          # repo root: code, docs, figures
BASE = DATA_ROOT     # data root: scans, caches, exemplar DB
from sf3d import motion as M
from scipy.spatial import cKDTree

G = M.G
CACHE = os.environ.get('CACHE', 'annot_points_r11')

# ---------------------------------------------------------------- primitives
def plane_svd(pts):
    return np.linalg.svd(pts - pts.mean(0), full_matrices=False)[2][2]

def plane_robust(pts, n_iter=3, k=2.5, min_keep=30):
    P = pts; n = plane_svd(P)
    for _ in range(n_iter):
        r = np.abs((P - P.mean(0)) @ n)
        mad = np.median(np.abs(r - np.median(r)))
        if mad < 1e-9: break
        keep = r < np.median(r) + k * 1.4826 * mad
        if keep.sum() < min_keep: break
        P = P[keep]; n = plane_svd(P)
    return n

def parent_normal(nbr, cen, extent, robust=True, adaptive=True,
                  lo=1.5, hi=4.0, min_pts=50, max_hi=12.0):
    if nbr is None or len(nbr) == 0:
        return None
    d = np.linalg.norm(nbr - cen, axis=1)
    h = hi
    sh = nbr[(d > extent * lo) & (d < extent * h)]
    if adaptive:
        while len(sh) < min_pts and h < max_hi:
            h *= 1.5
            sh = nbr[(d > extent * lo) & (d < extent * h)]
    if len(sh) < min_pts:
        return None
    return plane_robust(sh) if robust else plane_svd(sh)

def panel_component(nbr, cen, extent, pn, hi=40, plane_tol=0.05, link=0.02, min_pts=150):
    if nbr is None or pn is None: return None
    d = np.linalg.norm(nbr - cen, axis=1)
    near = nbr[(d > extent * 1.2) & (d < extent * hi)]
    if len(near) < min_pts: return None
    cop = near[np.abs((near - cen) @ pn) < plane_tol]
    if len(cop) < min_pts: return None
    tree = cKDTree(cop)
    seed = int(np.argmin(np.linalg.norm(cop - cen, axis=1)))
    seen = np.zeros(len(cop), bool); seen[seed] = True; fr = [seed]
    while fr:
        nx = []
        for i in fr:
            for j in tree.query_ball_point(cop[i], link):
                if not seen[j]: seen[j] = True; nx.append(j)
        fr = nx
    return cop[seen] if seen.sum() >= min_pts else None

def cop_points(nbr, cen, extent, pn, hi=40, plane_tol=0.05):
    if nbr is None or pn is None: return None
    d = np.linalg.norm(nbr - cen, axis=1)
    near = nbr[(d > extent * 1.2) & (d < extent * hi)]
    if len(near) < 30: return None
    cop = near[np.abs((near - cen) @ pn) < plane_tol]
    return cop if len(cop) >= 30 else None

def panel_axes(pn):
    v = G - (G @ pn) * pn
    nv = np.linalg.norm(v)
    if nv < 1e-6: return None
    v = v / nv
    h = np.cross(pn, v); h /= np.linalg.norm(h) + 1e-9
    return v, h

def panel_hw(nbr, cen, extent, pn, hi=20, plane_tol=0.03):
    panel = panel_component(nbr, cen, extent, pn, hi=hi, plane_tol=plane_tol)
    ax = panel_axes(pn) if panel is not None else None
    if ax is None: return None, None
    v, h = ax
    height = float(np.ptp((panel - cen) @ v))
    hp = (panel - cen) @ h; w = float(np.ptp(hp))
    offset = abs(hp.min() + hp.max()) / w if w > 1e-6 else None
    return height, offset

# ---------------------------------------------------------------- predictor
def predict(label, pts, nbr, cfg):
    # The FULL METHOD row must reflect the DEPLOYED code, not this module's
    # parallel copy of the rules. They drifted once already: EDGE_ASYM_THRESH was
    # adopted into sf3d/motion.py and this table did not move, because every rule
    # below is reimplemented here for the ablation switches. Delegating the
    # unablated path keeps the baseline honest; the ablations still use the local
    # variants, which is the only reason they exist.
    if not cfg:
        _p = M.predict_motion(label, pts, nbr)
        return _p['motion_type'], _p['motion_dir'], _p['motion_origin']
    cen = pts.mean(0)
    extent = float(np.linalg.norm(pts - cen, axis=1).mean())
    own = M.local_frame(pts)[0]
    pn = parent_normal(nbr, cen, extent,
                       robust=cfg.get('robust', True),
                       adaptive=cfg.get('adaptive', True))

    # ---- type ----
    if label in M.DETERMINED_TYPE:
        t = M.DETERMINED_TYPE[label]
    elif label in M.MIXED:
        mode = cfg.get('type_rule', 'combined')
        if mode == 'majority':
            t = 'trans'
        else:
            is_rot = False
            if pn is not None:
                hgt, off = panel_hw(nbr, cen, extent, pn)
                hh = hgt is not None and hgt > M.TYPE_HEIGHT_THRESH
                oo = off is not None and off > M.TYPE_OFFSET_THRESH
                is_rot = {'combined': hh or oo, 'height': hh, 'offset': oo}[mode]
            t = 'rot' if is_rot else 'trans'
    else:
        t = 'trans'

    # ---- direction ----
    if cfg.get('own_normal', False):
        axis = own
    elif label in M.LEVER and cfg.get('gravity', True):
        axis = G
    elif label in M.MIXED and t == 'rot' and cfg.get('gravity', True):
        axis = G
    else:
        axis = pn if pn is not None else own

    # ---- origin ----
    org = None
    if t == 'rot':
        if label in M.KNOB or pn is None:
            org = cen
        else:
            panel = panel_component(nbr, cen, extent, pn, hi=40)
            src = panel
            if src is None and cfg.get('fallback', True):
                src = cop_points(nbr, cen, extent, pn, hi=40)
            if src is None:
                org = cen
            else:
                ax = np.cross(axis, pn); n = np.linalg.norm(ax)
                if n < 1e-6:
                    org = cen
                else:
                    ax = ax / n
                    proj = (src - cen) @ ax
                    far, near = cen + ax * proj.max(), cen + ax * proj.min()
                    mode = cfg.get('edge', 'corner')
                    if mode == 'corner':
                        org = far if M.corner_score(nbr, far, pn) >= M.corner_score(nbr, near, pn) else near
                    elif mode == 'far':
                        org = far if np.linalg.norm(far - cen) >= np.linalg.norm(near - cen) else near
                    else:
                        org = cen
    return t, axis, org

# ---------------------------------------------------------------- data
ann = {os.path.basename(os.path.dirname(f)): f
       for f in glob.glob(f'{BASE}/data/*/*_annotations.json')}
mot = {os.path.basename(os.path.dirname(f)): f
       for f in glob.glob(f'{BASE}/data/*/*_motions.json')}

ROWS = []
for v in sorted(set(ann) & set(mot)):
    pf = f'{BASE}/{CACHE}/{v}.npz'
    if not os.path.exists(pf): continue
    pz = np.load(pf)
    oz = np.load(f'{BASE}/annot_origins/{v}.npz') if os.path.exists(f'{BASE}/annot_origins/{v}.npz') else None
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
        r = dict(visit=v, label=lab.get(aid), mtype=str(m.get('motion_type')),
                 dirv=d / np.linalg.norm(d), pts=pts,
                 nbr=pz['NBR_' + aid].astype(np.float64) if 'NBR_' + aid in pz.files else None)
        if oz is not None and aid in oz.files:
            r['org'] = oz[aid].astype(np.float64)
        ROWS.append(r)

isrot = lambda r: 'rot' in r['mtype'].lower()
print(f'cache={CACHE}  n={len(ROWS)}  scenes={len({r["visit"] for r in ROWS})}', flush=True)

# ---------------------------------------------------------------- run
def evaluate(cfg):
    res = []
    for r in ROWS:
        t, axis, org = predict(r['label'], r['pts'], r['nbr'], cfg)
        oe = M.OE(r['dirv'], axis)
        md = (M.MD(org, axis, r['org'], r['dirv'])
              if (org is not None and 'org' in r) else None)
        gt_rot = isrot(r)
        gate = (False if oe >= 15 else True if not gt_rot
                else None if md is None else bool(md < 0.25))
        res.append(dict(visit=r['visit'], label=r['label'], gt_rot=gt_rot,
                        p_type=t, oe=oe, md=md, gate=gate))
    return res

def summarize(res):
    R = [x for x in res if x['gate'] is not None]
    gate = np.mean([x['gate'] for x in R])
    ty = np.mean([('rot' if x['gt_rot'] else 'trans') == x['p_type'] for x in res])
    tra = [x['oe'] for x in res if not x['gt_rot']]
    rot = [x['oe'] for x in res if x['gt_rot']]
    md = [x['md'] for x in res if x['md'] is not None]
    return dict(gate=gate*100, type=ty*100,
                oe_tra=float(np.median(tra)) if tra else float('nan'),
                oe_rot=float(np.median(rot)) if rot else float('nan'),
                md=float(np.median(md)) if md else float('nan'), n=len(R))

ABLATIONS = [
    ('FULL METHOD',                      {}),
    ('A  - parent normal (use own)',     dict(own_normal=True)),
    ('B  - gravity prior for hinges',    dict(gravity=False)),
    ('C  - type rule (majority)',        dict(type_rule='majority')),
    ('C1   type: height only',           dict(type_rule='height')),
    ('C2   type: offset only',           dict(type_rule='offset')),
    ('D  - corner score (far edge)',     dict(edge='far')),
    ('D1 - edge choice (centroid)',      dict(edge='centroid')),
    ('E  - origin fallback',             dict(fallback=False)),
    ('F  - robust fit (plain SVD)',      dict(robust=False)),
    ('G  - adaptive shell',              dict(adaptive=False)),
    ('F+G both (original rule)',         dict(robust=False, adaptive=False)),
]

full = None
print(f'\n{"ablation":<32} {"gate%":>7} {"Δ":>7} {"type%":>7} {"OEtra":>7} {"OErot":>7} {"MD":>7}')
print('-' * 82)
out = {}
for name, cfg in ABLATIONS:
    s = summarize(evaluate(cfg))
    out[name] = s
    if full is None:
        full = s['gate']; delta = ''
    else:
        delta = f'{s["gate"]-full:+.1f}'
    print(f'{name:<32} {s["gate"]:7.1f} {delta:>7} {s["type"]:7.1f} '
          f'{s["oe_tra"]:7.2f} {s["oe_rot"]:7.2f} {s["md"]:7.3f}', flush=True)

json.dump(out, open(f'{CODE}/docs/motion_ablations_{CACHE}.json', 'w'), indent=1)
print(f'\nn={len(ROWS)} elements. Negative Δ = removing the component HURTS,')
print('i.e. the component was doing real work.')

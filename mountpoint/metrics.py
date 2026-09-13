"""Metrics and evaluation loops. Kept out of pipeline.py so the pipeline stays
importable without any evaluation assumptions."""
import numpy as np, cv2, os
from . import config as C, pipeline as P, retrieval as RT

def closed_gt(mask, k=21):
    """SceneFun3D GT is a sparse point set: projected and dilated it fills a
    median of 6.1% of its own bounding box, so raw IoU structurally penalises
    correct segmentations. Report against a morphologically closed GT."""
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                            np.ones((k, k), np.uint8)).astype(bool)

def gt_full(e, H, W):
    x0, y0, x1, y1 = e['box']
    g = np.zeros((H, W), bool)
    sub = e['mask'][:y1-y0, :x1-x0]
    g[y0:y0+sub.shape[0], x0:x0+sub.shape[1]] = sub
    return g

def pool_oracle(db, key, k_ex=None):
    """Same label, other scenes. NOTE: uses the ground-truth label."""
    t = db[key]
    return [db[k] for k, e in db.items()
            if e['visit'] != t['visit'] and e['label'] == t['label']][:(k_ex or C.K_EX)]

def pool_retrieved(db, key, k_ex=5):
    """Retrieval-driven: no GT label anywhere in the prediction path."""
    e = db[key]
    desc = (e.get('desc') or [''])[0]
    if not desc: return [], None
    ks = RT.retrieve(desc, db, k=k_ex, exclude_visit=e['visit'])
    return [db[k] for k in ks], RT.guess_label(db, ks)

def evaluate(db, keys, n=80, seed=0, oracle_box=False, oracle_label=True,
             oracle_hint=True, topk=None, verbose=False):
    """Returns per-item records. Three complementary metrics per handoff 5.1:
    IoU against closed GT, GT coverage, and peak distance in pixels."""
    rng = np.random.default_rng(seed)
    sel = rng.choice(keys, min(n, len(keys)), replace=False)
    R = []
    for k in sel:
        e = db[k]
        fp = f"{C.FRAMES}/{e['frame_file']}"
        if not os.path.exists(fp): continue

        if oracle_label:
            pool, guess = pool_oracle(db, k), e['label']
        else:
            pool, guess = pool_retrieved(db, k)
        if not pool: continue
        proto = P.build_proto(pool)
        if proto is None: continue

        img = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        g = gt_full(e, H, W)
        if g.sum() < 10: continue
        gc = closed_gt(g); gpts = np.argwhere(g)

        hint = C.PARENT_HINT.get(guess, C.DEFAULT_HINT) if oracle_hint else C.DEFAULT_HINT
        text = f"{(e.get('desc') or [''])[0]}. {hint}"

        if oracle_box:
            x0, y0, x1, y1 = e['box']
            pw, ph = (x1-x0)*.4, (y1-y0)*.4
            X0, Y0 = int(max(0, x0-pw)), int(max(0, y0-ph))
            X1, Y1 = int(min(W, x1+pw)), int(min(H, y1+ph))
            smap, (gh, gw) = P.score_image(img[Y0:Y1, X0:X1], proto)
            pi, pj = np.unravel_index(int(np.argmax(smap)), smap.shape)
            pt = (X0+(pj+.5)*(X1-X0)/gw, Y0+(pi+.5)*(Y1-Y0)/gh)
            up = cv2.resize(smap, (X1-X0, Y1-Y0), interpolation=cv2.INTER_CUBIC)
            full = np.full((H, W), float(smap.min()), np.float32)
            full[Y0:Y1, X0:X1] = up
            mask, _, _ = P.sam_mask(img, pt, full, region=(X0,Y0,X1,Y1),
                                    area_prior=C.MEDIAN_AREA)
            res = dict(mask=mask, point=pt, peak=float(smap.max()), conf=1.0)
        else:
            res = P.run(img, proto, text, topk=topk)
            if res is None: continue

        m = res['mask']
        py, px = int(np.clip(res['point'][1], 0, H-1)), int(np.clip(res['point'][0], 0, W-1))
        R.append(dict(key=k, label=e['label'], guess=guess,
                      iou=(m & gc).sum() / max((m | gc).sum(), 1),
                      coverage=(m & g).sum() / g.sum(),
                      peak_px=float(np.min(np.linalg.norm(
                          gpts - np.array([py, px]), axis=1))),
                      in_gt=bool(gc[py, px]),
                      mask_px=int(m.sum()), peak=res.get('peak', np.nan)))
        if verbose: print(f"  {R[-1]['iou']:.3f}  {R[-1]['coverage']:.2f}")
    return R

def summarise(R, name=''):
    if not R:
        print(f'{name}: no results'); return {}
    g = lambda f: np.array([r[f] for r in R])
    s = dict(n=len(R), iou=float(g('iou').mean()), iou_med=float(np.median(g('iou'))),
             coverage=float(g('coverage').mean()),
             peak15=float(np.mean(g('peak_px') < 15)),
             in_gt=float(g('in_gt').mean()),
             peak_med=float(np.median(g('peak_px'))),
             zeros=float(np.mean(g('iou') < 0.02)),
             good=float(np.mean(g('iou') > 0.15)))
    print(f"{name:<26} n={s['n']:>3}  IoU {s['iou']:.3f}  cov {s['coverage']:.3f}  "
          f"peak<15px {s['peak15']:.0%}  in-GT {s['in_gt']:.0%}  "
          f"zeros {s['zeros']:.0%}  good {s['good']:.0%}")
    return s

def distribution(R, name=''):
    """The mean is misleading here: two binary gates (box hit, correspondence
    hit) multiply into a bimodal outcome, not a cluster around the mean."""
    v = np.array([r['iou'] for r in R])
    print(f'\n{name}  n={len(v)}  mean {v.mean():.3f}  median {np.median(v):.3f}')
    bins = [0, .02, .05, .10, .15, .20, .30, .50, 1.01]
    hist, _ = np.histogram(v, bins=bins)
    for lo, hi, c in zip(bins[:-1], bins[1:], hist):
        bar = '#' * int(c / max(hist.max(), 1) * 40)
        print(f'  [{lo:.2f},{hi:.2f})  {c:>3}  {bar}')
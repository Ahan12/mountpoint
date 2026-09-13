"""Backbone, dense features, prototype scoring, Grounding DINO, SAM."""
import os, re, numpy as np, cv2, torch
from . import config as C

_M = {}   # model singletons

# ------------------------------------------------------------------ backbone
def dino():
    if 'dino' in _M: return _M['dino']
    from transformers import AutoConfig, AutoModel
    sd = torch.load(C.CKPT, map_location='cpu', weights_only=False)
    sd = sd.get('model', sd.get('state_dict', sd))
    assert not any('pos_embed' in k for k in sd), 'learned pos_embed -> this is DINOv2'
    pw = next(k for k in sd if k.endswith(('patch_embeddings.weight',
                                           'patch_embed.proj.weight')))
    dim, patch = int(sd[pw].shape[0]), int(sd[pw].shape[-1])
    nl = 1 + max(int(m.group(1)) for k in sd if (m := re.search(r'layers?\.([^.]+)\.', k)))
    reg = next((k for k in sd if 'register' in k), None)
    nprefix = 1 + (int(sd[reg].shape[1]) if reg else 0)

    cfg = AutoConfig.from_pretrained(C.HUB_ID)
    assert (cfg.hidden_size, cfg.patch_size, cfg.num_hidden_layers) == (dim, patch, nl)
    m = AutoModel.from_pretrained(C.HUB_ID, config=cfg); tgt = set(m.state_dict())
    cand, name = min((({f(k): v for k, v in sd.items()}, n) for n, f in {
        'as-is'       : lambda k: k,
        'strip model.': lambda k: re.sub(r'^model\.', '', k),
        'strip 1'     : lambda k: k.split('.', 1)[1] if '.' in k else k}.items()),
        key=lambda c: len(tgt - set(c[0])))
    # strict=False without checking `missing` is how you end up running a
    # randomly-initialised model that produces plausible numbers.
    assert not (tgt - set(cand)), f'remap "{name}" missing {len(tgt-set(cand))} keys'
    m.load_state_dict(cand, strict=False)
    m = m.to('cuda').eval()

    blocks = None
    for n, mod in m.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == nl:
            assert blocks is None, 'ambiguous block list'
            blocks = mod
    assert blocks is not None
    _M['dino'] = dict(model=m, blocks=blocks, dim=dim, patch=patch,
                      n_layers=nl, n_prefix=nprefix)
    print(f'DINOv3 dim={dim} patch={patch} blocks={nl} prefix={nprefix} via "{name}"')
    return _M['dino']

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

class _Stop(Exception): pass

def patch_feats(img, layer=None, short=None):
    """(gh, gw, D) unit-norm patch grid.

    Uses an early-exit forward hook rather than output_hidden_states=True: at
    short=1036 that materialised 25 x ~5600 x 1024 floats (~575 MB) per call
    to use one slice of it, which is where the OOMs came from.
    """
    d = dino(); layer = layer or C.LAYER; short = short or C.QUERY_SHORT
    P = d['patch']; H, W = img.shape[:2]; s = short / min(H, W)
    gh, gw = max(round(H*s/P), 1), max(round(W*s/P), 1)
    im = (cv2.resize(img, (gw*P, gh*P), interpolation=cv2.INTER_CUBIC)
            .astype(np.float32)/255. - MEAN) / STD
    t = torch.from_numpy(im).permute(2,0,1)[None].float().cuda()
    cap = {}
    def hook(mod, i, o):
        cap['x'] = o[0] if isinstance(o, (tuple, list)) else o
        raise _Stop
    h = d['blocks'][d['n_layers']-layer].register_forward_hook(hook)
    try: d['model'](pixel_values=t)
    except _Stop: pass
    finally: h.remove()
    n = gh*gw; tok = cap['x']
    # take the LAST h*w tokens: robust to however many prefix tokens exist
    assert tok.shape[1] == n + d['n_prefix'], f'{tok.shape[1]} != {n}+{d["n_prefix"]}'
    f = tok[:, -n:, :].reshape(gh, gw, -1).float().cpu().detach().numpy()
    del tok, t
    return f / (np.linalg.norm(f, axis=-1, keepdims=True) + 1e-8)

def patchify(mask, gh, gw):
    return cv2.resize(mask.astype(np.uint8), (gw, gh),
                      interpolation=cv2.INTER_NEAREST).astype(bool)

def _erode1(m):
    """Boundary patches mix hardware with background and carry the residual
    reprojection offset, so drop one patch of border."""
    if m.sum() <= 6: return m
    e = cv2.erode(m.astype(np.uint8), np.ones((3,3), np.uint8)).astype(bool)
    return e if e.sum() >= 2 else m

# ------------------------------------------------------- prototype scoring
def build_proto(entries, layer=None, short=None, bg_samples=128):
    """Mean-pooled foreground prototype + background prototype.

    Measured best of four scorers (45.3% vs 38.9% for a top-q bank, n=120).
    The bank's multimodality advantage does not appear at k_ex=3, plausibly
    because three same-label exemplars rarely span multiple modes.
    """
    FG, BG = [], []
    rng = np.random.default_rng(0)
    for e in entries:
        for v in e.get('views', [e])[:3]:
            f = patch_feats(v['crop'], layer, short or C.BANK_SHORT)
            m = patchify(v['mask'], *f.shape[:2])
            fg, bg = f[_erode1(m)], f[~m]
            if len(fg): FG.append(fg)
            if len(bg):
                k = min(bg_samples, len(bg))
                BG.append(bg[rng.choice(len(bg), k, replace=False)])
    if not FG: return None
    u = lambda A: A / (np.linalg.norm(A) + 1e-8)
    return dict(fg=u(np.vstack(FG).mean(0)),
                bg=u(np.vstack(BG).mean(0)) if BG else None,
                layer=layer or C.LAYER, short=short or C.BANK_SHORT)

def score_image(img, proto):
    """Similarity map. Background subtraction is worth ~19.5 points and is the
    single most important component in the pipeline."""
    f = patch_feats(img, proto['layer'], proto['short'])
    f2 = f.reshape(-1, f.shape[-1])
    s = f2 @ proto['fg']
    if proto['bg'] is not None: s = s - f2 @ proto['bg']
    return s.reshape(f.shape[:2]), f.shape[:2]

# ------------------------------------------------------------ Grounding DINO
def gdino():
    if 'gd' in _M: return _M['gd']
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    mid = 'IDEA-Research/grounding-dino-tiny'
    _M['gd'] = (AutoProcessor.from_pretrained(mid),
                 AutoModelForZeroShotObjectDetection.from_pretrained(mid).cuda().eval())
    return _M['gd']

def gd_boxes(img, text, topk=None):
    from PIL import Image
    proc, mod = gdino(); topk = topk or C.GD_TOPK
    inp = proc(images=Image.fromarray(img), text=text.lower().strip(),
               return_tensors='pt').to('cuda')
    out = mod(**inp)
    r = proc.post_process_grounded_object_detection(
        out, inp.input_ids, threshold=0.15, text_threshold=0.15,
        target_sizes=[img.shape[:2]])[0]
    b, s = r['boxes'].cpu().numpy(), r['scores'].cpu().numpy()
    o = np.argsort(-s)[:topk]
    return b[o], s[o]

# ------------------------------------------------------------------- SAM
def sam():
    if 'sam' in _M: return _M['sam']
    from transformers import SamProcessor, SamModel
    mid = 'facebook/sam-vit-base'
    _M['sam'] = (SamProcessor.from_pretrained(mid),
                 SamModel.from_pretrained(mid).cuda().eval())
    return _M['sam']

def _hot_box(hot, pad_frac=0.15):
    """Bounding box of the high-similarity region, padded a bit. Gives SAM a
    scale hint so it doesn't have to guess between 'the handle' and 'the whole
    cabinet' from a bare point -- see BOX PROMPT note below."""
    ys, xs = np.where(hot)
    if len(xs) == 0: return None
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max()+1, ys.max()+1
    px, py = pad_frac*(x1-x0), pad_frac*(y1-y0)
    H, W = hot.shape
    return [float(max(0, x0-px)), float(max(0, y0-py)),
            float(min(W, x1+px)), float(min(H, y1+py))]

def sam_mask(img, pt, smap=None, region=None, area_prior=None):
    """SAM prompted on the BOX CROP, not the full frame.

    On a 1920x1440 frame the whole-cabinet mask is the natural segmentation:
    measured failures had coverage 1.0 with IoU 0.005 (correct peak, ~200x too
    large). Cropping first, weighting purity above cover, and rejecting masks
    far larger than a typical annotation doubled the GT-box ceiling (0.117 ->
    0.235).

    BOX PROMPT: point-only prompting still let SAM propose all-oversized mask
    triples for small hardware on a large flat panel -- measured cases where
    every one of the 3 candidates was 30-250x the median GT area, so the area
    penalty below had nothing smaller to fall back to (it penalises all three
    equally and changes nothing). Passing the high-similarity region's own
    bounding box as an additional prompt gives SAM an explicit scale hint.
    Verified n=30 local (MPS): mean IoU 0.029 -> 0.062, good rate 3% -> 20%.
    Toggle with config.SAM_BOX_PROMPT for an A/B check.
    """
    from PIL import Image
    proc, mod = sam()
    if region is not None:
        X0, Y0, X1, Y1 = region
        sub, spt = img[Y0:Y1, X0:X1], (pt[0]-X0, pt[1]-Y0)
    else:
        X0 = Y0 = 0; sub, spt = img, pt

    hot = None
    if smap is not None:
        h = smap[Y0:Y1, X0:X1] if region is not None else smap
        hot = h >= np.percentile(h, 92)
    box_prompt = _hot_box(hot) if (hot is not None and C.SAM_BOX_PROMPT) else None

    kw = dict(input_points=[[[float(spt[0]), float(spt[1])]]])
    if box_prompt is not None:
        kw['input_boxes'] = [[box_prompt]]
    inp = proc(Image.fromarray(sub), return_tensors='pt', **kw).to('cuda')
    out = mod(**inp)
    masks = proc.image_processor.post_process_masks(
        out.pred_masks.cpu(), inp['original_sizes'].cpu(),
        inp['reshaped_input_sizes'].cpu())[0][0].numpy()
    iou = out.iou_scores.cpu().numpy().ravel()

    best, bs = 0, -1e9
    for i, m in enumerate(masks):
        if m.sum() < 20: continue
        s = 0.3 * iou[i]
        if hot is not None:
            cov = (m & hot).sum() / max(hot.sum(), 1)
            pur = (m & hot).sum() / max(m.sum(), 1)
            # combine so that cover 1.0 / purity 0.2 cannot tie 0.6 / 0.6
            s += 2.0*pur + 0.5*cov
        if area_prior and m.sum() > C.SAM_AREA_MULT * area_prior:
            s -= 3.0
        if s > bs: bs, best = s, i
    full = np.zeros(img.shape[:2], bool)
    if region is not None: full[Y0:Y1, X0:X1] = masks[best]
    else: full = masks[best]
    return full, masks, iou

# ------------------------------------------------------------- full 2D run
def run(img, proto, text, topk=None, area_prior=None):
    """Multi-box: score correspondence inside each of the top-K boxes, keep the
    box holding the globally strongest match. Parameter-free -- the peak IS the
    criterion. Distinct from the reranker that failed (55.0% vs 58.3% null),
    which scored boxes on region statistics BEFORE localising within them."""
    topk = topk or C.MULTIBOX_K
    area_prior = area_prior if area_prior is not None else C.MEDIAN_AREA
    boxes, confs = gd_boxes(img, text)
    if len(boxes) == 0: return None
    H, W = img.shape[:2]
    best = None
    for b, cf in zip(boxes[:topk], confs[:topk]):
        x0, y0, x1, y1 = b
        pw, ph = (x1-x0)*C.BOX_PAD, (y1-y0)*C.BOX_PAD
        X0, Y0 = int(max(0, x0-pw)), int(max(0, y0-ph))
        X1, Y1 = int(min(W, x1+pw)), int(min(H, y1+ph))
        if X1-X0 < 16 or Y1-Y0 < 16: continue
        smap, (gh, gw) = score_image(img[Y0:Y1, X0:X1], proto)
        peak = float(smap.max())
        if best is None or peak > best['peak']:
            pi, pj = np.unravel_index(int(np.argmax(smap)), smap.shape)
            best = dict(peak=peak, box=b, conf=float(cf), region=(X0,Y0,X1,Y1),
                        smap=smap,
                        point=(X0+(pj+.5)*(X1-X0)/gw, Y0+(pi+.5)*(Y1-Y0)/gh))
    if best is None: return None
    X0, Y0, X1, Y1 = best['region']
    up = cv2.resize(best['smap'], (X1-X0, Y1-Y0), interpolation=cv2.INTER_CUBIC)
    full = np.full((H, W), float(best['smap'].min()), np.float32)
    full[Y0:Y1, X0:X1] = up
    mask, cands, iou = sam_mask(img, best['point'], full,
                                region=best['region'], area_prior=area_prior)
    return dict(box=best['box'], conf=best['conf'], region=best['region'],
                point=best['point'], mask=mask, smap=full, peak=best['peak'],
                sam_cands=cands, sam_iou=iou)

def init_stats(db):
    """Median annotation area -- a dataset constant, not a per-item oracle."""
    C.MEDIAN_AREA = float(np.median([e['mask'].sum() for e in db.values()]))
    return C.MEDIAN_AREA


# torch.set_grad_enabled() is THREAD-LOCAL. Gradio dispatches handlers to a
# worker thread, so the global setting from the notebook does not apply there.
# Decorating the entry points makes it thread-independent.
import torch as _t
patch_feats = _t.no_grad()(patch_feats)
build_proto = _t.no_grad()(build_proto)
score_image = _t.no_grad()(score_image)
gd_boxes    = _t.no_grad()(gd_boxes)
sam_mask    = _t.no_grad()(sam_mask)
run         = _t.no_grad()(run)
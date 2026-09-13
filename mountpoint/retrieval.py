"""CLIP hybrid retrieval. Embeddings cached to Drive; re-encoding the whole
exemplar DB per query is otherwise the pipeline bottleneck."""
import os, numpy as np, torch
from . import config as C

_R = {}

def _clip():
    if 'm' not in _R:
        import open_clip
        m, _, pp = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        _R['m'], _R['pp'] = m.cuda().eval(), pp
        _R['tok'] = open_clip.get_tokenizer('ViT-L-14')
    return _R['m'], _R['pp'], _R['tok']

def _unit(a, ax=-1):
    return a / (np.linalg.norm(a, axis=ax, keepdims=True) + 1e-8)

def build_index(db, force=False):
    if not force and os.path.exists(C.CLIP_CACHE):
        z = np.load(C.CLIP_CACHE, allow_pickle=True)
        _R['keys'] = list(z['keys']); _R['I'] = z['I']
        _R['T'] = z['T']; _R['has_t'] = z['has_t']
        print(f'loaded CLIP cache: {len(_R["keys"])} exemplars')
        return
    from PIL import Image
    m, pp, tok = _clip()
    keys = list(db.keys()); I, T, has_t = [], [], []
    for i, k in enumerate(keys):
        e = db[k]
        with torch.no_grad():
            im = pp(Image.fromarray(e['crop'])).unsqueeze(0).cuda()
            I.append(m.encode_image(im).float().cpu().numpy()[0])
            d = (e.get('desc') or [''])[0]
            if d:
                T.append(m.encode_text(tok([d]).cuda()).float().cpu().numpy()[0])
                has_t.append(True)
            else:
                T.append(np.zeros_like(I[-1])); has_t.append(False)
        if (i+1) % 100 == 0: print(f'  {i+1}/{len(keys)}')
    _R['keys'] = keys; _R['I'] = _unit(np.stack(I))
    _R['T'] = _unit(np.stack(T)); _R['has_t'] = np.array(has_t)
    np.savez_compressed(C.CLIP_CACHE, keys=np.array(keys, object),
                        I=_R['I'], T=_R['T'], has_t=_R['has_t'])
    print(f'encoded and cached {len(keys)} exemplars')

def encode_query(text):
    m, _, tok = _clip()
    with torch.no_grad():
        q = m.encode_text(tok([text]).cuda()).float().cpu().numpy()[0]
    return _unit(q)

def retrieve(query_text, db, k=5, alpha=0.5, max_per_scene=2, exclude_visit=None):
    """s(q,e) = a*<t(q), i(e)> + (1-a)*<t(q), tbar(e)>

    Each modality is centred by its OWN gallery mean and both sides are
    renormalised. Without this, CLIP hubness makes every query return the same
    five results. Exemplars with no description fall back to the image score
    rather than being structurally excluded by a zero text vector.
    """
    keys, I, T, has_t = _R['keys'], _R['I'], _R['T'], _R['has_t']
    q = encode_query(query_text)

    Im = I.mean(0)
    img_s = _unit(I - Im) @ _unit(q - Im)
    txt_s = img_s.copy()
    if has_t.any():
        Tm = T[has_t].mean(0)
        txt_s[has_t] = _unit(T[has_t] - Tm) @ _unit(q - Tm)

    score = alpha*img_s + (1-alpha)*txt_s
    out, seen = [], {}
    for i in np.argsort(-score):
        kk = keys[i]; vis = db[kk]['visit']
        if exclude_visit and vis == exclude_visit: continue
        if seen.get(vis, 0) >= max_per_scene: continue
        out.append(kk); seen[vis] = seen.get(vis, 0) + 1
        if len(out) >= k: break
    return out

def guess_label(db, ex_keys):
    labs = [db[k]['label'] for k in ex_keys]
    return max(set(labs), key=labs.count) if labs else None
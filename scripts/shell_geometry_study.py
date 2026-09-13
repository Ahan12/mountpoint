"""Where the shell bounds come from, and what the plane fit actually computes.

Two outputs:
  1. A fully worked numeric example on one real element -- every intermediate
     quantity, so the SVD step can be checked by hand.
  2. The (lo, hi) sweep the deployed shell (1.5, 4.0) is chosen from, scored on
     direction error and on the end-to-end motion gate.

Run: python scripts/shell_geometry_study.py
"""
import sys, io, json, contextlib
import numpy as np

sys.path.insert(0, f'{CODE}/scripts'); sys.path.insert(0, f'{CODE}/scenefun3d 2')
with contextlib.redirect_stdout(io.StringIO()):
    import motion_ablations as AB
from _paths import REPO, DATA_ROOT, TOOLKIT, DOCS  # noqa: F401
CODE = REPO          # repo root: code, docs, figures
BASE = DATA_ROOT     # data root: scans, caches, exemplar DB
from sf3d import motion as M

ROWS = AB.ROWS
print(f'{len(ROWS)} elements\n')

# ---------------------------------------------------------------- 1. example
def worked_example(rec):
    pts, nbr, cen = rec['pts'], rec['nbr'], rec['pts'].mean(0)
    extent = float(np.linalg.norm(pts - cen, axis=1).mean())
    print('='*68); print(f'WORKED EXAMPLE  {rec["visit"]}  label={rec["label"]}  type={rec["mtype"]}')
    print('='*68)
    print(f'  region points        {pts.shape}')
    print(f'  centroid             {np.round(cen,4)}')
    print(f'  extent (mean |p-c|)  {extent:.4f} m   <- the ONLY scale in the method')
    print(f'  neighbourhood        {nbr.shape}')

    # element's own SVD
    Q = pts - cen
    U, s, Vt = np.linalg.svd(Q, full_matrices=False)
    print(f'\n  -- element own SVD --')
    print(f'  singular values      {np.round(s,4)}')
    print(f'  ratio s3/s1          {s[2]/s[0]:.4f}   (planarity; ~0 = flat, ~1 = blob)')
    own_n = Vt[2]
    print(f'  own normal (v3)      {np.round(own_n,4)}')
    print(f'  OE vs GT             {M.OE(rec["dirv"], own_n):.2f} deg')

    # shell
    d = np.linalg.norm(nbr - cen, axis=1)
    lo, hi = 1.5, 4.0
    sel = (d > extent*lo) & (d < extent*hi)
    shell = nbr[sel]
    print(f'\n  -- shell [{lo}, {hi}] x extent = [{extent*lo:.4f}, {extent*hi:.4f}] m --')
    print(f'  candidates in nbr    {len(nbr)}')
    print(f'  points in shell      {len(shell)}   (min_pts = 50)')

    Qs = shell - shell.mean(0)
    _, ss, Vts = np.linalg.svd(Qs, full_matrices=False)
    print(f'  shell singular vals  {np.round(ss,4)}')
    print(f'  ratio s3/s1          {ss[2]/ss[0]:.4f}   <- small = genuinely planar')
    pn_plain = Vts[2]
    print(f'  plain normal         {np.round(pn_plain,4)}   OE {M.OE(rec["dirv"], pn_plain):.2f} deg')

    # robust trim, iteration by iteration
    P = shell.copy(); n = M.plane_normal(P)
    print(f'\n  -- MAD-trimmed refit (k = 2.5) --')
    for it in range(3):
        r = np.abs((P - P.mean(0)) @ n)
        med = np.median(r); mad = np.median(np.abs(r - med))
        cut = med + 2.5 * 1.4826 * mad
        keep = r < cut
        print(f'  iter {it}: n={len(P):5d}  median|r|={med:.4f}  MAD={mad:.4f}  '
              f'cut={cut:.4f}  keep={keep.sum():5d}  OE={M.OE(rec["dirv"], n):.2f} deg')
        if mad < 1e-9 or keep.sum() < 30: break
        P = P[keep]; n = M.plane_normal(P)
    print(f'  final robust normal  {np.round(n,4)}   OE {M.OE(rec["dirv"], n):.2f} deg')
    print(f'  GT direction         {np.round(rec["dirv"],4)}')
    deployed = M.parent_normal(nbr, cen, extent)
    print(f'  module output        {np.round(deployed,4)}   OE {M.OE(rec["dirv"], deployed):.2f} deg')

# pick a clean prismatic example where the parent normal clearly wins
cands = [r for r in ROWS if r['mtype'] == 'trans' and r.get('nbr') is not None
         and len(r['pts']) > 60 and len(r['nbr']) > 4000]
scored = []
for r in cands:
    cen = r['pts'].mean(0); ext = float(np.linalg.norm(r['pts']-cen, axis=1).mean())
    pn = M.parent_normal(r['nbr'], cen, ext)
    if pn is None: continue
    own = M.local_frame(r['pts'])[0]
    scored.append((M.OE(r['dirv'], own) - M.OE(r['dirv'], pn), r))
scored.sort(key=lambda t: -t[0])
worked_example(scored[len(scored)//8][1])

# ---------------------------------------------------------------- 2. sweep
def run_shell(lo, hi):
    oes, gate_ok, gate_n = [], 0, 0
    for r in ROWS:
        pts, nbr = r['pts'], r.get('nbr')
        if nbr is None: continue
        cen = pts.mean(0); ext = float(np.linalg.norm(pts-cen, axis=1).mean())
        d = np.linalg.norm(nbr - cen, axis=1)
        h = hi; sel = (d > ext*lo) & (d < ext*h)
        while sel.sum() < 50 and h < 12.0:
            h *= 1.5; sel = (d > ext*lo) & (d < ext*h)
        pn = M.plane_normal_robust(nbr[sel]) if sel.sum() >= 50 else None
        own = M.local_frame(pts)[0]
        axis_src = pn if pn is not None else own
        if r['mtype'] == 'trans':
            oes.append(M.OE(r['dirv'], axis_src))
        # end-to-end gate with this shell
        t = M.predict_type(r['label'], nbr, cen, ext)
        if r['label'] in M.LEVER or (r['label'] in M.MIXED and t == 'rot'):
            axis = M.G
        else:
            axis = axis_src
        isrot = 'rot' in r['mtype'].lower()
        if t != r['mtype'] or M.OE(r['dirv'], axis) >= 15:
            gate_n += 1; continue
        if not isrot:
            gate_n += 1; gate_ok += 1; continue
        org = M.predict_origin(r['label'], axis, nbr, cen, ext)
        if org is None or 'org' not in r: continue
        gate_n += 1
        gate_ok += M.MD(org, axis, r['org'], r['dirv']) < 0.25
    return float(np.median(oes)), 100*gate_ok/max(gate_n,1)

print('\n' + '='*68)
print('SHELL SWEEP  (median prismatic OE, deg  |  end-to-end gate %)')
print('='*68)
LOS = [1.0, 1.25, 1.5, 2.0, 2.5]
HIS = [2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
print('        ' + ''.join(f'hi={h:<11}' for h in HIS))
grid = {}
for lo in LOS:
    cells = []
    for hi in HIS:
        if hi <= lo + 0.5: cells.append('     -      '); continue
        oe, g = run_shell(lo, hi); grid[(lo,hi)] = (oe, g)
        cells.append(f'{oe:5.2f}/{g:5.1f}%  ')
    print(f'lo={lo:<5}' + ''.join(cells), flush=True)
json.dump({f'{k[0]}_{k[1]}': v for k, v in grid.items()},
          open(f'{CODE}/docs/shell_sweep.json','w'), indent=1)
print(f'\nwritten docs/shell_sweep.json')

"""Is there Articulate3D-style headroom in SceneFun3D's hinge origin?

On Articulate3D the recovered panel was the WALL (median span 1.44 m for a
~0.8 m door), which capped oracle edge choice at 46.3%. Handing the method the
real part mask took origin from 33.1% -> 93.8%.

SceneFun3D annotates no parent part, so the same fix is not available. Before
concluding there is nothing to gain, measure whether its recovered panel is
inflated the same way -- if oracle edge choice is already high, the panel is
fine and only selection is left; if oracle is low, a predicted panel would pay.

Run: python scripts/sf3d_panel_headroom.py
"""
import sys, io, contextlib, json
import numpy as np

sys.path.insert(0, f'{CODE}/scripts'); sys.path.insert(0, f'{CODE}/scenefun3d 2')
with contextlib.redirect_stdout(io.StringIO()):
    import motion_ablations as AB
from _paths import REPO, DATA_ROOT, TOOLKIT, DOCS  # noqa: F401
CODE = REPO          # repo root: code, docs, figures
BASE = DATA_ROOT     # data root: scans, caches, exemplar DB
from sf3d import motion as M

ROWS = [r for r in AB.ROWS if 'rot' in r['mtype'].lower() and 'org' in r
        and r.get('nbr') is not None and r['label'] not in M.KNOB]
print(f'{len(ROWS)} hinge elements (rotations, excluding knobs whose origin is '
      f'the element itself)\n', flush=True)

out = []
for r in ROWS:
    pts, nbr = r['pts'], r['nbr']
    cen = pts.mean(0); extent = float(np.linalg.norm(pts - cen, axis=1).mean())
    pn = M.parent_normal(nbr, cen, extent)
    if pn is None:
        out.append(dict(state='no_plane')); continue
    panel = M.panel_component(nbr, cen, extent, hi=40)
    state = 'panel'; src = panel
    if src is None:
        src = M.cop_points(nbr, cen, extent, hi=40); state = 'coplanar'
    if src is None:
        out.append(dict(state='centroid',
                        corner=M.MD(cen, M.G, r['org'], r['dirv']))); continue
    axis = M.G if r['label'] in M.LEVER or r['label'] in M.MIXED else pn
    ax = np.cross(axis, pn); n = np.linalg.norm(ax)
    if n < 1e-6:
        out.append(dict(state='degenerate')); continue
    ax /= n
    proj = (src - cen) @ ax
    far, near = cen + ax*proj.max(), cen + ax*proj.min()
    e_f = M.MD(far, axis, r['org'], r['dirv']); e_n = M.MD(near, axis, r['org'], r['dirv'])
    sf = M.corner_score(nbr, far, pn); sn = M.corner_score(nbr, near, pn)
    out.append(dict(state=state, span=float(proj.max()-proj.min()),
                    corner   = e_f if sf >= sn else e_n,
                    ergonomic= e_f if abs(proj.max()) >= abs(proj.min()) else e_n,
                    oracle   = min(e_f, e_n),
                    centroid = M.MD(cen, axis, r['org'], r['dirv'])))

states = {}
for o in out: states[o['state']] = states.get(o['state'], 0) + 1
print('panel isolation:', states)
sp = np.array([o['span'] for o in out if 'span' in o])
print(f'recovered panel span: median {np.median(sp):.3f} m   '
      f'(Articulate3D recovered 1.440 m vs true part 0.606 m)\n')

print(f'{"rule":<12} {"n":>5} {"median err":>12} {"<0.25 m":>10}')
for rule in ('centroid', 'ergonomic', 'corner', 'oracle'):
    e = np.array([o[rule] for o in out if rule in o])
    tag = '  <-- deployed' if rule == 'corner' else ''
    print(f'{rule:<12} {len(e):5d} {np.median(e):11.3f} m {100*np.mean(e<0.25):9.1f}%{tag}')

e_c = np.array([o['corner'] for o in out if 'corner' in o and 'oracle' in o])
e_o = np.array([o['oracle'] for o in out if 'corner' in o and 'oracle' in o])
print(f'\nheadroom if edge SELECTION were perfect: '
      f'{100*np.mean(e_c<0.25):.1f}% -> {100*np.mean(e_o<0.25):.1f}% '
      f'({100*np.mean(e_o<0.25)-100*np.mean(e_c<0.25):+.1f} pts)')
print(f'corner picks the worse edge: {100*np.mean(e_c > e_o + 1e-9):.1f}%')
json.dump(out, open(f'{CODE}/docs/sf3d_panel_headroom.json','w'), indent=1, default=float)
print('\nwritten docs/sf3d_panel_headroom.json')

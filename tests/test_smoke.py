"""A synthetic scene with a known answer, so a broken install is distinguishable
from a real result.

Every number in the paper comes from data that cannot be redistributed. Without
a test like this, someone who installs the package wrong gets plausible-looking
numbers and no signal that anything is amiss. These cases are built from
geometry alone -- no dataset, no checkpoint, no network -- and each asserts a
value that follows from the construction rather than from a previous run.

Run: python -m pytest tests/ -q       (or: python tests/test_smoke.py)
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mountpoint import motion as M

RNG = np.random.default_rng(0)

def panel_scene(width=0.80, height=2.00, handle_x=0.70, density=140):
    """A door in the x-z plane at y=0, hinged at x=0, with a handle near x=0.70.

    The hinge is therefore the FAR edge from the handle -- which is exactly what
    the ergonomic prior is supposed to recover -- and the axis is vertical.
    """
    nx, nz = int(width*density), int(height*density)
    gx, gz = np.meshgrid(np.linspace(0, width, nx), np.linspace(0, height, nz))
    panel = np.stack([gx.ravel(), np.zeros(gx.size), gz.ravel()], 1)
    panel += RNG.normal(0, 0.0015, panel.shape)          # scanner noise

    # Handle: a vertical bar standing proud of the panel in -y. A scanner sees
    # the whole protrusion, not a surface at one depth, so the points spread
    # from the panel outward -- which matters, because panel_height measures
    # coplanarity from the ELEMENT centroid with a 30 mm tolerance. A handle
    # modelled as a sheet at a single depth of 35 mm puts the panel outside
    # that tolerance and the door reads as a drawer.
    n_h = 120
    hz = RNG.uniform(0.95, 1.15, n_h)
    hy = RNG.uniform(-0.030, 0.0, n_h)
    handle = np.stack([np.full(n_h, handle_x), hy, hz], 1)
    handle += RNG.normal(0, 0.0015, handle.shape)

    # a floor, so the neighbourhood is not a single plane
    fx, fy = np.meshgrid(np.linspace(-0.5, 1.3, 80), np.linspace(-1.0, 0.4, 60))
    floor = np.stack([fx.ravel(), fy.ravel(), np.zeros(fx.size)], 1)
    return handle, np.vstack([panel, floor, handle])

def test_hinge_axis_is_vertical():
    pts, nbr = panel_scene()
    p = M.predict_motion('hook_pull', pts, nbr)
    assert p['motion_type'] == 'rot', p['motion_type']
    oe = M.OE(M.G, p['motion_dir'])
    assert oe < 15.0, f'hinge axis {oe:.1f} deg from vertical'

def test_hinge_origin_is_the_far_edge():
    """The handle sits at x=0.70 on a 0.80 m door, so the hinge is at x=0."""
    pts, nbr = panel_scene()
    p = M.predict_motion('hook_pull', pts, nbr)
    assert p['motion_origin'] is not None
    x = float(p['motion_origin'][0])
    assert x < 0.35, f'origin at x={x:.3f}, expected the far edge near x=0'

def test_parent_normal_beats_the_element_normal():
    """The claim the whole method rests on, in miniature: a thin vertical handle
    has almost no usable normal of its own, while the panel behind it does."""
    pts, nbr = panel_scene()
    cen = pts.mean(0)
    extent = float(np.linalg.norm(pts - cen, axis=1).mean())
    pn = M.parent_normal(nbr, cen, extent)
    assert pn is not None, 'parent_normal returned None on a clean panel'
    ang = min(M.OE(pn, [0, 1, 0]), M.OE(pn, [0, -1, 0]))
    assert ang < 10.0, f'parent normal {ang:.1f} deg off the panel normal'

def test_prismatic_slides_along_the_parent_normal():
    pts, nbr = panel_scene()
    p = M.predict_motion('tip_push', pts, nbr)
    assert p['motion_type'] == 'trans'
    assert p['motion_origin'] is None, 'translation is origin-invariant'
    ang = min(M.OE(p['motion_dir'], [0, 1, 0]), M.OE(p['motion_dir'], [0, -1, 0]))
    assert ang < 15.0, f'slide direction {ang:.1f} deg off the panel normal'

if __name__ == '__main__':
    fails = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith('test_'): continue
        try:
            fn(); print(f'  PASS  {name}')
        except AssertionError as e:
            fails += 1; print(f'  FAIL  {name}: {e}')
    print(f'\n{"OK" if not fails else str(fails) + " FAILED"}')
    sys.exit(1 if fails else 0)

"""Orthographic point-cloud rendering for paper figures.

WHY NOT mpl_toolkits.mplot3d. Its painter's algorithm sorts whole artists, not
points, so an element sitting on a panel is drawn either wholly in front of or
wholly behind it depending on artist order -- exactly the relationship these
figures exist to show. We project ourselves and depth-sort per point, which is
correct and also lets us size and fade points by depth for a legible sense of
volume.

Convention: the scan's up axis is +Z (sf3d.motion.G), so `up` defaults to that.
"""
import numpy as np

UP = np.array([0.0, 0.0, 1.0])

def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])

def camera(view_dir, up=UP):
    """Right-handed basis with -view_dir pointing at the camera.
    Returns (right, camera_up, forward); forward is the depth axis."""
    f = _unit(np.asarray(view_dir, float))
    if abs(f @ _unit(up)) > 0.99:            # degenerate: looking straight down
        up = np.array([0.0, 1.0, 0.0])
    r = _unit(np.cross(f, up))
    u = _unit(np.cross(r, f))
    return r, u, f

def project(pts, basis, centre):
    """-> (x, y, depth). Larger depth = farther from the camera."""
    r, u, f = basis
    q = np.asarray(pts, float) - centre
    return q @ r, q @ u, q @ f

def orbit(normal, az_deg=28.0, el_deg=16.0, up=UP):
    """A view a little off a surface normal, so the surface reads as a surface
    rather than as a line. Rotate about `up`, then lift by el_deg."""
    n = _unit(np.asarray(normal, float))
    a = np.deg2rad(az_deg)
    K = np.array([[0, -up[2], up[1]], [up[2], 0, -up[0]], [-up[1], up[0], 0]])
    R = np.eye(3) + np.sin(a)*K + (1-np.cos(a))*(K @ K)      # Rodrigues
    v = R @ n
    e = np.deg2rad(el_deg)
    side = _unit(np.cross(up, v))
    K2 = np.array([[0, -side[2], side[1]], [side[2], 0, -side[0]], [-side[1], side[0], 0]])
    R2 = np.eye(3) + np.sin(e)*K2 + (1-np.cos(e))*(K2 @ K2)
    return -_unit(R2 @ v)          # camera looks back along the surface normal

def scatter(ax, pts, basis, centre, color, s=1.0, alpha=1.0, zorder=2,
            depth_fade=True, rng=None, max_pts=None, lw=0):
    """Depth-sorted scatter. Far points are drawn first, smaller and fainter."""
    p = np.asarray(pts, float)
    if max_pts and len(p) > max_pts:
        rng = rng or np.random.default_rng(0)
        p = p[rng.choice(len(p), max_pts, replace=False)]
    if len(p) == 0:
        return
    x, y, d = project(p, basis, centre)
    o = np.argsort(-d)                        # far -> near
    x, y, d = x[o], y[o], d[o]
    if depth_fade and len(d) > 1 and np.ptp(d) > 1e-6:
        t = (d - d.min()) / np.ptp(d)           # 0 near, 1 far
        sz = s * (1.25 - 0.55 * t)
        al = alpha * (1.0 - 0.45 * t)
    else:
        sz = np.full(len(x), s); al = np.full(len(x), alpha)
    ax.scatter(x, y, s=sz, c=color, alpha=al, linewidths=lw,
               edgecolors='none', zorder=zorder, rasterized=True)

def segment(ax, p0, p1, basis, centre, **kw):
    x, y, _ = project(np.stack([p0, p1]), basis, centre)
    ax.plot(x, y, **kw)

def axis_line(ax, origin, direction, basis, centre, half_len, **kw):
    d = _unit(np.asarray(direction, float))
    o = np.asarray(origin, float)
    segment(ax, o - d*half_len, o + d*half_len, basis, centre, **kw)

def marker(ax, pt, basis, centre, **kw):
    x, y, _ = project(np.asarray(pt, float)[None], basis, centre)
    ax.plot(x, y, **kw)

def frame(ax, xr=None, yr=None):
    ax.set_aspect('equal'); ax.axis('off')
    if xr: ax.set_xlim(*xr)
    if yr: ax.set_ylim(*yr)

def fit_view(ax, pts, basis, centre, pad=0.08):
    x, y, _ = project(np.asarray(pts, float), basis, centre)
    dx, dy = np.ptp(x), np.ptp(y)
    m = max(dx, dy) * (0.5 + pad)
    cx, cy = (x.min()+x.max())/2, (y.min()+y.max())/2
    ax.set_xlim(cx-m, cx+m); ax.set_ylim(cy-m, cy+m)
    ax.set_aspect('equal'); ax.axis('off')

def crop_view(pts, basis, centre, hx, hy, hz=None):
    """Keep points inside a box defined in VIEW space, not a ball in world
    space. A spherical crop renders as a disc with a curved edge, which reads
    as a lens artefact rather than as a deliberate framing."""
    x, y, d = project(pts, basis, centre)
    m = (np.abs(x) < hx) & (np.abs(y) < hy)
    if hz is not None:
        m &= np.abs(d) < hz
    return np.asarray(pts, float)[m]

def readable_view(normal, axis, up=UP, els=(14.0,), azs=(40, 58, 72, 86)):
    """Pick a camera that shows BOTH the mounting surface and the motion axis.

    A prismatic element's direction is its parent normal, so a view chosen to
    show the surface looks straight down the axis and collapses it to a point.
    We widen the azimuth until the axis projects to a usable fraction of its
    true length, preferring the smallest rotation that works.
    """
    a = _unit(np.asarray(axis, float))
    best, best_len = None, -1.0
    for el in els:
        for az in azs:
            v = orbit(normal, az_deg=az, el_deg=el, up=up)
            r, u, f = camera(v, up)
            proj = np.hypot(a @ r, a @ u)          # in-image length of a unit axis
            if proj >= 0.55:
                return camera(v, up)
            if proj > best_len:
                best_len, best = proj, camera(v, up)
    return best

def axis_through(origin, direction, near_pt):
    """The point on the axis line closest to `near_pt`.

    Drawing a segment centred on the ORIGIN pushes the line off-frame whenever
    the origin is far from the element -- which for a hinge is exactly the
    normal case. Centre the drawn segment here instead; the line is the same
    line, and the origin marker still shows where it actually is.
    """
    d = _unit(np.asarray(direction, float))
    o = np.asarray(origin, float)
    return o + d * ((np.asarray(near_pt, float) - o) @ d)

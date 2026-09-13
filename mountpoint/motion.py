"""Training-free motion parameter estimation: motion_type, motion_dir,
motion_origin from a 3D region + its neighbourhood. Every rule is fixed
geometry -- nothing learned, nothing retrieved (MOTION.md Sec.1 hard
constraint). See MOTION.md for the full derivation and validation of every
rule below; this module only packages the final, validated functions so
they're importable from outside 02_motion_parameters.ipynb's own kernel
(needed by 03_3d_lifting.ipynb, which runs in a separate Colab runtime).

Functions take explicit arrays (region points, neighbourhood points, centroid,
extent) rather than an internal row-dict, so they work on ANY 3D region --
a ground-truth annotation or a Stage-2/3 predicted lift -- not just the
cached exemplar rows this project's own notebooks use internally.
"""
import numpy as np
from scipy.spatial import cKDTree

G = np.array([0.0, 0.0, 1.0])  # gravity = the scan's own up axis (hard constraint, never fitted)

# ---- fixed label categorisation, SceneFun3D's 9-category affordance taxonomy ----
# (MOTION.md Sec.4/7b: 7/9 categories are >98%-pure for one motion type)
KNOB = {'rotate'}
LEVER = {'hook_turn'}
MIXED = {'hook_pull', 'pinch_pull'}
DETERMINED_TYPE = {
    'foot_push': 'trans', 'hook_turn': 'rot', 'key_press': 'trans',
    'plug_in': 'trans', 'rotate': 'rot', 'tip_push': 'trans', 'unplug': 'trans',
}

# ---- fixed thresholds for the mixed-label type tie-break (MOTION.md Sec.7b) ----
# Both stable to within 0.01 across all leave-one-scene-out folds -- treated
# as measured constants, not per-scene-adaptive parameters.
TYPE_HEIGHT_THRESH = 1.054   # m; parent panel taller than this -> rotation
TYPE_OFFSET_THRESH = 0.456   # element's in-plane offset within the panel -> rotation

# Edge-selection threshold for the hinge origin (Sec.8c). The corner score alone
# picks the worse of the two panel edges 42.5% of the time -- barely better than
# chance -- while the ergonomic prior (a handle is mounted for leverage, so the
# hinge is the far edge) is only informative when the element is decisively
# off-centre. Below this normalised asymmetry the prior carries no signal and the
# corner score decides. Selected on DEV scenes in 19 of 20 random 50/50 splits;
# held-out TEST delta +1.04 pts, paired scene bootstrap +1.09 pts, 95% CI
# [+0.59, +1.63], positive in 100% of resamples. Same status as the two
# thresholds above: measured and stable, not fitted to the reported set.
EDGE_ASYM_THRESH = 0.6

# ---- metrics (MultiScan protocol) ----
def OE(a, b):
    """Orientation error, degrees. Unsigned: an axis and its flip are the same axis."""
    a = a / (np.linalg.norm(a) + 1e-9); b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(abs(float(a @ b)), 0, 1))))

def MD(p1, d1, p2, d2):
    """Minimum distance between two 3D lines, metres."""
    d1 = d1 / (np.linalg.norm(d1) + 1e-9); d2 = d2 / (np.linalg.norm(d2) + 1e-9)
    n = np.cross(d1, d2); nn = np.linalg.norm(n)
    if nn < 1e-6:
        w = np.asarray(p2) - np.asarray(p1)
        return float(np.linalg.norm(w - (w @ d1) * d1))
    return float(abs((np.asarray(p2) - np.asarray(p1)) @ (n / nn)))

def local_frame(pts):
    """Region's own normal, long in-plane axis (e1), planarity, centroid."""
    c = pts.mean(0)
    _, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    n, e1 = vt[2], vt[0]
    e1 = e1 - (e1 @ n) * n
    e1 /= np.linalg.norm(e1) + 1e-9
    return n, e1, float(s[2] / (s[0] + 1e-9)), c

def plane_normal(pts):
    _, _, vt = np.linalg.svd(pts - pts.mean(0), full_matrices=False)
    return vt[2]

def plane_normal_robust(pts, n_iter=3, k=2.5, min_keep=30):
    """Iteratively trimmed plane fit: fit, drop residuals beyond k MAD, refit.

    The shell around a mounting surface is not purely the mounting surface --
    it also contains the element's own shaft, the frame, and clutter in front
    of the panel. A least-squares fit cannot down-weight those; every point
    pulls on the normal. MAD-based trimming is the standard robust-estimator
    response and introduces no dataset-specific constant (k=2.5 MAD and the
    1.4826 consistency factor are the textbook Gaussian-equivalent values)."""
    P = pts
    n = plane_normal(P)
    for _ in range(n_iter):
        r = np.abs((P - P.mean(0)) @ n)
        mad = np.median(np.abs(r - np.median(r)))
        if mad < 1e-9:
            break
        keep = r < np.median(r) + k * 1.4826 * mad
        if keep.sum() < min_keep:
            break
        P = P[keep]
        n = plane_normal(P)
    return n

# ---- direction: parent-surface plane fit (MOTION.md Sec.2) ----
def parent_normal(nbr, cen, extent, lo=1.5, hi=4.0, min_pts=50, max_hi=12.0):
    """Fits a plane to a shell of neighbourhood points 1.5-4.0x the region's
    own extent, deliberately excluding the region's own points. Validated:
    2.13 deg median on prismatic elements vs 11.50 deg for the region's own
    normal -- a 5.4x reduction, flat across the shell-radius sweep. (The 3.21
    deg once quoted here predates the robust fit and adaptive shell below;
    scripts/eval_motion.py re-measures it on every run.)

    Two well-posedness changes, both held-out validated (scripts/
    improve_direction.py, scripts/eval_gate_variant.py; 504 elements,
    34 scenes):

    1. ADAPTIVE SHELL. `hi` grows (x1.5, capped at max_hi) only when the shell
       holds fewer than min_pts. Previously such a fit returned None and the
       caller fell back to the region's OWN normal -- which is far worse
       (48.21 deg vs 30.08 deg median on knobs). This only fires where the
       old fit was ill-posed; where the shell was already adequate it is a
       no-op, so it cannot regress an element that previously worked.
    2. ROBUST FIT (see plane_normal_robust).

    Selected on DEV scenes in 20/20 splits. Held-out TEST direction median
    4.02 -> 2.64 deg; paired scene bootstrap +1.31 deg, 95% CI [+0.62, +2.14].
    End-to-end at the motion gate (this function also feeds predict_type and
    predict_origin, so it is judged there): 60.5% -> 63.9% overall on the wide
    neighbourhood cache, 95% CI [+1.46, +5.19] pts, improving in 100% of 20
    held-out splits; hook_turn 36.9% -> 40.0%, revolute 31.3% -> 33.5%.
    """
    if nbr is None or len(nbr) == 0:
        return None
    d = np.linalg.norm(nbr - cen, axis=1)
    h = hi
    shell = nbr[(d > extent * lo) & (d < extent * h)]
    while len(shell) < min_pts and h < max_hi:
        h *= 1.5
        shell = nbr[(d > extent * lo) & (d < extent * h)]
    return plane_normal_robust(shell) if len(shell) >= min_pts else None

# ---- origin: panel isolation (MOTION.md Sec.5/8/8b) ----
def panel_component(nbr, cen, extent, hi, plane_tol=0.05, link=0.02, min_pts=150):
    """Coplanar region-growing from the point nearest the element, capped at
    hi x extent. hi=40 (not the original hi=20) -- Sec.8: hi=20 caps the
    search shorter than the true origin for 49% of hinges, independent of
    neighbourhood-cache radius."""
    if nbr is None:
        return None
    pn = parent_normal(nbr, cen, extent)
    if pn is None:
        return None
    d = np.linalg.norm(nbr - cen, axis=1)
    near = nbr[(d > extent * 1.2) & (d < extent * hi)]
    if len(near) < min_pts:
        return None
    cop = near[np.abs((near - cen) @ pn) < plane_tol]
    if len(cop) < min_pts:
        return None
    tree = cKDTree(cop)
    seed = int(np.argmin(np.linalg.norm(cop - cen, axis=1)))
    seen = np.zeros(len(cop), bool); seen[seed] = True
    frontier = [seed]
    while frontier:
        nxt = []
        for i in frontier:
            for j in tree.query_ball_point(cop[i], link):
                if not seen[j]:
                    seen[j] = True; nxt.append(j)
        frontier = nxt
    return cop[seen] if seen.sum() >= min_pts else None

def cop_points(nbr, cen, extent, hi=40, plane_tol=0.05):
    """Raw coplanar points -- same filter as panel_component but WITHOUT the
    flood-fill connectivity requirement. Fallback source when connectivity
    fails (Sec.8b: failing regions are genuinely fragmented, median 24
    connected points -- not a near-miss on the threshold). Rescues coverage
    from ~70% to ~98% and is the single biggest lever found for hinge origin."""
    if nbr is None:
        return None
    pn = parent_normal(nbr, cen, extent)
    if pn is None:
        return None
    d = np.linalg.norm(nbr - cen, axis=1)
    near = nbr[(d > extent * 1.2) & (d < extent * hi)]
    if len(near) < 30:
        return None
    cop = near[np.abs((near - cen) @ pn) < plane_tol]
    return cop if len(cop) >= 30 else None

def panel_axes(pn):
    """In-plane (v=vertical, h=horizontal) axes of a panel with normal pn."""
    v = G - (G @ pn) * pn
    nv = np.linalg.norm(v)
    if nv < 1e-6:
        return None
    v = v / nv
    h = np.cross(pn, v); h /= np.linalg.norm(h) + 1e-9
    return v, h

def panel_height(nbr, cen, extent, pn, hi=20, plane_tol=0.03):
    """Absolute vertical span of the isolated parent panel (Sec.7b's own
    hi=20/tol=0.03 params -- kept distinct from the hi=40 origin config)."""
    panel = panel_component(nbr, cen, extent, hi=hi, plane_tol=plane_tol)
    ax = panel_axes(pn) if panel is not None else None
    if ax is None:
        return None
    v, h = ax
    return float(np.ptp((panel - cen) @ v))

def panel_offset(nbr, cen, extent, pn, hi=20, plane_tol=0.03):
    panel = panel_component(nbr, cen, extent, hi=hi, plane_tol=plane_tol)
    ax = panel_axes(pn) if panel is not None else None
    if ax is None:
        return None
    v, h = ax
    hp = (panel - cen) @ h
    w = float(np.ptp(hp))
    return abs(hp.min() + hp.max()) / w if w > 1e-6 else None

def corner_score(nbr, pt, pn, radius=0.12, thresh=0.06):
    """Count of nearby neighbourhood points NOT coplanar with the panel --
    evidence of an adjacent wall/frame corner. A hinge is structurally
    anchored; a free edge more often has open space in front of it instead.
    Scene-CV validated (Sec.8): real gains on two independent held-out halves."""
    if nbr is None:
        return 0
    d = np.linalg.norm(nbr - pt, axis=1)
    near = nbr[d < radius]
    if len(near) == 0:
        return 0
    return int((np.abs((near - pt) @ pn) > thresh).sum())

# ---- top-level predictors ----
def predict_type(label, nbr, cen, extent):
    """rot/trans. Determined categories: fixed lookup (>98% pure). Mixed
    categories: combined height-OR-offset rule (MOTION.md Sec.7b) -- LOSO
    76.8% vs 69.6% majority baseline, scene-bootstrap 95% CI [+1.8%,+13.6%].
    Recall on the hard cases: 31/85 (36%) at 11/195 false positives -- real
    but conservative; most ambiguous elements still default to translation."""
    if label in DETERMINED_TYPE:
        return DETERMINED_TYPE[label]
    if label in MIXED:
        pn = parent_normal(nbr, cen, extent)
        is_rot = False
        if pn is not None:
            h = panel_height(nbr, cen, extent, pn)
            o = panel_offset(nbr, cen, extent, pn)
            is_rot = (h is not None and h > TYPE_HEIGHT_THRESH) or \
                     (o is not None and o > TYPE_OFFSET_THRESH)
        return 'rot' if is_rot else 'trans'
    return 'trans'  # unrecognised label: conservative default, majority class overall

def predict_dir(label, nbr, cen, extent, normal):
    """Rotation axis (revolute) or slide direction (prismatic). hook_turn and
    type-predicted-rotation mixed-label elements -> gravity (SceneFun3D hinge
    axes are canonically vertical -- a dataset/annotation convention as much
    as a geometric fact, MOTION.md Sec.4). Everything else -> parent_normal."""
    if label in LEVER:
        return G
    if label in MIXED and predict_type(label, nbr, cen, extent) == 'rot':
        return G
    p = parent_normal(nbr, cen, extent)
    return p if p is not None else normal

def predict_origin(label, axis, nbr, cen, extent):
    """Only meaningful when predicted type is 'rot'. Knob -> region centroid
    (essentially solved, 4mm median). Hinge -> corner-score-picked panel edge
    with a cascading fallback (connected panel -> raw coplanar extent ->
    centroid, Sec.8b) -- coverage ~98%, though picking accuracy itself still
    trails its own oracle ceiling (Sec.9)."""
    if label in KNOB:
        return cen
    pn = parent_normal(nbr, cen, extent)
    if pn is None:
        return cen
    panel = panel_component(nbr, cen, extent, hi=40)
    src = panel if panel is not None else cop_points(nbr, cen, extent, hi=40)
    if src is None:
        return cen
    ax = np.cross(axis, pn); n = np.linalg.norm(ax)
    if n < 1e-6:
        return cen
    ax = ax / n
    proj = (src - cen) @ ax
    far_pt, near_pt = cen + ax * proj.max(), cen + ax * proj.min()
    span = float(proj.max() - proj.min())
    asym = (abs(abs(proj.max()) - abs(proj.min())) / span) if span > 1e-6 else 0.0
    if asym >= EDGE_ASYM_THRESH:
        # the element sits decisively toward one edge of the panel; a handle is
        # placed for leverage, so the hinge is the edge it is FARTHEST from
        return far_pt if abs(proj.max()) >= abs(proj.min()) else near_pt
    sf = corner_score(nbr, far_pt, pn)
    sn = corner_score(nbr, near_pt, pn)
    return far_pt if sf >= sn else near_pt

def predict_motion(label, pts, nbr):
    """Top-level entry point. `pts`: the region's own 3D points (a
    ground-truth annotation OR a Stage-2/3 predicted lift -- this function
    doesn't care which). `nbr`: neighbourhood points out to >=1.1m radius
    around the region's centroid (Sec.8: 0.60m cuts off 37% of true hinge
    origins). Returns dict(motion_type, motion_dir, motion_origin) --
    motion_origin is None for predicted-translation elements (origin-invariant)."""
    cen = pts.mean(0)
    extent = float(np.linalg.norm(pts - cen, axis=1).mean())
    normal, e1, planar, _ = local_frame(pts)
    t = predict_type(label, nbr, cen, extent)
    axis = predict_dir(label, nbr, cen, extent, normal)
    origin = predict_origin(label, axis, nbr, cen, extent) if t == 'rot' else None
    return dict(motion_type=t, motion_dir=axis, motion_origin=origin)

# ---- panel-given variant (Sec. transfer): when the mounting surface is SEGMENTED ----
def predict_motion_given_panel(pts, panel, kind, nbr=None):
    """Same principle as predict_motion -- read motion off the mounting surface --
    for the case where that surface is GIVEN rather than recovered.

    SceneFun3D annotates only the functional element, so parent_normal must
    reconstruct the panel from a shell of neighbourhood points. Benchmarks that
    segment the movable part (Articulate3D; and any part-segmentation model, which
    is what the hybrid consumes) hand us the panel directly, and recovering it is
    then both unnecessary and harmful: a door set flush in a wall is COPLANAR with
    that wall, so region growing returns the wall (measured: median span 1.44 m for
    a ~0.8 m door, capping oracle edge choice at 46.3%).

    `pts`   the interactable element (handle), or the part itself if unknown
    `panel` the movable part's points -- the mounting surface
    `kind`  'rot' | 'trans', taken as given (from the benchmark's class prediction)
    `nbr`   optional neighbourhood, only used for the corner-score fallback

    Rules, unchanged in spirit and free of new constants:
      trans -> the panel's own robust plane normal (a drawer slides out of its face)
      rot   -> gravity, origin at a panel edge along cross(axis, panel normal).
               Which edge: the one FARTHER from the element, because a handle is
               mounted for leverage and therefore opposite the hinge. Without an
               element to reference, fall back to the corner score.

    Measured on Articulate3D validation (42 scenes, ground-truth part masks):
    gate 88.5% with a handle, 95% CI [83.9, 92.9]; 78.6% without; 83.3% overall.
    Degrading the panel to 50% of its points costs 2.2 points, so the result does
    not depend on a perfect mask.
    """
    pn = plane_normal_robust(panel) if len(panel) >= 10 else None
    if pn is None:
        pn = local_frame(pts)[0]
    if kind != 'rot':
        return dict(motion_type='trans', motion_dir=pn, motion_origin=None)

    cen = pts.mean(0)
    axis = G
    ax = np.cross(axis, pn)
    n = np.linalg.norm(ax)
    if n < 1e-6:
        return dict(motion_type='rot', motion_dir=axis, motion_origin=cen)
    ax = ax / n
    proj = (panel - cen) @ ax
    far, near = cen + ax * proj.max(), cen + ax * proj.min()

    have_element = len(pts) < len(panel)      # a handle, not the panel itself
    if have_element:
        org = far if abs(proj.max()) >= abs(proj.min()) else near
    elif nbr is not None:
        org = far if corner_score(nbr, far, pn) >= corner_score(nbr, near, pn) else near
    else:
        org = far if abs(proj.max()) >= abs(proj.min()) else near
    return dict(motion_type='rot', motion_dir=axis, motion_origin=org)

# ---- signed direction (Sec. 4.2) --------------------------------------------
# predict_dir returns an AXIS: a plane fit determines a normal only up to sign,
# so the deployed direction is sign-correct about half the time. The composed
# metric is unsigned so this costs nothing there, but the annotations are signed
# and meaningful, so we recover the sign from geometry already in hand.
#
# An element protrudes from the surface it is mounted on, so the sign of
# (element centroid - shell centroid) . n identifies the outward side. The
# affordance then supplies the sense: pull goes outward, press and plug inward.
# Revolute elements are excluded -- a hinge axis has no preferred end.
#
# Measured, n=432 prismatic: unsigned 84.3% (the ceiling), deployed 40.7%,
# with this rule 76.9%.

_SENSE = {'hook_pull': +1, 'pinch_pull': +1, 'unplug': +1,
          'key_press': -1, 'tip_push': -1, 'foot_push': -1, 'plug_in': -1}

def outward_normal(pts, nbr, cen, extent, pn):
    """Orient `pn` away from the mounting surface. None if the shell is too thin."""
    d = np.linalg.norm(nbr - cen, axis=1)
    h = 4.0
    shell = nbr[(d > extent * 1.5) & (d < extent * h)]
    while len(shell) < 50 and h < 12.0:
        h *= 1.5
        shell = nbr[(d > extent * 1.5) & (d < extent * h)]
    if len(shell) < 50:
        return None
    return pn if (cen - shell.mean(0)) @ pn > 0 else -pn

def signed_dir(label, pts, nbr):
    """Direction as a VECTOR rather than an axis. None where undefined:
    revolute elements, and labels with no canonical sense."""
    if label not in _SENSE:
        return None
    cen = pts.mean(0)
    extent = float(np.linalg.norm(pts - cen, axis=1).mean())
    if predict_type(label, nbr, cen, extent) != 'trans':
        return None
    pn = parent_normal(nbr, cen, extent)
    if pn is None:
        return None
    on = outward_normal(pts, nbr, cen, extent, pn)
    return None if on is None else on * _SENSE[label]

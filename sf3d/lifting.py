"""Multi-view 2D-to-3D lifting for Stage 2 predictions, with metric-aware SAM
candidate re-selection. See notebook header for the diagnosis this fixes.
"""
import numpy as np
import cv2
from scipy.spatial import cKDTree
from plyfile import PlyData
from utils.fusion_util import PointCloudToImageMapper

from . import config as C
from . import pipeline as P


def load_laser_scan_xyz(data_root, visit_id):
    """plyfile-based (not open3d.io.read_point_cloud): avoids an open3d
    dependency for the one thing we actually need -- XYZ positions -- and
    matches the same loader the motion module already uses."""
    path = f'{data_root}/{visit_id}/{visit_id}_laser_scan.ply'
    vx = PlyData.read(path)['vertex']
    return np.stack([vx['x'], vx['y'], vx['z']], 1).astype(np.float64)


def pick_frames(parser, visit_id, video_id, center_ts, n_frames=6):
    """Frames spread around the exemplar's own annotation timestamp, restricted
    to timestamps that have an RGB frame, a depth frame, AND intrinsics."""
    rgb_paths = parser.get_rgb_frames(visit_id, video_id, data_asset_identifier='hires_wide')
    depth_paths = parser.get_depth_frames(visit_id, video_id, data_asset_identifier='hires_depth')
    intr_paths = parser.get_camera_intrinsics(visit_id, video_id, data_asset_identifier='hires_wide_intrinsics')
    common = sorted(set(rgb_paths) & set(depth_paths) & set(intr_paths), key=float)
    if not common:
        return [], rgb_paths, depth_paths, intr_paths
    center = min(common, key=lambda t: abs(float(t) - float(center_ts)))
    idx = common.index(center)
    offsets = sorted(range(-n_frames // 2, n_frames // 2 + 1), key=abs)
    picked = []
    for off in offsets:
        j = idx + off
        if 0 <= j < len(common) and common[j] not in picked:
            picked.append(common[j])
        if len(picked) >= n_frames:
            break
    return picked, rgb_paths, depth_paths, intr_paths


def rescore_candidates_by_metric(sam_cands, hot, mapping, laser_xyz, prior_extent, region, max_mult=4.0):
    """Re-rank SAM's raw candidate masks (it always proposes 3, at different
    granularities) by REAL physical extent, using the laser-scan points
    actually visible in this frame -- not pixel count, which is blind to
    depth. Returns None if no candidate is physically plausible: we do not
    fall back to an oversized one, we skip this frame's contribution.

    sam_cands are CROP-sized (SAM only returns masks at the box-crop's own
    resolution), so mapping/hot are shifted into crop-relative coordinates.
    """
    X0, Y0, X1, Y1 = region
    valid = mapping[:, 2] == 1
    rows, cols = mapping[valid, 0] - Y0, mapping[valid, 1] - X0
    global_idx = np.where(valid)[0]
    in_crop = (rows >= 0) & (cols >= 0) & (rows < (Y1 - Y0)) & (cols < (X1 - X0))
    rows, cols, global_idx = rows[in_crop], cols[in_crop], global_idx[in_crop]

    scored = []
    for i, m in enumerate(sam_cands):
        if m.sum() < 20:
            continue
        H, W = m.shape
        in_bounds = (rows < H) & (cols < W)
        hit = m[rows[in_bounds], cols[in_bounds]].astype(bool)
        pt_idx = global_idx[in_bounds][hit]
        if len(pt_idx) < 5:
            continue
        pts = laser_xyz[pt_idx]
        extent = float(np.linalg.norm(pts - pts.mean(0), axis=1).mean())
        cov = (m & hot).sum() / max(hot.sum(), 1) if hot is not None else 0.0
        pur = (m & hot).sum() / max(m.sum(), 1) if hot is not None else 0.0
        scored.append(dict(extent=extent, pt_idx=pt_idx,
                            plausible=(extent <= max_mult * prior_extent), pur=pur, cov=cov))

    pool = [s for s in scored if s['plausible']]
    if not pool:
        return None  # never fall back to an implausibly-large candidate
    return max(pool, key=lambda s: (2.0 * s['pur'] + 0.5 * s['cov'], -s['extent']))


def largest_cluster(points, link=0.05, min_pts=3):
    """Keep only the single largest spatially-coherent cluster (BFS via
    radius-linking -- same pattern as the motion module's panel_component). A
    gripper needs one coherent region, not scattered agreement across unrelated
    parts of the room."""
    if len(points) < min_pts:
        return np.zeros(len(points), bool)
    tree = cKDTree(points)
    pairs = tree.query_pairs(link, output_type='ndarray')
    parent = np.arange(len(points))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(i) for i in range(len(points))])
    labels, counts = np.unique(roots, return_counts=True)
    return roots == labels[np.argmax(counts)]


def lift_instance(parser, e, gt_indices, proto, prior_extent, n_frames=6,
                   agree_thresh=0.5, min_views=1, min_flags=2, cluster_link=0.05,
                   verbose=False):
    """Lift one exemplar's Stage 2 prediction into a 3D point mask.

    Args:
        parser: utils.data_parser.DataParser
        e: exemplar_db entry (needs visit, video, frame, label, desc)
        gt_indices: ground-truth point indices for this instance (for scoring;
            omit/None at real inference time when GT isn't known)
        proto: output of pipeline.build_proto(pool) for this label
        prior_extent: physical size prior in metres (see notebook section 3)
    Returns dict with pred_mask, iou3d (if gt_indices given), used_frames, etc.
    """
    visit_id, video_id, center_ts = e['visit'], e['video'], e['frame']
    xyz = load_laser_scan_xyz(C.ROOT + '/data', visit_id)
    n_points = xyz.shape[0]

    timestamps, rgb_paths, depth_paths, intr_paths = pick_frames(parser, visit_id, video_id, center_ts, n_frames)
    if not timestamps:
        return dict(ok=False, reason='no posed frames available for this visit')

    poses = parser.get_camera_trajectory(visit_id, video_id, pose_source='colmap')
    w, h, fx, fy, hw, hh = parser.read_camera_intrinsics(next(iter(intr_paths.values())))
    mapper = PointCloudToImageMapper((int(w), int(h)), visibility_threshold=0.25, cut_bound=0)

    acc = np.zeros(n_points, np.float32)
    seen = np.zeros(n_points, np.float32)
    used_frames, frame_report = 0, []
    for ts in timestamps:
        pose = parser.get_nearest_pose(ts, poses)
        if pose is None:
            continue
        img = cv2.cvtColor(cv2.imread(rgb_paths[ts]), cv2.COLOR_BGR2RGB)
        text = f"{(e.get('desc') or [''])[0]}. {C.PARENT_HINT.get(e['label'], C.DEFAULT_HINT)}"
        res = P.run(img, proto, text)
        if res is None:
            continue
        depth = parser.read_depth_frame(depth_paths[ts])
        intr = parser.read_camera_intrinsics(intr_paths[ts], format='matrix')
        mapping = mapper.compute_mapping(pose, xyz, depth, intr)
        if (mapping[:, 2] == 1).sum() == 0:
            continue

        region = res['region']
        X0, Y0, X1, Y1 = region
        region_smap = res['smap'][Y0:Y1, X0:X1]
        hot_region = region_smap >= np.percentile(region_smap, 92)

        best = rescore_candidates_by_metric(res['sam_cands'], hot_region, mapping, xyz, prior_extent, region)
        if best is None:
            frame_report.append((ts, 'no metric-plausible candidate')); continue

        acc[best['pt_idx']] += 1
        seen[mapping[:, 2] == 1] += 1
        used_frames += 1
        frame_report.append((ts, f"extent={best['extent']:.4f}m n_pts={len(best['pt_idx'])}"))

    result = dict(ok=used_frames > 0, visit=visit_id, annot_id=e.get('annot_id'), label=e['label'],
                  used_frames=used_frames, n_frames_tried=len(timestamps), frame_report=frame_report)
    if used_frames == 0:
        return result

    with np.errstate(divide='ignore', invalid='ignore'):
        agreement = np.where(seen > 0, acc / seen, 0)
    raw_pred_mask = (agreement >= agree_thresh) & (seen >= min_views) & (acc >= min_flags)

    pred_idx = np.where(raw_pred_mask)[0]
    if len(pred_idx) >= 3:
        keep = largest_cluster(xyz[pred_idx], link=cluster_link)
        pred_mask = np.zeros(n_points, bool); pred_mask[pred_idx[keep]] = True
    else:
        pred_mask = raw_pred_mask

    result['pred_mask'] = pred_mask
    result['pred_pts'] = int(pred_mask.sum())
    if gt_indices is not None:
        gt_mask = np.zeros(n_points, bool); gt_mask[gt_indices] = True
        inter = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        result['iou3d'] = float(inter / union) if union > 0 else 0.0
        result['gt_pts'] = int(gt_mask.sum())
    if verbose:
        print(f"  {visit_id} {e['label']:<10} frames={used_frames}/{len(timestamps)} "
              f"pred_pts={result['pred_pts']}" +
              (f" gt_pts={result['gt_pts']} IoU={result['iou3d']:.3f}" if gt_indices is not None else ''))
    return result
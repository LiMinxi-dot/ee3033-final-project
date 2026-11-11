"""Pure geometry helpers for the target search mission (no ROS imports, unit-testable).

Functions taking ROS messages only read their fields, so tests can pass simple stand-ins.
"""

import math

import cv2
import numpy as np


def wrap_angle(a):
    """Wrap an angle (or array of angles) to [-pi, pi)."""
    return (np.asarray(a) + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_to_quaternion(yaw):
    """(x, y, z, w) of a rotation by yaw about z."""
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def transform_to_xyyaw(tf):
    """Planar pose (x, y, yaw) of a geometry_msgs/TransformStamped."""
    t, q = tf.transform.translation, tf.transform.rotation
    return t.x, t.y, yaw_from_quaternion(q.x, q.y, q.z, q.w)


def scan_to_points(scan, x0=0.0, y0=0.0, yaw=0.0):
    """Valid hits of a sensor_msgs/LaserScan as an (N, 2) array, moved by the scan frame pose (x0, y0, yaw)."""
    r = np.asarray(scan.ranges, dtype=float)
    a = scan.angle_min + scan.angle_increment * np.arange(r.size)
    ok = np.isfinite(r) & (r > scan.range_min) & (r < scan.range_max)
    r, a = r[ok], a[ok] + yaw
    return np.column_stack((x0 + r * np.cos(a), y0 + r * np.sin(a)))


def apply_body_offset(x, y, yaw, dx, dy, dyaw):
    """Pose (x, y, yaw) moved by (dx, dy) in its own frame and rotated by dyaw."""
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * dx - s * dy, y + s * dx + c * dy, yaw + dyaw


def focal_length_px(image_width, hfov_rad):
    """Pinhole focal length (pixels) from the horizontal field of view."""
    return (image_width / 2.0) / math.tan(hfov_rad / 2.0)


def intrinsics(K, image_width, hfov_rad):
    """(fx, cx) from the calibrated camera matrix K, or from the field of view if K is None."""
    if K is not None:
        return K[0, 0], K[0, 2]
    return focal_length_px(image_width, hfov_rad), image_width / 2.0


def pixel_to_bearing(u, cx, fx):
    """Horizontal bearing (rad) of pixel column u in the camera body frame.

    The camera frame follows the ROS body convention (x forward, y left), so a
    target to the right of the image centre has a negative bearing.
    """
    return math.atan2(-(u - cx), fx)


def ray_range_from_scan(origin_xy, ray_yaw, points_xy, half_width_rad,
                        min_range=0.15, max_range=6.0, cluster_depth=0.15):
    """Distance along a ray to the nearest object seen by the LiDAR: (range, n_points) or (None, 0).

    See nearest_cluster_bearing; points_xy are LiDAR hits in the frame of the ray origin.
    """
    _, r, n = nearest_cluster_bearing(origin_xy, ray_yaw, points_xy, half_width_rad,
                                      min_range, max_range, cluster_depth)
    return r, n


def nearest_cluster_bearing(origin_xy, ray_yaw, points_xy, half_width_rad,
                            min_range=0.3, max_range=3.0, cluster_depth=0.15):
    """Bearing and range of the nearest LiDAR cluster within half_width_rad of a ray.

    Points whose bearing (seen from the origin) is within half_width_rad of the
    ray are kept; the nearest cluster of them (within cluster_depth of the
    closest one) gives the median range and bearing. For the camera/LiDAR yaw
    calibration the camera gives ray_yaw and the LiDAR tells where the object
    really is. Returns (bearing, range, n_points) with the bearing in the same
    frame as ray_yaw, or (None, None, 0).
    """
    pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    if pts.size == 0:
        return None, None, 0
    d = pts - np.asarray(origin_xy, dtype=float)
    r = np.hypot(d[:, 0], d[:, 1])
    ang = np.arctan2(d[:, 1], d[:, 0])
    rel = wrap_angle(ang - ray_yaw)
    keep = (np.abs(rel) <= half_width_rad) & (r >= min_range) & (r <= max_range)
    if not keep.any():
        return None, None, 0
    r, rel = r[keep], rel[keep]
    near = r <= r.min() + cluster_depth
    return float(ray_yaw + np.median(rel[near])), float(np.median(r[near])), int(near.sum())


def range_from_object_height(bbox_height_px, object_height_m, fy):
    """Monocular range estimate from a known object height (fallback when the LiDAR misses)."""
    if bbox_height_px <= 1.0 or object_height_m <= 0.0:
        return None
    return object_height_m * fy / bbox_height_px


def fuse_observations(points_xy, cluster_radius):
    """Robustly fuse target observations.

    Finds the largest group of observations lying within cluster_radius of one
    observation and returns (estimate_xy, n_support, spread) using the median of
    that group, which rejects outliers such as a wrong LiDAR hit.
    """
    pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    if pts.shape[0] == 0:
        return None, 0, 0.0
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.hypot(diff[..., 0], diff[..., 1])
    support = (dist <= cluster_radius).sum(axis=1)
    best = int(np.argmax(support))
    members = pts[dist[best] <= cluster_radius]
    est = np.median(members, axis=0)
    spread = float(np.median(np.hypot(*(members - est).T)))
    return (float(est[0]), float(est[1])), int(members.shape[0]), spread


def grid_to_world(row, col, origin_xy, resolution):
    """Cell centre of OccupancyGrid cell (row, col) in the map frame (map yaw assumed 0)."""
    return (origin_xy[0] + (col + 0.5) * resolution,
            origin_xy[1] + (row + 0.5) * resolution)


def world_to_grid(x, y, origin_xy, resolution):
    return (int(math.floor((y - origin_xy[1]) / resolution)),
            int(math.floor((x - origin_xy[0]) / resolution)))


def generate_viewpoints(grid, resolution, origin_xy, start_xy, spacing,
                        min_clearance, occupied_thresh=50, skip_radius=0.0):
    """Pick search viewpoints that cover the free space of an occupancy grid.

    grid: 2-D int array (OccupancyGrid data reshaped to height x width,
    -1 unknown, 0..100 occupancy). The free space is divided into blocks of
    spacing x spacing metres; in each block the reachable free cell with the
    largest clearance from obstacles/unknown space becomes a viewpoint. Only
    cells connected to the start are used, so every viewpoint is reachable.
    Viewpoints within skip_radius of the start are dropped (the robot already
    looks around there). Returns a list of (x, y) in the map frame.
    """
    grid = np.asarray(grid)
    free = ((grid >= 0) & (grid < occupied_thresh)).astype(np.uint8)
    # Distance (m) from each free cell to the nearest occupied or unknown cell.
    clearance = cv2.distanceTransform(free, cv2.DIST_L2, 5) * resolution
    ok = (clearance >= min_clearance).astype(np.uint8)
    if not ok.any():
        return []

    _, labels = cv2.connectedComponents(ok, connectivity=8)
    sr, sc = world_to_grid(start_xy[0], start_xy[1], origin_xy, resolution)
    rows, cols = np.nonzero(ok)
    # Component of the start cell, or of the nearest valid cell if the start itself is too tight.
    nearest = int(np.argmin((rows - sr) ** 2 + (cols - sc) ** 2))
    reachable = labels == labels[rows[nearest], cols[nearest]]

    step = max(1, int(round(spacing / resolution)))
    h, w = grid.shape
    candidates = []   # (clearance, x, y)
    for r0 in range(0, h, step):
        for c0 in range(0, w, step):
            block = np.where(reachable[r0:r0 + step, c0:c0 + step],
                             clearance[r0:r0 + step, c0:c0 + step], -1.0)
            if block.max() < 0:
                continue
            br, bc = np.unravel_index(int(np.argmax(block)), block.shape)
            x, y = grid_to_world(r0 + br, c0 + bc, origin_xy, resolution)
            if math.hypot(x - start_xy[0], y - start_xy[1]) < skip_radius:
                continue
            candidates.append((float(block.max()), x, y))

    # Neighbouring blocks can pick cells on either side of their shared border; keep the
    # most open of any viewpoints closer than half the spacing.
    viewpoints = []
    for _, x, y in sorted(candidates, reverse=True):
        if all(math.hypot(x - vx, y - vy) >= 0.5 * spacing for vx, vy in viewpoints):
            viewpoints.append((x, y))
    return viewpoints


def plan_coverage(grid, resolution, origin_xy, start_xy, view_range, min_clearance, candidate_spacing=0.25,
                  occupied_thresh=50, sample=0.10, min_gain=0.01):
    """Viewpoints from which the whole free space is seen (the robot spins 360 deg at each one).

    Candidates come from generate_viewpoints() on a fine grid (reachable, clear of walls). A free cell
    (sampled every `sample` m) is seen from a point if it is within view_range and the straight line
    to it crosses no occupied or unknown cell. The start counts as a viewpoint if it is clear enough
    for the initial spin. Greedy set cover: take the candidate that sees the most still unseen cells
    until no candidate adds min_gain of the cells.
    Returns (viewpoints, fraction of the free cells seen, fraction seeable from any candidate).
    """
    grid = np.asarray(grid)
    free = (grid >= 0) & (grid < occupied_thresh)
    h, w = grid.shape
    step = max(1, int(round(sample / resolution)))
    rows, cols = np.nonzero(free)
    keep = (rows % step == 0) & (cols % step == 0)
    tx = origin_xy[0] + (cols[keep] + 0.5) * resolution
    ty = origin_xy[1] + (rows[keep] + 0.5) * resolution
    if tx.size == 0:
        return [], 0.0, 0.0
    k = np.linspace(0.0, 1.0, int(view_range / (resolution / 2)) + 2)[1:-1]   # samples between the ends

    def seen_from(x, y):
        near = np.hypot(tx - x, ty - y) <= view_range
        px = x + np.outer(tx[near] - x, k)
        py = y + np.outer(ty[near] - y, k)
        c = np.clip(np.floor((px - origin_xy[0]) / resolution).astype(int), 0, w - 1)
        r = np.clip(np.floor((py - origin_xy[1]) / resolution).astype(int), 0, h - 1)
        seen = np.zeros(tx.size, bool)
        seen[near] = free[r, c].all(axis=1)
        return seen

    candidates = generate_viewpoints(grid, resolution, origin_xy, start_xy, candidate_spacing, min_clearance,
                                     occupied_thresh)
    views = [seen_from(x, y) for x, y in candidates]
    seeable = np.logical_or.reduce(views) if views else np.zeros(tx.size, bool)
    clear = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5) * resolution
    sr, sc = world_to_grid(start_xy[0], start_xy[1], origin_xy, resolution)
    start_ok = 0 <= sr < h and 0 <= sc < w and clear[sr, sc] >= min_clearance
    covered = seen_from(*start_xy) if start_ok else np.zeros(tx.size, bool)
    picked = []
    while views:
        gains = [int((v & ~covered).sum()) for v in views]
        i = int(np.argmax(gains))
        if gains[i] < max(1, min_gain * tx.size):
            break
        picked.append(candidates.pop(i))
        covered |= views.pop(i)
    return picked, float(covered.mean()), float((seeable | covered).mean())


def wall_sdf(grid, resolution, occupied_thresh=50):
    """Signed distance (m) of every cell to the map walls: outside a wall the distance to the nearest wall cell,
    inside one minus the distance to the nearest cell that is not a wall."""
    wall = np.asarray(grid) >= occupied_thresh
    out = cv2.distanceTransform((~wall).astype(np.uint8), cv2.DIST_L2, 5)
    inside = cv2.distanceTransform(wall.astype(np.uint8), cv2.DIST_L2, 5)
    return (out - inside) * resolution


def line_min_sdf(sdf, resolution, origin_xy, a, b):
    """Smallest wall_sdf value (m) on the straight line from a to b (x, y), sampled every half cell; -inf if the
    line leaves the map."""
    n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / (resolution / 2)) + 2)
    xs, ys = np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n)
    rows = np.floor((ys - origin_xy[1]) / resolution).astype(int)
    cols = np.floor((xs - origin_xy[0]) / resolution).astype(int)
    h, w = sdf.shape
    if rows.min() < 0 or cols.min() < 0 or rows.max() >= h or cols.max() >= w:
        return float('-inf')
    return float(sdf[rows, cols].min())


def escape_move(grid, sdf, resolution, origin_xy, pose, target, max_forward, max_backward, occupied_thresh=50):
    """Straight move along the robot's heading away from the map walls, as (d, clearance at the end): d > 0
    forward, d < 0 backward. The shortest move that ends at least `target` from the walls (forward unless
    backward is clearly shorter); if none gets that far, the one that gains the most (at least one cell).
    None if no move helps.

    Next to a wall Nav2 does not move the robot at all: the planner says "Starting point in lethal space" and its
    spin and back-up say "Collision Ahead". A move never gets closer to a wall than it already got on the way
    (half a cell of tolerance), so it cannot cross a thin wall to its other side, and it ends on a free cell.
    sdf: wall_sdf() of grid.
    """
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    h, w = sdf.shape
    step = resolution / 2

    def at(px, py):
        row, col = world_to_grid(px, py, origin_xy, resolution)
        return (float(sdf[row, col]), int(grid[row, col])) if 0 <= row < h and 0 <= col < w else (None, None)

    start, _ = at(x, y)
    if start is None:
        return None
    options = []                      # (d, clearance) of the moves that end on a free cell
    for sign, limit in ((1, max_forward), (-1, max_backward)):
        top, d = start, step
        while d <= limit + 1e-9:
            v, g = at(x + sign * d * c, y + sign * d * s)
            if v is None or v < top - step:
                break
            top = max(top, v)
            if 0 <= g < occupied_thresh:
                options.append((sign * d, v))
                if v >= target:
                    break
            d += step
    reach = [o for o in options if o[1] >= target]
    if reach:
        fwd = min((o for o in reach if o[0] > 0), default=None)
        back = max((o for o in reach if o[0] < 0), default=None)
        return fwd if fwd and (back is None or fwd[0] <= -back[0] + 0.03) else back
    if not options:
        return None
    d, v = max(options, key=lambda o: (o[1], -abs(o[0])))
    return (d, v) if v >= start + resolution else None


def _turned(points, angle):
    """Points (robot frame) as seen by the robot after it has turned by angle."""
    c, s = math.cos(angle), math.sin(angle)
    return np.column_stack((c * points[:, 0] + s * points[:, 1], -s * points[:, 0] + c * points[:, 1]))


def free_ahead(points, half_length, half_width, turn=0.0):
    """Free distance (m) in front of the robot's front edge after turning by `turn`: the nearest of the points
    (robot frame, e.g. LiDAR hits) in the strip the robot's width sweeps forward. inf if none."""
    p = _turned(points, turn)
    ahead = p[(p[:, 0] > 0) & (np.abs(p[:, 1]) <= half_width), 0]
    return max(0.0, float(ahead.min()) - half_length) if ahead.size else float('inf')


def rotation_clear(points, half_length, half_width, turn, step=math.radians(3)):
    """True if turning in place by `turn` (rad) sweeps the robot's rectangle past none of the points."""
    for a in np.linspace(0.0, turn, max(2, int(abs(turn) / step) + 2)):
        p = _turned(points, a)
        if np.any((np.abs(p[:, 0]) <= half_length) & (np.abs(p[:, 1]) <= half_width)):
            return False
    return True


def order_nearest_neighbour(start_xy, points):
    """Greedy nearest-neighbour tour starting from start_xy."""
    remaining = list(points)
    tour = []
    cur = start_xy
    while remaining:
        i = min(range(len(remaining)),
                key=lambda k: (remaining[k][0] - cur[0]) ** 2 + (remaining[k][1] - cur[1]) ** 2)
        cur = remaining.pop(i)
        tour.append(cur)
    return tour


def scale_odom_step(prev_raw, raw, corrected, yaw_scale, lin_scale=1.0, yaw_scale_cw=None):
    """Next corrected odometry pose (x, y, yaw) from two consecutive raw poses: the raw step is taken in the
    robot frame of prev_raw (forward, lateral, turn), the turn scaled by yaw_scale (left turns; right turns by
    yaw_scale_cw, default the same) and the shift by lin_scale, and applied to `corrected`. The shift keeps its
    direction relative to the step's midpoint heading, so it turns by (scale - 1) * raw turn / 2 beyond the corrected
    start heading (scales 1: the raw path exactly). On our car the base over-reports left turns by ~2.5 % and right
    turns by ~4.5 % (scan-to-map heading over 630 two-second windows of two site missions, 11-11)."""
    px, py, pa = prev_raw
    dx, dy = raw[0] - px, raw[1] - py
    fwd = math.cos(pa) * dx + math.sin(pa) * dy
    lat = -math.sin(pa) * dx + math.cos(pa) * dy
    raw_turn = float(wrap_angle(raw[2] - pa))
    k = yaw_scale if raw_turn >= 0.0 or yaw_scale_cw is None else yaw_scale_cw
    cx, cy, ca = corrected
    rot = ca + (k - 1.0) * raw_turn / 2.0
    return (cx + lin_scale * (math.cos(rot) * fwd - math.sin(rot) * lat),
            cy + lin_scale * (math.sin(rot) * fwd + math.cos(rot) * lat),
            float(wrap_angle(ca + k * raw_turn)))

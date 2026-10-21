"""Pure geometry helpers for the target search mission (no ROS imports, unit-testable)."""

import math

import numpy as np


def wrap_angle(a):
    """Wrap an angle (or array of angles) to [-pi, pi)."""
    return (np.asarray(a) + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def apply_body_offset(x, y, yaw, dx, dy, dyaw):
    """Pose (x, y, yaw) moved by (dx, dy) in its own frame and rotated by dyaw."""
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * dx - s * dy, y + s * dx + c * dy, yaw + dyaw


def focal_length_px(image_width, hfov_rad):
    """Pinhole focal length (pixels) from the horizontal field of view."""
    return (image_width / 2.0) / math.tan(hfov_rad / 2.0)


def pixel_to_bearing(u, cx, fx):
    """Horizontal bearing (rad) of pixel column u in the camera body frame.

    The camera frame follows the ROS body convention (x forward, y left), so a
    target to the right of the image centre has a negative bearing.
    """
    return math.atan2(-(u - cx), fx)


def ray_range_from_scan(origin_xy, ray_yaw, points_xy, half_width_rad,
                        min_range=0.15, max_range=6.0, cluster_depth=0.15):
    """Distance along a ray to the nearest object seen by the LiDAR.

    points_xy are LiDAR hits already in the same frame as the ray origin.
    Points whose bearing (seen from the origin) is within half_width_rad of the
    ray are kept, and the nearest cluster of them (within cluster_depth of the
    closest one) gives the range. Returns (range, n_points) or (None, 0).
    """
    pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    if pts.size == 0:
        return None, 0
    d = pts - np.asarray(origin_xy, dtype=float)
    r = np.hypot(d[:, 0], d[:, 1])
    ang = np.arctan2(d[:, 1], d[:, 0])
    keep = (np.abs(wrap_angle(ang - ray_yaw)) <= half_width_rad) & (r >= min_range) & (r <= max_range)
    r = r[keep]
    if r.size == 0:
        return None, 0
    r_min = r.min()
    near = r[r <= r_min + cluster_depth]
    return float(np.median(near)), int(near.size)


def nearest_cluster_bearing(origin_xy, ray_yaw, points_xy, half_width_rad,
                            min_range=0.3, max_range=3.0, cluster_depth=0.15):
    """Bearing and range of the nearest LiDAR cluster within half_width_rad of a ray.

    Used for camera/LiDAR yaw calibration: the camera gives ray_yaw, the LiDAR
    tells where the object really is. Returns (bearing, range, n_points) with the
    bearing in the same frame as ray_yaw, or (None, None, 0).
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
    import cv2  # local import keeps the module importable without OpenCV for pure tests

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

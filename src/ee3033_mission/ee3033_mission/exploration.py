"""Frontier-based exploration helpers for the self-mapping step (no ROS imports, unit-testable).

A frontier is a free cell next to unknown space. The robot drives to a safe,
reachable cell near a frontier; the LiDAR then turns the unknown cells behind
it into free or occupied cells. Mapping is complete when no reachable frontier
is left.

Grids are OccupancyGrid data reshaped to (height, width): -1 unknown, 0..100
occupancy, row 0 at the lowest y. The map yaw is assumed to be 0 (true for
SLAM Toolbox maps).
"""

import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import grid_to_world, world_to_grid

# Frontier status values (also used for the RViz colours)
OK, BLACKLISTED, UNREACHABLE, TOO_FAR = 'ok', 'blacklisted', 'unreachable', 'too_far'

_CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
_SQUARE = np.ones((3, 3), np.uint8)


@dataclass
class Frontier:
    cells: np.ndarray            # (N, 2) grid (row, col) of the frontier cells
    size: float                  # approximate length (m)
    centre: tuple                # (x, y) of the frontier cell closest to the cluster mean
    status: str = OK
    goal: tuple = None           # (x, y) safe reachable cell to drive to
    distance: float = math.inf   # path length robot -> goal (m)
    cost: float = math.inf
    points: list = field(default_factory=list)   # (x, y) of all cells, for RViz


def classify(grid, occupied_thresh=50):
    """Boolean masks (free, occupied, unknown)."""
    grid = np.asarray(grid)
    unknown = grid < 0
    occupied = grid >= occupied_thresh
    return ~unknown & ~occupied, occupied, unknown


def frontier_mask(grid, occupied_thresh=50):
    """Free cells with an unknown 4-neighbour."""
    free, _, unknown = classify(grid, occupied_thresh)
    near_unknown = cv2.dilate(unknown.astype(np.uint8), _CROSS) > 0
    return free & near_unknown


def traversable_mask(grid, resolution, robot_radius, occupied_thresh=50):
    """Free cells at least robot_radius from any occupied cell.

    Unknown cells are not traversable but do not count as obstacles, otherwise
    the cells next to a frontier (the ones we want to drive to) would be excluded.
    """
    free, occupied, _ = classify(grid, occupied_thresh)
    clearance = cv2.distanceTransform((~occupied).astype(np.uint8), cv2.DIST_L2, 5) * resolution
    return free & (clearance >= robot_radius)


def path_distance(mask, seed_rc, max_iter=100000):
    """8-connected wavefront over mask from seed_rc.

    Returns an int32 array with the number of steps from the seed (Chebyshev
    path length in cells), -1 where unreachable.
    """
    h, w = mask.shape
    dist = np.full((h, w), -1, np.int32)
    if not (0 <= seed_rc[0] < h and 0 <= seed_rc[1] < w) or not mask[seed_rc]:
        return dist
    allowed = mask.astype(np.uint8)
    reached = np.zeros((h, w), np.uint8)
    reached[seed_rc] = 1
    dist[seed_rc] = 0
    for step in range(1, max_iter):
        grown = cv2.dilate(reached, _SQUARE) & allowed
        new = (grown > 0) & (reached == 0)
        if not new.any():
            break
        dist[new] = step
        reached = grown
    return dist


def nearest_cell(mask, rc, max_cells):
    """Cell of mask nearest to rc (within max_cells), or None."""
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return None
    d2 = (rows - rc[0]) ** 2 + (cols - rc[1]) ** 2
    i = int(np.argmin(d2))
    if d2[i] > max_cells ** 2:
        return None
    return int(rows[i]), int(cols[i])


def descend_path(dist, goal_rc):
    """Cells from goal_rc back to the seed following decreasing path distance (for the simulator)."""
    h, w = dist.shape
    path = [goal_rc]
    r, c = goal_rc
    while dist[r, c] > 0:
        best = None
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and 0 <= dist[rr, cc] < dist[r, c]:
                    if best is None or dist[rr, cc] < dist[best]:
                        best = (rr, cc)
        if best is None:
            break
        r, c = best
        path.append(best)
    return path[::-1]


def near_any(xy, points, radius):
    return any(math.hypot(xy[0] - px, xy[1] - py) <= radius for px, py in points)


def plan_frontiers(grid, resolution, origin_xy, robot_xy, *, robot_radius=0.15, min_frontier_size=0.20,
                   goal_reach=0.5, gain_weight=0.3, blacklist=(), blacklist_radius=0.3,
                   home_xy=None, max_radius=0.0, occupied_thresh=50, seed_radius=0.5):
    """Find frontiers, check them and rank them.

    Returns (frontiers, seeded). frontiers is sorted by cost, the best one with
    status OK first. Each frontier's goal is the traversable cell reachable from
    the robot that is closest to the frontier centre, at most goal_reach away.
    cost = path length to the goal - gain_weight * frontier size, so near and
    large frontiers win. seeded is False if no traversable cell was found near
    the robot (the robot pose or the map is not ready): then nothing is reachable
    and the caller must not conclude that exploration is finished.
    """
    grid = np.asarray(grid)
    fmask = frontier_mask(grid, occupied_thresh)
    trav = traversable_mask(grid, resolution, robot_radius, occupied_thresh)

    robot_rc = world_to_grid(robot_xy[0], robot_xy[1], origin_xy, resolution)
    seed = nearest_cell(trav, robot_rc, int(round(seed_radius / resolution)))
    dist = path_distance(trav, seed) if seed is not None else np.full(grid.shape, -1, np.int32)
    reachable = dist >= 0

    n, labels, stats, _ = cv2.connectedComponentsWithStats(fmask.astype(np.uint8), connectivity=8)
    reach_cells = int(math.ceil(goal_reach / resolution))
    h, w = grid.shape
    frontiers = []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] * resolution < min_frontier_size:
            continue
        rows, cols = np.nonzero(labels == k)
        cells = np.stack([rows, cols], axis=1)
        mean = cells.mean(axis=0)
        i = int(np.argmin(((cells - mean) ** 2).sum(axis=1)))
        cr, cc = int(rows[i]), int(cols[i])
        f = Frontier(cells=cells, size=float(cells.shape[0] * resolution),
                     centre=grid_to_world(cr, cc, origin_xy, resolution),
                     points=[grid_to_world(r, c, origin_xy, resolution) for r, c in cells])

        if near_any(f.centre, blacklist, blacklist_radius):
            f.status = BLACKLISTED
        elif home_xy is not None and max_radius > 0 and \
                math.hypot(f.centre[0] - home_xy[0], f.centre[1] - home_xy[1]) > max_radius:
            f.status = TOO_FAR
        else:
            r0, r1 = max(0, cr - reach_cells), min(h, cr + reach_cells + 1)
            c0, c1 = max(0, cc - reach_cells), min(w, cc + reach_cells + 1)
            win = reachable[r0:r1, c0:c1]
            rr, ccs = np.mgrid[r0:r1, c0:c1]
            d2 = (rr - cr) ** 2 + (ccs - cc) ** 2
            d2 = np.where(win & (d2 <= reach_cells ** 2), d2, np.iinfo(np.int64).max)
            j = np.unravel_index(int(np.argmin(d2)), d2.shape)
            if d2[j] == np.iinfo(np.int64).max:
                f.status = UNREACHABLE
            else:
                gr, gc = r0 + int(j[0]), c0 + int(j[1])
                f.goal = grid_to_world(gr, gc, origin_xy, resolution)
                f.distance = float(dist[gr, gc] * resolution)
                f.cost = f.distance - gain_weight * f.size
        frontiers.append(f)

    frontiers.sort(key=lambda f: (f.status != OK, f.cost))
    return frontiers, seed is not None


def save_map_files(path_base, grid, resolution, origin, occupied_thresh=0.65, free_thresh=0.25):
    """Write <path_base>.pgm and <path_base>.yaml in the map_server (trinary) format.

    origin is (x, y, yaw) of cell (0, 0). Returns the yaml path.
    """
    grid = np.asarray(grid)
    img = np.full(grid.shape, 205, np.uint8)                   # unknown: grey
    img[(grid >= 0) & (grid <= free_thresh * 100)] = 254       # free: white
    img[grid >= occupied_thresh * 100] = 0                     # occupied: black
    img = img[::-1]                                            # image rows run top-down (highest y first)

    os.makedirs(os.path.dirname(os.path.abspath(path_base)), exist_ok=True)
    pgm, yml = path_base + '.pgm', path_base + '.yaml'
    if not cv2.imwrite(pgm, img):
        raise OSError(f'cannot write {pgm}')
    with open(yml, 'w') as f:
        f.write(f'image: {os.path.basename(pgm)}\n'
                'mode: trinary\n'
                f'resolution: {round(float(resolution), 6)}\n'
                f'origin: [{origin[0]}, {origin[1]}, {origin[2]}]\n'
                'negate: 0\n'
                f'occupied_thresh: {occupied_thresh}\n'
                f'free_thresh: {free_thresh}\n')
    return yml

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


def frontier_left(frontiers, centre, radius):
    """Size (m) of the largest OK frontier whose centre is within radius of centre (0 if there is none)."""
    return max((f.size for f in frontiers if f.status == OK and near_any(f.centre, [centre], radius)), default=0.0)


def _goal_in_sight(free, cand_r, cand_c, cr, cc, near, reach_cells, samples=12, max_checks=150):
    """Index of the candidate goal (within reach_cells of the frontier `near`) closest to the frontier centre
    (cr, cc) from which at least one of `samples` cells of the frontier is in sight over known free cells; None if
    there is none. (Candidates further away were tried as well: in test mazes they often saw too little of their
    frontier, got it blacklisted, and the blacklist circle hid a neighbouring frontier.)"""
    fr, fc = np.nonzero(near)
    pick = np.linspace(0, len(fr) - 1, min(samples, len(fr))).round().astype(int)
    fr, fc = fr[pick], fc[pick]
    order = np.argsort((cand_r - cr) ** 2 + (cand_c - cc) ** 2)
    reach = np.sqrt(((cand_r[:, None] - fr[None]) ** 2 + (cand_c[:, None] - fc[None]) ** 2).min(axis=1))
    idx = order[reach[order] <= reach_cells]
    for i in idx[np.unique(np.linspace(0, len(idx) - 1, min(max_checks, len(idx))).round().astype(int))] \
            if len(idx) else []:
        n = int(max(np.abs(fr - cand_r[i]).max(), np.abs(fc - cand_c[i]).max())) * 2 + 2
        t = np.linspace(0.0, 1.0, n)[:, None]
        rows = np.rint(cand_r[i] + t * (fr - cand_r[i])[None]).astype(int)
        cols = np.rint(cand_c[i] + t * (fc - cand_c[i])[None]).astype(int)
        if free[rows, cols].all(axis=0).any():
            return int(i)
    return None


def plan_frontiers(grid, resolution, origin_xy, robot_xy, *, robot_radius=0.15, min_frontier_size=0.15,
                   goal_reach=0.5, gain_weight=0.3, blacklist=(), blacklist_radius=0.3,
                   home_xy=None, max_radius=0.0, occupied_thresh=50, seed_radius=0.5, goal_clearance=0.15,
                   unknown_clearance=0.20, pass_clearance=0.09):
    """Find frontiers, check them and rank them.

    Returns (frontiers, seeded). frontiers is sorted by cost, the best one with
    status OK first. Each frontier's goal is the traversable cell reachable from
    the robot within goal_reach of any cell of the frontier that is closest to the
    frontier centre (so a thin frontier seen through a gap, whose centre is far
    from every safe cell, is approached at its near end), at least goal_clearance
    from occupied cells, from which part of the frontier is in sight if there is such a cell (a goal behind a
    wall end was reached, saw nothing of a frontier in a pocket, and the pocket was blacklisted unmapped) (a goal next to a wall leaves the robot where Nav2 can neither
    plan nor recover) and at least unknown_clearance from unknown cells (unknown space
    may hide a wall the LiDAR has not seen: the robot drove into one at a goal 15 cm
    from the unknown edge; 0.20 explores test mazes as completely as 0.15, 0.25 and 0.30
    do not). A frontier without such a goal is UNREACHABLE.
    Only frontier cells connected to the robot through free cells at least pass_clearance from walls
    (a way the robot could pass) count, otherwise the frontier is UNREACHABLE: free cells seen beyond a wall
    (LiDAR beams through a crack or over the wall) gave frontiers the robot drove to the wall for, could never
    resolve, and had to turn back from.
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
    free, occupied, unknown = classify(grid, occupied_thresh)
    to_wall = cv2.distanceTransform((~occupied).astype(np.uint8), cv2.DIST_L2, 5) * resolution
    side = np.zeros(grid.shape, bool)       # cells on the robot's side of the walls (or next to it)
    if seed is not None:
        _, comp = cv2.connectedComponents((free & (to_wall >= pass_clearance)).astype(np.uint8), connectivity=8)
        side = cv2.dilate((comp == comp[seed]).astype(np.uint8), _SQUARE) > 0
    to_unknown = cv2.distanceTransform((~unknown).astype(np.uint8), cv2.DIST_L2, 5) * resolution
    reachable = (dist >= 0) & (to_wall >= goal_clearance) & (to_unknown >= unknown_clearance)

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
        elif not side[rows, cols].any():
            f.status = UNREACHABLE
        else:
            near = (labels == k) & side
            to_frontier = cv2.distanceTransform((~near).astype(np.uint8), cv2.DIST_L2, 5)   # cells
            cand_r, cand_c = np.nonzero(reachable & (to_frontier <= reach_cells))
            if cand_r.size == 0:
                f.status = UNREACHABLE
            else:
                i = _goal_in_sight(free, cand_r, cand_c, cr, cc, near, reach_cells)
                if i is None:            # in sight from nowhere: as before, the safe cell closest to the centre
                    i = int(np.argmin((cand_r - cr) ** 2 + (cand_c - cc) ** 2))
                gr, gc = int(cand_r[i]), int(cand_c[i])
                f.goal = grid_to_world(gr, gc, origin_xy, resolution)
                f.distance = float(dist[gr, gc] * resolution)
                f.cost = f.distance - gain_weight * f.size
        frontiers.append(f)

    frontiers.sort(key=lambda f: (f.status != OK, f.cost))
    return frontiers, seed is not None


def goal_reachable(grid, resolution, origin_xy, robot_xy, goal_xy, *, robot_radius=0.15, goal_clearance=0.15,
                   unknown_clearance=0.20, occupied_thresh=50, seed_radius=0.5):
    """Is goal_xy still a goal that plan_frontiers could choose on this map: a traversable cell connected to the
    robot, at least goal_clearance from walls and unknown_clearance from unknown space."""
    grid = np.asarray(grid)
    trav = traversable_mask(grid, resolution, robot_radius, occupied_thresh)
    seed = nearest_cell(trav, world_to_grid(robot_xy[0], robot_xy[1], origin_xy, resolution),
                        int(round(seed_radius / resolution)))
    g = world_to_grid(goal_xy[0], goal_xy[1], origin_xy, resolution)
    if seed is None or not (0 <= g[0] < grid.shape[0] and 0 <= g[1] < grid.shape[1]) or not trav[g]:
        return False
    _, occupied, unknown = classify(grid, occupied_thresh)
    to_wall = cv2.distanceTransform((~occupied).astype(np.uint8), cv2.DIST_L2, 5)[g] * resolution
    to_unknown = cv2.distanceTransform((~unknown).astype(np.uint8), cv2.DIST_L2, 5)[g] * resolution
    if to_wall < goal_clearance or to_unknown < unknown_clearance:
        return False
    _, comp = cv2.connectedComponents(trav.astype(np.uint8), connectivity=8)
    return bool(comp[g] == comp[seed])


def plan_route(grid, resolution, origin_xy, start_xy, goal_xy, *, clearance=0.15, max_leg=0.8, leg_clearance=0.22,
               min_leg=0.35, centre_weight=3.0, centre_clearance=0.35, occupied_thresh=50, seed_radius=0.5):
    """Short straight legs from start to goal through known free space, kept to the middle of corridors.

    Usable cells are known free cells at least `clearance` from occupied or unknown cells. Dijkstra over them
    with a step cost that grows as the clearance drops below centre_clearance (so the route follows corridor
    centres), then string pulling: each leg runs from a waypoint to the farthest route cell that is at most
    max_leg away and in line of sight over cells at least leg_clearance from walls (so that legs do not cut
    a corner next to a wall end: the robot brushed walls there), or over usable cells where that gives
    no leg of at least min_leg (narrow corridors: fewer stops instead of many short legs). Waypoints fall
    in the middle of the bends. Returns the waypoints [(x, y), ...] ending
    exactly at goal_xy, or [] if there is no route.
    """
    import heapq

    grid = np.asarray(grid)
    free, _, _ = classify(grid, occupied_thresh)
    clear = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5) * resolution
    usable = free & (clear >= clearance)
    seed_cells = int(round(seed_radius / resolution))
    start = nearest_cell(usable, world_to_grid(start_xy[0], start_xy[1], origin_xy, resolution), seed_cells)
    goal = nearest_cell(usable, world_to_grid(goal_xy[0], goal_xy[1], origin_xy, resolution), seed_cells)
    if start is None or goal is None:
        return []
    h, w = grid.shape
    penalty = 1.0 + centre_weight * np.clip((centre_clearance - clear) / centre_clearance, 0.0, 1.0)
    cost = np.full((h, w), np.inf)
    cost[start] = 0.0
    prev, queue = {}, [(0.0, start)]
    while queue:
        c, (r, q) = heapq.heappop(queue)
        if (r, q) == goal:
            break
        if c > cost[r, q]:
            continue
        for dr in (-1, 0, 1):
            for dq in (-1, 0, 1):
                a, b = r + dr, q + dq
                if (dr or dq) and 0 <= a < h and 0 <= b < w and usable[a, b]:
                    nc = c + (1.4142 if dr and dq else 1.0) * penalty[a, b]
                    if nc < cost[a, b]:
                        cost[a, b], prev[(a, b)] = nc, (r, q)
                        heapq.heappush(queue, (nc, (a, b)))
    if not np.isfinite(cost[goal]):
        return []
    cells = [goal]
    while cells[-1] != start:
        cells.append(prev[cells[-1]])
    cells.reverse()

    wide = usable & (clear >= leg_clearance)

    def visible(a, b, mask):
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2) + 2
        rows = np.rint(np.linspace(a[0], b[0], n)).astype(int)
        cols = np.rint(np.linspace(a[1], b[1], n)).astype(int)
        return bool(mask[rows, cols].all())

    def farthest(i, mask):
        j = i + 1
        while j + 1 < len(cells) and math.hypot(cells[j + 1][0] - cells[i][0], cells[j + 1][1] - cells[i][1]) <= \
                leg_cells and visible(cells[i], cells[j + 1], mask):
            j += 1
        return j

    leg_cells = max_leg / resolution
    waypoints, i = [], 0
    while i < len(cells) - 1:
        j = farthest(i, wide)
        if math.hypot(cells[j][0] - cells[i][0], cells[j][1] - cells[i][1]) * resolution < min_leg:
            j = max(j, farthest(i, usable))      # too narrow for leg_clearance here: usable cells will do
        waypoints.append(grid_to_world(cells[j][0], cells[j][1], origin_xy, resolution))
        i = j
    if not waypoints:
        waypoints = [tuple(goal_xy)]
    waypoints[-1] = (float(goal_xy[0]), float(goal_xy[1]))
    return waypoints


def save_map_files(path_base, grid, resolution, origin, occupied_thresh=0.65, free_thresh=0.196):
    """Write <path_base>.pgm and <path_base>.yaml in the map_server (trinary) format.

    origin is (x, y, yaw) of cell (0, 0). Returns the yaml path. free_thresh must stay below
    (255 - 205) / 255 = 0.196078: map_server reads grey 205 as free otherwise (with the Nav2 default 0.25
    every unknown cell of the site map became free, and the mission planned for 45 m2 instead of 3.8 m2).
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

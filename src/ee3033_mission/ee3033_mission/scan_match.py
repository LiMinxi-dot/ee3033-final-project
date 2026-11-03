"""Does the LiDAR scan fit the saved map at the pose the robot thinks it has? (no ROS imports, unit-testable)

AMCL only corrects its pose after the odometry reports enough motion, so a robot
that turns without the odometry noticing (balance drift, wheel slip, moved by
hand) stays mislocalized. The check transforms the scan into the map with the
believed pose and counts the points that land on a wall. If too few do, a grid
search finds the pose correction that fits best, to tell the user what is wrong.
"""

import math

import cv2
import numpy as np


def wall_distance(grid, resolution, occupied_thresh=50):
    """Distance (m) from every cell of an occupancy grid (row = y) to the nearest occupied cell."""
    not_wall = (np.asarray(grid) < occupied_thresh).astype(np.uint8)   # free and unknown (-1)
    return cv2.distanceTransform(not_wall, cv2.DIST_L2, 5) * resolution


def transform(points, x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.column_stack((x + c * points[:, 0] - s * points[:, 1], y + s * points[:, 0] + c * points[:, 1]))


def distances(dist, resolution, origin_xy, points):
    """Distance to the nearest wall at each point (any leading shape, last axis = x, y); outside the map: inf."""
    h, w = dist.shape
    c = np.floor((points[..., 0] - origin_xy[0]) / resolution).astype(int)
    r = np.floor((points[..., 1] - origin_xy[1]) / resolution).astype(int)
    inside = (c >= 0) & (c < w) & (r >= 0) & (r < h)
    d = np.full(points.shape[:-1], np.inf)
    d[inside] = dist[r[inside], c[inside]]
    return d


def match_fraction(dist, resolution, origin_xy, points_local, pose, tol=0.07):
    """Fraction of the scan points (sensor frame) within tol of a wall when the sensor is at pose (x, y, yaw)."""
    if len(points_local) == 0:
        return 0.0
    return float(np.mean(distances(dist, resolution, origin_xy, transform(points_local, *pose)) <= tol))


def best_alignment(dist, resolution, origin_xy, points_local, pose, max_yaw=math.radians(90), max_xy=0.4):
    """Correction (dx, dy, dyaw) of pose that fits the scan to the walls best (coarse-to-fine grid search).

    dyaw > 0: the sensor is turned further counter-clockwise (left) than pose says.
    """
    def search(yaws, steps, centre):
        offsets = np.stack(np.meshgrid(steps, steps), -1).reshape(-1, 1, 2)
        best = (np.inf, 0.0, 0.0, 0.0)
        for dyaw in yaws:
            p = transform(points_local, centre[0], centre[1], pose[2] + dyaw)
            cost = np.minimum(distances(dist, resolution, origin_xy, p[None] + offsets), 0.15).mean(axis=1)
            i = int(np.argmin(cost))
            if cost[i] < best[0]:
                best = (cost[i], centre[0] + offsets[i, 0, 0] - pose[0], centre[1] + offsets[i, 0, 1] - pose[1], dyaw)
        return best[1:]

    dx, dy, dyaw = search(np.arange(-max_yaw, max_yaw + 1e-9, math.radians(3)),
                          np.arange(-max_xy, max_xy + 1e-9, 0.04), pose)
    return search(dyaw + np.radians(np.arange(-3.0, 3.01, 0.5)), np.arange(-0.04, 0.041, 0.01),
                  (pose[0] + dx, pose[1] + dy))


def describe_correction(pose, dx, dy, dyaw):
    """The pose correction in plain words, in the robot frame the program believes (pose)."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    fwd, left = c * dx + s * dy, -s * dx + c * dy
    deg = math.degrees(dyaw)
    turn = f'turned {abs(deg):.0f} deg further {"LEFT" if deg > 0 else "RIGHT"}' if abs(deg) >= 3 else 'heading OK'
    shift = (f'{math.hypot(dx, dy) * 100:.0f} cm off ({abs(fwd) * 100:.0f} cm {"forward" if fwd >= 0 else "back"}, '
             f'{abs(left) * 100:.0f} cm {"left" if left >= 0 else "right"})' if math.hypot(dx, dy) >= 0.03
             else 'position OK')
    return f'{turn}, {shift}'


def localization_verdict(walls, resolution, origin_xy, points_local, pose, need=0.75, margin=0.10):
    """(ok, fraction, lines): does the scan fit the walls at pose, and if not, how is the sensor really placed.

    OK needs at least `need` of the points on the walls and no nearby pose that fits clearly better (by
    `margin`): in a small arena many points land near some wall even 20-40 deg off, so the absolute
    fraction alone is not enough. lines say it in plain words for the operator (one line if ok).
    """
    frac = match_fraction(walls, resolution, origin_xy, points_local, pose)
    dx, dy, dyaw = best_alignment(walls, resolution, origin_xy, points_local, pose)
    fit = match_fraction(walls, resolution, origin_xy, points_local, (pose[0] + dx, pose[1] + dy, pose[2] + dyaw))
    correction = describe_correction(pose, dx, dy, dyaw)
    if frac >= need and fit < frac + margin:
        return True, frac, [f'best fit {fit:.0%}: {correction}']
    lines = [f'only {frac:.0%} of the LiDAR points lie on the map walls'
             + (f' (need {need:.0%}).' if frac < need else f', {fit:.0%} would at a slightly different pose.'),
             'The robot is NOT where the program thinks it is.']
    if fit >= need:
        lines.append(f'The scan fits the map ({fit:.0%}) if the robot really is: {correction}, '
                     'compared with where the program thinks it is.')
    else:
        lines.append(f'The scan does not fit the map anywhere nearby (best {fit:.0%}): is the robot inside the '
                     'mapped area, and is the arena the same as when it was mapped?')
    return False, frac, lines

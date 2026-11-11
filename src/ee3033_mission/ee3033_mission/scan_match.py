"""Does the LiDAR scan fit the saved map at the pose the robot thinks it has? (no ROS imports, unit-testable)

AMCL only corrects its pose after the odometry reports enough motion, so a robot
that turns without the odometry noticing (balance drift, wheel slip, moved by
hand) stays mislocalized. The check transforms the scan into the map with the
believed pose and counts the points that land on a wall. If too few do, a grid
search finds the pose correction that fits best, to tell the user what is wrong,
and calibration() says whether that correction is clear enough to apply by itself.
"""

import math
from collections import namedtuple

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


def _grid_costs(dist, resolution, origin_xy, points_local, centre, yaw, dyaws, steps):
    """Rows (dyaw, x, y, cost) for the sensor at every grid pose around (centre, yaw): cost = mean distance of the
    scan points to the nearest wall, capped at 0.15 m."""
    offsets = np.stack(np.meshgrid(steps, steps), -1).reshape(-1, 1, 2)
    rows = []
    for dyaw in dyaws:
        p = transform(points_local, centre[0], centre[1], yaw + dyaw)
        cost = np.minimum(distances(dist, resolution, origin_xy, p[None] + offsets), 0.15).mean(axis=1)
        rows.append(np.column_stack((np.full(len(cost), dyaw), centre[0] + offsets[:, 0, 0],
                                     centre[1] + offsets[:, 0, 1], cost)))
    return np.concatenate(rows)


def alignments(dist, resolution, origin_xy, points_local, pose, max_yaw=math.radians(90), max_xy=0.4,
               apart_yaw=math.radians(20), dip=0.05):
    """Best correction (dx, dy, dyaw) of pose (coarse-to-fine grid search), and the fit of the best other heading.

    The other heading is the highest separate peak of the fit over the heading (each heading at its best
    position), at least apart_yaw from the best one, with the fit dropping by at least `dip` on the way to it:
    the shoulders of the best peak do not count (site map with thick, blurred walls: turned 10-25 deg, 90 % of
    the points still lie near a wall). If that peak fits about as well, the arena looks alike from both
    headings (e.g. a square room). Positions are not compared: along a corridor a shifted pose fits nearly as
    well, and AMCL corrects the position itself once the heading is right.
    """
    yaws, steps = np.arange(-max_yaw, max_yaw + 1e-9, math.radians(3)), np.arange(-max_xy, max_xy + 1e-9, 0.04)
    coarse = _grid_costs(dist, resolution, origin_xy, points_local, pose[:2], pose[2], yaws, steps)
    per_yaw = coarse.reshape(len(yaws), -1, 4)
    best_per_yaw = per_yaw[np.arange(len(yaws)), np.argmin(per_yaw[:, :, 3], axis=1)]
    fits = np.array([match_fraction(dist, resolution, origin_xy, points_local, (x, y, pose[2] + dyaw))
                     for dyaw, x, y, _ in best_per_yaw])
    k0 = int(np.argmin(best_per_yaw[:, 3]))
    other = 0.0
    for k in range(len(yaws)):
        peak = (k == 0 or fits[k - 1] <= fits[k]) and (k == len(yaws) - 1 or fits[k + 1] <= fits[k])
        lo, hi = min(k, k0), max(k, k0)
        if peak and abs(yaws[k] - yaws[k0]) >= apart_yaw and fits[lo:hi + 1].min() <= fits[k] - dip:
            other = max(other, float(fits[k]))
    b = best_per_yaw[k0]
    fine = _grid_costs(dist, resolution, origin_xy, points_local, b[1:3], pose[2],
                       b[0] + np.radians(np.arange(-3.0, 3.01, 0.5)), np.arange(-0.04, 0.041, 0.01))
    f = fine[np.argmin(fine[:, 3])]
    return (f[1] - pose[0], f[2] - pose[1], f[0]), other


def best_alignment(dist, resolution, origin_xy, points_local, pose, max_yaw=math.radians(90), max_xy=0.4):
    """Correction (dx, dy, dyaw) of pose that fits the scan to the walls best (coarse-to-fine grid search).

    dyaw > 0: the sensor is turned further counter-clockwise (left) than pose says.
    """
    return alignments(dist, resolution, origin_xy, points_local, pose, max_yaw, max_xy)[0]


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


Verdict = namedtuple('Verdict', 'ok frac lines correction fit other_fit')


def scan_verdict(walls, resolution, origin_xy, points_local, pose, need=0.75, margin=0.10):
    """Verdict(ok, frac, lines, correction, fit, other_fit): does the scan fit the walls at pose, and if not, how
    is the sensor really placed.

    OK needs at least `need` of the points on the walls and no nearby pose that fits clearly better (by
    `margin`): in a small arena many points land near some wall even 20-40 deg off, so the absolute
    fraction alone is not enough. lines say it in plain words for the operator (one line if ok).
    correction (dx, dy, dyaw) gives the best fit `fit`; other_fit is the fit of the best separate peak at a
    heading at least 20 deg away (alignments).
    """
    frac = match_fraction(walls, resolution, origin_xy, points_local, pose)
    (dx, dy, dyaw), other = alignments(walls, resolution, origin_xy, points_local, pose)
    fit = match_fraction(walls, resolution, origin_xy, points_local, (pose[0] + dx, pose[1] + dy, pose[2] + dyaw))
    correction = describe_correction(pose, dx, dy, dyaw)
    if frac >= need and fit < frac + margin:
        return Verdict(True, frac, [f'best fit {fit:.0%}: {correction}'], (dx, dy, dyaw), fit, other)
    lines = [f'only {frac:.0%} of the LiDAR points lie on the map walls'
             + (f' (need {need:.0%}).' if frac < need else f', {fit:.0%} would at a slightly different pose.'),
             'The robot is NOT where the program thinks it is.']
    if fit >= need:
        lines.append(f'The scan fits the map ({fit:.0%}) if the robot really is: {correction}, '
                     'compared with where the program thinks it is.')
    else:
        lines.append(f'The scan does not fit the map anywhere nearby (best {fit:.0%}): is the robot inside the '
                     'mapped area, and is the arena the same as when it was mapped?')
    return Verdict(False, frac, lines, (dx, dy, dyaw), fit, other)


def localization_verdict(walls, resolution, origin_xy, points_local, pose, need=0.75, margin=0.10):
    """(ok, fraction, lines) of scan_verdict()."""
    v = scan_verdict(walls, resolution, origin_xy, points_local, pose, need, margin)
    return v.ok, v.frac, v.lines


def calibration(verdict, min_fit=0.90, margin=0.10):
    """(correction, why_not): the correction (dx, dy, dyaw) of a failed check that is safe to apply without the
    operator, or None and the reason in plain words.

    The scan must fit at the corrected pose (min_fit of the points on walls) and clearly better (by margin) than
    at any separate peak 20 deg or more away (alignments): where the arena looks alike from two headings, the
    wrong one could be chosen.
    """
    if verdict.ok:
        return None, 'the check passed'
    if verdict.fit < min_fit:
        return None, f'no nearby pose fits well enough ({verdict.fit:.0%}, need {min_fit:.0%})'
    if verdict.fit - verdict.other_fit < margin:
        return None, (f'another heading 20 deg or more away fits almost as well ({verdict.other_fit:.0%}): the '
                      'arena looks alike from both')
    return verdict.correction, ''


def _nearest(ref, pts, chunk=2000):
    """Index of the nearest ref point and its distance for every point of pts (brute force, small scans)."""
    idx, dist = np.empty(len(pts), int), np.empty(len(pts))
    for i in range(0, len(pts), chunk):
        d = np.hypot(pts[i:i + chunk, None, 0] - ref[None, :, 0], pts[i:i + chunk, None, 1] - ref[None, :, 1])
        idx[i:i + chunk] = np.argmin(d, axis=1)
        dist[i:i + chunk] = d[np.arange(len(d)), idx[i:i + chunk]]
    return idx, dist


def icp_2d(ref, cur, init=(0.0, 0.0, 0.0), iters=40, max_dist=0.10):
    """Rigid transform (x, y, yaw) that maps the scan points cur onto ref (point-to-point ICP from init), the RMS
    distance of the matched points (m) and the share of cur points matched within max_dist."""
    x, y, yaw = init
    rms, share = float('inf'), 0.0
    for _ in range(iters):
        p = transform(cur, x, y, yaw)
        idx, d = _nearest(ref, p)
        ok = d < max_dist
        if ok.sum() < 10:
            break
        a, b = cur[ok], ref[idx[ok]]
        ma, mb = a.mean(axis=0), b.mean(axis=0)
        h = (a - ma).T @ (b - mb)
        new_yaw = math.atan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
        c, s = math.cos(new_yaw), math.sin(new_yaw)
        nx, ny = mb[0] - (c * ma[0] - s * ma[1]), mb[1] - (s * ma[0] + c * ma[1])
        done = abs(new_yaw - yaw) < 1e-5 and math.hypot(nx - x, ny - y) < 1e-5
        x, y, yaw = nx, ny, new_yaw
        rms, share = float(np.sqrt(np.mean(d[ok] ** 2))), float(ok.mean())
        if done:
            break
    return (x, y, yaw), rms, share


def scan_rotation(ref, cur, guess_yaw, search=math.radians(25), step=math.radians(0.5), cap=0.10):
    """Rotation (rad) of the robot from the scan ref to the scan cur (both in the sensor frame, robot turning about
    roughly the same place), searched within `search` of guess_yaw and refined by ICP; with the ICP RMS (m) and
    matched share. For a full turn the guess is 0 and the result is the over- or under-turn."""
    yaws = guess_yaw + np.arange(-search, search + 1e-9, step)
    costs = [np.minimum(_nearest(ref, transform(cur, 0.0, 0.0, a))[1], cap).mean() for a in yaws]
    (x, y, yaw), rms, share = icp_2d(ref, cur, (0.0, 0.0, float(yaws[int(np.argmin(costs))])))
    return yaw, rms, share

import math
import re

import numpy as np
import pytest

from ee3033_mission.geometry import grid_to_world
from ee3033_mission.scan_match import (best_alignment, calibration, icp_2d, localization_verdict, match_fraction,
                                       scan_rotation, scan_verdict, transform, wall_distance)

RES, ORIGIN = 0.05, (-0.5, -0.5)


def room():
    """2.0 m x 1.5 m room with a box near one corner (so it has no rotational symmetry)."""
    g = np.zeros((32, 42), np.int16)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 100
    g[20:26, 30:36] = 100
    return g


def scan_from(grid, pose):
    """Every wall cell centre, as seen from the sensor at pose (sensor frame)."""
    rows, cols = np.nonzero(grid >= 50)
    pts = np.array([grid_to_world(r, c, ORIGIN, RES) for r, c in zip(rows, cols)])
    x, y, yaw = pose
    return transform(pts - (x, y), 0.0, 0.0, -yaw)


@pytest.fixture(scope='module')
def setup():
    g = room()
    true_pose = (0.6, 0.4, 0.3)
    return wall_distance(g, RES), scan_from(g, true_pose), true_pose


def test_scan_fits_at_true_pose(setup):
    dist, pts, pose = setup
    assert match_fraction(dist, RES, ORIGIN, pts, pose) > 0.95


def test_turned_robot_is_detected_and_explained(setup):
    dist, pts, (x, y, yaw) = setup
    believed = (x, y, yaw + math.radians(60))         # the robot is really turned 60 deg to the right
    assert match_fraction(dist, RES, ORIGIN, pts, believed) < 0.5
    dx, dy, dyaw = best_alignment(dist, RES, ORIGIN, pts, believed)
    assert math.degrees(dyaw) == pytest.approx(-60, abs=1.5)
    assert math.hypot(dx, dy) < 0.03


def test_shifted_robot_is_explained(setup):
    dist, pts, (x, y, yaw) = setup
    dx, dy, dyaw = best_alignment(dist, RES, ORIGIN, pts, (x + 0.15, y - 0.10, yaw))
    assert (dx, dy) == pytest.approx((-0.15, 0.10), abs=0.02)
    assert abs(math.degrees(dyaw)) < 1.5


def test_points_outside_the_map_do_not_match(setup):
    dist, _, _ = setup
    assert match_fraction(dist, RES, ORIGIN, np.array([[5.0, 5.0]]), (0.0, 0.0, 0.0)) == 0.0


def test_verdict_in_plain_words(setup):
    dist, pts, (x, y, yaw) = setup
    ok, frac, lines = localization_verdict(dist, RES, ORIGIN, pts, (x, y, yaw))
    assert ok and frac > 0.95 and lines == ['best fit 100%: heading OK, position OK']
    ok, frac, lines = localization_verdict(dist, RES, ORIGIN, pts, (x, y, yaw - math.radians(45)))
    assert not ok
    assert re.search(r'turned 4[456] deg further LEFT, position OK', lines[-1])


def test_verdict_says_back_and_right(setup):
    dist, pts, (x, y, yaw) = setup
    # believed pose 15 cm ahead and 10 cm left of the true one (robot frame): really 15 cm back, 10 cm right
    c, s = math.cos(yaw), math.sin(yaw)
    ok, _, lines = localization_verdict(dist, RES, ORIGIN, pts, (x + 0.15 * c - 0.10 * s, y + 0.15 * s + 0.10 * c, yaw))
    assert not ok
    assert re.search(r'heading OK, 1[78] cm off \(1[456] cm back, (9|10|11) cm right\)', lines[-1]), lines[-1]


def test_small_error_fails_when_a_nearby_pose_fits_clearly_better(setup):
    dist, pts, (x, y, yaw) = setup
    ok, frac, lines = localization_verdict(dist, RES, ORIGIN, pts, (x, y, yaw + math.radians(12)), need=0.0)
    assert not ok                                   # passes any absolute threshold, but 12 deg off
    assert re.search(r'turned 1[123] deg further RIGHT', lines[-1]), lines[-1]


def test_robot_placed_askew_is_calibrated(setup):
    dist, pts, (x, y, yaw) = setup
    believed = (x + 0.08, y - 0.05, yaw - math.radians(30))     # really 30 deg further left and a bit off
    v = scan_verdict(dist, RES, ORIGIN, pts, believed)
    fix, why = calibration(v)
    assert not v.ok and fix is not None, why
    corrected = (believed[0] + fix[0], believed[1] + fix[1], believed[2] + fix[2])
    assert corrected == pytest.approx((x, y, yaw), abs=0.03)
    assert localization_verdict(dist, RES, ORIGIN, pts, corrected)[0]


def test_no_calibration_where_the_arena_looks_alike():
    g = np.zeros((40, 40), np.int16)                  # square room: turned by 90 deg it looks the same
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 100
    true_pose = (0.475, 0.475, 0.0)                   # centre of the room
    dist, pts = wall_distance(g, RES), scan_from(g, true_pose)
    v = scan_verdict(dist, RES, ORIGIN, pts, (0.475, 0.475, math.radians(30)))
    assert not v.ok and v.fit > 0.9 and v.other_fit > 0.9
    fix, why = calibration(v)
    assert fix is None and 'looks alike' in why


def test_no_calibration_when_nothing_fits_or_all_is_well(setup):
    dist, pts, pose = setup
    assert calibration(scan_verdict(dist, RES, ORIGIN, pts, pose))[0] is None              # check passes
    rng = np.random.default_rng(0)
    noise = rng.uniform(-0.3, 0.3, (200, 2))                                               # fits no wall
    fix, why = calibration(scan_verdict(dist, RES, ORIGIN, noise, pose))
    assert fix is None and 'well enough' in why


def test_corridor_shift_does_not_block_calibration():
    g = np.zeros((20, 80), np.int16)                  # 3.9 m long, 0.9 m wide corridor, closed at both ends
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 100
    true_pose = (0.5, 0.0, 0.0)                       # near the left end
    dist, pts = wall_distance(g, RES), scan_from(g, true_pose)
    v = scan_verdict(dist, RES, ORIGIN, pts, (0.5, 0.0, math.radians(-25)))
    fix, why = calibration(v)
    assert fix is not None, why
    assert math.degrees(fix[2]) == pytest.approx(25, abs=1.5)


def outline_points(n=400, seed=0):
    """Wall points of an asymmetric room (2.0 x 1.5 m, a box in one corner) around the origin, with 5 mm noise."""
    rng = np.random.default_rng(seed)
    segs = [((-1.0, -0.75), (1.0, -0.75)), ((1.0, -0.75), (1.0, 0.75)), ((1.0, 0.75), (-1.0, 0.75)),
            ((-1.0, 0.75), (-1.0, -0.75)), ((0.5, 0.3), (0.8, 0.3)), ((0.5, 0.3), (0.5, 0.75))]
    pts = []
    for (a, b) in segs:
        t = rng.random(n // len(segs))[:, None]
        pts.append(np.array(a) + t * (np.array(b) - np.array(a)))
    return np.concatenate(pts) + rng.normal(0, 0.005, (n // len(segs) * len(segs), 2))


def in_frame(world, x, y, yaw):
    """World points in the frame of a sensor at (x, y, yaw)."""
    c, s = math.cos(yaw), math.sin(yaw)
    d = world - np.array([x, y])
    return np.column_stack((c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]))


def test_icp_2d_recovers_transform():
    w = outline_points()
    ref = in_frame(w, 0.0, 0.0, 0.0)
    cur = in_frame(w, 0.03, -0.02, math.radians(7))
    (x, y, yaw), rms, share = icp_2d(ref, cur, (0.0, 0.0, math.radians(5)))
    assert math.degrees(yaw) == pytest.approx(7.0, abs=0.1)
    assert (x, y) == pytest.approx((0.03, -0.02), abs=0.005)
    assert share > 0.95


def test_scan_rotation_full_turn_short():
    # the odometry says 360 deg, the robot really turned 345.6 deg (and its LiDAR moved 1.5 cm): -14.4 deg
    w = outline_points(seed=1)
    ref = in_frame(w, 0.015, 0.0, 0.0)
    true = math.radians(345.6)
    cur = in_frame(w, 0.015 * math.cos(true), 0.015 * math.sin(true), true)
    yaw, rms, share = scan_rotation(ref, cur, 0.0)
    assert math.degrees(yaw) == pytest.approx(-14.4, abs=0.1)
    assert rms < 0.01

import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from ee3033_mission.geometry import (apply_body_offset, escape_move, focal_length_px, free_ahead,
                                     fuse_observations, generate_viewpoints, intrinsics, line_min_sdf,
                                     nearest_cluster_bearing,
                                     order_nearest_neighbour, pixel_to_bearing, plan_coverage,
                                     range_from_object_height, ray_range_from_scan, rotation_clear, scale_odom_step,
                                     scan_to_points, transform_to_xyyaw, wall_sdf, wrap_angle,
                                     yaw_from_quaternion, yaw_to_quaternion)


def test_wrap_angle():
    assert wrap_angle(3 * math.pi) == pytest.approx(-math.pi)
    assert wrap_angle(-0.5) == pytest.approx(-0.5)


def test_yaw_from_quaternion():
    yaw = 0.7
    assert yaw_from_quaternion(0, 0, math.sin(yaw / 2), math.cos(yaw / 2)) == pytest.approx(yaw)


def test_yaw_to_quaternion_round_trip():
    for yaw in (-3.0, -0.4, 0.0, 1.2, 3.1):
        assert yaw_from_quaternion(*yaw_to_quaternion(yaw)) == pytest.approx(yaw)


def test_transform_to_xyyaw():
    _, _, z, w = yaw_to_quaternion(0.5)
    tf = NS(transform=NS(translation=NS(x=1.0, y=-2.0, z=0.3), rotation=NS(x=0.0, y=0.0, z=z, w=w)))
    assert transform_to_xyyaw(tf) == pytest.approx((1.0, -2.0, 0.5))


def test_scan_to_points_drops_invalid_and_applies_pose():
    scan = NS(ranges=[1.0, float('inf'), 0.01, 2.0], angle_min=0.0, angle_increment=math.pi / 2,
              range_min=0.05, range_max=10.0)
    assert scan_to_points(scan) == pytest.approx(np.array([[1.0, 0.0], [0.0, -2.0]]), abs=1e-9)  # beams 0 and 270 deg
    # scan frame at (1, 1) facing +y: the first beam (straight ahead) ends at (1, 2)
    assert scan_to_points(scan, 1.0, 1.0, math.pi / 2)[0] == pytest.approx((1.0, 2.0))


def test_intrinsics_prefers_calibration():
    K = np.array([[500.0, 0, 330.0], [0, 500.0, 240.0], [0, 0, 1]])
    assert intrinsics(K, 640, math.radians(60)) == (500.0, 330.0)
    assert intrinsics(None, 640, math.radians(60)) == pytest.approx((focal_length_px(640, math.radians(60)), 320.0))


def test_apply_body_offset():
    # robot facing +y: an offset 1.5 cm to its left is towards -x in the map
    x, y, yaw = apply_body_offset(1.0, 2.0, math.pi / 2, 0.0, 0.015, math.radians(2))
    assert (x, y) == pytest.approx((0.985, 2.0))
    assert yaw == pytest.approx(math.pi / 2 + math.radians(2))


def test_bearing_sign_and_fov_edge():
    fx = focal_length_px(640, math.radians(60))
    assert pixel_to_bearing(320, 320, fx) == pytest.approx(0.0)
    assert pixel_to_bearing(0, 320, fx) == pytest.approx(math.radians(30))     # left edge -> +30 deg
    assert pixel_to_bearing(640, 320, fx) == pytest.approx(math.radians(-30))  # right edge -> -30 deg


def test_ray_range_picks_nearest_object_on_ray():
    # wall 3 m ahead, object 1.5 m ahead on the ray, clutter off to the side
    wall = [(3.0, y) for y in np.linspace(-1, 1, 41)]
    obj = [(1.5, -0.02), (1.5, 0.0), (1.52, 0.02)]
    side = [(1.0, 1.0)]
    r, n = ray_range_from_scan((0, 0), 0.0, wall + obj + side, math.radians(3))
    assert r == pytest.approx(1.5, abs=0.03)
    assert n == 3


def test_ray_range_none_when_nothing_on_ray():
    r, n = ray_range_from_scan((0, 0), math.pi / 2, [(2.0, 0.0)], math.radians(3))
    assert r is None and n == 0


def test_nearest_cluster_bearing_finds_object_off_the_ray():
    # object at +8 deg, 1.2 m; the camera ray says 0 deg; wall at 3 m
    obj_ang = math.radians(8)
    obj = [(1.2 * math.cos(obj_ang + e), 1.2 * math.sin(obj_ang + e)) for e in (-0.02, 0.0, 0.02)]
    wall = [(3.0, y) for y in np.linspace(-1, 1, 41)]
    b, r, n = nearest_cluster_bearing((0, 0), 0.0, obj + wall, math.radians(15))
    assert math.degrees(b) == pytest.approx(8.0, abs=0.5)
    assert r == pytest.approx(1.2, abs=0.02)
    assert n == 3


def test_range_from_object_height():
    fx = 500.0
    assert range_from_object_height(100, 0.2, fx) == pytest.approx(1.0)
    assert range_from_object_height(0, 0.2, fx) is None


def test_fuse_rejects_outliers():
    rng = np.random.default_rng(0)
    good = rng.normal([2.0, 1.0], 0.03, size=(8, 2))
    bad = np.array([[0.5, -1.0], [3.5, 2.5]])
    est, n, spread = fuse_observations(np.vstack([good, bad]), cluster_radius=0.3)
    assert est == pytest.approx((2.0, 1.0), abs=0.05)
    assert n == 8
    assert spread < 0.1


def test_viewpoints_cover_room_and_are_reachable():
    res = 0.05
    grid = np.full((80, 120), 100, dtype=np.int16)   # 4 m x 6 m, walls everywhere
    grid[2:-2, 2:-2] = 0                              # free room inside the walls
    grid[:, 60] = 100                                 # wall splitting the room ...
    grid[30:50, 60] = 0                               # ... with a 1 m doorway
    grid[5:15, 100:110] = -1                          # unknown patch
    island = (slice(70, 78), slice(5, 15))
    grid[island] = 100
    vps = generate_viewpoints(grid, res, (0.0, 0.0), start_xy=(1.0, 2.0), spacing=1.0, min_clearance=0.25)
    assert len(vps) >= 10
    xs = [x for x, _ in vps]
    assert min(xs) < 1.5 and max(xs) > 4.5            # both halves of the room are covered
    for x, y in vps:
        r, c = int(y / res), int(x / res)
        assert grid[r, c] == 0                        # every viewpoint is in free space


def test_viewpoints_skip_radius():
    grid = np.zeros((60, 60), dtype=np.int16)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 100
    vps = generate_viewpoints(grid, 0.05, (0.0, 0.0), (1.5, 1.5), 0.5, 0.2, skip_radius=0.6)
    assert all(math.hypot(x - 1.5, y - 1.5) >= 0.6 for x, y in vps)


def test_viewpoints_are_not_duplicated():
    grid = np.zeros((100, 100), dtype=np.int16)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 100
    vps = generate_viewpoints(grid, 0.05, (0.0, 0.0), (0.5, 0.5), 1.0, 0.2)
    for i, (x1, y1) in enumerate(vps):
        for x2, y2 in vps[i + 1:]:
            assert math.hypot(x1 - x2, y1 - y2) >= 0.5


def test_order_nearest_neighbour():
    tour = order_nearest_neighbour((0, 0), [(5, 0), (1, 0), (3, 0)])
    assert tour == [(1, 0), (3, 0), (5, 0)]


def room(w_m, h_m, res=0.05):
    g = np.zeros((int(h_m / res), int(w_m / res)), np.int16)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 100
    return g


def test_coverage_open_room_needs_no_extra_viewpoint():
    g = room(1.5, 1.5)
    vps, seen, _ = plan_coverage(g, 0.05, (0, 0), (0.75, 0.75), view_range=1.5, min_clearance=0.2)
    assert vps == [] and seen > 0.99            # the spin at the start sees everything


def test_coverage_adds_a_viewpoint_behind_a_wall_and_near_a_tight_start():
    g = room(3.0, 2.5)
    g[0:35, 30] = 100                            # partition from the bottom wall, 1.75 m long, at x = 1.5
    vps, seen, seeable = plan_coverage(g, 0.05, (0, 0), (0.5, 0.5), view_range=1.5, min_clearance=0.2)
    assert seen > 0.95 and seeable > 0.95
    assert any(x > 1.55 for x, _ in vps)         # one on the other side of the partition
    vps2, _, _ = plan_coverage(g, 0.05, (0, 0), (0.08, 0.5), view_range=1.5, min_clearance=0.2)
    assert len(vps2) >= len(vps)                 # start against the wall: its spin does not count


def wall_x(width_cells=3):
    g = np.zeros((40, 40), np.int16)
    g[:, 20:20 + width_cells] = 100              # wall from x = 1.00 m, 15 cm thick
    return g, wall_sdf(g, 0.05)


def test_wall_sdf_is_negative_inside_walls():
    g, sdf = wall_x()
    assert sdf[10, 19] == pytest.approx(0.05) and sdf[10, 21] == pytest.approx(-0.10)
    assert sdf[10, 20] == pytest.approx(-0.05) and sdf[10, 23] == pytest.approx(0.05)


def test_line_min_sdf_finds_the_closest_approach_to_a_wall():
    g, sdf = wall_x()                            # wall cells from x = 1.00 m (sdf between cell centres)
    assert line_min_sdf(sdf, 0.05, (0, 0), (0.52, 0.5), (0.82, 0.5)) == pytest.approx(0.20)   # towards it
    assert line_min_sdf(sdf, 0.05, (0, 0), (0.87, 0.2), (0.87, 1.8)) == pytest.approx(0.15)   # along it
    assert line_min_sdf(sdf, 0.05, (0, 0), (0.52, 0.5), (1.42, 0.5)) < 0                      # through it
    assert line_min_sdf(sdf, 0.05, (0, 0), (0.52, 0.5), (2.52, 0.5)) == float('-inf')         # off the map


def test_escape_moves_straight_away_from_the_wall():
    g, sdf = wall_x()
    pose = (0.94, 0.5)                           # 0.10 m from the wall (cell centres), Nav2's inscribed zone
    d, clear = escape_move(g, sdf, 0.05, (0, 0), (*pose, math.pi), 0.14, 0.25, 0.15)     # facing away
    assert d == pytest.approx(0.05) and clear >= 0.14
    d, clear = escape_move(g, sdf, 0.05, (0, 0), (*pose, 0.0), 0.14, 0.25, 0.15)         # facing the wall
    assert d == pytest.approx(-0.05) and clear >= 0.14
    assert escape_move(g, sdf, 0.05, (0, 0), (*pose, math.pi / 2), 0.14, 0.25, 0.15) is None   # along it


def test_escape_never_crosses_a_wall():
    g, sdf = wall_x()
    d, _ = escape_move(g, sdf, 0.05, (0, 0), (1.02, 0.5, 0.0), 0.14, 0.25, 0.15)  # centre in the wall, facing across
    assert d < 0                                 # back out the way it came, not through to x > 1.15
    # LiDAR sees an obstacle ahead (no forward move allowed) and behind is the wall: nothing to do
    assert escape_move(g, sdf, 0.05, (0, 0), (0.94, 0.5, math.pi), 0.14, 0.0, 0.15) is None


def test_escape_ends_on_known_free_cells_only():
    g, sdf = wall_x()
    g[:, :19] = -1                               # unknown space right behind the robot
    assert escape_move(g, sdf, 0.05, (0, 0), (0.94, 0.5, math.pi), 0.14, 0.25, 0.15) is None


def side_wall(gap, side=1):
    """LiDAR hits of a wall along the robot's left (side=1) or right side, `gap` m from its 0.10 m half width."""
    return np.column_stack((np.linspace(-0.5, 0.5, 201), np.full(201, side * (0.10 + gap))))


def test_free_ahead_counts_only_what_the_robot_width_sweeps():
    pts = np.array([[0.30, 0.05], [0.20, 0.15], [-0.20, 0.0]])        # ahead, beside the path, behind
    assert free_ahead(pts, 0.07, 0.11) == pytest.approx(0.23)
    assert free_ahead(pts, 0.07, 0.11, turn=math.radians(37)) == pytest.approx(0.25 - 0.07, abs=0.01)
    assert free_ahead(pts[2:], 0.07, 0.11) == float('inf')


def test_rotation_clear_depends_on_the_gap_to_the_wall():
    for side in (1, -1):
        assert rotation_clear(side_wall(0.04, side), 0.07, 0.11, math.radians(90))     # corners sweep 0.13 m
        assert rotation_clear(side_wall(0.04, side), 0.07, 0.11, math.radians(-90))
        assert not rotation_clear(side_wall(0.02, side), 0.07, 0.11, math.radians(30))  # a corner hits it
        assert not rotation_clear(side_wall(0.02, side), 0.07, 0.11, math.radians(-30))


def _integrate(raw_poses, yaw_scale, lin_scale=1.0, yaw_scale_cw=None):
    c = raw_poses[0]
    for a, b in zip(raw_poses, raw_poses[1:]):
        c = scale_odom_step(a, b, c, yaw_scale, lin_scale, yaw_scale_cw)
    return c


def test_scale_odom_step_identity():
    raw = [(0.1 * i, 0.02 * i * i, 0.3 * i) for i in range(10)]
    assert _integrate(raw, 1.0) == pytest.approx((raw[-1][0], raw[-1][1], float(wrap_angle(raw[-1][2]))), abs=0.01)


def test_scale_odom_step_spin_and_straight():
    # in place: 180 deg reported in 36 steps -> 171 deg with 0.95, position stays
    spin = [(0.0, 0.0, float(wrap_angle(math.radians(5 * i)))) for i in range(37)]
    x, y, yaw = _integrate(spin, 0.95)
    assert (x, y) == pytest.approx((0.0, 0.0), abs=1e-9)
    assert math.degrees(yaw) == pytest.approx(171.0, abs=1e-6)
    # straight 1 m along the reported heading 90 deg; distances x 0.9
    line = [(0.0, 0.1 * i, math.pi / 2) for i in range(11)]
    assert _integrate(line, 0.95, 0.9) == pytest.approx((0.0, 0.9, math.pi / 2), abs=1e-9)


def test_scale_odom_step_turn_then_drive():
    # reported: turn 90 deg left, then 1 m forward (to (0, 1)); with turns x 0.9 the corrected heading is 81 deg
    raw = [(0.0, 0.0, math.radians(10 * i)) for i in range(10)] + [(0.0, 0.1 * i, math.pi / 2) for i in range(1, 11)]
    x, y, yaw = _integrate(raw, 0.9)
    assert math.degrees(yaw) == pytest.approx(81.0, abs=1e-6)
    assert (x, y) == pytest.approx((math.cos(math.radians(81)), math.sin(math.radians(81))), abs=1e-6)


def test_scale_odom_step_left_right():
    left = [(0.0, 0.0, math.radians(5 * i)) for i in range(19)]       # 90 deg left
    right = [(0.0, 0.0, math.radians(-5 * i)) for i in range(19)]     # 90 deg right
    assert math.degrees(_integrate(left, 0.98, 1.0, 0.95)[2]) == pytest.approx(88.2)
    assert math.degrees(_integrate(right, 0.98, 1.0, 0.95)[2]) == pytest.approx(-85.5)
    assert math.degrees(_integrate(right, 0.98)[2]) == pytest.approx(-88.2)          # cw default: the same scale

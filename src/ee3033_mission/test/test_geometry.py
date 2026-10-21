import math

import numpy as np
import pytest

from ee3033_mission.geometry import (apply_body_offset, focal_length_px, fuse_observations, generate_viewpoints,
                                     order_nearest_neighbour, pixel_to_bearing, range_from_object_height,
                                     ray_range_from_scan, wrap_angle, yaw_from_quaternion)


def test_wrap_angle():
    assert wrap_angle(3 * math.pi) == pytest.approx(-math.pi)
    assert wrap_angle(-0.5) == pytest.approx(-0.5)


def test_yaw_from_quaternion():
    yaw = 0.7
    assert yaw_from_quaternion(0, 0, math.sin(yaw / 2), math.cos(yaw / 2)) == pytest.approx(yaw)


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

import math

import numpy as np
import pytest

from ee3033_mission.exploration import (BLACKLISTED, OK, TOO_FAR, UNREACHABLE, descend_path, frontier_mask,
                                        path_distance, plan_frontiers, save_map_files, traversable_mask)

RES = 0.05


def room(h=60, w=60):
    """3 m x 3 m room: free inside, walls on the border."""
    g = np.zeros((h, w), np.int16)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 100
    return g


def test_frontier_mask_is_free_next_to_unknown():
    g = np.full((5, 5), -1, np.int16)
    g[2, 1:4] = 0
    g[2, 2] = 0
    g[1, 2] = 100
    f = frontier_mask(g)
    assert f[2, 1] and f[2, 3]            # free with unknown neighbours
    assert not f[1, 2]                    # occupied is never a frontier
    assert not f[0, 0]                    # unknown is never a frontier


def test_fully_known_room_has_no_frontier():
    frontiers, seeded = plan_frontiers(room(), RES, (0, 0), (1.5, 1.5))
    assert seeded and frontiers == []


def test_traversable_keeps_robot_radius_from_walls():
    t = traversable_mask(room(), RES, robot_radius=0.15)
    assert t[30, 30]
    assert not t[30, 2]                   # 0.1 m from the wall
    assert t[30, 4]                       # 0.2 m from the wall


def test_path_distance_goes_around_a_wall():
    g = room()
    g[:45, 30] = 100                      # wall with a gap at the top
    t = traversable_mask(g, RES, 0.1)
    d = path_distance(t, (10, 10))
    assert d[10, 50] > 0
    assert d[10, 50] > 40                 # longer than the straight line (40 cells): goes through the gap
    path = descend_path(d, (10, 50))
    assert path[0] == (10, 10) and path[-1] == (10, 50)
    assert all(t[r, c] for r, c in path)


def test_open_side_gives_reachable_frontier():
    g = room()
    g[:, 40:] = -1                        # right part not seen yet
    frontiers, seeded = plan_frontiers(g, RES, (0, 0), (0.5, 1.5))
    assert seeded and len(frontiers) == 1
    f = frontiers[0]
    assert f.status == OK
    assert f.centre[0] == pytest.approx(1.975, abs=0.05)        # column 39
    assert f.size == pytest.approx(58 * RES, abs=0.1)
    assert math.hypot(f.goal[0] - f.centre[0], f.goal[1] - f.centre[1]) <= 0.5
    assert f.distance == pytest.approx(1.45, abs=0.15)


def test_ranking_prefers_near_frontier():
    g = room(60, 120)                     # 6 m x 3 m
    g[:, 1:10] = -1                       # unknown strip on the left (near the robot)
    g[:, 110:119] = -1                    # unknown strip on the right (far)
    frontiers, _ = plan_frontiers(g, RES, (0, 0), (1.0, 1.5))
    assert len(frontiers) == 2 and all(f.status == OK for f in frontiers)
    assert frontiers[0].centre[0] < frontiers[1].centre[0]


def test_unreachable_blacklisted_and_too_far():
    g = room(60, 120)
    g[:, 60] = 100                        # closed wall: right half unreachable
    g[20:40, 100:110] = -1                # unknown patch in the unreachable half
    g[20:40, 20:30] = -1                  # unknown patch near the robot
    g[20:40, 45:52] = -1                  # unknown patch 1.6 m from home
    frontiers, _ = plan_frontiers(g, RES, (0, 0), (0.5, 1.5), blacklist=[(1.25, 1.5)])
    status = sorted((round(f.centre[0], 1), f.status) for f in frontiers)
    assert [s for _, s in status] == [BLACKLISTED, OK, UNREACHABLE]
    frontiers, _ = plan_frontiers(g, RES, (0, 0), (0.5, 1.5), blacklist=[(1.25, 1.5)], home_xy=(0.5, 1.5),
                                  max_radius=1.5)
    assert sorted(f.status for f in frontiers) == sorted([BLACKLISTED, TOO_FAR, TOO_FAR])


def test_robot_off_the_map_is_not_seeded():
    g = room()
    g[:, 40:] = -1
    frontiers, seeded = plan_frontiers(g, RES, (0, 0), (10.0, 10.0))
    assert not seeded
    assert all(f.status == UNREACHABLE for f in frontiers)


def test_save_map_files(tmp_path):
    import cv2
    g = np.full((4, 6), -1, np.int16)
    g[0, 0] = 0          # bottom-left free
    g[3, 5] = 100        # top-right occupied
    yml = save_map_files(str(tmp_path / 'm'), g, 0.05, (-1.0, -2.0, 0.0))
    img = cv2.imread(str(tmp_path / 'm.pgm'), cv2.IMREAD_GRAYSCALE)
    assert img.shape == (4, 6)
    assert img[3, 0] == 254              # image rows are flipped: bottom row last
    assert img[0, 5] == 0
    assert img[1, 1] == 205
    text = open(yml).read()
    assert 'image: m.pgm' in text and 'origin: [-1.0, -2.0, 0.0]' in text and 'resolution: 0.05' in text

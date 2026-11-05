import math

import numpy as np
import pytest

from ee3033_mission.exploration import (BLACKLISTED, OK, TOO_FAR, UNREACHABLE, descend_path, frontier_mask,
                                        goal_reachable, path_distance, plan_frontiers, save_map_files,
                                        traversable_mask)

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
    assert f.distance == pytest.approx(1.2, abs=0.15)       # goal 0.3 m back from the unknown edge


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
    free_thresh = float(text.split('free_thresh:')[1].split()[0])
    assert 0.0039 < free_thresh < (255 - 205) / 255     # map_server: 254 free, 205 unknown (not free)


def test_goal_keeps_clear_of_walls_and_unknown():
    # room with the right half unknown; the only way to the frontier is a 0.3 m wide corridor
    g = np.full((40, 60), -1, np.int16)
    g[:, :30] = 0
    g[0, :30] = g[-1, :30] = g[:, 0] = 100
    frontiers, _ = plan_frontiers(g, RES, (0, 0), (0.5, 1.0), goal_clearance=0.20)
    f = [f for f in frontiers if f.status == OK][0]
    c, r = int(f.goal[0] / RES), int(f.goal[1] / RES)
    assert c <= 30 - 4 and 4 <= r <= 40 - 1 - 4          # >= 0.20 m from unknown (x = 1.5) and from walls
    g[:, 15:30] = 100                                    # 0.75 m thick wall before the unknown, with a 0.25 m
    g[18:23, 15:30] = 0                                  # wide tunnel through it
    frontiers, _ = plan_frontiers(g, RES, (0, 0), (0.5, 1.0), goal_clearance=0.20)
    assert all(f.status != OK for f in frontiers)        # no goal >= 0.20 m from everything within 0.5 m
    frontiers, _ = plan_frontiers(g, RES, (0, 0), (0.5, 1.0), goal_clearance=0.0)
    assert any(f.status == OK for f in frontiers)        # (the old rule would drive into the tunnel)


def s_maze():
    """3 m x 3 m room with two partitions (an S-shaped corridor), 5 cm cells, origin (0, 0)."""
    g = np.zeros((60, 60), np.int16)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 100
    g[20, 0:45] = 100            # partition from the left wall at y = 1.0
    g[40, 15:60] = 100           # partition from the right wall at y = 2.0
    return g


def test_route_home_through_an_s_maze_uses_short_straight_legs():
    from ee3033_mission.exploration import plan_route
    g = s_maze()
    start, home = (0.5, 0.5), (0.5, 2.5)              # bottom corridor -> top corridor, two U-turns
    route = plan_route(g, RES, (0, 0), start, home, clearance=0.15, max_leg=0.8)
    assert route[-1] == home and len(route) >= 4
    pts = [start] + route
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        assert math.hypot(x1 - x0, y1 - y0) <= 0.8 + 0.08
        for t in np.linspace(0, 1, 30):                   # each leg stays in the corridor, off the partitions
            x, y = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
            assert min(abs(y - 1.025), abs(y - 2.025)) > 0.12 or not (
                (abs(y - 1.025) <= 0.12 and x < 2.3) or (abs(y - 2.025) <= 0.12 and x > 0.7))
    assert any(x > 2.3 for x, y in route) and any(x < 0.7 and 1.1 < y < 1.9 for x, y in route)   # both U-turns


def test_no_route_when_the_goal_is_walled_off():
    from ee3033_mission.exploration import plan_route
    g = s_maze()
    g[20, :] = 100                                       # close the first passage
    assert plan_route(g, RES, (0, 0), (0.5, 0.5), (0.5, 2.5)) == []


def test_frontier_seen_through_a_crack_in_a_wall_is_ignored():
    """Free cells beyond a wall (LiDAR beams through a crack) give no frontier; through a door they do."""
    g = room()
    g[30, 1:-1] = 100                     # wall across the room at y = 1.5 m
    g[31:, 1:-1] = -1                     # unknown beyond it ...
    g[31:50, 28:30] = 0                   # ... except a strip seen through a crack
    g[30, 29] = 0                         # the crack (one cell)
    frontiers, seeded = plan_frontiers(g, RES, (0, 0), (1.5, 0.75))
    assert seeded and not any(f.status == OK for f in frontiers)

    g[30, 22:32] = 0                      # a 0.5 m door instead
    frontiers, _ = plan_frontiers(g, RES, (0, 0), (1.5, 0.75))
    assert frontiers and frontiers[0].status == OK


def test_goal_reachable():
    g = room(60, 120)
    g[:, 60] = 100                        # closed wall: right half cut off
    g[20:30, 20:30] = -1                  # unknown patch
    robot = (0.5, 1.5)
    assert goal_reachable(g, RES, (0, 0), robot, (2.0, 2.0))
    assert not goal_reachable(g, RES, (0, 0), robot, (2.0, 0.1))        # 5 cm from the bottom wall
    assert not goal_reachable(g, RES, (0, 0), robot, (4.5, 1.5))        # behind the closed wall
    assert not goal_reachable(g, RES, (0, 0), robot, (1.25, 1.6))       # 0.1 m from the unknown patch


def test_goal_has_the_frontier_in_sight():
    """A frontier in a pocket too narrow to enter: the goal looks into the pocket from its opening, not from
    beside it (closer, but behind its wall: the site robot saw nothing there and blacklisted the pocket)."""
    from ee3033_mission.geometry import world_to_grid
    g = room()
    g[10:41, 20] = 100                    # pocket 0.2 m wide inside, open at the bottom (row 10)
    g[10:41, 25] = 100
    g[40, 20:26] = 100
    g[15:40, 21:25] = -1                  # its far part not seen yet
    frontiers, _ = plan_frontiers(g, RES, (0, 0), (1.5, 0.3))
    ok = [f for f in frontiers if f.status == OK]
    assert ok
    gr, gc = world_to_grid(*ok[0].goal, (0, 0), RES)
    assert 21 <= gc <= 24 and gr < 10, (gr, gc)


def test_frontier_of_a_narrow_opening_is_kept():
    """Seen at an angle, a narrow opening's frontier is split by a wall cell; its 3-cell piece (0.15 m) counts."""
    g = room()
    g[:, 40] = 100                        # wall with a narrow opening at rows 28-33 ...
    g[28:34, 40] = 0
    g[30, 40] = 100                       # ... with one wall cell left in it
    g[:, 41:59] = -1                      # unknown beyond the wall
    frontiers, _ = plan_frontiers(g, RES, (0, 0), (1.0, 1.5))
    assert [f for f in frontiers if f.status == OK]
    assert not plan_frontiers(g, RES, (0, 0), (1.0, 1.5), min_frontier_size=0.20)[0]    # the old 0.20 lost it

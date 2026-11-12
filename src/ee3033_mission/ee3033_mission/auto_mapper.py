"""Autonomous mapping (step 1): frontier exploration on the live SLAM map, then return home and save.

  WAIT_SYSTEM -> IDLE --(/explore/start)--> EXPLORE -> RETURN -> DONE

IDLE      dry run: frontiers and the goal it would choose are published on
          /explore/frontiers, but the robot does not move.
EXPLORE   every replan_period: find frontiers on /map and drive to the goal of
          the cheapest reachable one along a route (straight legs along corridor
          centres, turning in place at the corners, as for the way home: a long
          Nav2 goal cut the corner of a 0.45 m gap and the robot hit the wall
          there). The goal is kept until its
          frontier disappears (explored), Nav2 fails or the robot stops making
          progress; failed frontiers are blacklisted. After a goal is reached
          the node waits for a fresh /map; if the same frontier is still there
          the LiDAR cannot resolve it from there, so it is blacklisted.
          Exploration ends when no reachable frontier is left for finish_wait
          seconds, when the free area has not grown by coverage_step for
          coverage_timeout seconds (the area is complete; driving on only
          thickens the walls), or after max_duration.
RETURN    navigates back to the start pose (same heading, so the mission in
          step 2 can use the map origin as the initial pose).
DONE      the map from the end of EXPLORE stays in map_file.{pgm,yaml} (saved
          before the return); the map after the return goes to
          map_file_after_return.{pgm,yaml} only: on the way home the robot once
          went faster on a long straight leg, tilted, and distorted the map.
          The map with the largest free area during EXPLORE (thinnest walls,
          but a wall seen only later is missing) goes to map_file_peak.{pgm,yaml}.
/explore/stop cancels the Nav2 goal and stops the robot at any time (/explore/start
then resumes with the same home pose and blacklist);
/explore/save_map saves the current map (also usable while mapping by teleop).
"""

import math
import time

from geometry_msgs.msg import Point
from nav2_msgs.action import Spin
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from .exploration import (BLACKLISTED, OK, TOO_FAR, UNREACHABLE, frontier_left, goal_reachable, near_any,
                          plan_frontiers, save_map_files)
from .geometry import wrap_angle, yaw_from_quaternion
from .nav_node import NavStateNode, run
from .ros_helpers import marker

STATUS_COLOURS = {OK: (0.1, 0.9, 0.1), BLACKLISTED: (0.9, 0.1, 0.1),
                  UNREACHABLE: (0.5, 0.5, 0.5), TOO_FAR: (0.9, 0.6, 0.1)}

PARAMS = {
    'robot_radius': 0.15,        # min distance of a path cell from walls (m)
    'goal_clearance': 0.15,      # min distance of a goal from walls (m)
    'unknown_clearance': 0.20,   # min distance of a goal from unknown space (m)
    'min_frontier_size': 0.15,   # ignore shorter frontiers (m)
    'goal_reach': 0.5,           # goal at most this far from the frontier (m)
    'small_frontier_size': 0.0,  # shorter frontiers are looked at from open space, not driven into (m), 0 = off
    'small_goal_clearance': 0.17,  # min distance of such a frontier's goal from walls (m)
    'small_fallback': False,     # no such goal in sight: use any safe goal in sight (True) or skip it (False)
    'blind_from_deg': 125.0,     # LiDAR blind sector in the robot frame (deg, CCW from forward, from -> to through
    'blind_to_deg': -175.0,      # 180): at a goal with the frontier there, turn to face it first. Both 0 = off
    'gain_weight': 0.3,          # cost = path length - gain_weight * frontier size
    'blacklist_radius': 0.3,
    'blacklist_shrink': 0.7,     # this share of the frontier still left after reaching its goal -> blacklist
    'max_radius': 0.0,           # ignore frontiers further from home (m), 0 = off
    'replan_period': 1.0,
    'progress_timeout': 15.0,    # s without getting 0.1 m closer -> give up goal
    'finish_goals': True,        # frontier explored on the way: drive on to its goal if it is still safe
    'settle_timeout': 6.0,       # max wait for a fresh /map after reaching a goal
    'finish_wait': 6.0,          # s with no frontier before exploration ends
    'coverage_timeout': 45.0,    # s without coverage_step more free area before exploration ends (0 = off)
    'coverage_step': 0.05,       # m2
    'max_duration': 300.0,       # s, then return home anyway
    'return_home': True,
    'home_at_map_origin': 0.3,   # robot this close (m) to the map origin at the start: home is the origin with
                                 # heading 0 (0 = off: the pose at the start of the exploration)
    'map_file': '/home/ros2_ws/maps/sandbox',   # without extension
}


class AutoMapper(NavStateNode):

    def __init__(self):
        super().__init__('auto_mapper', 'explore', PARAMS)
        self.pub_markers = self.create_publisher(MarkerArray, '/explore/frontiers', 10)
        self.create_service(Trigger, '/explore/save_map', self.on_save)
        self.blacklist = []           # frontier centres (x, y) never to try again
        self.target = None            # frontier centre currently driven to
        self.goal = None              # its Nav2 goal (x, y)
        self.reached = None           # (frontier centre, size) of the last goal reached, checked after the next /map
        self.target_size = 0.0
        self.settle_until = None      # (map_count, deadline) while waiting for a fresh map
        self.no_frontier_since = None
        self.seen_on_the_way = False  # the current target frontier was explored before its goal was reached
        self.progress = None          # (best distance to goal, time it was set)
        self.n_goals = 0
        self.last_plan = 0.0
        self.coverage = (0.0, 0.0)    # (free area at its last growth by coverage_step, time of that growth)
        self.coverage_count = -1      # map_count of the last map checked
        self.peak = None              # (free area, /map, s into the exploration) of the map with the largest free area

    # ---------------------------------------------------------------- helpers
    def plan(self, robot_xy):
        grid, resolution, origin = self.map_grid()
        c = self.cfg
        return plan_frontiers(grid, resolution, origin, robot_xy,
                              robot_radius=float(c['robot_radius']), min_frontier_size=float(c['min_frontier_size']),
                              goal_reach=float(c['goal_reach']), gain_weight=float(c['gain_weight']),
                              blacklist=self.blacklist, blacklist_radius=float(c['blacklist_radius']),
                              home_xy=self.home[:2] if self.home else None, max_radius=float(c['max_radius']),
                              goal_clearance=float(c['goal_clearance']),
                              unknown_clearance=float(c['unknown_clearance']),
                              small_frontier_size=float(c['small_frontier_size']),
                              small_goal_clearance=float(c['small_goal_clearance']),
                              small_fallback=bool(c['small_fallback']))

    def save_map(self, path_base=None, msg=None):
        msg = msg or self.map
        if msg is None:
            return False, 'no map received yet'
        grid, resolution, (x0, y0) = self.map_grid(msg)
        q = msg.info.origin.orientation
        origin = (x0, y0, yaw_from_quaternion(q.x, q.y, q.z, q.w))
        try:
            path = save_map_files(path_base or self.cfg['map_file'], grid, resolution, origin)
        except OSError as e:
            self.get_logger().error(f'Map not saved: {e}')
            return False, str(e)
        free = int(((grid >= 0) & (grid < 50)).sum()) * resolution ** 2
        self.get_logger().info(f'Map saved: {path} ({grid.shape[1]}x{grid.shape[0]} cells, free {free:.1f} m2)')
        return True, path

    def blacklist_point(self, xy, why):
        self.blacklist.append(xy)
        self.get_logger().warn(f'Blacklisted frontier ({xy[0]:.2f}, {xy[1]:.2f}): {why}')

    # ---------------------------------------------------------------- inputs
    def on_save(self, request, response):
        response.success, response.message = self.save_map()
        return response

    # ---------------------------------------------------------------- exploration
    def can_resume(self):
        return self.state == 'STOPPED' and self.home is not None

    def begin(self):
        if self.state == 'STOPPED':                 # resume, keeping home and blacklist
            self.target, self.settle_until, self.no_frontier_since = None, None, None
            self.coverage = (self.coverage[0], time.time())
            self.set_state('EXPLORE')
            return True
        pose = self.robot_pose()
        if pose is None:
            self.get_logger().error('No map -> robot TF; is SLAM Toolbox running?')
            return False
        self.t_start = time.time()
        self.home = pose
        near = float(self.cfg['home_at_map_origin'])
        if near > 0 and math.hypot(pose[0], pose[1]) <= near:
            # The map origin is the robot's pose when the base came up (SLAM starts on the odometry frame), i.e.
            # where it was placed; the mission starts from it (initial_pose 0 0 0). The balancing robot turned
            # 10.6 deg on its own in the 14 s before the exploration began (1112_111012), so the old home (the pose
            # at the start of the exploration) left it 11 deg off that start heading after mapping.
            self.home = (0.0, 0.0, 0.0)
        self.get_logger().info('Home pose ({:.2f}, {:.2f}, {:.0f} deg){}; robot at ({:.2f}, {:.2f}, {:.0f} deg)'.format(
            self.home[0], self.home[1], math.degrees(self.home[2]), ' = map origin' if self.home != pose else '',
            pose[0], pose[1], math.degrees(pose[2])))
        self.blacklist, self.target, self.reached = [], None, None
        self.settle_until, self.no_frontier_since, self.n_goals = None, None, 0
        self.coverage, self.coverage_count, self.peak = (0.0, self.t_start), -1, None
        self.set_state('EXPLORE')
        return True

    def explore_step(self, pose):
        frontiers, seeded = self.plan(pose[:2])
        radius = float(self.cfg['blacklist_radius'])

        if self.target is not None:                 # driving to a frontier
            if any(f.status == OK and near_any(f.centre, [self.target], radius) for f in frontiers) or \
                    self.drive_on(pose):
                self.check_progress(pose)
                self.publish_markers(frontiers)
                return
            self.get_logger().info('Current frontier explored or no longer reachable; choosing the next one')
            self.cancel_action()
            self.target = None
            self.coverage_progress()

        if self.settle_until is not None:           # goal reached: wait for a map that includes the new view
            count, deadline = self.settle_until
            if self.map_count <= count and time.time() < deadline:
                self.publish_markers(frontiers)
                return
            self.settle_until = None
            if self.reached is not None:
                (centre, size), left = self.reached, frontier_left(frontiers, self.reached[0], radius)
                if left >= float(self.cfg['blacklist_shrink']) * size:
                    self.blacklist_point(centre, f'still a frontier after reaching it ({left:.2f} of {size:.2f} m)')
                    frontiers, seeded = self.plan(pose[:2])
                else:
                    self.coverage_progress()
                    if left > 0:     # a large frontier is seldom resolved from one place: keep the rest open
                        self.get_logger().info(f'Frontier shrank from {size:.2f} to {left:.2f} m; the rest stays open')
            self.reached = None

        best = frontiers[0] if frontiers and frontiers[0].status == OK else None
        self.publish_markers(frontiers, best)
        if best is None:
            if not seeded:
                self.get_logger().warn('Robot is not on traversable free space of the map; waiting',
                                       throttle_duration_sec=3.0)
                return
            if self.no_frontier_since is None:
                self.no_frontier_since = time.time()
                self.get_logger().info(f'No reachable frontier ({len(frontiers)} rejected); confirming '
                                       f'for {self.cfg["finish_wait"]:.0f}s')
            elif time.time() - self.no_frontier_since >= float(self.cfg['finish_wait']):
                self.finish_exploration('no reachable frontier left')
            return
        self.no_frontier_since = None

        self.target, self.goal, self.target_size = best.centre, best.goal, best.size
        self.seen_on_the_way = False
        self.n_goals += 1
        yaw = math.atan2(best.centre[1] - best.goal[1], best.centre[0] - best.goal[0])
        if math.hypot(best.centre[0] - best.goal[0], best.centre[1] - best.goal[1]) < 0.05:
            yaw = math.atan2(best.goal[1] - pose[1], best.goal[0] - pose[0])
        self.get_logger().info(f'Goal {self.n_goals}: frontier ({best.centre[0]:.2f}, {best.centre[1]:.2f}) '
                               f'size {best.size:.2f} m -> drive to ({best.goal[0]:.2f}, {best.goal[1]:.2f}), '
                               f'path {best.distance:.2f} m, {sum(f.status == OK for f in frontiers)} open frontiers')
        target = self.target
        self.send_route(best.goal[0], best.goal[1], yaw, lambda ok: self.on_goal_result(ok, target),
                        f'Goal {self.n_goals}')
        self.progress = (self.remaining(pose), time.time())

    def coverage_stalled(self):
        """Track the free area of /map: True when it has not grown by coverage_step for coverage_timeout s. Site
        11-06: the area was complete 61 s into the exploration (walls 8.7 cm thick); two small frontiers in corners
        took another 104 s without any new area, and the walls ended up 11.2 cm thick. Also keeps the map with the
        largest free area (self.peak; the free area shrinks again as the walls thicken)."""
        if self.map_count != self.coverage_count:
            self.coverage_count = self.map_count
            grid, resolution, _ = self.map_grid()
            free = int(((grid >= 0) & (grid < 50)).sum()) * resolution ** 2
            now = time.time()
            if self.peak is None or free > self.peak[0]:
                self.peak = (free, self.map, now - self.t_start)
            if free >= self.coverage[0] + float(self.cfg['coverage_step']):
                self.coverage = (free, now)
        timeout = float(self.cfg['coverage_timeout'])
        return timeout > 0 and time.time() - self.coverage[1] > timeout

    def coverage_progress(self):
        """A frontier was resolved (explored on the way or from its goal): progress for coverage_timeout as well.
        Lab maze 11-12: two 0.25 m frontiers added less than coverage_step each, and the timeout ended the
        exploration on the way to the last one."""
        self.coverage = (self.coverage[0], time.time())

    def remaining(self, pose):
        """Distance left to the goal along the route (straight line if there is none)."""
        d = self.route_remaining(pose)
        return d if d is not None else math.hypot(self.goal[0] - pose[0], self.goal[1] - pose[1])

    def drive_on(self, pose):
        """The target frontier is gone (explored from afar): keep driving to its goal while it is still a safe,
        reachable goal. A thin wall seen end-on from afar shows only its end; the site map lost a whole wall when
        the goals of a far side were all cancelled because the LiDAR had seen past it."""
        if not self.cfg['finish_goals']:
            return False
        grid, resolution, origin = self.map_grid()
        c = self.cfg
        if not goal_reachable(grid, resolution, origin, pose[:2], self.goal, robot_radius=float(c['robot_radius']),
                              goal_clearance=float(c['goal_clearance']),
                              unknown_clearance=float(c['unknown_clearance'])):
            return False
        if not self.seen_on_the_way:
            self.seen_on_the_way = True
            self.coverage_progress()
            self.get_logger().info('Frontier explored on the way; driving on to its goal to see the area up close')
        return True

    def check_progress(self, pose):
        d = self.remaining(pose)
        best, since = self.progress
        if d < best - 0.1:
            self.progress = (d, time.time())
            self.coverage_progress()     # driving to a far frontier adds no area for a while (lab maze 1112_095156:
                                         # the timeout ended the exploration on the way to goal 8, 2 frontiers left)
        elif time.time() - since > float(self.cfg['progress_timeout']):
            self.cancel_action()
            self.blacklist_point(self.target, f'no progress for {self.cfg["progress_timeout"]:.0f}s')
            self.target = None

    def on_goal_result(self, success, target):
        if self.state != 'EXPLORE' or target != self.target:
            return
        self.target = None
        if not success and self.goal_rejected:   # Nav2 not active: wait for it, do not blacklist
            self.get_logger().warn('Nav2 rejected the goal; waiting until Nav2 is active again')
            self.nav_active = False
            self.set_state('WAIT_SYSTEM')
        elif success:
            self.reached = (target, self.target_size)
            if not self.face_frontier(target):
                self.start_settle()
        else:
            self.blacklist_point(target, 'Nav2 failed or timed out')

    def start_settle(self, turned=True):
        """Wait for a /map that includes the view from the goal (then the frontier is checked)."""
        if self.state == 'EXPLORE':
            self.settle_until = (self.map_count, time.time() + float(self.cfg['settle_timeout']))

    def face_frontier(self, centre):
        """At a goal with the frontier in the LiDAR's blind sector (behind the robot to the left, where the robot's
        own body blocks the beams), turn in place to face it, then settle. Lab maze 1112_095156: the goal next to the
        start was skipped (within leg_skip), the robot never turned, the frontier 160 deg behind it stayed unseen
        and was blacklisted. Returns True if a turn was started."""
        a, b = math.radians(float(self.cfg['blind_from_deg'])), math.radians(float(self.cfg['blind_to_deg']))
        pose = self.robot_pose()
        if a == b or pose is None or not self.spin.server_is_ready():
            return False
        rel = math.atan2(centre[1] - pose[1], centre[0] - pose[0]) - pose[2]
        if float(wrap_angle(rel - a)) % (2 * math.pi) > float(wrap_angle(b - a)) % (2 * math.pi):
            return False
        spin = Spin.Goal()
        spin.target_yaw = float(wrap_angle(rel))
        spin.time_allowance.sec = 15
        self.get_logger().info(f'Frontier {math.degrees(float(wrap_angle(rel))):+.0f} deg off the heading, in the '
                               f'LiDAR blind sector: turning to face it')
        self.settle_until = (math.inf, time.time() + 30.0)   # no frontier check or new goal while turning
        self.run_action(self.spin, spin, 18.0, self.start_settle)
        return True

    def finish_exploration(self, why):
        self.cancel_action()
        self.target = None
        self.get_logger().info(f'Exploration finished ({why}) after {time.time() - self.t_start:.1f}s, '
                               f'{self.n_goals} goals, {len(self.blacklist)} blacklisted')
        self.save_map()
        if self.peak is not None:
            free, msg, t = self.peak
            self.get_logger().info(f'Largest free area {free:.2f} m2, {t:.0f} s into the exploration:')
            self.save_map(self.cfg['map_file'] + '_peak', msg)
        if self.cfg['return_home']:
            self.return_home()
        else:
            self.stop_robot()
            self.set_state('DONE')

    def on_home_reached(self):
        ok, path = self.save_map(self.cfg['map_file'] + '_after_return')
        if ok:
            self.get_logger().info(f'The mission uses the map from the end of the exploration '
                                   f'({self.cfg["map_file"]}.yaml), not this one')

    # ---------------------------------------------------------------- visualisation
    def publish_markers(self, frontiers, best=None):
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr = MarkerArray(markers=[clear])
        stamp = self.get_clock().now().to_msg()
        for i, f in enumerate(frontiers):
            mk = marker(self.frame, f'frontier_{f.status}', i, Marker.POINTS, 0.04, (*STATUS_COLOURS[f.status], 0.9),
                        stamp=stamp)
            mk.points = [Point(x=x, y=y, z=0.02) for x, y in f.points]
            arr.markers.append(mk)
        goal = best.goal if best is not None else (self.goal if self.target is not None else None)
        if goal is not None:
            arr.markers.append(marker(self.frame, 'goal', 0, Marker.SPHERE, 0.12, (0.2, 0.4, 1.0, 0.9), goal, 0.06,
                                      stamp))
        d = 2 * float(self.cfg['blacklist_radius'])
        arr.markers += [marker(self.frame, 'blacklist', i, Marker.CYLINDER, (d, d, 0.01), (0.9, 0.1, 0.1, 0.25), xy,
                               stamp=stamp) for i, xy in enumerate(self.blacklist)]
        self.pub_markers.publish(arr)

    # ---------------------------------------------------------------- tick
    def tick(self):
        s = self.state
        if s == 'WAIT_SYSTEM':
            if not self.system_ready(self.robot_pose() is not None):
                return
            if self.home is not None:                # Nav2 is back after rejecting a goal: continue
                self.get_logger().info('Nav2 active again; continuing the exploration')
                self.set_state('EXPLORE')
            else:
                self.set_state('IDLE')
                self.get_logger().info('System ready (dry run: frontiers on /explore/frontiers). '
                                       'Start with: ros2 service call /explore/start std_srvs/srv/Trigger')
            return

        if time.time() - self.last_plan < float(self.cfg['replan_period']):
            return
        self.last_plan = time.time()
        pose = self.robot_pose()
        if pose is None or self.map is None:
            return
        if s == 'IDLE':
            if self.cfg['auto_start']:
                self.begin()
                return
            frontiers, _ = self.plan(pose[:2])
            best = frontiers[0] if frontiers and frontiers[0].status == OK else None
            self.publish_markers(frontiers, best)
        elif s == 'EXPLORE':
            if time.time() - self.t_start > float(self.cfg['max_duration']):
                self.finish_exploration(f'max_duration {self.cfg["max_duration"]:.0f}s')
            elif self.coverage_stalled():
                self.finish_exploration(f'no new area for {self.cfg["coverage_timeout"]:.0f}s')
            else:
                self.explore_step(pose)


def main(args=None):
    run(AutoMapper, args)


if __name__ == '__main__':
    main()

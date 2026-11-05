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
          seconds or after max_duration.
RETURN    navigates back to the start pose (same heading, so the mission in
          step 2 can use the map origin as the initial pose).
DONE      the map from the end of EXPLORE stays in map_file.{pgm,yaml} (saved
          before the return); the map after the return goes to
          map_file_after_return.{pgm,yaml} only: on the way home the robot once
          went faster on a long straight leg, tilted, and distorted the map.
/explore/stop cancels the Nav2 goal and stops the robot at any time (/explore/start
then resumes with the same home pose and blacklist);
/explore/save_map saves the current map (also usable while mapping by teleop).
"""

import math
import time

from geometry_msgs.msg import Point
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from .exploration import (BLACKLISTED, OK, TOO_FAR, UNREACHABLE, frontier_left, goal_reachable, near_any,
                          plan_frontiers, save_map_files)
from .geometry import yaw_from_quaternion
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
    'gain_weight': 0.3,          # cost = path length - gain_weight * frontier size
    'blacklist_radius': 0.3,
    'blacklist_shrink': 0.7,     # this share of the frontier still left after reaching its goal -> blacklist
    'max_radius': 0.0,           # ignore frontiers further from home (m), 0 = off
    'replan_period': 1.0,
    'progress_timeout': 15.0,    # s without getting 0.1 m closer -> give up goal
    'finish_goals': True,        # frontier explored on the way: drive on to its goal if it is still safe
    'settle_timeout': 6.0,       # max wait for a fresh /map after reaching a goal
    'finish_wait': 6.0,          # s with no frontier before exploration ends
    'max_duration': 300.0,       # s, then return home anyway
    'return_home': True,
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
                              unknown_clearance=float(c['unknown_clearance']))

    def save_map(self, path_base=None):
        if self.map is None:
            return False, 'no map received yet'
        grid, resolution, (x0, y0) = self.map_grid()
        q = self.map.info.origin.orientation
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
            self.set_state('EXPLORE')
            return True
        pose = self.robot_pose()
        if pose is None:
            self.get_logger().error('No map -> robot TF; is SLAM Toolbox running?')
            return False
        self.t_start = time.time()
        self.home = pose
        self.get_logger().info('Home pose ({:.2f}, {:.2f}, {:.0f} deg)'.format(
            pose[0], pose[1], math.degrees(pose[2])))
        self.blacklist, self.target, self.reached = [], None, None
        self.settle_until, self.no_frontier_since, self.n_goals = None, None, 0
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
                elif left > 0:       # a large frontier is seldom resolved from one place: keep the rest open
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
            self.get_logger().info('Frontier explored on the way; driving on to its goal to see the area up close')
        return True

    def check_progress(self, pose):
        d = self.remaining(pose)
        best, since = self.progress
        if d < best - 0.1:
            self.progress = (d, time.time())
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
            self.settle_until = (self.map_count, time.time() + float(self.cfg['settle_timeout']))
        else:
            self.blacklist_point(target, 'Nav2 failed or timed out')

    def finish_exploration(self, why):
        self.cancel_action()
        self.target = None
        self.get_logger().info(f'Exploration finished ({why}) after {time.time() - self.t_start:.1f}s, '
                               f'{self.n_goals} goals, {len(self.blacklist)} blacklisted')
        self.save_map()
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
            else:
                self.explore_step(pose)


def main(args=None):
    run(AutoMapper, args)


if __name__ == '__main__':
    main()

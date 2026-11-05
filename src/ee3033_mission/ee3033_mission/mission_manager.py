"""Mission state machine for the EE3033 target search.

  WAIT_SYSTEM -> IDLE --(/mission/start)--> LOCALIZE -> SEARCH -> REPORT -> RETURN -> DONE

LOCALIZE  publishes the initial pose for AMCL (the start position is the map
          origin because mapping starts there) and lets AMCL settle with
          no-motion updates. Then the localization check: the LiDAR scan must
          fit the map walls at the pose AMCL reports. If it does not but the scan
          clearly fits one nearby pose (robot placed askew or turned while
          waiting), that pose becomes AMCL's initial pose and the check is
          repeated (auto-calibration, at most max_calibrations times). Otherwise
          the robot does not move; the log says how the robot is really placed,
          and the check is repeated until it passes (robot put back on the start
          mark), or /mission/start is called again to start anyway.
          Then it records the home pose and spins once in place, which both
          helps AMCL converge and looks for the target around the start.
SEARCH    plans viewpoints from which the whole free space of the saved map is
          seen within view_range (the algorithm decides how many), visits
          them in nearest-neighbour order with Nav2 and spins at each one,
          checking the localization again at each viewpoint (warning only).
          As soon as the target localizer confirms the target, the current
          Nav2 goal is cancelled.
REPORT    prints/publishes the (x, y) estimate and the last localization check
          and writes them to a YAML file.
RETURN    navigates back to the home pose and stops.
/mission/stop cancels everything and stops the robot at any time.
"""

import math
import time

import numpy as np
import yaml
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped
from nav2_msgs.action import Spin
from rclpy.time import Time
from std_srvs.srv import Empty
from tf2_ros import TransformException
from visualization_msgs.msg import Marker, MarkerArray

from ee3033_interfaces.msg import TargetEstimate

from .geometry import apply_body_offset, order_nearest_neighbour, plan_coverage, scan_to_points, transform_to_xyyaw
from .nav_node import NavStateNode, run
from .ros_helpers import LATCHED, marker
from .scan_match import calibration, scan_verdict, wall_distance

PARAMS = {
    'set_initial_pose': True,
    'initial_pose': [0.0, 0.0, 0.0],     # x, y, yaw in the map
    'initial_xy_std': 0.10,              # m, AMCL initial pose uncertainty
    'initial_yaw_std_deg': 20.0,         # deg; wide enough for AMCL to correct a robot placed a bit askew
    'localize_wait': 2.0,                # s of AMCL no-motion updates after the initial pose
    'localization_check': True,          # the LiDAR scan must fit the map before the robot moves
    'min_scan_match': 0.75,              # fraction of LiDAR points that must lie on the map walls
    'scan_match_margin': 0.10,           # fail if a nearby pose fits better by this much
    'auto_calibrate': True,              # failed check at the start: use the pose that clearly fits the scan
    'calibrate_min_fit': 0.90,           # ... if this fraction of the points lie on the walls there
    'calibrate_margin': 0.10,            # ... and any other heading >= 20 deg away fits worse by this much
    'max_calibrations': 3,               # per /mission/start; then the operator has to place the robot
    'initial_spin': True,
    'spin_angle_deg': 360.0,
    'view_range': 1.5,                   # m; the target is found reliably up to this distance
    'candidate_spacing': 0.25,           # m; grid of candidate viewpoints for the coverage planner
    'min_clearance': 0.15,               # m; viewpoints keep this far from walls and unknown space
    'max_viewpoints': 40,
    'spin_timeout': 25.0,
    'report_unconfirmed': True,
    'result_file': '/home/ros2_ws/mission_result.yaml',
}


class MissionManager(NavStateNode):

    def __init__(self):
        super().__init__('mission_manager', 'mission', PARAMS)
        self.estimate = None
        self.create_subscription(TargetEstimate, '/target/estimate', self.on_estimate, LATCHED)
        self.pub_target = self.create_publisher(PointStamped, '/mission/target', LATCHED)
        self.pub_init = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.pub_vp = self.create_publisher(MarkerArray, '/mission/viewpoints', LATCHED)
        self.reset_localizer = self.create_client(Empty, '/target_localizer/reset')
        self.nomotion = self.create_client(Empty, '/request_nomotion_update')   # AMCL update without motion
        self.walls = None             # (map_count, distance to the nearest wall per cell)
        self.loc_match = None         # (fraction of scan points on walls, where) of the last check
        self.loc_since = 0.0          # initial pose sent at
        self.loc_retry = None         # failed check: try again at
        self.loc_failed = False       # the check at the start has failed (the operator may start anyway)
        self.loc_override = False     # operator started anyway (/mission/start while the check fails)
        self.calibrations = 0         # initial poses set from the scan (auto-calibration) since begin()
        self.loc_calibrated = False   # the last check set the initial pose from the scan
        self.last_nomotion = 0.0
        self.viewpoints = []
        self.vp_index = 0
        self.last_init_pose = 0.0

    # ---------------------------------------------------------------- inputs
    def on_estimate(self, msg):
        self.estimate = msg
        if msg.confirmed and (self.state == 'SEARCH' or (self.state == 'LOCALIZE' and self.home is not None)):
            self.get_logger().info('Target confirmed by the localizer; ending the search')
            self.cancel_action()
            self.set_state('REPORT')

    def on_start(self, request, response):
        if self.state == 'LOCALIZE' and self.home is None and self.loc_failed:   # check failing: start anyway
            self.loc_retry, self.loc_override = None, True
            self.get_logger().warn('[loc] Started anyway by /mission/start: the localization check at the start is '
                                   'skipped; target positions may be wrong if the robot is not on the start mark')
            response.success, response.message = True, 'starting without the localization check'
            return response
        return super().on_start(request, response)

    # ---------------------------------------------------------------- mission
    def send_spin(self, on_done):
        goal = Spin.Goal()
        goal.target_yaw = math.radians(float(self.cfg['spin_angle_deg']))
        goal.time_allowance.sec = int(self.cfg['spin_timeout'])
        self.run_action(self.spin, goal, float(self.cfg['spin_timeout']) + 5.0, on_done)

    def publish_initial_pose(self, pose=None):
        x, y, yaw = pose if pose is not None else self.cfg['initial_pose']
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = self.frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose = self.make_pose(x, y, yaw).pose
        cov = np.zeros(36)
        cov[0] = cov[7] = float(self.cfg['initial_xy_std']) ** 2
        cov[35] = math.radians(float(self.cfg['initial_yaw_std_deg'])) ** 2
        msg.pose.covariance = cov.tolist()
        for _ in range(3):
            self.pub_init.publish(msg)
        self.get_logger().info(f'Initial pose set to ({x:.2f}, {y:.2f}, {math.degrees(yaw):.0f} deg)')

    def begin(self):
        self.t_start = time.time()
        self.calibrations = 0
        self.start_localization()
        self.set_state('LOCALIZE')
        return True

    # ---------------------------------------------------------------- localization
    def start_localization(self):
        """(Re)start at the initial pose; observations made with a wrong pose are dropped."""
        if self.reset_localizer.service_is_ready():
            self.reset_localizer.call_async(Empty.Request())
        if self.cfg['set_initial_pose']:
            self.publish_initial_pose()
        self.loc_since = time.time()

    def localize_step(self):
        """AMCL no-motion updates for localize_wait, then the localization check; the robot moves only if it passes."""
        if self.home is not None:               # check passed: the initial spin is running
            return
        if self.loc_retry is not None:          # check failed: the robot waits, then starts over
            if time.time() >= self.loc_retry:
                self.loc_retry = None
                self.start_localization()
            return
        if time.time() - self.loc_since < float(self.cfg['localize_wait']):
            if self.nomotion.service_is_ready() and time.time() - self.last_nomotion >= 0.5:
                self.last_nomotion = time.time()
                self.nomotion.call_async(Empty.Request())
            return
        pose = self.robot_pose()
        if pose is None:
            self.get_logger().warn('No map->robot TF yet', throttle_duration_sec=2.0)
            return
        if not self.loc_override and not self.check_localization('at the start', [
                'Put the robot on the start mark, facing the start direction. It will not move; checking again in 5 s.',
                'To start anyway: ros2 service call /mission/start std_srvs/srv/Trigger'],
                calibrate=bool(self.cfg['auto_calibrate'])):
            if not self.loc_calibrated:         # calibrated: checked again after localize_wait
                self.loc_retry, self.loc_failed = time.time() + 5.0, True
            return
        self.home = pose
        self.get_logger().info('Home pose ({:.2f}, {:.2f}, {:.0f} deg)'.format(
            pose[0], pose[1], math.degrees(pose[2])))
        if self.cfg['initial_spin']:
            self.send_spin(lambda ok: self.start_search())
        else:
            self.start_search()

    def check_localization(self, where, advice, calibrate=False):
        """True if the LiDAR scan fits the map walls at the pose AMCL reports (advice: lines shown if not).

        Otherwise logs, in plain words, how the robot is really placed compared with what the program
        believes (best-fitting pose correction). All lines start with [loc] (shown by demo.sh).
        With calibrate, a correction that is clear (scan_match.calibration) is applied instead of showing the
        advice: AMCL gets the corrected robot pose as its initial pose (self.loc_calibrated is then True).
        """
        self.loc_calibrated = False
        if not self.cfg['localization_check']:
            return True
        log = self.get_logger()
        try:
            if self.scan is None:
                raise TransformException(f'no scan on {self.cfg["scan_topic"]}')
            sensor = self.scan.header.frame_id
            pose = transform_to_xyyaw(self.tf_buffer.lookup_transform(self.frame, sensor, Time()))
            robot_in_sensor = transform_to_xyyaw(self.tf_buffer.lookup_transform(sensor, self.robot_frame, Time()))
        except TransformException as e:
            log.error(f'[loc] Cannot check the localization ({where}): {e}')
            return False
        grid, res, origin = self.map_grid()
        if self.walls is None or self.walls[0] != self.map_count:
            self.walls = (self.map_count, wall_distance(grid, res))
        points = scan_to_points(self.scan)
        points = points[np.hypot(points[:, 0], points[:, 1]) >= float(self.cfg['scan_min_range'])]
        v = scan_verdict(self.walls[1], res, origin, points, pose,
                         float(self.cfg['min_scan_match']), float(self.cfg['scan_match_margin']))
        self.loc_match = (v.frac, where)
        if v.ok:
            log.info(f'[loc] Localization check OK ({where}): {v.frac:.0%} of the LiDAR points lie on the map walls '
                     f'({v.lines[0]})')
            return True
        fix = None
        if calibrate and self.calibrations >= int(self.cfg['max_calibrations']):
            advice = [f'No auto-calibration: already done {self.calibrations} times.'] + advice
        elif calibrate:
            fix, why = calibration(v, float(self.cfg['calibrate_min_fit']), float(self.cfg['calibrate_margin']))
            if fix is None:
                advice = [f'No auto-calibration: {why}.'] + advice
        if fix is not None:
            sensor_pose = (pose[0] + fix[0], pose[1] + fix[1], pose[2] + fix[2])
            x, y, yaw = apply_body_offset(*sensor_pose, *robot_in_sensor)
            self.calibrations += 1
            advice = [f'AUTO-CALIBRATION {self.calibrations}/{int(self.cfg["max_calibrations"])}: no other pose fits '
                      f'as well (next best {v.other_fit:.0%}), so the program now uses that pose: '
                      f'({x:.2f}, {y:.2f}, {math.degrees(yaw):.0f} deg). Checking again in '
                      f'{float(self.cfg["localize_wait"]):.0f} s.']
        lines = ['!' * 70, f'LOCALIZATION CHECK FAILED ({where}): ' + v.lines[0]] + v.lines[1:] + advice + ['!' * 70]
        for line in lines:
            if where == 'at the start':       # rclpy: one severity per call site
                log.error('[loc] ' + line)
            else:
                log.warn('[loc] ' + line)
        if fix is not None:
            if self.reset_localizer.service_is_ready():     # observations made with the wrong pose
                self.reset_localizer.call_async(Empty.Request())
            self.publish_initial_pose((x, y, yaw))
            self.loc_since = time.time()
            self.loc_calibrated = True
        return False

    def plan_viewpoints(self):
        grid, resolution, origin = self.map_grid()
        view_range = float(self.cfg['view_range'])
        vps, seen, seeable = plan_coverage(grid, resolution, origin, self.home[:2], view_range,
                                           float(self.cfg['min_clearance']), float(self.cfg['candidate_spacing']))
        here = self.robot_pose() or self.home
        self.viewpoints = order_nearest_neighbour(here[:2], vps)[:int(self.cfg['max_viewpoints'])]
        self.vp_index = 0
        log = self.get_logger()
        log.info(f'[plan] {len(self.viewpoints)} viewpoints (plus the spin at the start) see {seen:.0%} of the free '
                 f'space within {view_range:.1f} m: ' + ', '.join(f'({x:.2f},{y:.2f})' for x, y in self.viewpoints))
        if seeable < 0.95:
            log.warn(f'[plan] {1 - seeable:.0%} of the free space cannot be seen from any place the robot can '
                     'reach (narrow gaps, behind obstacles): a target there may be missed')
        self.publish_viewpoints()

    def publish_viewpoints(self):
        self.pub_vp.publish(MarkerArray(markers=[
            marker(self.frame, 'viewpoints', i, Marker.SPHERE, 0.1,
                   (0.5, 0.5, 0.5, 0.6) if i < self.vp_index else (0.2, 0.4, 1.0, 0.9), xy, 0.05)
            for i, xy in enumerate(self.viewpoints)]))

    def start_search(self):
        if self.state != 'LOCALIZE':
            return
        self.set_state('SEARCH')
        self.plan_viewpoints()
        self.next_viewpoint()

    def next_viewpoint(self):
        if self.state != 'SEARCH':
            return
        if self.vp_index >= len(self.viewpoints):
            self.get_logger().warn('All viewpoints visited')
            self.set_state('REPORT')
            return
        x, y = self.viewpoints[self.vp_index]
        here = self.robot_pose() or self.home
        yaw = math.atan2(y - here[1], x - here[0])
        self.send_route(x, y, yaw, self.on_viewpoint_reached,
                        f'Viewpoint {self.vp_index + 1}/{len(self.viewpoints)} -> ({x:.2f}, {y:.2f})')

    def on_viewpoint_reached(self, success):
        if not success:
            self.get_logger().warn(f'Could not reach viewpoint {self.vp_index + 1}; spinning here instead')
        self.send_spin(self.on_viewpoint_spun)

    def on_viewpoint_spun(self, success):
        self.check_localization(f'viewpoint {self.vp_index + 1}',
                                ['Target positions measured now may be wrong; the search continues.'])
        self.vp_index += 1
        self.publish_viewpoints()
        self.next_viewpoint()

    def report(self):
        est = self.estimate
        elapsed = time.time() - self.t_start
        if est is not None and (est.confirmed or (self.cfg['report_unconfirmed'] and est.total_observations > 0)):
            x, y = est.position.x, est.position.y
            tag = 'CONFIRMED' if est.confirmed else 'UNCONFIRMED (best guess)'
            banner = (f'TARGET {tag}: {est.class_name} at (x, y) = ({x:.2f}, {y:.2f}) m in frame {self.frame}, '
                      f'{est.num_observations}/{est.total_observations} observations, spread {est.spread:.2f} m, '
                      f't={elapsed:.1f}s')
            pt = PointStamped()
            pt.header.frame_id = self.frame
            pt.header.stamp = self.get_clock().now().to_msg()
            pt.point.x, pt.point.y = x, y
            self.pub_target.publish(pt)
            result = {'found': True, 'confirmed': bool(est.confirmed), 'class': est.class_name,
                      'x': round(x, 3), 'y': round(y, 3), 'observations': int(est.num_observations),
                      'spread': round(float(est.spread), 3), 'time_s': round(elapsed, 1)}
        else:
            banner = f'TARGET NOT FOUND after {elapsed:.1f}s'
            result = {'found': False, 'time_s': round(elapsed, 1)}
        lines = [banner]
        if self.loc_match is not None:
            frac, where = self.loc_match
            ok = frac >= float(self.cfg['min_scan_match'])
            lines.append(f'Last localization check ({where}): {frac:.0%} of the LiDAR points on the map walls'
                         + ('' if ok else ' -- POOR: the reported position may be wrong'))
            result.update(scan_match=round(frac, 2), scan_match_at=where)
        self.get_logger().info('=' * 70)
        for line in lines:
            self.get_logger().info(line)
        self.get_logger().info('=' * 70)
        try:
            with open(self.cfg['result_file'], 'w') as f:
                f.write(f'# EE3033 mission result, frame {self.frame}\n')
                yaml.safe_dump(result, f, sort_keys=False)
        except OSError as e:
            self.get_logger().warn(f'Cannot write {self.cfg["result_file"]}: {e}')

    # ---------------------------------------------------------------- tick
    def tick(self):
        s = self.state
        if s == 'WAIT_SYSTEM':
            # AMCL publishes map -> odom only after an initial pose, and the Nav2 costmaps do not
            # activate without that TF, so the initial pose must be set before Nav2 can be ready.
            if (self.cfg['set_initial_pose'] and self.map is not None and self.robot_pose() is None
                    and time.time() - self.last_init_pose > 2.0):
                self.last_init_pose = time.time()
                self.publish_initial_pose()
            if self.system_ready(self.spin.server_is_ready()):
                self.set_state('IDLE')
                self.get_logger().info('System ready. Start with: ros2 service call /mission/start std_srvs/srv/Trigger')
        elif s == 'IDLE':
            if self.cfg['auto_start']:
                self.begin()
        elif s == 'LOCALIZE':
            self.localize_step()
        elif s == 'REPORT':
            self.report()
            if self.home is None:   # target confirmed before the home pose was recorded
                self.home = tuple(float(v) for v in self.cfg['initial_pose'])
            self.return_home()


def main(args=None):
    run(MissionManager, args)


if __name__ == '__main__':
    main()

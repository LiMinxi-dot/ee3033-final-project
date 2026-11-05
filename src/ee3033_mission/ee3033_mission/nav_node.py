"""Shared base of auto_mapper and mission_manager: a state machine that drives Nav2.

It provides the parameters, the robot pose from TF, the latched /map, the
<ns>/state topic, the <ns>/start and <ns>/stop services, one Nav2 action at a
time (with timeout and cancel), the WAIT_SYSTEM readiness check and the return
to the home pose. Subclasses implement begin() and tick().
"""

import math
import signal
import time

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose, Spin
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from .exploration import plan_route
from .geometry import (escape_move, free_ahead, rotation_clear, scan_to_points, transform_to_xyyaw, wall_sdf,
                       world_to_grid, wrap_angle, yaw_to_quaternion)
from .scan_match import transform
from .ros_helpers import LATCHED


class NavStateNode(Node):

    def __init__(self, name, ns, params):
        super().__init__(name)
        params = {'global_frame': 'map', 'robot_frame': 'base_footprint', 'auto_start': False,
                  'nav_timeout': 30.0, 'home_retries': 2, 'home_tolerance': 0.25,
                  'stuck_time': 8.0,            # s; a Nav2 goal during which the robot moves less than stuck_move
                  'stuck_move': 0.05,           # m  ... ends: reached if within stuck_accept, else failed
                  'stuck_accept': 0.20,         # m
                  'leg_skip': 0.15,             # m; a route leg whose waypoint is this close already is skipped
                  'route_clearance': 0.15, 'route_max_leg': 0.8, 'turn_first_deg': 45.0,
                  'scan_topic': '/scan_fixed',
                  'scan_min_range': 0.13,       # m; closer LiDAR hits are the robot itself (up to 12 cm behind it)
                  'escape': True,               # move away from a wall Nav2 cannot leave (see start_escape)
                  'escape_trigger': 0.11,       # m from the map walls; Nav2's inscribed zone: robot_radius 0.10 + 0.01
                  'escape_target': 0.14,
                  'escape_max_forward': 0.25,   # m; also limited by the LiDAR (body_half_*)
                  'escape_max_backward': 0.15,  # m; the LiDAR cannot see behind the robot, only the map counts
                  'escape_speed': 0.05,         # m/s, through the velocity smoother
                  'escape_turn_speed': 0.3,     # rad/s
                  'escape_margin': 0.01,        # m around the robot's rectangle for the LiDAR checks
                  'body_half_length': 0.06,     # m; the robot is 0.115 m long and 0.20 m wide (camera cable)
                  'body_half_width': 0.10,
                  **params}
        self.cfg = {n: self.declare_parameter(n, v).value for n, v in params.items()}
        self.frame, self.robot_frame = self.cfg['global_frame'], self.cfg['robot_frame']
        self.ns = ns

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.spin = ActionClient(self, Spin, 'spin')
        # The action server exists as soon as bt_navigator is configured, but it rejects goals until
        # the Nav2 lifecycle manager has activated all nodes (its bringup can also abort).
        self.nav_active_client = self.create_client(Trigger, '/lifecycle_manager_navigation/is_active')
        self.nav_active = False
        self.nav_query = 0.0

        self.map = None
        self.map_count = 0
        self.create_subscription(OccupancyGrid, '/map', self.on_map, LATCHED)
        self.pub_state = self.create_publisher(String, f'/{ns}/state', LATCHED)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_nav_cmd = self.create_publisher(Twist, '/cmd_vel_nav', 10)     # velocity smoother input
        self.scan = None
        self.create_subscription(LaserScan, self.cfg['scan_topic'], lambda m: setattr(self, 'scan', m),
                                 qos_profile_sensor_data)
        self.sdf = None               # (map_count, wall_sdf of /map)
        self.escaping = None          # the escape move being driven (start_escape)
        self.create_service(Trigger, f'/{ns}/start', self.on_start)
        self.create_service(Trigger, f'/{ns}/stop', self.on_stop)

        self.t_start = None
        self.home = None
        self.home_attempts = 0
        self.route = None             # legs of the route being driven (send_route)
        self.goal_handle = None
        self.goal_rejected = False
        self.pending_goal = None      # (client, goal, seq) held back while Nav2 finishes a cancel
        self.last_cancel = 0.0
        self.action_seq = 0
        self.action_deadline = None
        self.action_done_cb = None
        self.watch = None             # progress of the Nav2 goal being driven (see _check_stuck)
        self.state = None
        self.set_state('WAIT_SYSTEM')
        self.create_timer(0.2, self._tick)

    # ---------------------------------------------------------------- helpers
    def set_state(self, s):
        if s != self.state:
            elapsed = f' (t={time.time() - self.t_start:.1f}s)' if self.t_start else ''
            self.get_logger().info(f'[state] {self.state} -> {s}{elapsed}')
            self.state = s
            self.state_enter = time.time()
            self.pub_state.publish(String(data=s))

    def robot_pose(self):
        try:
            return transform_to_xyyaw(self.tf_buffer.lookup_transform(self.frame, self.robot_frame, Time()))
        except TransformException:
            return None

    def map_grid(self):
        """The last /map as a 2-D int array (row = y), its resolution and origin (x, y)."""
        m = self.map
        grid = np.array(m.data, dtype=np.int16).reshape(m.info.height, m.info.width)
        return grid, m.info.resolution, (m.info.origin.position.x, m.info.origin.position.y)

    def make_pose(self, x, y, yaw):
        ps = PoseStamped()
        ps.header.frame_id = self.frame
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x, ps.pose.position.y = float(x), float(y)
        (ps.pose.orientation.x, ps.pose.orientation.y,
         ps.pose.orientation.z, ps.pose.orientation.w) = yaw_to_quaternion(yaw)
        return ps

    def stop_robot(self):
        for _ in range(5):
            self.pub_cmd.publish(Twist())
            self.pub_nav_cmd.publish(Twist())       # else the smoother keeps an escape move's speed for 1 s

    def system_ready(self, *extra):
        """WAIT_SYSTEM check: map received, Nav2 active and all extra conditions true. Logs every 10 s."""
        if time.time() - self.nav_query >= 1.0 and self.nav_active_client.service_is_ready():
            self.nav_query = time.time()
            future = self.nav_active_client.call_async(Trigger.Request())
            future.add_done_callback(lambda f: setattr(self, 'nav_active', bool(f.result() and f.result().success)))
        if self.map is not None and self.nav_active and self.nav.server_is_ready() and all(extra):
            return True
        if time.time() - self.state_enter > 10.0:
            self.state_enter = time.time()
            self.get_logger().info(f'Waiting: map={self.map is not None} tf={self.robot_pose() is not None} '
                                   f'nav2_active={self.nav_active}')
        return False

    # ---------------------------------------------------------------- inputs
    def on_map(self, msg):
        self.map = msg
        self.map_count += 1

    def on_start(self, request, response):
        if self.state == 'WAIT_SYSTEM':      # too early: start as soon as the system is ready
            self.cfg['auto_start'] = True
            response.success, response.message = True, 'will start when the system is ready'
        elif self.state == 'IDLE' or self.can_resume():
            response.success = self.begin()
            response.message = 'started' if response.success else 'cannot start, see the log'
        else:
            response.success, response.message = False, f'cannot start in state {self.state}'
        self.get_logger().info(f'/{self.ns}/start: {response.message}')
        return response

    def can_resume(self):
        return False

    def on_stop(self, request, response):
        self.cancel_action()
        self.stop_robot()
        self.set_state('STOPPED')
        response.success, response.message = True, 'stopped'
        return response

    # ---------------------------------------------------------------- actions
    def run_action(self, client, goal, timeout, on_done, watch_xy=None):
        """Send an action goal; on_done(success) is called once (result, rejection or timeout). With watch_xy
        (a Nav2 goal's position) the goal also ends when the robot stops making progress (_check_stuck)."""
        self.action_seq += 1
        seq = self.action_seq
        self.action_done_cb = on_done
        self.action_deadline = time.time() + timeout
        self.watch = {'xy': watch_xy, 'pose': None, 't': time.time()} if watch_xy is not None else None
        self.goal_rejected = False
        self.pending_goal = (client, goal, seq)
        self._send_pending()

    def _send_pending(self):
        """Send the pending goal, but not within 1 s of cancelling one: Nav2 also drops a goal that arrives
        while it is still cancelling the previous one, and the new goal comes back as failed."""
        if self.pending_goal is None or time.time() - self.last_cancel < 1.0:
            return
        client, goal, seq = self.pending_goal
        self.pending_goal = None
        if seq == self.action_seq:
            client.send_goal_async(goal).add_done_callback(lambda f: self._on_goal_response(f, seq))

    def _on_goal_response(self, future, seq):
        handle = future.result()
        if seq != self.action_seq:
            if handle is not None and handle.accepted:      # cancelled before Nav2 had accepted it
                handle.cancel_goal_async()
                self.last_cancel = time.time()
            return
        if handle is None or not handle.accepted:
            self.get_logger().warn('Goal rejected')
            self.goal_rejected = True
            self._finish_action(seq, False)
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(lambda f: self._on_result(f, seq))

    def _on_result(self, future, seq):
        status = future.result().status if future.result() else GoalStatus.STATUS_UNKNOWN
        self._finish_action(seq, status == GoalStatus.STATUS_SUCCEEDED)

    def _finish_action(self, seq, success):
        if seq != self.action_seq or self.action_done_cb is None:
            return
        cb, self.action_done_cb = self.action_done_cb, None
        self.goal_handle, self.action_deadline, self.watch = None, None, None
        if not success:
            self.stop_robot()     # the base keeps the last speed command (no timeout), e.g. when Nav2 goes down
        cb(success)

    def cancel_action(self):
        self.action_seq += 1          # results of the old goal are ignored from now on
        self.action_done_cb, self.action_deadline, self.pending_goal, self.watch = None, None, None, None
        self.route = None
        if self.escaping is not None:
            self.escaping = None
            self.pub_nav_cmd.publish(Twist())
        self._cancel_goal()

    def _cancel_goal(self):
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
            self.goal_handle = None
            self.last_cancel = time.time()

    def send_nav(self, x, y, yaw, on_done, via=None, escaped=False):
        """Nav2 goal to (x, y, yaw); on_done(success) is called once.

        If the way there (towards via, by default the first waypoint of plan_route) starts more than
        turn_first_deg off the robot's heading, the robot first turns in place (Nav2 spin, no forward
        speed). DWB turned around in a forward arc instead, and twice the robot drove into the wall it
        stood next to when the next goal was behind it. Before that, a robot too close to a map wall for Nav2
        first moves away from it (start_escape); if the goal fails and the robot has got there meanwhile (e.g.
        drifted while turning), it moves away and the goal is sent once more.
        """
        def retry():
            self.send_nav(x, y, yaw, on_done, via, escaped=True)

        if not escaped and self.start_escape(retry):
            return
        goal = NavigateToPose.Goal()
        goal.pose = self.make_pose(x, y, yaw)

        def done(success):
            if success or escaped or not self.start_escape(retry):
                on_done(success)

        def drive(turned=True):
            if not turned:
                self.get_logger().warn('Turn in place failed (Nav2 spin aborted, e.g. "Collision Ahead"); '
                                       'driving anyway')
            self.run_action(self.nav, goal, float(self.cfg['nav_timeout']), done, watch_xy=(x, y))

        turn = self.turn_first(x, y, via)
        if turn is None:
            drive()
            return
        spin = Spin.Goal()
        spin.target_yaw = turn
        spin.time_allowance.sec = 15                      # 180 deg at the spin speed of 0.5 rad/s: 6.3 s
        self.get_logger().info(f'Turning {math.degrees(turn):.0f} deg in place before driving')
        self.run_action(self.spin, spin, 18.0, drive)     # drive also if the turn fails

    def turn_first(self, x, y, via=None):
        """Signed angle (rad) to turn in place before driving to (x, y), or None if not needed."""
        limit = math.radians(float(self.cfg['turn_first_deg']))
        pose = self.robot_pose()
        if limit <= 0 or pose is None or not self.spin.server_is_ready():
            return None
        if via is None:
            via = (x, y)
            if self.map is not None:
                grid, res, origin = self.map_grid()
                legs = plan_route(grid, res, origin, pose[:2], (x, y), clearance=float(self.cfg['route_clearance']),
                                  max_leg=float(self.cfg['route_max_leg']))
                via = legs[0] if legs else via
        if math.hypot(via[0] - pose[0], via[1] - pose[1]) < 0.1:
            return None
        turn = float(wrap_angle(math.atan2(via[1] - pose[1], via[0] - pose[0]) - pose[2]))
        return turn if abs(turn) > limit else None

    def _check_stuck(self):
        """End a Nav2 goal during which the robot has moved less than stuck_move for stuck_time s: reached if it
        is within stuck_accept of the goal, else failed (the caller may escape and retry). On site the robot stood
        0.14 m from such goals while Nav2 kept replanning ("Starting point in lethal space", "Failed to make
        progress") until nav_timeout: 96 s for a 0.14 m leg next to a wall corner, 60 s for another one."""
        w, pose, now = self.watch, self.robot_pose(), time.time()
        if pose is None:
            return
        if w['pose'] is None or math.hypot(pose[0] - w['pose'][0], pose[1] - w['pose'][1]) >= float(self.cfg['stuck_move']):
            w['pose'], w['t'] = pose, now
            return
        if now - w['t'] < float(self.cfg['stuck_time']):
            return
        dist = math.hypot(pose[0] - w['xy'][0], pose[1] - w['xy'][1])
        reached = dist <= float(self.cfg['stuck_accept'])
        self.get_logger().warn(f'No progress for {now - w["t"]:.0f} s, {dist:.2f} m from the goal: '
                               + ('counts as reached' if reached else 'giving up this goal'))
        seq = self.action_seq
        self._cancel_goal()
        self._finish_action(seq, reached)

    # ---------------------------------------------------------------- escape from walls
    def start_escape(self, then):
        """Before a Nav2 goal: if the robot stands within escape_trigger of the map walls, move it to where Nav2
        can plan from, then call then(). True if a move started.

        After the 360 deg spin at a viewpoint the balancing robot had drifted 15 cm towards a wall that the map
        shows thicker than it is (the real wall was 5-8 cm from the robot, the map wall under its centre): Nav2
        refused every goal ("Starting point in lethal space") and every turn ("Collision Ahead"), so the robot
        stood there until the mission gave up, its return home included.
        The move is a straight drive along the heading (geometry.escape_move: never through a map wall), forward
        only as far as the LiDAR shows the robot's rectangle (body_half_*) free, backward at most
        escape_max_backward (the LiDAR cannot see behind the robot). If no straight move gets clear (e.g. the
        robot stands along the wall), it first turns in place by the smallest angle that the LiDAR shows its
        corners can sweep (geometry.rotation_clear; behind it, where the LiDAR is blind, the map must be free).
        Everything goes through the velocity smoother (its acceleration limits).
        """
        c = self.cfg
        pose = self.robot_pose()
        if not c['escape'] or self.map is None or pose is None:
            return False
        grid, res, origin = self.map_grid()
        if self.sdf is None or self.sdf[0] != self.map_count:
            self.sdf = (self.map_count, wall_sdf(grid, res))
        sdf = self.sdf[1]
        row, col = world_to_grid(pose[0], pose[1], origin, res)
        if not (0 <= row < sdf.shape[0] and 0 <= col < sdf.shape[1]):
            return False
        here = float(sdf[row, col])
        if here >= float(c['escape_trigger']):
            return False
        points = self.body_points()
        target, back = float(c['escape_target']), float(c['escape_max_backward'])
        hl, hw = self.body_box()

        def straight(turn):
            ahead = free_ahead(points, hl, hw, turn) - float(c['escape_margin']) if points is not None else 0.0
            fwd = max(0.0, min(float(c['escape_max_forward']), ahead))
            return escape_move(grid, sdf, res, origin, (pose[0], pose[1], pose[2] + turn), target, fwd, back)

        plan, best = None, straight(0.0)
        if best is not None and best[1] >= target:
            plan = (0.0, best)
        elif points is not None and self.rear_free(pose, sdf, res, origin):
            for deg in (20, -20, 35, -35, 50, -50, 70, -70, 90, -90):
                turn = math.radians(deg)
                if not rotation_clear(points, hl, hw, turn):
                    continue
                m = straight(turn)
                if m is not None and m[1] >= target:
                    plan = (turn, m)
                    break
                if m is not None and (best is None or m[1] > best[1] + res):
                    best, plan = m, (turn, m)
        if plan is None and best is not None:
            plan = (0.0, best)
        if plan is None:
            self.get_logger().warn(f'Escape: the robot is {here:.2f} m from the map walls (Nav2 may not plan from '
                                   'here); no straight move gets it clear and no turn is free of LiDAR hits')
            return False
        turn, (d, clear) = plan
        segments = ([('turn', turn)] if turn else []) + [('drive', d)]
        self.get_logger().warn(f'Escape: the robot is {here:.2f} m from the map walls; '
                               + (f'turning {math.degrees(turn):+.0f} deg, then ' if turn else '')
                               + f'driving {abs(d):.2f} m {"forward" if d > 0 else "backward"} ({clear:.2f} m there)')
        self.escaping = {'segments': segments, 'i': 0, 'start': None, 'stop_at': None, 'abort': False,
                         'then': then, 'done': []}
        return True

    def body_box(self):
        """Half length and half width (m) of the robot's rectangle plus escape_margin."""
        m = float(self.cfg['escape_margin'])
        return float(self.cfg['body_half_length']) + m, float(self.cfg['body_half_width']) + m

    def body_points(self):
        """LiDAR hits of the last scan in the robot frame, None without a scan or its TF. Hits closer than
        scan_min_range are dropped only behind the robot (its own body shows up there), not ahead or beside it."""
        if self.scan is None:
            return None
        try:
            lx, ly, lyaw = transform_to_xyyaw(self.tf_buffer.lookup_transform(self.robot_frame,
                                                                              self.scan.header.frame_id, Time()))
        except TransformException:
            return None
        points = scan_to_points(self.scan)
        r = np.hypot(points[:, 0], points[:, 1])
        points = transform(points, lx, ly, lyaw)
        side = np.abs(np.arctan2(points[:, 1], points[:, 0])) <= math.radians(150)
        return points[(r >= float(self.cfg['scan_min_range'])) | (side & (r >= 0.07))]

    def rear_free(self, pose, sdf, res, origin):
        """No map wall straight behind the robot within the circle its corners sweep, where the LiDAR is blind
        (the robot's body). A wall counts only if the map gets deeper into it there than at the robot itself by
        more than a cell: a robot that AMCL puts into the wall beside it (the case to escape from) would
        otherwise find that wall behind it too."""
        hl, hw = self.body_box()
        row, col = world_to_grid(pose[0], pose[1], origin, res)
        here = float(sdf[row, col])
        for r in np.arange(0.06, math.hypot(hl, hw) + 0.01, 0.02):
            row, col = world_to_grid(pose[0] - r * math.cos(pose[2]), pose[1] - r * math.sin(pose[2]), origin, res)
            if not (0 <= row < sdf.shape[0] and 0 <= col < sdf.shape[1]) or sdf[row, col] < here - res:
                return False
        return True

    def _escape_step(self):
        e, now = self.escaping, time.time()
        if e['stop_at'] is not None:                   # between segments: the smoother ramps down, the robot settles
            self.pub_nav_cmd.publish(Twist())
            if now - e['stop_at'] < 0.6:
                return
            e['stop_at'], e['start'] = None, None
            e['i'] += 1
            if e['abort'] or e['i'] >= len(e['segments']):
                self.escaping = None
                self.get_logger().warn('Escape: ' + ', '.join(e['done']))
                e['then']()
                return
        kind, amount = e['segments'][e['i']]
        pose = self.robot_pose()
        if pose is None:
            e['done'].append(f'{kind} stopped: no pose')
            e['abort'], e['stop_at'] = True, now
            return
        if e['start'] is None:
            speed = float(self.cfg['escape_turn_speed'] if kind == 'turn' else self.cfg['escape_speed'])
            e['start'], e['deadline'] = pose, now + abs(amount) / speed + 4.0
        x0, y0, yaw0 = e['start']
        points = self.body_points()
        hl, hw = self.body_box()
        cmd = Twist()
        if kind == 'turn':
            progress = abs(float(wrap_angle(pose[2] - yaw0)))
            blocked = points is None or not rotation_clear(points, hl, hw, math.copysign(math.radians(5), amount))
            finished = progress >= abs(amount) - 0.03
            cmd.angular.z = math.copysign(float(self.cfg['escape_turn_speed']), amount)
            text = f'turned {math.degrees(progress):.0f} deg'
        else:
            progress = abs((pose[0] - x0) * math.cos(yaw0) + (pose[1] - y0) * math.sin(yaw0))
            blocked = amount > 0 and (points is None or free_ahead(points, hl, hw) <= 0.0)
            finished = progress >= abs(amount) - 0.005
            cmd.linear.x = math.copysign(float(self.cfg['escape_speed']), amount)
            text = f'drove {progress:.2f} m {"forward" if amount > 0 else "backward"}'
        if finished or blocked or now > e['deadline']:
            if not finished:
                text += ' (stopped: LiDAR hit)' if blocked else ' (stopped: timeout)'
                e['abort'] = True
            e['done'].append(text)
            e['stop_at'] = now
            self.pub_nav_cmd.publish(Twist())
            return
        self.pub_nav_cmd.publish(cmd)

    # ---------------------------------------------------------------- routes
    def send_route(self, x, y, yaw, on_done, label):
        """Drive to (x, y, yaw) in short straight legs along corridor centres (exploration.plan_route).

        At each corner the robot first turns in place to face the next leg. One long goal through a maze
        often failed: the controller got stuck at the U-turns ("Failed to make progress"), while the short
        goals of the exploration were reached. on_done(success) is called once; a leg that ends within
        home_tolerance of its waypoint counts as reached.
        """
        pose = self.robot_pose()
        legs = []
        if self.map is not None and pose is not None:
            grid, res, origin = self.map_grid()
            legs = plan_route(grid, res, origin, pose[:2], (x, y), clearance=float(self.cfg['route_clearance']),
                              max_leg=float(self.cfg['route_max_leg']))
        if not legs:                  # no route on the map: let Nav2 try the whole way
            legs = [(float(x), float(y))]
        self.route = {'legs': legs, 'yaw': yaw, 'done': on_done, 'total': len(legs)}
        self.get_logger().info(f'{label}: {len(legs)} legs via ' + ' '.join(f'({a:.2f},{b:.2f})' for a, b in legs))
        self._next_leg()

    def route_remaining(self, pose):
        """Distance (m) left along the route being driven, or None without a route."""
        if not self.route or not self.route['legs']:
            return None
        pts = [tuple(pose[:2])] + list(self.route['legs'])
        return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))

    def _next_leg(self):
        r, pose, skip = self.route, self.robot_pose(), float(self.cfg['leg_skip'])
        while r['legs'] and pose is not None and math.hypot(pose[0] - r['legs'][0][0], pose[1] - r['legs'][0][1]) <= skip:
            self.get_logger().info(f'Leg {r["total"] - len(r["legs"]) + 1}/{r["total"]} skipped: the robot is within '
                                   f'{skip:.2f} m of it')
            r['legs'].pop(0)
        if not r['legs']:
            r['done'](True)
            return
        x, y = r['legs'][0]
        yaw = r['yaw'] if len(r['legs']) == 1 else math.atan2(r['legs'][1][1] - y, r['legs'][1][0] - x)
        self.get_logger().info(f'Leg {r["total"] - len(r["legs"]) + 1}/{r["total"]} -> ({x:.2f}, {y:.2f})')
        self.send_nav(x, y, yaw, self._on_leg, via=(x, y))       # legs are straight

    def _on_leg(self, success):
        r = self.route
        pose = self.robot_pose()
        wx, wy = r['legs'][0]
        near = pose is not None and math.hypot(pose[0] - wx, pose[1] - wy) <= float(self.cfg['home_tolerance'])
        if not (success or near):
            r['done'](False)
            return
        r['legs'].pop(0)
        if r['legs']:
            self._next_leg()
        else:
            r['done'](True)

    # ---------------------------------------------------------------- return home
    def return_home(self):
        self.home_attempts = 0
        self.set_state('RETURN')
        self.go_home()

    def go_home(self):
        """Route home from where the robot is now (a failed attempt replans from there)."""
        self.home_attempts += 1
        x, y, yaw = self.home
        self.send_route(x, y, yaw, self.on_home_result,
                        f'Returning home to ({x:.2f}, {y:.2f}), attempt {self.home_attempts}')

    def on_home_result(self, success):
        pose = self.robot_pose()
        dist = math.hypot(pose[0] - self.home[0], pose[1] - self.home[1]) if pose else float('inf')
        if success or dist <= float(self.cfg['home_tolerance']):
            self.stop_robot()
            self.get_logger().info(f'Home reached ({dist:.2f} m from start) after {time.time() - self.t_start:.1f}s')
            self.on_home_reached()
            self.set_state('DONE')
        elif self.home_attempts <= int(self.cfg['home_retries']):
            self.go_home()
        else:
            self.stop_robot()
            self.get_logger().error(f'Could not return home ({dist:.2f} m away)')
            self.set_state('FAILED')

    def on_home_reached(self):
        pass

    # ---------------------------------------------------------------- tick
    def _tick(self):
        self._send_pending()
        if self.escaping is not None:
            self._escape_step()
        if self.watch is not None and self.action_done_cb is not None:
            self._check_stuck()
        if self.action_deadline is not None and time.time() > self.action_deadline:
            self.get_logger().warn('Action timed out; cancelling')
            self._cancel_goal()
            self._finish_action(self.action_seq, False)
        self.tick()


def run(node_class, args=None):
    # Python's own SIGINT handler instead of rclpy's: Ctrl-C raises KeyboardInterrupt while the context
    # is still valid, so stop_robot() below can still publish (rclpy's handler shuts the context down first).
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    node = node_class()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.try_shutdown()

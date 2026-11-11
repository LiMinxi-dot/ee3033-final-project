"""Odometry turn test: how far the robot really turns when the odometry says it turned N degrees.

The robot turns in place by each angle of `turns` (by /odom, at `speed`), stops, and the LiDAR scans from before
and after the turn are aligned (scan_match.scan_rotation): after a full turn they should coincide, and any rotation
left between them is the odometry's over- or under-turn. Left and right turns are reported separately, with the
ratio true / odometry turn for each direction: without odom_scaler these are the odom_yaw_scale (left) and
odom_yaw_scale_cw (right) to use; with odom_scaler running, multiply its current scales by them (1.000 = right).

Run with only the base and the LiDAR up (base_sensors.launch.py, no Nav2, no teleop):

  ros2 run ee3033_mission odom_turn_test                  # dry run: checks, plan, scan noise; the robot does not move
  ros2 run ee3033_mission odom_turn_test --ros-args -p go:=true

Safety: nothing may publish /cmd_vel but this node; no LiDAR hit within `clearance` of the LiDAR (closer than
scan_min_range is the robot itself); the turn stops if the pitch (/robot_euler) changes by more than max_pitch_deg
from the start, or after twice the expected time. The robot is stopped on every exit.
"""

import math
import signal
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, Vector3Stamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import LaserScan

from .geometry import scan_to_points, wrap_angle, yaw_from_quaternion
from .scan_match import scan_rotation


class OdomTurnTest(Node):

    def __init__(self):
        super().__init__('odom_turn_test')
        for name, default in (('turns', [360.0, -360.0, 360.0, -360.0]), ('speed', 0.5), ('settle', 2.0),
                              ('scans', 3), ('go', False), ('scan_topic', '/scan_fixed'), ('scan_min_range', 0.13),
                              ('clearance', 0.20), ('max_pitch_deg', 25.0), ('cmd_topic', '/cmd_vel')):
            self.declare_parameter(name, default)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.turns = [math.radians(float(t)) for t in p('turns')]
        self.speed, self.settle, self.n_scans = float(p('speed')), float(p('settle')), int(p('scans'))
        self.go, self.min_range, self.clearance = bool(p('go')), float(p('scan_min_range')), float(p('clearance'))
        self.max_pitch = math.radians(float(p('max_pitch_deg')))
        self.cmd_topic = p('cmd_topic')
        self.pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.create_subscription(Odometry, '/odom', self.on_odom, 50)
        self.create_subscription(LaserScan, p('scan_topic'), self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Vector3Stamped, '/robot_euler', self.on_euler, 20)
        self.yaw = None               # unwrapped odometry yaw
        self.last_raw = None
        self.scan_q = []              # (odometry yaw, points) of the scans since the robot stood still
        self.pitch = self.pitch0 = None
        self.closest = None           # nearest LiDAR hit beyond scan_min_range (m)
        self.state, self.t_state = 'WAIT', time.time()
        self.i = 0                    # index of the turn being done
        self.before = None            # (odometry yaw, [points]) before the turn
        self.results = []             # (odometry turn, true turn, rms, share)
        self.done = False
        self.create_timer(0.05, self.tick)
        mode = 'GO: the robot will turn' if self.go else 'dry run: the robot will NOT move (add -p go:=true)'
        self.get_logger().info(f'Turns {[round(math.degrees(t)) for t in self.turns]} deg at {self.speed} rad/s; {mode}')

    # ------------------------------------------------------------------ inputs
    def on_odom(self, msg):
        q = msg.pose.pose.orientation
        raw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        if self.last_raw is None:
            self.yaw = raw
        else:
            self.yaw += float(wrap_angle(raw - self.last_raw))
        self.last_raw = raw

    def on_scan(self, msg):
        if self.yaw is None:
            return
        pts = scan_to_points(msg)
        r = np.hypot(pts[:, 0], pts[:, 1])
        self.closest = float(r[r >= self.min_range].min()) if (r >= self.min_range).any() else float('inf')
        if self.state == 'COLLECT':
            self.scan_q.append((self.yaw, pts[r >= self.min_range]))

    def on_euler(self, msg):
        self.pitch = msg.vector.y

    # ------------------------------------------------------------------ helpers
    def stop(self):
        for _ in range(3):
            self.pub.publish(Twist())

    def set_state(self, s):
        self.state, self.t_state = s, time.time()

    def finish(self, why=None):
        self.stop()
        if why:
            self.get_logger().error(f'Stopped: {why}')
        self.report()
        self.done = True

    def safe_to_turn(self):
        others = [i.node_name for i in self.get_publishers_info_by_topic(self.cmd_topic)
                  if i.node_name != self.get_name()]
        if others:
            return f'{self.cmd_topic} is also published by {others} (Nav2 or teleop running?)'
        if self.closest < self.clearance:
            return (f'LiDAR hit {self.closest:.2f} m away (< clearance {self.clearance:.2f} m): '
                    f'move the robot away from the walls')
        return None

    def measure(self, before, after, guess):
        """Median over the after-scans of the rotation from the first before-scan, near guess."""
        rots = [scan_rotation(before[1][0], pts, guess) for pts in after[1]]
        k = int(np.argsort([r[0] for r in rots])[len(rots) // 2])
        return rots[k]

    # ------------------------------------------------------------------ state machine
    def tick(self):
        if self.done:
            return
        now = time.time()
        if self.state == 'WAIT':
            if self.yaw is None or self.pitch is None or self.closest is None:
                if now - self.t_state > 10.0:
                    self.finish('no /odom, /robot_euler or LiDAR scan after 10 s (base and LiDAR running?)')
                return
            why = self.safe_to_turn()
            if why:
                self.finish(why)
                return
            self.pitch0 = self.pitch
            self.get_logger().info(f'Checks OK: nearest LiDAR hit {self.closest:.2f} m, pitch '
                                   f'{math.degrees(self.pitch0):+.1f} deg, '
                                   f'odom_scaler {"running" if self.count_publishers("/odom_raw") else "not running"}')
            self.set_state('SETTLE')
        elif self.state == 'SETTLE':
            if now - self.t_state >= self.settle:
                self.scan_q = []
                self.set_state('COLLECT')
        elif self.state == 'COLLECT':
            if len(self.scan_q) < self.n_scans:
                return
            sample = (float(np.mean([y for y, _ in self.scan_q])), [p for _, p in self.scan_q])
            if self.before is None:
                self.before = sample
                if not self.go:       # dry run: scan noise standing still
                    rot, rms, share = scan_rotation(sample[1][0], sample[1][-1], 0.0)
                    self.get_logger().info(f'Standing still: scans {math.degrees(rot):+.2f} deg apart (noise), ICP '
                                           f'RMS {rms * 100:.1f} cm, {share:.0%} matched. Dry run done.')
                    self.finish()
                    return
                self.start_turn()
            else:
                odom_turn = sample[0] - self.before[0]
                rot, rms, share = self.measure(self.before, sample, float(wrap_angle(odom_turn)))
                true = odom_turn + float(wrap_angle(rot - odom_turn))
                self.results.append((odom_turn, true, rms, share))
                self.get_logger().info(f'Turn {self.i + 1}/{len(self.turns)}: odometry {math.degrees(odom_turn):+.1f} '
                                       f'deg, LiDAR {math.degrees(true):+.1f} deg (true / odometry '
                                       f'{true / odom_turn:.3f}; ICP RMS {rms * 100:.1f} cm, {share:.0%} matched)')
                self.i += 1
                self.before = sample
                if self.i >= len(self.turns):
                    self.finish()
                else:
                    self.start_turn()
        elif self.state == 'TURN':
            if abs(self.pitch - self.pitch0) > self.max_pitch:
                self.finish(f'pitch {math.degrees(self.pitch):+.1f} deg (start {math.degrees(self.pitch0):+.1f})')
                return
            if now - self.t_state > 2.0 * abs(self.target) / self.speed + 5.0:
                self.finish('turn took too long')
                return
            left = self.target - (self.yaw - self.turn_start)
            if left * math.copysign(1.0, self.target) <= math.radians(0.5):
                self.stop()
                self.set_state('SETTLE')
                return
            cmd = Twist()
            cmd.angular.z = math.copysign(min(self.speed, max(0.15, 1.5 * abs(left))), left)
            self.pub.publish(cmd)

    def start_turn(self):
        why = self.safe_to_turn()
        if why:
            self.finish(why)
            return
        self.target, self.turn_start = self.turns[self.i], self.yaw
        self.get_logger().info(f'Turn {self.i + 1}/{len(self.turns)}: {math.degrees(self.target):+.0f} deg')
        self.set_state('TURN')

    def report(self):
        if not self.results:
            return
        self.get_logger().info('=' * 70)
        for name, sign in (('left (CCW)', 1.0), ('right (CW)', -1.0)):
            r = [(o, t) for o, t, _, _ in self.results if math.copysign(1.0, o) == sign]
            if r:
                ratios = [t / o for o, t in r]
                self.get_logger().info(f'{name}: {len(r)} turns, odometry {math.degrees(sum(o for o, _ in r)):+.1f} '
                                       f'deg, LiDAR {math.degrees(sum(t for _, t in r)):+.1f} deg: true / odometry '
                                       f'{sum(t for _, t in r) / sum(o for o, _ in r):.3f} '
                                       f'(each {", ".join(f"{k:.3f}" for k in ratios)})')
        self.get_logger().info('Without odom_scaler: use these as odom_yaw_scale (left) / odom_yaw_scale_cw (right). '
                               'With it: multiply its scales by them.')
        self.get_logger().info('=' * 70)


def main(args=None):
    # Python's SIGINT handler, as nav_node.run: Ctrl-C keeps the context valid so that stop() can still publish
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    node = OdomTurnTest()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.try_shutdown()

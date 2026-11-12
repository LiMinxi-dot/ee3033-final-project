"""Heading hold while the system comes up.

The balancing robot turns on its own while it stands: 10.6 deg in the 14 s between the base coming up and the
start of the exploration (lab maze 1112_111012, ~0.7 deg/s). The map frame is the odometry frame, i.e. the pose
the robot had when the base came up (where it was placed), so the map ended up turned against the arena walls and
against the start heading, and the robot came home from mapping 11 deg off the mission's initial pose.

This node turns the robot in place (odometry yaw, P control, no forward speed) back to the heading it had when
the odometry started, until auto_mapper or mission_manager leaves WAIT_SYSTEM / IDLE. It then sends one zero
command and publishes nothing more. It also lets go for good when anything else commands /cmd_vel (a message it
did not send), when the robot tilts more than max_pitch_deg (a fall) and after max_time s.
"""

import math
import time

import rclpy
from geometry_msgs.msg import Twist, Vector3Stamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String

from .geometry import wrap_angle, yaw_from_quaternion

WAITING = ('WAIT_SYSTEM', 'IDLE')


class HeadingHold(Node):

    def __init__(self):
        super().__init__('heading_hold')
        self.declare_parameter('gain', 1.0)              # rad/s per rad of heading error
        self.declare_parameter('max_rate', 0.3)          # rad/s
        self.declare_parameter('deadband_deg', 1.5)      # start turning back beyond this,
        self.declare_parameter('settle_deg', 0.5)        # and stop within this
        self.declare_parameter('max_pitch_deg', 25.0)
        self.declare_parameter('max_time', 180.0)        # s
        self.declare_parameter('state_topics', ['/explore/state', '/mission/state'])
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.gain, self.max_rate = float(p('gain')), float(p('max_rate'))
        self.deadband = math.radians(float(p('deadband_deg')))
        self.settle = math.radians(float(p('settle_deg')))
        self.turning = False
        self.max_pitch = math.radians(float(p('max_pitch_deg')))
        self.max_time = float(p('max_time'))
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odom', self.on_odom, 20)
        self.create_subscription(Twist, '/cmd_vel', self.on_cmd, 20)
        self.create_subscription(Vector3Stamped, '/robot_euler', self.on_euler, 20)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        for topic in p('state_topics'):
            self.create_subscription(String, topic, lambda m, t=topic: self.on_state(t, m), latched)
        self.ref = None              # heading (rad) to hold: the first odometry heading
        self.yaw = None
        self.sent = []               # angular speeds sent recently (to recognise our own messages)
        self.active = True
        self.t0 = time.time()
        self.timer = self.create_timer(0.1, self.step)
        self.get_logger().info('Holding the start heading until the exploration / mission starts')

    def release(self, why):
        if not self.active:
            return
        self.active = False
        self.timer.cancel()
        self.pub.publish(Twist())
        err = '' if self.ref is None or self.yaw is None else \
            f' (heading {math.degrees(float(wrap_angle(self.yaw - self.ref))):+.1f} deg from the start)'
        self.get_logger().info(f'Heading hold off: {why}{err}')

    def on_odom(self, msg):
        q = msg.pose.pose.orientation
        self.yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        if self.ref is None:
            self.ref = self.yaw

    def on_cmd(self, msg):
        if not self.active:
            return
        if msg.linear.x == 0.0 and any(abs(msg.angular.z - w) < 1e-9 for w in self.sent):
            return                   # ours
        self.release('another node commands /cmd_vel')

    def on_euler(self, msg):
        if abs(msg.vector.y) > self.max_pitch:
            self.release(f'pitch {math.degrees(msg.vector.y):+.0f} deg')

    def on_state(self, topic, msg):
        if msg.data not in WAITING:
            self.release(f'{topic} {msg.data}')

    def step(self):
        if not self.active:
            return
        if time.time() - self.t0 > self.max_time:
            self.release(f'{self.max_time:.0f} s passed')
            return
        if self.ref is None or self.yaw is None:
            return
        err = float(wrap_angle(self.ref - self.yaw))
        if abs(err) > self.deadband:
            self.turning = True
        elif abs(err) < self.settle:
            self.turning = False
        w = max(-self.max_rate, min(self.max_rate, self.gain * err)) if self.turning else 0.0
        self.sent = (self.sent + [w])[-20:]
        out = Twist()
        out.angular.z = w
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = HeadingHold()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.active:
            node.pub.publish(Twist())
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

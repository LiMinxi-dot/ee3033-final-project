"""Velocity watchdog for remote teleoperation.

The balance base keeps driving with the last /cmd_vel it received, so a teleop
command sent over Wi-Fi keeps the robot moving if the link drops. This node
runs on the car, relays the teleop topic to /cmd_vel and sends a stop when no
command has arrived for `timeout` seconds. After the stop it stays silent, so
it does not fight Nav2 when Nav2 also publishes /cmd_vel (explore mode).
"""

import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class CmdVelWatchdog(Node):

    def __init__(self):
        super().__init__('cmd_vel_watchdog')
        self.declare_parameter('input_topic', '/cmd_vel_teleop')
        self.declare_parameter('output_topic', '/cmd_vel')
        self.declare_parameter('timeout', 0.5)
        self.declare_parameter('max_linear', 0.25)
        self.declare_parameter('max_angular', 1.2)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.timeout = float(p('timeout'))
        self.max_lin, self.max_ang = float(p('max_linear')), float(p('max_angular'))
        self.pub = self.create_publisher(Twist, p('output_topic'), 10)
        self.create_subscription(Twist, p('input_topic'), self.on_cmd, 10)
        self.last_time = None
        self.active = False
        self.create_timer(0.05, self.check)
        self.get_logger().info(f'Relaying {p("input_topic")} -> {p("output_topic")}, stop after {self.timeout}s silence')

    def on_cmd(self, msg):
        out = Twist()
        out.linear.x = max(-self.max_lin, min(self.max_lin, msg.linear.x))
        out.angular.z = max(-self.max_ang, min(self.max_ang, msg.angular.z))
        self.pub.publish(out)
        self.last_time = time.time()
        self.active = True

    def check(self):
        if self.active and time.time() - self.last_time > self.timeout:
            for _ in range(3):
                self.pub.publish(Twist())
            self.active = False
            self.get_logger().info('No teleop command: robot stopped')


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelWatchdog()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(Twist())
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

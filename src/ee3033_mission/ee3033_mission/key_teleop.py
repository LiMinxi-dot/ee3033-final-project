"""Keyboard driving for manual mapping (demo.sh manual), in the terminal it runs in.

Hold a key to drive (the terminal repeats it); release it and the robot stops after `hold` s:
  w / s   forward / backward          a / d   turn left / right on the spot
  q / e   forward while turning left / right
  space   stop at once                Ctrl-C  stop and quit (demo.sh then saves the map)
Publishes /cmd_vel_teleop at 20 Hz (cmd_vel_watchdog relays it to /cmd_vel and stops the robot 0.6 s after the
last message). The speed ramps up and down (accel / angular_accel): a step from 0 to 0.15 m/s made the
balancing robot lurch (nav2.launch.py, velocity smoother).
"""

import math
import select
import sys
import termios
import time
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions

KEYS = {'w': (1, 0), 's': (-1, 0), 'a': (0, 1), 'd': (0, -1), 'q': (1, 1), 'e': (1, -1)}


class KeyTeleop(Node):

    def __init__(self):
        super().__init__('key_teleop')
        self.declare_parameter('speed', 0.10)          # m/s
        self.declare_parameter('turn', 0.4)            # rad/s
        self.declare_parameter('accel', 0.2)           # m/s2
        self.declare_parameter('angular_accel', 1.0)   # rad/s2
        self.declare_parameter('hold', 0.6)            # s after the last key press before stopping
        p = lambda n: float(self.get_parameter(n).value)  # noqa: E731
        self.speed, self.turn, self.acc, self.aacc, self.hold = p('speed'), p('turn'), p('accel'), \
            p('angular_accel'), p('hold')
        self.pub = self.create_publisher(Twist, '/cmd_vel_teleop', 10)
        self.target = (0.0, 0.0)
        self.v = self.w = 0.0
        self.last_key = 0.0

    def key(self, c):
        if c == ' ':
            self.target, self.v, self.w = (0.0, 0.0), 0.0, 0.0
            self.last_key = 0.0
        elif c in KEYS:
            lin, ang = KEYS[c]
            self.target = (lin * self.speed, ang * self.turn)
            self.last_key = time.time()

    def step(self, dt):
        if time.time() - self.last_key > self.hold:
            self.target = (0.0, 0.0)
        tv, tw = self.target
        self.v += max(-self.acc * dt, min(self.acc * dt, tv - self.v))
        self.w += max(-self.aacc * dt, min(self.aacc * dt, tw - self.w))
        out = Twist()
        out.linear.x, out.angular.z = self.v, self.w
        self.pub.publish(out)

    def stop(self):
        for _ in range(5):
            self.pub.publish(Twist())
            time.sleep(0.05)


def main(args=None):
    if not sys.stdin.isatty():
        sys.exit('key_teleop needs a terminal (ssh -t / docker exec -it)')
    # Ctrl-C raises KeyboardInterrupt here (not rclpy's handler, which would shut ROS down first, so the final
    # stop commands could not be sent)
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = KeyTeleop()
    print(__doc__.split('\n\n')[1], flush=True)
    old = termios.tcgetattr(sys.stdin)
    shown = None
    try:
        tty.setcbreak(sys.stdin.fileno())
        last = time.time()
        while rclpy.ok():
            if select.select([sys.stdin], [], [], 0.05)[0]:
                node.key(sys.stdin.read(1).lower())
            now = time.time()
            node.step(now - last)
            last = now
            state = (round(node.v, 2), round(node.w, 2))
            if state != shown:
                shown = state
                print(f'\r  speed {node.v:+.2f} m/s  turn {math.degrees(node.w):+5.1f} deg/s   ', end='', flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        node.stop()
        print('\nkey_teleop: stopped', flush=True)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

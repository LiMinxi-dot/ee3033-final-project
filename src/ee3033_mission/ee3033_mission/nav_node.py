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
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from .geometry import transform_to_xyyaw, yaw_to_quaternion
from .ros_helpers import LATCHED


class NavStateNode(Node):

    def __init__(self, name, ns, params):
        super().__init__(name)
        params = {'global_frame': 'map', 'robot_frame': 'base_footprint', 'auto_start': False,
                  'nav_timeout': 60.0, 'home_retries': 2, 'home_tolerance': 0.25, **params}
        self.cfg = {n: self.declare_parameter(n, v).value for n, v in params.items()}
        self.frame, self.robot_frame = self.cfg['global_frame'], self.cfg['robot_frame']
        self.ns = ns

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')
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
        self.create_service(Trigger, f'/{ns}/start', self.on_start)
        self.create_service(Trigger, f'/{ns}/stop', self.on_stop)

        self.t_start = None
        self.home = None
        self.home_attempts = 0
        self.goal_handle = None
        self.goal_rejected = False
        self.action_seq = 0
        self.action_deadline = None
        self.action_done_cb = None
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
    def run_action(self, client, goal, timeout, on_done):
        """Send an action goal; on_done(success) is called once (result, rejection or timeout)."""
        self.action_seq += 1
        seq = self.action_seq
        self.action_done_cb = on_done
        self.action_deadline = time.time() + timeout
        self.goal_rejected = False
        future = client.send_goal_async(goal)
        future.add_done_callback(lambda f: self._on_goal_response(f, seq))

    def _on_goal_response(self, future, seq):
        if seq != self.action_seq:
            return
        handle = future.result()
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
        self.goal_handle, self.action_deadline = None, None
        cb(success)

    def cancel_action(self):
        self.action_seq += 1          # results of the old goal are ignored from now on
        self.action_done_cb, self.action_deadline = None, None
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
            self.goal_handle = None

    def send_nav(self, x, y, yaw, on_done):
        goal = NavigateToPose.Goal()
        goal.pose = self.make_pose(x, y, yaw)
        self.run_action(self.nav, goal, float(self.cfg['nav_timeout']), on_done)

    # ---------------------------------------------------------------- return home
    def return_home(self):
        self.home_attempts = 0
        self.set_state('RETURN')
        self.go_home()

    def go_home(self):
        self.home_attempts += 1
        x, y, yaw = self.home
        self.get_logger().info(f'Returning home to ({x:.2f}, {y:.2f}), attempt {self.home_attempts}')
        self.send_nav(x, y, yaw, self.on_home_result)

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
        if self.action_deadline is not None and time.time() > self.action_deadline:
            self.get_logger().warn('Action timed out; cancelling')
            if self.goal_handle is not None:
                self.goal_handle.cancel_goal_async()
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

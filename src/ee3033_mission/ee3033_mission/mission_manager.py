"""Mission state machine for the EE3033 target search.

  WAIT_SYSTEM -> IDLE --(/mission/start)--> LOCALIZE -> SEARCH -> REPORT -> RETURN -> DONE

LOCALIZE  publishes the initial pose for AMCL (the start position is the map
          origin because mapping starts there), records the home pose and
          spins once in place, which both helps AMCL converge and looks for the
          target around the start.
SEARCH    builds viewpoints that cover the free space of the saved map, visits
          them in nearest-neighbour order with Nav2 and spins at each one.
          As soon as the target localizer confirms the target, the current
          Nav2 goal is cancelled.
REPORT    prints/publishes the (x, y) estimate and writes it to a YAML file.
RETURN    navigates back to the home pose and stops.
/mission/stop cancels everything and stops the robot at any time.
"""

import math
import time

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PointStamped, PoseStamped, PoseWithCovarianceStamped, Twist
from nav2_msgs.action import NavigateToPose, Spin
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import String
from std_srvs.srv import Empty, Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from ee3033_interfaces.msg import TargetEstimate

from .geometry import generate_viewpoints, order_nearest_neighbour, yaw_from_quaternion


def yaw_to_quaternion(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class MissionManager(Node):

    def __init__(self):
        super().__init__('mission_manager')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('robot_frame', 'base_footprint')
        self.declare_parameter('auto_start', False)
        self.declare_parameter('set_initial_pose', True)
        self.declare_parameter('initial_pose', [0.0, 0.0, 0.0])   # x, y, yaw in the map
        self.declare_parameter('localize_wait', 2.0)              # s after publishing the initial pose
        self.declare_parameter('initial_spin', True)
        self.declare_parameter('spin_angle_deg', 360.0)
        self.declare_parameter('viewpoint_spacing', 0.6)
        self.declare_parameter('min_clearance', 0.20)
        self.declare_parameter('skip_radius', 0.4)
        self.declare_parameter('max_viewpoints', 40)
        self.declare_parameter('nav_timeout', 60.0)
        self.declare_parameter('spin_timeout', 25.0)
        self.declare_parameter('home_tolerance', 0.25)
        self.declare_parameter('home_retries', 2)
        self.declare_parameter('report_unconfirmed', True)
        self.declare_parameter('result_file', '/home/ros2_ws/mission_result.yaml')

        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.frame, self.robot_frame = p('global_frame'), p('robot_frame')
        self.cfg = {n: p(n) for n in (
            'auto_start', 'set_initial_pose', 'initial_pose', 'localize_wait', 'initial_spin',
            'spin_angle_deg', 'viewpoint_spacing', 'min_clearance', 'skip_radius', 'max_viewpoints',
            'nav_timeout', 'spin_timeout', 'home_tolerance', 'home_retries', 'report_unconfirmed',
            'result_file')}

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.spin = ActionClient(self, Spin, 'spin')

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.map = None
        self.estimate = None
        self.create_subscription(OccupancyGrid, '/map', self.on_map, latched)
        self.create_subscription(TargetEstimate, '/target/estimate', self.on_estimate, latched)
        self.pub_state = self.create_publisher(String, '/mission/state', latched)
        self.pub_target = self.create_publisher(PointStamped, '/mission/target', latched)
        self.pub_init = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_vp = self.create_publisher(MarkerArray, '/mission/viewpoints', latched)
        self.reset_localizer = self.create_client(Empty, '/target_localizer/reset')
        self.create_service(Trigger, '/mission/start', self.on_start)
        self.create_service(Trigger, '/mission/stop', self.on_stop)

        self.state = None
        self.set_state('WAIT_SYSTEM')
        self.t_start = None
        self.home = None
        self.viewpoints = []
        self.vp_index = 0
        self.home_attempts = 0
        self.goal_handle = None
        self.action_seq = 0
        self.action_deadline = None
        self.action_done_cb = None
        self.state_enter = time.time()
        self.create_timer(0.2, self.tick)

    # ---------------------------------------------------------------- helpers
    def set_state(self, s):
        if s != self.state:
            elapsed = f' (t={time.time() - self.t_start:.1f}s)' if getattr(self, 't_start', None) else ''
            self.get_logger().info(f'[state] {self.state} -> {s}{elapsed}')
            self.state = s
            self.state_enter = time.time()
            self.pub_state.publish(String(data=s))

    def robot_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(self.frame, self.robot_frame, Time())
        except TransformException:
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        return t.x, t.y, yaw_from_quaternion(q.x, q.y, q.z, q.w)

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

    # ---------------------------------------------------------------- inputs
    def on_map(self, msg):
        self.map = msg

    def on_estimate(self, msg):
        self.estimate = msg
        if msg.confirmed and self.state in ('LOCALIZE', 'SEARCH'):
            self.get_logger().info('Target confirmed by the localizer; ending the search')
            self.cancel_action()
            self.set_state('REPORT')

    def on_start(self, request, response):
        if self.state != 'IDLE':
            response.success, response.message = False, f'cannot start in state {self.state}'
            return response
        self.begin_mission()
        response.success, response.message = True, 'mission started'
        return response

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
        future = client.send_goal_async(goal)
        future.add_done_callback(lambda f: self._on_goal_response(f, seq))

    def _on_goal_response(self, future, seq):
        if seq != self.action_seq:
            return
        handle = future.result()
        if handle is None or not handle.accepted:
            self.get_logger().warn('Goal rejected')
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

    def send_spin(self, on_done):
        goal = Spin.Goal()
        goal.target_yaw = math.radians(float(self.cfg['spin_angle_deg']))
        goal.time_allowance.sec = int(self.cfg['spin_timeout'])
        self.run_action(self.spin, goal, float(self.cfg['spin_timeout']) + 5.0, on_done)

    def send_nav(self, x, y, yaw, on_done):
        goal = NavigateToPose.Goal()
        goal.pose = self.make_pose(x, y, yaw)
        self.run_action(self.nav, goal, float(self.cfg['nav_timeout']), on_done)

    # ---------------------------------------------------------------- mission
    def begin_mission(self):
        self.t_start = time.time()
        if self.reset_localizer.service_is_ready():
            self.reset_localizer.call_async(Empty.Request())
        if self.cfg['set_initial_pose']:
            x, y, yaw = self.cfg['initial_pose']
            msg = PoseWithCovarianceStamped()
            msg.header.frame_id = self.frame
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.pose.pose = self.make_pose(x, y, yaw).pose
            cov = np.zeros(36)
            cov[0] = cov[7] = 0.05 ** 2
            cov[35] = math.radians(8.0) ** 2
            msg.pose.covariance = cov.tolist()
            for _ in range(3):
                self.pub_init.publish(msg)
            self.get_logger().info(f'Initial pose set to ({x:.2f}, {y:.2f}, {math.degrees(yaw):.0f} deg)')
        self.set_state('LOCALIZE')

    def plan_viewpoints(self):
        m = self.map
        grid = np.array(m.data, dtype=np.int16).reshape(m.info.height, m.info.width)
        origin = (m.info.origin.position.x, m.info.origin.position.y)
        vps = generate_viewpoints(grid, m.info.resolution, origin, self.home[:2],
                                  float(self.cfg['viewpoint_spacing']), float(self.cfg['min_clearance']),
                                  skip_radius=float(self.cfg['skip_radius']))
        here = self.robot_pose() or self.home
        self.viewpoints = order_nearest_neighbour(here[:2], vps)[:int(self.cfg['max_viewpoints'])]
        self.vp_index = 0
        self.get_logger().info(f'Planned {len(self.viewpoints)} viewpoints: '
                               + ', '.join(f'({x:.2f},{y:.2f})' for x, y in self.viewpoints))
        self.publish_viewpoints()

    def publish_viewpoints(self):
        arr = MarkerArray()
        for i, (x, y) in enumerate(self.viewpoints):
            mk = Marker()
            mk.header.frame_id = self.frame
            mk.ns, mk.id, mk.type = 'viewpoints', i, Marker.SPHERE
            mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = x, y, 0.05
            mk.pose.orientation.w = 1.0
            mk.scale.x = mk.scale.y = mk.scale.z = 0.1
            done = i < self.vp_index
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = (0.5, 0.5, 0.5, 0.6) if done else (0.2, 0.4, 1.0, 0.9)
            arr.markers.append(mk)
        self.pub_vp.publish(arr)

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
        self.get_logger().info(f'Viewpoint {self.vp_index + 1}/{len(self.viewpoints)} -> ({x:.2f}, {y:.2f})')
        self.send_nav(x, y, yaw, self.on_viewpoint_reached)

    def on_viewpoint_reached(self, success):
        if not success:
            self.get_logger().warn(f'Could not reach viewpoint {self.vp_index + 1}; spinning here instead')
        self.send_spin(self.on_viewpoint_spun)

    def on_viewpoint_spun(self, success):
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
        self.get_logger().info('=' * 70)
        self.get_logger().info(banner)
        self.get_logger().info('=' * 70)
        try:
            with open(self.cfg['result_file'], 'w') as f:
                f.write(f'# EE3033 mission result, frame {self.frame}\n')
                for k, v in result.items():
                    f.write(f'{k}: {v}\n')
        except OSError as e:
            self.get_logger().warn(f'Cannot write {self.cfg["result_file"]}: {e}')

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
            self.get_logger().info(f'Home reached ({dist:.2f} m from start). Mission time '
                                   f'{time.time() - self.t_start:.1f}s')
            self.set_state('DONE')
        elif self.home_attempts <= int(self.cfg['home_retries']):
            self.go_home()
        else:
            self.stop_robot()
            self.get_logger().error(f'Could not return home ({dist:.2f} m away)')
            self.set_state('FAILED')

    # ---------------------------------------------------------------- tick
    def tick(self):
        if self.action_deadline is not None and time.time() > self.action_deadline:
            seq = self.action_seq
            self.get_logger().warn('Action timed out; cancelling')
            if self.goal_handle is not None:
                self.goal_handle.cancel_goal_async()
            self._finish_action(seq, False)

        s = self.state
        if s == 'WAIT_SYSTEM':
            ready = (self.map is not None and self.nav.server_is_ready() and self.spin.server_is_ready())
            if ready:
                self.set_state('IDLE')
                self.get_logger().info('System ready. Start with: ros2 service call /mission/start std_srvs/srv/Trigger')
            elif time.time() - self.state_enter > 10.0:
                self.state_enter = time.time()
                self.get_logger().info(f'Waiting: map={self.map is not None} nav={self.nav.server_is_ready()} '
                                       f'spin={self.spin.server_is_ready()}')
        elif s == 'IDLE':
            if self.cfg['auto_start']:
                self.begin_mission()
        elif s == 'LOCALIZE':
            if self.home is None and time.time() - self.state_enter >= float(self.cfg['localize_wait']):
                self.home = self.robot_pose()
                if self.home is None:
                    self.get_logger().warn('No map->robot TF yet', throttle_duration_sec=2.0)
                    return
                self.get_logger().info('Home pose ({:.2f}, {:.2f}, {:.0f} deg)'.format(
                    self.home[0], self.home[1], math.degrees(self.home[2])))
                if self.cfg['initial_spin']:
                    self.send_spin(lambda ok: self.start_search())
                else:
                    self.start_search()
        elif s == 'REPORT':
            self.report()
            if self.home is None:   # target confirmed before the home pose was recorded
                self.home = tuple(float(v) for v in self.cfg['initial_pose'])
            self.home_attempts = 0
            self.set_state('RETURN')
            self.go_home()

    def start_search(self):
        if self.state != 'LOCALIZE':
            return
        self.set_state('SEARCH')
        self.plan_viewpoints()
        self.next_viewpoint()


def main(args=None):
    rclpy.init(args=args)
    node = MissionManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

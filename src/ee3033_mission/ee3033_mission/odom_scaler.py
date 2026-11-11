"""Odometry with the base's turn (and distance) over-reporting scaled out.

balance_robot_node integrates the STM32's velocities without a scale parameter, and our car over-reports turns:
true / odometry turn 0.976 to the left and 0.956 to the right (scan-to-map heading over 630 two-second windows of two
site missions, 11-11; scan-to-scan ICP of three mapping runs on 11-06: 0.941-0.961 both ways). After every
in-place turn AMCL's heading was then 8-15 deg off and needed 5-15 s of driving to recover (mission 1111_045745),
and SLAM drew the walls thicker after turns. base_sensors.launch.py starts the base with its frames renamed
(odom_raw -> base_raw) and its /odom remapped to /odom_raw; this node re-integrates the raw pose steps with the
turns scaled by yaw_scale (left) / yaw_scale_cw (right) and the shifts by lin_scale (geometry.scale_odom_step),
and publishes odom_combined -> base_footprint (TF) and /odom (same stamps, twist scaled) as the base did.
"""

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster

from .geometry import scale_odom_step, yaw_from_quaternion, yaw_to_quaternion


class OdomScaler(Node):

    def __init__(self):
        super().__init__('odom_scaler')
        self.declare_parameter('yaw_scale', 1.0)
        self.declare_parameter('yaw_scale_cw', 1.0)
        self.declare_parameter('lin_scale', 1.0)
        self.declare_parameter('odom_frame', 'odom_combined')
        self.declare_parameter('robot_frame', 'base_footprint')
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.k_yaw, self.k_cw, self.k_lin = float(p('yaw_scale')), float(p('yaw_scale_cw')), float(p('lin_scale'))
        self.odom_frame, self.robot_frame = p('odom_frame'), p('robot_frame')
        self.raw = None
        self.pose = (0.0, 0.0, 0.0)
        self.tf = TransformBroadcaster(self)
        self.pub = self.create_publisher(Odometry, '/odom', 50)
        self.create_subscription(Odometry, '/odom_raw', self.on_raw, 50)
        self.get_logger().info(f'/odom_raw -> /odom + TF {self.odom_frame} -> {self.robot_frame}: '
                               f'left turns x {self.k_yaw:.3f}, right turns x {self.k_cw:.3f}, '
                               f'distances x {self.k_lin:.3f}')

    def on_raw(self, msg):
        q = msg.pose.pose.orientation
        raw = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw_from_quaternion(q.x, q.y, q.z, q.w))
        if self.raw is None:
            self.pose = raw
        else:
            self.pose = scale_odom_step(self.raw, raw, self.pose, self.k_yaw, self.k_lin, self.k_cw)
        self.raw = raw
        x, y, yaw = self.pose
        qx, qy, qz, qw = yaw_to_quaternion(yaw)

        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id, t.child_frame_id = self.odom_frame, self.robot_frame
        t.transform.translation.x, t.transform.translation.y = x, y
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = qx, qy, qz, qw
        self.tf.sendTransform(t)

        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id, out.child_frame_id = self.odom_frame, self.robot_frame
        out.pose.pose.position.x, out.pose.pose.position.y = x, y
        out.pose.pose.orientation = t.transform.rotation
        out.pose.covariance = msg.pose.covariance
        out.twist.twist.linear.x = self.k_lin * msg.twist.twist.linear.x
        out.twist.twist.linear.y = self.k_lin * msg.twist.twist.linear.y
        w = msg.twist.twist.angular.z
        out.twist.twist.angular.z = (self.k_yaw if w >= 0.0 else self.k_cw) * w
        out.twist.covariance = msg.twist.covariance
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = OdomScaler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

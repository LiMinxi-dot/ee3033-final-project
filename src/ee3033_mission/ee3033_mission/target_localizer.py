"""Estimate the target position in the map frame from camera detections and LiDAR.

For each detection:
  1. The bounding-box centre column gives the bearing of the target in the
     camera frame (pinhole model: camera_info K if calibrated, else the
     configured horizontal field of view).
  2. TF at the image time gives the camera pose in the map, so the bearing
     becomes a ray in the map.
  3. The LiDAR scan closest in time is transformed into the map; the nearest
     hits around the ray give the range. If the LiDAR sees nothing there, an
     optional known object height gives a monocular range instead.
  4. Range + target radius along the ray gives one observation.
All observations are fused robustly (largest consistent cluster, median) and
published as ee3033_interfaces/TargetEstimate plus RViz markers.
"""

import math
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, LaserScan
from std_srvs.srv import Empty
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from ee3033_interfaces.msg import TargetDetectionArray, TargetEstimate

from .geometry import (apply_body_offset, focal_length_px, fuse_observations, pixel_to_bearing,
                       range_from_object_height, ray_range_from_scan, yaw_from_quaternion)


def transform_to_xyyaw(tf):
    t, q = tf.transform.translation, tf.transform.rotation
    return t.x, t.y, yaw_from_quaternion(q.x, q.y, q.z, q.w)


class TargetLocalizer(Node):

    def __init__(self):
        super().__init__('target_localizer')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('scan_topic', '/scan_fixed')
        self.declare_parameter('camera_info_topic', '/camera_info')
        self.declare_parameter('hfov_deg', 62.0)
        # Correction of the measured camera mount relative to the vendor TF camera_link
        # (x forward, y left, yaw counter-clockwise); keeps the vendor launch files untouched.
        self.declare_parameter('camera_offset_x', 0.0)
        self.declare_parameter('camera_offset_y', 0.0)
        self.declare_parameter('camera_offset_yaw_deg', 0.0)
        self.declare_parameter('use_camera_info', True)
        self.declare_parameter('min_score', 0.5)
        self.declare_parameter('min_range', 0.15)
        self.declare_parameter('max_range', 4.0)
        self.declare_parameter('ray_half_width_deg', 2.0)
        self.declare_parameter('cluster_depth', 0.15)
        self.declare_parameter('target_radius', 0.04)
        self.declare_parameter('target_height', 0.0)      # m; > 0 enables the monocular fallback
        self.declare_parameter('max_scan_age', 0.25)      # s between image and scan stamps
        self.declare_parameter('cluster_radius', 0.30)
        self.declare_parameter('min_observations', 4)
        self.declare_parameter('max_spread', 0.20)

        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.frame = p('global_frame')
        self.hfov = math.radians(float(p('hfov_deg')))
        self.cam_offset = (float(p('camera_offset_x')), float(p('camera_offset_y')),
                           math.radians(float(p('camera_offset_yaw_deg'))))
        self.use_camera_info = bool(p('use_camera_info'))
        self.min_score = float(p('min_score'))
        self.min_range, self.max_range = float(p('min_range')), float(p('max_range'))
        self.half_width = math.radians(float(p('ray_half_width_deg')))
        self.cluster_depth = float(p('cluster_depth'))
        self.target_radius = float(p('target_radius'))
        self.target_height = float(p('target_height'))
        self.max_scan_age = float(p('max_scan_age'))
        self.cluster_radius = float(p('cluster_radius'))
        self.min_obs = int(p('min_observations'))
        self.max_spread = float(p('max_spread'))

        self.K = None
        self.scans = deque(maxlen=20)
        self.observations = []   # (x, y, class_name, source)
        self.confirmed_logged = False

        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(LaserScan, p('scan_topic'), self.scans.append, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, p('camera_info_topic'), self.on_camera_info, 10)
        self.create_subscription(TargetDetectionArray, '/target_detector/detections', self.on_detections, 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_est = self.create_publisher(TargetEstimate, '/target/estimate', latched)
        self.pub_markers = self.create_publisher(MarkerArray, '/target/markers', 10)
        self.create_service(Empty, '/target_localizer/reset', self.on_reset)
        self.get_logger().info(f'Target localizer ready (frame {self.frame}, hfov {math.degrees(self.hfov):.0f} deg)')

    # ---------------------------------------------------------------- inputs
    def on_camera_info(self, msg):
        k = np.array(msg.k, dtype=float).reshape(3, 3)
        if self.use_camera_info and k[0, 0] > 1.0:
            if self.K is None:
                self.get_logger().info(f'Using calibrated intrinsics fx={k[0, 0]:.1f} cx={k[0, 2]:.1f}')
            self.K = k

    def on_reset(self, request, response):
        self.observations.clear()
        self.confirmed_logged = False
        self.get_logger().info('Observations cleared')
        return response

    def closest_scan(self, stamp):
        t = Time.from_msg(stamp).nanoseconds
        best = min(self.scans, key=lambda s: abs(Time.from_msg(s.header.stamp).nanoseconds - t), default=None)
        if best is None or abs(Time.from_msg(best.header.stamp).nanoseconds - t) > self.max_scan_age * 1e9:
            return None
        return best

    def lookup(self, target, source, stamp):
        """TF at the given time, falling back to the latest transform."""
        try:
            return self.tf_buffer.lookup_transform(target, source, Time.from_msg(stamp), Duration(seconds=0.1))
        except TransformException:
            try:
                return self.tf_buffer.lookup_transform(target, source, Time())
            except TransformException as e:
                self.get_logger().warn(f'TF {target}<-{source}: {e}', throttle_duration_sec=5.0)
                return None

    def scan_points_in_frame(self, scan):
        tf = self.lookup(self.frame, scan.header.frame_id, scan.header.stamp)
        if tf is None:
            return None
        x0, y0, yaw = transform_to_xyyaw(tf)
        r = np.asarray(scan.ranges, dtype=float)
        a = scan.angle_min + scan.angle_increment * np.arange(r.size)
        ok = np.isfinite(r) & (r > scan.range_min) & (r < scan.range_max)
        r, a = r[ok], a[ok] + yaw
        return np.column_stack((x0 + r * np.cos(a), y0 + r * np.sin(a)))

    # ---------------------------------------------------------------- core
    def on_detections(self, msg):
        dets = [d for d in msg.detections if d.score >= self.min_score]
        if not dets:
            return
        cam_tf = self.lookup(self.frame, msg.header.frame_id, msg.header.stamp)
        if cam_tf is None:
            return
        ox, oy, cam_yaw = apply_body_offset(*transform_to_xyyaw(cam_tf), *self.cam_offset)

        if self.K is not None:
            fx, cx = self.K[0, 0], self.K[0, 2]
        else:
            fx, cx = focal_length_px(msg.image_width, self.hfov), msg.image_width / 2.0

        scan = self.closest_scan(msg.header.stamp)
        points = self.scan_points_in_frame(scan) if scan is not None else None

        det = max(dets, key=lambda d: d.score)  # one target: use the most confident box
        bearing = pixel_to_bearing(det.center_u, cx, fx)
        ray_yaw = cam_yaw + bearing
        # Search window: the inner half of the box, clipped to a sensible angular range.
        half = min(max(self.half_width, 0.25 * det.width / fx), math.radians(6.0))

        rng, source = None, ''
        if points is not None:
            rng, n = ray_range_from_scan((ox, oy), ray_yaw, points, half,
                                         self.min_range, self.max_range, self.cluster_depth)
            source = f'lidar({n})' if rng is not None else ''
        if rng is None and self.target_height > 0.0:
            rng = range_from_object_height(det.height, self.target_height, fx)
            source = 'size'
        if rng is None or not (self.min_range <= rng <= self.max_range):
            self.get_logger().info(f'{det.class_name} at bearing {math.degrees(bearing):+.0f} deg: no range',
                                   throttle_duration_sec=2.0)
            return

        d = rng + self.target_radius
        x, y = ox + d * math.cos(ray_yaw), oy + d * math.sin(ray_yaw)
        self.observations.append((x, y, det.class_name, source))
        self.get_logger().info(
            f'obs #{len(self.observations)} {det.class_name} ({det.score:.2f}) bearing '
            f'{math.degrees(bearing):+.1f} deg range {rng:.2f} m [{source}] -> ({x:.2f}, {y:.2f})')
        self.publish_estimate(msg.header.stamp)

    def publish_estimate(self, stamp):
        pts = [(o[0], o[1]) for o in self.observations]
        est, n_support, spread = fuse_observations(pts, self.cluster_radius)
        if est is None:
            return
        confirmed = n_support >= self.min_obs and spread <= self.max_spread
        out = TargetEstimate()
        out.header.stamp = stamp
        out.header.frame_id = self.frame
        out.position.x, out.position.y = est
        out.class_name = self.observations[-1][2]
        out.num_observations = n_support
        out.total_observations = len(self.observations)
        out.spread = spread
        out.confirmed = confirmed
        self.pub_est.publish(out)
        if confirmed and not self.confirmed_logged:
            self.confirmed_logged = True
            self.get_logger().info(
                f'TARGET CONFIRMED at ({est[0]:.2f}, {est[1]:.2f}) from {n_support} observations, spread {spread:.2f} m')
        self.publish_markers(est, confirmed, stamp)

    def publish_markers(self, est, confirmed, stamp):
        arr = MarkerArray()
        pts = Marker()
        pts.header.frame_id, pts.header.stamp = self.frame, stamp
        pts.ns, pts.id, pts.type = 'observations', 0, Marker.POINTS
        pts.scale.x = pts.scale.y = 0.04
        pts.color.r, pts.color.g, pts.color.b, pts.color.a = 1.0, 0.6, 0.0, 0.8
        for x, y, *_ in self.observations:
            pts.points.append(Point(x=x, y=y, z=0.05))
        arr.markers.append(pts)

        m = Marker()
        m.header.frame_id, m.header.stamp = self.frame, stamp
        m.ns, m.id, m.type = 'estimate', 1, Marker.CYLINDER
        m.pose.position.x, m.pose.position.y, m.pose.position.z = est[0], est[1], 0.1
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = 0.15
        m.scale.z = 0.2
        m.color.g = 1.0 if confirmed else 0.5
        m.color.r = 0.0 if confirmed else 1.0
        m.color.a = 0.8
        arr.markers.append(m)

        txt = Marker()
        txt.header = m.header
        txt.ns, txt.id, txt.type = 'estimate_label', 2, Marker.TEXT_VIEW_FACING
        txt.pose.position.x, txt.pose.position.y, txt.pose.position.z = est[0], est[1], 0.45
        txt.pose.orientation.w = 1.0
        txt.scale.z = 0.12
        txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
        txt.text = f'target ({est[0]:.2f}, {est[1]:.2f})' + (' OK' if confirmed else '?')
        arr.markers.append(txt)
        self.pub_markers.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = TargetLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

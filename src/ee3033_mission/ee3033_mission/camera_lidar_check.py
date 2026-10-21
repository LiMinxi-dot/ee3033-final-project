"""Camera/LiDAR yaw calibration for the target localizer.

Put ONE object that YOLO detects (one of target_classes) about 1-1.5 m in front
of the standing car, with nothing else within about 15 deg of it, and the
object's middle at the LiDAR height (about 19 cm). With the base, LiDAR, resampler,
camera and target_detector running:

  ros2 run ee3033_mission camera_lidar_check

For each detection the camera bearing is compared with the bearing of the
nearest LiDAR cluster around it, seen from the camera position. The median
difference is the camera_offset_yaw_deg to put in mission.yaml. An overlay of
the LiDAR points on the camera image (current vs suggested offset) is saved so
the result can be checked by eye.
"""

import math
import os

import cv2
import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from tf2_ros import Buffer, TransformException, TransformListener

from ee3033_interfaces.msg import TargetDetectionArray

from .geometry import apply_body_offset, focal_length_px, nearest_cluster_bearing, pixel_to_bearing, wrap_angle
from .target_detector import image_msg_to_bgr
from .target_localizer import transform_to_xyyaw


class CameraLidarCheck(Node):

    def __init__(self):
        super().__init__('camera_lidar_check')
        # Defaults: the target_localizer section of the installed mission.yaml (current calibration).
        cfg = os.path.join(get_package_share_directory('ee3033_mission'), 'config', 'mission.yaml')
        loc = yaml.safe_load(open(cfg))['target_localizer']['ros__parameters']
        for name, default in (('hfov_deg', 62.0), ('camera_offset_x', 0.0), ('camera_offset_y', 0.0),
                              ('camera_offset_yaw_deg', 0.0), ('use_camera_info', True),
                              ('scan_topic', '/scan_fixed'), ('camera_info_topic', '/camera_info'),
                              ('samples', 30), ('search_half_width_deg', 15.0),
                              ('image_topic', '/camera/image_raw'), ('output', '/tmp/camera_lidar_check.png')):
            self.declare_parameter(name, type(default)(loc.get(name, default)))
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.hfov = math.radians(float(p('hfov_deg')))
        self.offset = (float(p('camera_offset_x')), float(p('camera_offset_y')))
        self.current_yaw = float(p('camera_offset_yaw_deg'))
        self.samples = int(p('samples'))
        self.half = math.radians(float(p('search_half_width_deg')))
        self.output = p('output')
        self.use_camera_info = bool(p('use_camera_info'))

        self.K = None
        self.scan = None
        self.image = None
        self.diffs, self.ranges = [], []
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(LaserScan, p('scan_topic'), self.on_scan, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, p('camera_info_topic'), self.on_info, 10)
        self.create_subscription(Image, p('image_topic'), self.on_image,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_subscription(TargetDetectionArray, '/target_detector/detections', self.on_detections, 10)
        self.done = False
        self.get_logger().info(f'Collecting {self.samples} detections; current camera_offset_yaw_deg = '
                               f'{self.current_yaw:.1f}')

    def on_scan(self, msg):
        self.scan = msg

    def on_image(self, msg):
        self.image = msg

    def on_info(self, msg):
        k = np.array(msg.k, dtype=float).reshape(3, 3)
        if self.use_camera_info and k[0, 0] > 1.0:
            self.K = k

    def camera_in_laser(self, cam_frame):
        """Camera pose in the LiDAR frame (with the measured x/y mount offset, without yaw correction)."""
        tf = self.tf_buffer.lookup_transform(self.scan.header.frame_id, cam_frame, Time())
        x, y, yaw = transform_to_xyyaw(tf)
        return apply_body_offset(x, y, yaw, self.offset[0], self.offset[1], 0.0)

    def scan_points(self):
        r = np.asarray(self.scan.ranges, dtype=float)
        a = self.scan.angle_min + self.scan.angle_increment * np.arange(r.size)
        ok = np.isfinite(r) & (r > self.scan.range_min) & (r < self.scan.range_max)
        return np.column_stack((r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])))

    def intrinsics(self, width):
        if self.K is not None:
            return self.K[0, 0], self.K[0, 2]
        return focal_length_px(width, self.hfov), width / 2.0

    def on_detections(self, msg):
        if self.done or self.scan is None or not msg.detections:
            return
        try:
            cx_l, cy_l, cam_yaw = self.camera_in_laser(msg.header.frame_id)
        except TransformException as e:
            self.get_logger().warn(f'TF: {e}', throttle_duration_sec=5.0)
            return
        det = max(msg.detections, key=lambda d: d.score)
        fx, cx = self.intrinsics(msg.image_width)
        b_cam = pixel_to_bearing(det.center_u, cx, fx)
        bearing, rng, n = nearest_cluster_bearing((cx_l, cy_l), cam_yaw + b_cam, self.scan_points(), self.half)
        if bearing is None:
            self.get_logger().info('No LiDAR object near the detection', throttle_duration_sec=2.0)
            return
        diff = math.degrees(float(wrap_angle(bearing - cam_yaw - b_cam)))
        self.diffs.append(diff)
        self.ranges.append(rng)
        self.get_logger().info(f'[{len(self.diffs)}/{self.samples}] {det.class_name}: camera '
                               f'{math.degrees(b_cam):+.1f} deg, LiDAR {math.degrees(bearing - cam_yaw):+.1f} deg '
                               f'at {rng:.2f} m ({n} pts) -> offset {diff:+.1f} deg')
        if len(self.diffs) >= self.samples:
            self.finish()

    def finish(self):
        self.done = True
        d = np.array(self.diffs)
        med, spread = float(np.median(d)), float(np.median(np.abs(d - np.median(d))))
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'Suggested camera_offset_yaw_deg: {med:.1f}  (current {self.current_yaw:.1f}; '
                               f'spread {spread:.1f} deg, object at {np.median(self.ranges):.2f} m)')
        if spread > 1.5:
            self.get_logger().warn('Large spread: is there more than one object near the target?')
        self.save_overlay(med)
        self.get_logger().info('=' * 60)
        raise SystemExit

    def save_overlay(self, suggested):
        if self.image is None:
            return
        try:
            img = image_msg_to_bgr(self.image)
            cx_l, cy_l, cam_yaw = self.camera_in_laser('camera_link')
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'No overlay: {e}')
            return
        pts = self.scan_points() - np.array([cx_l, cy_l])
        rng = np.hypot(pts[:, 0], pts[:, 1])
        fx, cx = self.intrinsics(img.shape[1])
        panels = []
        for yaw_deg, label in ((self.current_yaw, 'current'), (suggested, 'suggested')):
            im = img.copy()
            b = wrap_angle(np.arctan2(pts[:, 1], pts[:, 0]) - cam_yaw - math.radians(yaw_deg))
            for bb, rr in zip(b, rng):
                if abs(bb) < math.radians(40) and 0.15 < rr < 4.0:
                    u = int(cx - fx * math.tan(bb))
                    v = int(np.clip(img.shape[0] * 0.9 - 90 * rr, 10, img.shape[0] - 10))
                    t = min(rr / 3.0, 1.0)
                    cv2.circle(im, (u, v), 4, (0, int(255 * t), int(255 * (1 - t))), -1)
            cv2.putText(im, f'{label}: yaw offset {yaw_deg:+.1f} deg', (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            panels.append(im)
        cv2.imwrite(self.output, np.hstack(panels))
        self.get_logger().info(f'Overlay saved to {self.output} (LiDAR points: red = near, green = far; '
                               'they should sit on the objects in the right panel)')


def main(args=None):
    rclpy.init(args=args)
    node = CameraLidarCheck()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

"""YOLO object detector for the target search.

Subscribes to the camera image (raw on the car, or compressed when run on another
machine), runs an Ultralytics YOLO model (PyTorch .pt or an exported NCNN folder)
on the newest frame at a fixed rate, and publishes the detections of the target
classes as ee3033_interfaces/TargetDetectionArray. An annotated JPEG is
published at a low rate for viewing in rqt/rviz from the VM.
"""

import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image

from ee3033_interfaces.msg import TargetDetection, TargetDetectionArray


def image_msg_to_bgr(msg):
    """sensor_msgs/Image -> BGR numpy array without cv_bridge."""
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    enc = msg.encoding.lower()
    if enc in ('rgb8', 'bgr8'):
        img = buf.reshape(msg.height, msg.step // 3, 3)[:, :msg.width]
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if enc == 'rgb8' else img
    if enc in ('yuv422_yuy2', 'yuyv', 'yuv422'):
        img = buf.reshape(msg.height, msg.step // 2, 2)[:, :msg.width]
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YUYV)
    if enc == 'mono8':
        return cv2.cvtColor(buf.reshape(msg.height, msg.step)[:, :msg.width], cv2.COLOR_GRAY2BGR)
    raise ValueError(f'unsupported image encoding {msg.encoding}')


class TargetDetector(Node):

    def __init__(self):
        super().__init__('target_detector')
        self.declare_parameter('model_path', '/home/ros2_ws/models/yolov8n.pt')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('use_compressed', False)
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('target_classes', ['bottle'])   # empty list = publish every class
        self.declare_parameter('conf_threshold', 0.45)
        self.declare_parameter('imgsz', 320)
        self.declare_parameter('max_rate', 10.0)
        self.declare_parameter('debug_image', True)
        self.declare_parameter('debug_rate', 2.0)

        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.camera_frame = p('camera_frame')
        self.conf = float(p('conf_threshold'))
        self.imgsz = int(p('imgsz'))
        self.debug = bool(p('debug_image'))
        self.debug_period = 1.0 / max(0.1, float(p('debug_rate')))
        self.last_debug = 0.0

        model_path = self.ensure_model(p('model_path'))
        from ultralytics import YOLO  # heavy import, keep it after the parameters are logged
        self.get_logger().info(f'Loading YOLO model {model_path} (imgsz={self.imgsz})')
        self.model = YOLO(model_path, task='detect')
        names = self.model.names  # {id: name}
        wanted = [c.strip() for c in p('target_classes') if c.strip()]
        unknown = [c for c in wanted if c not in names.values()]
        if unknown:
            self.get_logger().error(f'Unknown target classes {unknown}; model classes: {sorted(names.values())}')
        self.class_ids = [i for i, n in names.items() if n in wanted] or None
        self.get_logger().info(f'Target classes: {wanted or "ALL"} -> ids {self.class_ids}')

        self.min_period = 1.0 / max(0.5, float(p('max_rate')))
        self.last_process = 0.0
        topic = p('image_topic')
        # Depth 1: while a frame is being processed only the newest one is kept, so frames never queue up.
        newest_only = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        if p('use_compressed'):
            self.create_subscription(CompressedImage, topic + '/compressed', self.on_image, newest_only)
        else:
            self.create_subscription(Image, topic, self.on_image, newest_only)
        self.pub = self.create_publisher(TargetDetectionArray, '/target_detector/detections', 10)
        self.pub_dbg = self.create_publisher(CompressedImage, '/target_detector/debug_image/compressed', 1)
        self.n_frames = 0
        self.t_infer = 0.0
        self.create_timer(10.0, self.report_rate)

    def ensure_model(self, model_path):
        """Return a usable model path; download an official Ultralytics model if it is missing.

        Ultralytics downloads known model names (yolov8n.pt, yolo11n.pt, ...) into the current
        directory, so the download runs inside the model's target folder.
        """
        if os.path.exists(model_path):
            return model_path
        folder, name = os.path.split(os.path.abspath(model_path))
        self.get_logger().warn(f'{model_path} not found; downloading {name} (needs internet on the car)')
        try:
            os.makedirs(folder, exist_ok=True)
            cwd = os.getcwd()
            os.chdir(folder)
            try:
                from ultralytics import YOLO
                YOLO(name)
            finally:
                os.chdir(cwd)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f'Cannot load or download the YOLO model {model_path}: {e}. Copy {name} there by hand, '
                'e.g. scp yolov8n.pt to the car.') from e
        if not os.path.exists(model_path):
            raise RuntimeError(f'Download did not produce {model_path}')
        self.get_logger().info(f'Downloaded {model_path}')
        return model_path

    def on_image(self, msg):
        now = time.time()
        if now - self.last_process < self.min_period:
            return
        self.last_process = now
        self.process(msg)

    def process(self, msg):
        try:
            if isinstance(msg, CompressedImage):
                img = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
            else:
                img = image_msg_to_bgr(msg)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'Cannot decode image: {e}', throttle_duration_sec=5.0)
            return

        t0 = time.time()
        result = self.model.predict(img, imgsz=self.imgsz, conf=self.conf,
                                    classes=self.class_ids, verbose=False)[0]
        self.t_infer += time.time() - t0
        self.n_frames += 1

        out = TargetDetectionArray()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.camera_frame
        out.image_height, out.image_width = img.shape[:2]
        for box in result.boxes:
            cx, cy, w, h = (float(v) for v in box.xywh[0])
            det = TargetDetection()
            det.class_id = int(box.cls[0])
            det.class_name = self.model.names[det.class_id]
            det.score = float(box.conf[0])
            det.center_u, det.center_v, det.width, det.height = cx, cy, w, h
            out.detections.append(det)
        self.pub.publish(out)

        now = time.time()
        if self.debug and now - self.last_debug >= self.debug_period and self.pub_dbg.get_subscription_count() > 0:
            self.last_debug = now
            ok, jpg = cv2.imencode('.jpg', result.plot(), [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                dbg = CompressedImage()
                dbg.header = out.header
                dbg.format = 'jpeg'
                dbg.data = jpg.tobytes()
                self.pub_dbg.publish(dbg)

    def report_rate(self):
        if self.n_frames:
            self.get_logger().info(
                f'{self.n_frames / 10.0:.1f} frames/s, inference {1000 * self.t_infer / self.n_frames:.0f} ms/frame')
        else:
            self.get_logger().warn('No camera frames received in the last 10 s')
        self.n_frames, self.t_infer = 0, 0.0


def main(args=None):
    rclpy.init(args=args)
    node = TargetDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

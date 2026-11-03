"""Exit as soon as a TF transform is available (used by the launch files to order the bringup).

  ros2 run ee3033_mission wait_for_transform --ros-args -p target_frame:=map -p source_frame:=base_footprint

Nav2 aborts its whole bringup if the global costmap cannot get map -> base_footprint while
it is being activated, and SLAM Toolbox (or AMCL) only publishes map -> odom after a few
scans (or after the initial pose). The launch files start Nav2 when this node exits.
"""

import time

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener


def main(args=None):
    rclpy.init(args=args)
    node = Node('wait_for_transform')
    target = node.declare_parameter('target_frame', 'map').value
    source = node.declare_parameter('source_frame', 'base_footprint').value
    tf = TransformListener(Buffer(), node)
    log, t0 = node.get_logger(), time.time()
    try:
        while rclpy.ok() and not tf.buffer.can_transform(target, source, Time()):
            log.info(f'Waiting for {target} -> {source} ({time.time() - t0:.0f}s)', throttle_duration_sec=5.0)
            rclpy.spin_once(node, timeout_sec=0.2)
        if rclpy.ok():
            log.info(f'{target} -> {source} available after {time.time() - t0:.1f}s')
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

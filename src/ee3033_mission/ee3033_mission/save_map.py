"""Save the current /map (SLAM Toolbox) as <path>.pgm / <path>.yaml, the way auto_mapper does.

  ros2 run ee3033_mission save_map [path]       # default /home/ros2_ws/maps/sandbox

Used by demo.sh manual. /slam_toolbox/save_map runs Nav2's map_saver, which waits only 2 s for /map (11-12: "Failed
to spin map subscription", result 255) and writes free_thresh 0.25, at which map_server reads unknown cells as free
(see exploration.save_map_files). This waits up to `timeout` s and gets the latched map at once.
"""

import sys
import time

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node

from .exploration import save_map_files
from .geometry import yaw_from_quaternion
from .ros_helpers import LATCHED


def main(args=None):
    rclpy.init(args=args)
    argv = rclpy.utilities.remove_ros_args(sys.argv)[1:]
    path = argv[0] if argv else '/home/ros2_ws/maps/sandbox'
    node = Node('save_map')
    timeout = float(node.declare_parameter('timeout', 15.0).value)
    got = []
    node.create_subscription(OccupancyGrid, '/map', got.append, LATCHED)
    t0 = time.time()
    while not got and time.time() - t0 < timeout:
        rclpy.spin_once(node, timeout_sec=0.2)
    if not got:
        print(f'save_map: no /map within {timeout:.0f} s: NOT saved', flush=True)
        rclpy.try_shutdown()
        sys.exit(1)
    m = got[-1]
    grid = np.array(m.data, dtype=np.int16).reshape(m.info.height, m.info.width)
    q = m.info.origin.orientation
    origin = (m.info.origin.position.x, m.info.origin.position.y, yaw_from_quaternion(q.x, q.y, q.z, q.w))
    yml = save_map_files(path, grid, m.info.resolution, origin)
    free = int(((grid >= 0) & (grid < 50)).sum()) * m.info.resolution ** 2
    print(f'save_map: {yml} ({grid.shape[1]}x{grid.shape[0]} cells, free {free:.1f} m2)', flush=True)
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()

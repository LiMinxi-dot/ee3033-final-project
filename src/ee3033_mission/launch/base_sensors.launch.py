"""Robot base, LiDAR and scan resampler (shared by the mapping and mission launches).

The resampler is started with `ros2 run balance_mapping scan_resampler.py`
because the vendor bringup points to /home/ros2_ws/scan_resampler.py, which
does not exist on the car.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    base = IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(
        get_package_share_directory('balance_robot_base'), 'launch', 'balance_robot_base.launch.py')))
    lidar = IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(
        get_package_share_directory('lslidar_driver_n10p'), 'launch', 'lslidar_launch.py')))
    resampler = Node(
        package='balance_mapping', executable='scan_resampler.py', name='scan_resampler', output='screen',
        parameters=[{'input_topic': '/scan', 'output_topic': '/scan_fixed', 'number_of_bins': 450}])

    return LaunchDescription([
        base,
        TimerAction(period=2.0, actions=[LogInfo(msg='[ee3033] starting LiDAR'), lidar]),
        TimerAction(period=4.0, actions=[LogInfo(msg='[ee3033] starting scan resampler'), resampler]),
    ])

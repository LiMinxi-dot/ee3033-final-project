"""Camera, YOLO target detector and target localizer.

Can be run on its own (with base_sensors.launch.py) to test detection and
localization while the robot stands still, e.g. with global_frame:=odom_combined
when no map is available.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(get_package_share_directory('ee3033_mission'), 'config', 'mission.yaml')
    params = LaunchConfiguration('params_file')
    frame = LaunchConfiguration('global_frame')

    camera = Node(
        package='v4l2_camera', executable='v4l2_camera_node', name='v4l2_camera', output='screen',
        condition=IfCondition(LaunchConfiguration('start_camera')),
        remappings=[('image_raw', '/camera/image_raw')],
        parameters=[{'video_device': LaunchConfiguration('video_device'), 'image_size': [640, 480],
                     'camera_frame_id': 'camera_link'}])
    detector = Node(package='ee3033_mission', executable='target_detector', output='screen',
                    parameters=[params])
    localizer = Node(package='ee3033_mission', executable='target_localizer', output='screen',
                     parameters=[params, {'global_frame': frame}])

    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('global_frame', default_value='map'),
        DeclareLaunchArgument('video_device', default_value='/dev/video0'),
        DeclareLaunchArgument('start_camera', default_value='true'),
        camera, detector, localizer,
    ])

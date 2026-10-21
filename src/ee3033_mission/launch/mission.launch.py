"""Step 2: autonomous search mission on the saved map.

  ros2 launch ee3033_mission mission.launch.py map:=/home/ros2_ws/maps/sandbox.yaml
  ros2 service call /mission/start std_srvs/srv/Trigger     # when the robot is at the start
  ros2 service call /mission/stop std_srvs/srv/Trigger      # emergency stop

Everything runs on the car; the VM is only needed to start the mission and to watch in RViz.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def include(pkg, name, args=None):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory(pkg), 'launch', name)),
        launch_arguments=(args or {}).items())


def generate_launch_description():
    default_params = os.path.join(get_package_share_directory('ee3033_mission'), 'config', 'mission.yaml')
    params = LaunchConfiguration('params_file')
    manager = Node(package='ee3033_mission', executable='mission_manager', output='screen',
                   parameters=[params, {'auto_start': LaunchConfiguration('auto_start')}])

    return LaunchDescription([
        DeclareLaunchArgument('map', description='Map yaml saved in step 1'),
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('auto_start', default_value='false'),
        DeclareLaunchArgument('video_device', default_value='/dev/video0'),
        include('ee3033_mission', 'base_sensors.launch.py'),
        TimerAction(period=2.0, actions=[include('ee3033_mission', 'perception.launch.py', {
            'params_file': params, 'video_device': LaunchConfiguration('video_device')})]),
        TimerAction(period=6.0, actions=[LogInfo(msg='[ee3033] starting map server + AMCL'),
                                         include('balance_navigation', 'balance_amcl_localization.launch.py',
                                                 {'map': LaunchConfiguration('map')})]),
        TimerAction(period=10.0, actions=[LogInfo(msg='[ee3033] starting Nav2'),
                                          include('balance_navigation', 'balance_nav2_core.launch.py')]),
        TimerAction(period=14.0, actions=[LogInfo(msg='[ee3033] starting mission manager'), manager]),
    ])

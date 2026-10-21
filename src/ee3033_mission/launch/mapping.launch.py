"""Step 1: build the map of the sandbox with SLAM Toolbox.

  ros2 launch ee3033_mission mapping.launch.py               # drive with teleop from the VM
  ros2 launch ee3033_mission mapping.launch.py explore:=true # autonomous frontier exploration

Teleop on the VM must publish to /cmd_vel_teleop; the cmd_vel_watchdog on the car
relays it to /cmd_vel and stops the robot if commands stop arriving (Wi-Fi loss):
  ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel_teleop

Start this with the robot at the assigned starting position: the map origin is
then the start pose, which mission.launch.py uses as the initial pose.
Save the map at the end with
  ros2 service call /slam_toolbox/save_map slam_toolbox/srv/SaveMap "{name: {data: /home/ros2_ws/maps/<name>}}"
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def include(pkg, name, condition=None):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory(pkg), 'launch', name)),
        condition=condition)


def generate_launch_description():
    explore = IfCondition(LaunchConfiguration('explore'))
    params = os.path.join(get_package_share_directory('ee3033_mission'), 'config', 'mission.yaml')
    watchdog = Node(package='ee3033_mission', executable='cmd_vel_watchdog', output='screen', parameters=[params])
    return LaunchDescription([
        DeclareLaunchArgument('explore', default_value='false',
                              description='true: Nav2 + explore_lite explore the sandbox autonomously'),
        include('ee3033_mission', 'base_sensors.launch.py'),
        watchdog,
        TimerAction(period=7.0, actions=[LogInfo(msg='[ee3033] starting SLAM Toolbox'),
                                         include('balance_mapping', 'balance_mapping_slamtoolbox_async.launch.py')]),
        TimerAction(period=11.0, actions=[include('balance_navigation', 'balance_nav2_core.launch.py', explore)]),
        TimerAction(period=16.0, actions=[include('balance_mapping', 'balance_explorer_lite.launch.py', explore)]),
    ])

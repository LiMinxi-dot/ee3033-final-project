"""Step 2: autonomous search mission on the saved map.

  ros2 launch ee3033_mission mission.launch.py map:=/home/ros2_ws/maps/sandbox.yaml
  ros2 service call /mission/start std_srvs/srv/Trigger     # when the robot is at the start
  ros2 service call /mission/stop std_srvs/srv/Trigger      # emergency stop

Everything runs on the car; the VM is only needed to start the mission and to watch in RViz.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, LogInfo,
                            RegisterEventHandler, TimerAction)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter


def include(pkg, name, args=None):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory(pkg), 'launch', name)),
        launch_arguments=(args or {}).items())


def generate_launch_description():
    default_params = os.path.join(get_package_share_directory('ee3033_mission'), 'config', 'mission.yaml')
    params = LaunchConfiguration('params_file')
    speed = LaunchConfiguration('max_speed')
    manager = Node(package='ee3033_mission', executable='mission_manager', output='screen',
                   parameters=[params, {'auto_start': LaunchConfiguration('auto_start'), 'drive_speed': speed}])

    wait_tf = Node(package='ee3033_mission', executable='wait_for_transform', output='screen',
                   parameters=[{'target_frame': 'map', 'source_frame': 'base_footprint'}])
    after_amcl = RegisterEventHandler(OnProcessExit(target_action=wait_tf, on_exit=[
        LogInfo(msg='[ee3033] AMCL is localized: starting Nav2'),
        include('ee3033_mission', 'nav2.launch.py', {'max_speed': speed})]))

    return LaunchDescription([
        DeclareLaunchArgument('map', description='Map yaml saved in step 1'),
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('auto_start', default_value='false'),
        # m/s for Nav2 and our own straight drives; mapping keeps 0.10 (faster blurred the map), the mission uses
        # a fixed map
        DeclareLaunchArgument('max_speed', default_value='0.125'),
        DeclareLaunchArgument('video_device', default_value='/dev/video0'),
        DeclareLaunchArgument('amcl_tuned', default_value='false',
                              description='true: our AMCL laser-model overrides (localization.launch.py)'),
        include('ee3033_mission', 'base_sensors.launch.py'),
        TimerAction(period=2.0, actions=[include('ee3033_mission', 'perception.launch.py', {
            'params_file': params, 'video_device': LaunchConfiguration('video_device')})]),
        TimerAction(period=6.0, actions=[LogInfo(msg='[ee3033] starting map server + AMCL'), GroupAction([
            SetParameter(name='bond_timeout', value=0.0),  # no bond shutdowns, as for Nav2 (nav2.launch.py)
            include('ee3033_mission', 'localization.launch.py',
                    {'map': LaunchConfiguration('map'), 'amcl_tuned': LaunchConfiguration('amcl_tuned')})])]),
        # mission_manager sets the AMCL initial pose; Nav2 starts once AMCL publishes map -> base_footprint
        # (it aborts its bringup if that transform is missing while its costmaps activate).
        after_amcl,
        TimerAction(period=8.0, actions=[LogInfo(msg='[ee3033] starting mission manager'), manager, wait_tf]),
    ])

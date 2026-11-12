"""Step 2: autonomous search mission on the saved map.

  ros2 launch ee3033_mission mission.launch.py map:=/home/ros2_ws/maps/sandbox.yaml
  ros2 service call /mission/start std_srvs/srv/Trigger     # when the robot is at the start
  ros2 service call /mission/stop std_srvs/srv/Trigger      # emergency stop

Everything runs on the car; the VM is only needed to start the mission and to watch in RViz.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, GroupAction, IncludeLaunchDescription, LogInfo,
                            OpaqueFunction, RegisterEventHandler, TimerAction)
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
        include('ee3033_mission', 'nav2.launch.py', {'max_speed': speed, 'keep_off_walls': LaunchConfiguration('keep_off_walls')})]))

    perception = include('ee3033_mission', 'perception.launch.py', {
        'params_file': params, 'video_device': LaunchConfiguration('video_device')})
    robot = [  # the base driver and everything that needs it, at these delays (s) after the base starts
        include('ee3033_mission', 'base_sensors.launch.py'),
        TimerAction(period=6.0, actions=[LogInfo(msg='[ee3033] starting map server + AMCL'), GroupAction([
            SetParameter(name='bond_timeout', value=0.0),  # no bond shutdowns, as for Nav2 (nav2.launch.py)
            include('ee3033_mission', 'localization.launch.py',
                    {'map': LaunchConfiguration('map'), 'amcl_tuned': LaunchConfiguration('amcl_tuned')})])]),
        # mission_manager sets the AMCL initial pose; Nav2 starts once AMCL publishes map -> base_footprint
        # (it aborts its bringup if that transform is missing while its costmaps activate).
        after_amcl,
        TimerAction(period=8.0, actions=[LogInfo(msg='[ee3033] starting mission manager'), manager, wait_tf]),
    ]

    def order(context):
        """yolo_first: camera + YOLO first, the base driver and the rest after the detector's first result (at
        most 60 s). Loading torch and the model took the CPU and the SD card for 10-15 s and starved the base driver
        (/odom 22-32 Hz, gaps 0.6-0.8 s), then the base spun on its own (zero /cmd_vel): both missions of the 11-12
        evening fell, a third one caught itself (1112_115139, 121900, 123050). Mapping (no YOLO) never did."""
        if LaunchConfiguration('yolo_first').perform(context).lower() != 'true':
            return [*robot, TimerAction(period=2.0, actions=[perception])]     # as before 11-12
        wait = ('timeout 60 ros2 topic echo --once /target_detector/detections '
                'ee3033_interfaces/msg/TargetDetectionArray > /dev/null')
        start_file = LaunchConfiguration('start_file').perform(context)
        if start_file:      # demo.sh: the robot is placed while YOLO loads; Enter creates the file
            wait += f'; echo "[ee3033] YOLO is ready: waiting for {start_file}"; while [ ! -e {start_file} ]; do sleep 0.2; done'
        ready = ExecuteProcess(cmd=['bash', '-c', wait], output='screen')
        return [LogInfo(msg='[ee3033] starting camera + YOLO first; the base starts when YOLO is ready'),
                perception, ready,
                RegisterEventHandler(OnProcessExit(target_action=ready, on_exit=[
                    LogInfo(msg='[ee3033] YOLO is ready: starting the base, AMCL and the mission manager'), *robot]))]

    return LaunchDescription([
        DeclareLaunchArgument('map', description='Map yaml saved in step 1'),
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('auto_start', default_value='false'),
        # m/s for Nav2 and our own straight drives; mapping keeps 0.10 (faster blurred the map), the mission uses
        # a fixed map
        DeclareLaunchArgument('max_speed', default_value='0.125'),
        DeclareLaunchArgument('heading_hold', default_value='true',
                              description='hold the start heading until the robot starts (base_sensors.launch.py)'),
        DeclareLaunchArgument('keep_off_walls', default_value='true',
                              description='Nav2 keeps the robot off walls (nav2.launch.py); false: vendor DWB/inflation'),
        DeclareLaunchArgument('video_device', default_value='/dev/video0'),
        DeclareLaunchArgument('amcl_tuned', default_value='false',
                              description='true: our AMCL laser-model overrides (localization.launch.py)'),
        DeclareLaunchArgument('yolo_first', default_value='true',
                              description='start camera + YOLO first, the base when YOLO is ready (false: together)'),
        DeclareLaunchArgument('start_file', default_value='',
                              description='yolo_first: after YOLO is ready, also wait until this file exists (demo.sh: '
                                          'Enter once the robot is placed); empty: do not wait'),
        OpaqueFunction(function=order),
    ])

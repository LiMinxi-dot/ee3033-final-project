"""Step 1: build the map of the sandbox with SLAM Toolbox.

  ros2 launch ee3033_mission mapping.launch.py               # drive with teleop from the VM
  ros2 launch ee3033_mission mapping.launch.py explore:=true # autonomous: Nav2 + our auto_mapper
  ros2 launch ee3033_mission mapping.launch.py explore:=true explorer:=lite   # vendor explore_lite instead
  ros2 launch ee3033_mission mapping.launch.py explore:=true max_speed:=0.10   # Nav2 speed limit (m/s)

With explore:=true the auto_mapper starts in a dry run (frontiers shown on
/explore/frontiers, robot still). Start it with
  ros2 service call /explore/start std_srvs/srv/Trigger
or pass auto_start:=true. It explores, returns to the start pose and saves
the map to /home/ros2_ws/maps/sandbox.{pgm,yaml} (map_file in mission.yaml).

Teleop on the VM must publish to /cmd_vel_teleop; the cmd_vel_watchdog on the car
relays it to /cmd_vel and stops the robot if commands stop arriving (Wi-Fi loss):
  ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel_teleop

Start this with the robot at the assigned starting position: the map origin is
then the start pose, which mission.launch.py uses as the initial pose.
Save the map at the end (teleop or explore_lite) with
  ros2 service call /slam_toolbox/save_map slam_toolbox/srv/SaveMap "{name: {data: /home/ros2_ws/maps/<name>}}"
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, RegisterEventHandler,
                            TimerAction)
from launch.event_handlers import OnProcessExit
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def include(pkg, name, condition=None, args=None):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory(pkg), 'launch', name)),
        condition=condition, launch_arguments=(args or {}).items())


def generate_launch_description():
    explore = IfCondition(LaunchConfiguration('explore'))

    def explore_with(name):
        return IfCondition(PythonExpression(["'", LaunchConfiguration('explore'), "'.lower() in ('true', '1') and '",
                                             LaunchConfiguration('explorer'), f"' == '{name}'"]))
    ours, lite = explore_with('ee3033'), explore_with('lite')
    params = os.path.join(get_package_share_directory('ee3033_mission'), 'config', 'mission.yaml')
    # As balance_mapping_slamtoolbox_async.launch.py, but ignoring LiDAR hits closer than 13 cm: part of the robot
    # behind the LiDAR shows up in every scan (below 6.5 cm, and now and then at 7.5-12 cm), and SLAM marked it as an
    # obstacle wherever the robot paused (a trail of dots; one at the start made the start unreachable for Nav2).
    slam = Node(package='slam_toolbox', executable='async_slam_toolbox_node', name='slam_toolbox', output='screen',
                parameters=[os.path.join(get_package_share_directory('balance_mapping'), 'config',
                                         'mapper_params_online_async.yaml'),
                            {'use_sim_time': False, 'min_laser_range': 0.13,
                             # a cell is a wall if > this share of the rays reaching it end there (default 0.1):
                             # a thin wall mostly seen along its length (rays grazing past it in its cell) was
                             # mapped as free space but for 1-2 cells
                             'occupancy_threshold': 0.05,
                             # blur of the scan-matching score grid (default 0.1 m): with 0.1 poses a few cm or deg
                             # apart scored alike and the match stayed where the odometry put it (it over-reports
                             # turns by 4-10 %); 0.05 ~ the LiDAR noise of the swaying robot. Offline replays of a
                             # home and a site bag (11-06): walls 20-30 % thinner, inner walls no longer merged
                             'correlation_search_space_smear_deviation': 0.05}])
    watchdog = Node(package='ee3033_mission', executable='cmd_vel_watchdog', output='screen', parameters=[params])
    mapper = Node(package='ee3033_mission', executable='auto_mapper', output='screen', condition=ours,
                  parameters=[params, {'auto_start': LaunchConfiguration('auto_start')}])
    # Nav2 aborts its bringup if map -> base_footprint is missing while its costmaps activate,
    # so it starts only once SLAM Toolbox publishes that transform.
    wait_tf = Node(package='ee3033_mission', executable='wait_for_transform', output='screen', condition=explore,
                   parameters=[{'target_frame': 'map', 'source_frame': 'base_footprint'}])
    after_slam = RegisterEventHandler(OnProcessExit(target_action=wait_tf, on_exit=[
        LogInfo(msg='[ee3033] SLAM is publishing map -> base_footprint: starting Nav2'),
        include('ee3033_mission', 'nav2.launch.py', args={'max_speed': LaunchConfiguration('max_speed'),
                                                            'keep_off_walls': LaunchConfiguration('keep_off_walls')}),
        mapper,
        TimerAction(period=5.0, actions=[include('balance_mapping', 'balance_explorer_lite.launch.py', lite)]),
    ]))
    return LaunchDescription([
        DeclareLaunchArgument('explore', default_value='false',
                              description='true: Nav2 + an explorer map the sandbox autonomously'),
        DeclareLaunchArgument('explorer', default_value='ee3033',
                              description='ee3033: our auto_mapper (return home + save map); lite: explore_lite'),
        DeclareLaunchArgument('max_speed', default_value='0.125',
                              description='m/s, Nav2 (DWB and the velocity smoother) while exploring'),
        DeclareLaunchArgument('heading_hold', default_value='true',
                              description='hold the start heading until the robot starts (base_sensors.launch.py)'),
        DeclareLaunchArgument('keep_off_walls', default_value='true',
                              description='Nav2 keeps the robot off walls (nav2.launch.py); false: vendor DWB/inflation'),
        DeclareLaunchArgument('auto_start', default_value='false',
                              description='true: auto_mapper starts exploring as soon as Nav2 is ready'),
        include('ee3033_mission', 'base_sensors.launch.py'),
        watchdog,
        after_slam,
        TimerAction(period=7.0, actions=[LogInfo(msg='[ee3033] starting SLAM Toolbox'),
                                         slam, wait_tf]),
    ])

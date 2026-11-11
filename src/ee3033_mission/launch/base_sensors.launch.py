"""Robot base, LiDAR and scan resampler (shared by the mapping and mission launches).

The base is started as balance_robot_base.launch.py does (balance_robot_node with the vendor base_params.yaml,
static TFs from base_footprint to the laser, camera and gyro frames), except for the laser's: the vendor launch
hard-codes its yaw to 0, and on our car (balance4) the LiDAR is turned ~4 deg clockwise. Driving straight, it saw
itself move 3.7-4.1 deg to the left of its x axis (scan-to-scan ICP of three mapping runs on 11-06, one of them
before the firmware update), and in spins it moved as if 1-2 cm ahead of the turning centre (vendor TF: 0.2 cm).
In offline SLAM replays of a home and a site bag this TF with smear 0.05 (mapping.launch.py) gave the thinnest walls
and the best fit to the scans. camera_offset_yaw_deg (mission.yaml) was measured against the LiDAR, so it changed
with it (8.3 -> 4.3). Maps built before this change do not fit this TF: map again.

With odom_yaw_scale, odom_yaw_scale_cw or odom_lin_scale != 1 the base publishes its odometry as odom_raw -> base_raw on /odom_raw,
and odom_scaler (ee3033_mission) publishes odom_combined -> base_footprint and /odom with the base's left / right
turns (and distances) scaled: the base over-reports left turns by ~2.5 % and right turns by ~4.5 % (true / odometry
0.976 / 0.956, two site missions on 11-11), and after every in-place turn AMCL's heading was 8-15 deg off (mission
1111_045745). All 1: the base publishes odom_combined -> base_footprint itself, as before.

The resampler is started with `ros2 run balance_mapping scan_resampler.py`
because the vendor bringup points to /home/ros2_ws/scan_resampler.py, which
does not exist on the car.
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def static_tf(name, parent, child, x, y, z, yaw='0.0', pitch=0.0):
    return Node(package='tf2_ros', executable='static_transform_publisher', name=name,
                arguments=['--x', x, '--y', str(y), '--z', str(z), '--yaw', yaw, '--pitch', str(pitch),
                           '--roll', '0.0', '--frame-id', parent, '--child-frame-id', child])


def base_nodes(context, config, odom, robot):
    """balance_robot_node, and odom_scaler in front of it unless all scales are 1."""
    k_yaw = float(LaunchConfiguration('odom_yaw_scale').perform(context))
    k_cw = float(LaunchConfiguration('odom_yaw_scale_cw').perform(context))
    k_lin = float(LaunchConfiguration('odom_lin_scale').perform(context))
    if k_yaw == 1.0 and k_cw == 1.0 and k_lin == 1.0:
        return [Node(package='balance_robot_base', executable='balance_robot_node', name='balance_robot_node',
                     output='screen', parameters=[config])]
    return [
        Node(package='balance_robot_base', executable='balance_robot_node', name='balance_robot_node',
             output='screen', parameters=[config, {'odom_frame_id': 'odom_raw', 'robot_frame_id': 'base_raw'}],
             remappings=[('odom', 'odom_raw')]),
        Node(package='ee3033_mission', executable='odom_scaler', name='odom_scaler', output='screen',
             parameters=[{'yaw_scale': k_yaw, 'yaw_scale_cw': k_cw, 'lin_scale': k_lin, 'odom_frame': odom,
                          'robot_frame': robot}]),
    ]


def generate_launch_description():
    config = os.path.join(get_package_share_directory('balance_robot_base'), 'config', 'base_params.yaml')
    with open(config) as f:
        p = yaml.safe_load(f)['balance_robot_node']['ros__parameters']
    robot = p.get('robot_frame_id', 'base_footprint')
    tf = p.get('tf_parameters', {})
    laser, camera, gyro = tf.get('laser', {}), tf.get('camera', {}), tf.get('gyro', {})
    base = [
        OpaqueFunction(function=base_nodes, args=[config, p.get('odom_frame_id', 'odom_combined'), robot]),
        static_tf('laser_static_tf', robot, p.get('laser_frame_id', 'laser'), LaunchConfiguration('laser_x'),
                  laser.get('y', 0.0), laser.get('z', 0.1875), yaw=LaunchConfiguration('laser_yaw')),
        static_tf('camera_static_tf', robot, p.get('camera_frame_id', 'camera_link'), str(camera.get('x', -0.035)),
                  camera.get('y', 0.0), camera.get('z', 0.2655), pitch=camera.get('pitch', 0.0)),
        static_tf('gyro_static_tf', robot, p.get('gyro_frame_id', 'gyro_link'), str(gyro.get('x', -0.02)),
                  gyro.get('y', -0.04), gyro.get('z', 0.155)),
    ]
    lidar = IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(
        get_package_share_directory('lslidar_driver_n10p'), 'launch', 'lslidar_launch.py')))
    resampler = Node(
        package='balance_mapping', executable='scan_resampler.py', name='scan_resampler', output='screen',
        parameters=[{'input_topic': '/scan', 'output_topic': '/scan_fixed', 'number_of_bins': 450}])

    return LaunchDescription([
        DeclareLaunchArgument('laser_x', default_value='0.015',
                              description='m, LiDAR ahead of base_footprint (vendor 0.002)'),
        DeclareLaunchArgument('laser_yaw', default_value='-0.07',
                              description='rad, LiDAR yaw on base_footprint (vendor 0)'),
        DeclareLaunchArgument('odom_yaw_scale', default_value='1.0',
                              description='odometry left turns x this (odom_scaler); all three 1: off'),
        DeclareLaunchArgument('odom_yaw_scale_cw', default_value='1.0', description='odometry right turns x this'),
        DeclareLaunchArgument('odom_lin_scale', default_value='1.0', description='odometry distances x this'),
        *base,
        TimerAction(period=2.0, actions=[LogInfo(msg='[ee3033] starting LiDAR'), lidar]),
        TimerAction(period=4.0, actions=[LogInfo(msg='[ee3033] starting scan resampler'), resampler]),
    ])

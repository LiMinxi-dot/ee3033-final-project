"""Map server + AMCL for the mission, as balance_navigation's balance_amcl_localization.launch.py (vendor
localization_params.yaml, nav2_bringup localization_launch.py), optionally with our AMCL laser-model overrides.

amcl_tuned:=true: the vendor AMCL (mostly Nav2 defaults: sigma_hit 0.2 m, z_hit / z_rand 0.5 / 0.5, an update only
every 0.25 m or 0.2 rad, robot-body hits used) corrected the pose too weakly in our small arenas. Home mission
1111_133828: the AMCL position was 10-22 cm off next to the box for 30 s (scans fit 34-64 % there), and two escapes
drove backward by that wrong pose. Replays of that bag (scripts/amcl_replay/run_p.sh): position error mean / p90
vendor 10.4 / 16.1 cm, tuned (OVERRIDES) 3.8 / 5.4 cm; heading 7.0 / 11.4 -> 1.1 / 2.2 deg.
"""

import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

OVERRIDES = {
    'sigma_hit': 0.05,                 # m; the map walls are 1-3 cells (5 cm) thick
    'z_hit': 0.9, 'z_rand': 0.1,       # trust the beams that hit a wall
    'laser_min_range': 0.13,           # closer hits are the robot itself (as SLAM and the costmaps)
    'laser_max_range': 8.0,
    'max_beams': 120,                  # of the 450 beams of /scan
    'update_min_a': 0.05,              # rad, and
    'update_min_d': 0.05,              # m between filter updates (vendor 0.2 / 0.25: legs are 0.3-0.8 m)
    'laser_likelihood_max_dist': 0.5,
}


def localization(context):
    src = os.path.join(get_package_share_directory('balance_navigation'), 'config', 'localization_params.yaml')
    tuned = LaunchConfiguration('amcl_tuned').perform(context).lower() in ('1', 'true', 'yes')
    params_file, msg = src, '[ee3033] AMCL with the vendor parameters'
    if tuned:
        with open(src) as f:
            params = yaml.safe_load(f)
        params['amcl']['ros__parameters'].update(OVERRIDES)
        with tempfile.NamedTemporaryFile('w', prefix='ee3033_amcl_', suffix='.yaml', delete=False) as f:
            yaml.safe_dump(params, f)
        params_file, msg = f.name, f'[ee3033] AMCL tuned: {OVERRIDES}'
    return [LogInfo(msg=msg), IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('nav2_bringup'), 'launch',
                                                   'localization_launch.py')),
        launch_arguments={'use_sim_time': 'false', 'map': LaunchConfiguration('map'),
                          'params_file': params_file}.items())]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('map', description='Map yaml'),
        DeclareLaunchArgument('amcl_tuned', default_value='false',
                              description='true: our AMCL laser-model overrides (OVERRIDES), false: vendor'),
        OpaqueFunction(function=localization),
    ])

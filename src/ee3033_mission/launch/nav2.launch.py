"""Nav2 as started by balance_navigation/balance_nav2_core.launch.py, with our overrides of its parameters.

The vendor launch file hard-codes its params file, so the same Nav2 bringup is included here with the vendor
file rewritten (the vendor package stays unchanged):
  robot_radius 0.12 -> 0.10 (both costmaps; plus the default 0.01 m padding): the robot is 0.20 m wide (camera
  cable on the left) and 0.115 m long. With 0.12 the robot often stood inside the inscribed zone near walls and
  NavFn failed; with 0.08 and 0.09 its side brushed walls: on site it drove along a 0.37 m passage 10 cm off its
  centre and scraped a wall corner while DWB saw its trajectories as free. Tried: the robot's rectangle as
  footprint. Then the Nav2 spin aborted at every viewpoint ("Collision Ahead"), also 0.30-0.36 m from any wall
  in the map (cause not found; the spin's collision check maps the published footprint back into the robot
  frame, which matters for a rectangle only). The inscribed zone is now 0.11 m: escape_trigger in mission.yaml.
  DWB min_vel_x -0.05 -> 0: no reversing (the LiDAR cannot see behind the robot, and forward/backward
  alternation from the swaying odometry triggered "Trajectory is oscillating").
  Do NOT raise DWB acc_lim_x: it also limits how fast the speed command changes. With acc_lim_x 10 the robot
  was told 0 -> 0.25 m/s at once at the start of the return and the balancing robot fell over.
  Progress checker 0.2 m in 10 s -> 0.1 m in 20 s: the swaying robot was aborted while still moving slowly.
  Goal checker xy 0.15 -> 0.12 m: our routes are short legs along corridor centres, and with 0.15 each leg ended up
  to 0.15 m short or beside its waypoint; on site the offsets added up in a 0.37 m passage until the robot stood at
  a wall corner (twice, same corner: stuck once, scraped once). 0.08 was too tight: the swaying robot hovered
  0.06-0.10 m from the goal, the progress checker aborted ("Failed to make progress") and legs took 30-60 s.
  DWB's own xy_goal_tolerance 0.15 -> 0.10: within it DWB's RotateToGoal critic allows only turning, no driving.
  With 0.15 against the checker's 0.12 nobody drove the last 3 cm and the swaying robot hovered at the leg end until
  a sway took it inside 0.12 m (site mapping 11-06: 0.5-6 s per leg, 15 s once). Below the checker's, DWB drives in.
  obstacle_min_range 0 -> 0.13 (both costmaps), as min_laser_range for SLAM: the robot's own body is in every
  scan below 6.5 cm (rear +-28 deg), and ~0.14 returns per scan lie at 7.5-12 cm behind it (rear 12-24 deg off
  the axis). Points the costmap marks there are never cleared (the body blocks the beams that would clear them):
  they pile up behind a robot that stands still, the spin aborted with "Collision Ahead", and after the 6 s wait
  at the end of an exploration Smac said "Starting point in lethal space" for the whole way home although the
  map had 0.15 m clearance there (0.075 m let these through).
  Slower (site mapping run: walls 2-4 cells thick and blurred): DWB max_vel_x 0.25 / max_speed_xy 0.2 -> 0.10 m/s
  (0.15 at first: on the long straight legs home the robot then went twice as fast as while exploring, where legs
  are short, and the map got distorted), max_vel_theta 1.2 -> 0.5 rad/s; spin behaviour max_rotational_vel
  1.0 -> 0.5 rad/s. Turning smears a scan most (the robot turns during the ~0.1 s a scan takes). The speed is the
  launch argument max_speed (DWB and the smoother): 0.125 while mapping (mapping.launch.py; 0.10 until 11-06, when
  the spread of the wall hits turned out not to depend on the speed) and in the mission (mission.launch.py).
  Velocity smoother (between the controller / behaviours and the base, OPEN_LOOP: limits the change of the
  command itself, not affected by the swaying odometry): accel 1.0 -> 0.2 m/s2 and 2.0 -> 1.0 rad/s2, decel
  -1.0 -> -0.3 m/s2 and -2.0 -> -1.0 rad/s2 (0 -> 0.15 m/s in 0.15 s made the balancing robot lurch and tilt);
  its speed limits match DWB. Spin rotational_acc_lim 3.2 -> 1.0, as the smoother, so the spin slows down in time.
  The behaviour server (spin) published on /cmd_vel directly, past the smoother, which meanwhile sent zeros for
  1 s after each controller goal: the start of every turn got both. Now it publishes to the smoother too, so the
  smoother is the only publisher on /cmd_vel (apart from our nodes' stop commands).
  Behaviour tree without the recoveries that move the robot (behavior_trees/navigate_to_pose_no_motion_recoveries
  .xml: no Spin, Wait, BackUp; costmaps are still cleared before a retry). On site, FollowPath was acknowledged too
  late, the recoveries started and kept running after Nav2 had given up: backing up for 10 s while our own turn
  and the next goal drove the robot, which fell over. default_server_timeout 20 -> 500 ms: the BT waited only
  20 ms for an action server to accept a goal, often too short on the Pi ("Timed out while waiting for action
  server to acknowledge goal request").
  Global planner NavFn -> Smac 2D (same plugin id GridBased): NavFn traces its path down a potential field, and
  next to the inscribed zone that tracing failed ("Failed to create a plan from potential when a legal potential
  was found"): the robot stood 0.11 m from a wall, the map had a wide free way home, and every replan failed for
  40 s, so the return failed. Smac 2D searches the grid instead; cost_travel_multiplier keeps paths off walls.
  keep_off_walls (launch argument, default true): DWB BaseObstacle.scale 0.05 -> 0.25, PathDist.scale 16 -> 32,
  GoalDist.scale 24 -> 12, inflation cost_scaling_factor 3 -> 10 (both costmaps). With the vendor values DWB hardly
  minded the walls and was pulled straight at the leg's goal, and the inflation costs were nearly flat inside the
  0.20 m radius (0.12 m from a wall 229, 0.17 m 211), so neither DWB nor Smac kept the robot in the middle of a
  narrow passage. Lab maze 1112_104708: going round the corner into a 0.35 m passage the robot drifted 3-4 cm off
  the centre, a wheel hit the corner (LiDAR 0.14 m) and it fell. Now 229 / 139: a clear slope across a passage.
  bond_timeout 0 for the lifecycle manager (no bonds): in the site mission it missed controller_server's
  heartbeat for 4 s while YOLO loaded the CPU, shut Nav2 down ("CRITICAL FAILURE ... IS DOWN"), and its
  bringup after the reset never finished, so every goal of the mission was rejected.
"""

import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import SetParameter, SetRemap

DWB = 'controller_server.ros__parameters.FollowPath.'
SMOOTHER = 'velocity_smoother.ros__parameters.'
OVERRIDES = {                             # path in the vendor file -> value; no dots: every key of that name
    'robot_radius': 0.10,                 # both costmaps
    'obstacle_min_range': 0.13,           # the scan source of both costmaps (robot body returns reach 12 cm)
    DWB + 'min_vel_x': 0.0,
    DWB + 'max_vel_theta': 0.5,
    DWB + 'xy_goal_tolerance': 0.10,      # below the goal checker's 0.12 (see above)
    'behavior_server.ros__parameters.max_rotational_vel': 0.5,
    'behavior_server.ros__parameters.rotational_acc_lim': 1.0,
    'controller_server.ros__parameters.progress_checker.required_movement_radius': 0.1,
    'controller_server.ros__parameters.progress_checker.movement_time_allowance': 20.0,
    'controller_server.ros__parameters.general_goal_checker.xy_goal_tolerance': 0.12,
    'bt_navigator.ros__parameters.default_server_timeout': 500,      # ms
    'planner_server.ros__parameters.GridBased': {
        'plugin': 'nav2_smac_planner/SmacPlanner2D',
        'tolerance': 0.25,                # m; plan to the nearest free cell this close if the goal is blocked
        'downsample_costmap': False,
        'downsampling_factor': 1,
        'allow_unknown': True,
        'max_iterations': 1000000,
        'max_on_approach_iterations': 1000,
        'max_planning_time': 2.0,         # s
        'cost_travel_multiplier': 2.0,    # steer away from high costs (near walls)
        'use_final_approach_orientation': False,
        'smoother': {'max_iterations': 1000, 'w_smooth': 0.3, 'w_data': 0.2, 'tolerance': 1.0e-10},
    },
    SMOOTHER + 'max_accel': [0.2, 0.0, 1.0],
    SMOOTHER + 'max_decel': [-0.3, 0.0, -1.0],
}


def rewrite(params, overrides, new=()):
    """params (nested dicts) with overrides applied; a missing path is an error (a typo would go unnoticed),
    except for the paths in `new` (parameters the vendor file leaves at their default)."""
    def every(node, key, value):
        if isinstance(node, dict):
            for k in node:
                if k == key:
                    node[k] = value
                else:
                    every(node[k], key, value)

    for path, value in overrides.items():
        if '.' not in path:
            every(params, path, value)
            continue
        parts, node, i = path.split('.'), params, 0
        while i < len(parts) - 1 and isinstance(node.get(parts[i]), dict):
            node, i = node[parts[i]], i + 1
        leaf = '.'.join(parts[i:])             # DWB critic keys contain a dot themselves: 'BaseObstacle.scale'
        if leaf not in node and path not in new:
            raise KeyError(f'{path} is not in the vendor Nav2 params')
        node[leaf] = value
    return params


KEEP_OFF_WALLS = {                        # see keep_off_walls above
    DWB + 'BaseObstacle.scale': 0.25,
    DWB + 'PathDist.scale': 32.0,
    DWB + 'GoalDist.scale': 12.0,
    'cost_scaling_factor': 10.0,          # both inflation layers
}


def speed_overrides(speed):
    """Forward speed limit (m/s) of DWB and the velocity smoother."""
    return {DWB + 'max_vel_x': speed, DWB + 'max_speed_xy': speed,
            SMOOTHER + 'max_velocity': [speed, 0.0, 0.5],      # x, y, theta
            SMOOTHER + 'min_velocity': [-speed, 0.0, -0.5]}


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('max_speed', default_value='0.10', description='m/s, DWB and the velocity smoother'),
        DeclareLaunchArgument('keep_off_walls', default_value='true',
                              description='DWB and inflation keep the robot off walls (false: vendor values)'),
        OpaqueFunction(function=nav2_actions),
    ])


def nav2_actions(context):
    # Our own rewrite instead of nav2_common's RewrittenYaml, which cannot write lists (the smoother limits).
    speed = float(LaunchConfiguration('max_speed').perform(context))
    walls = KEEP_OFF_WALLS if LaunchConfiguration('keep_off_walls').perform(context).lower() == 'true' else {}
    src = os.path.join(get_package_share_directory('balance_navigation'), 'config', 'nav2_core_params.yaml')
    bt = os.path.join(get_package_share_directory('ee3033_mission'), 'behavior_trees',
                      'navigate_to_pose_no_motion_recoveries.xml')
    added = {'bt_navigator.ros__parameters.default_nav_to_pose_bt_xml': bt}
    with open(src) as f:
        params = rewrite(yaml.safe_load(f), {**OVERRIDES, **walls, **speed_overrides(speed), **added}, new=added)
    with tempfile.NamedTemporaryFile('w', prefix='ee3033_nav2_', suffix='.yaml', delete=False) as f:
        yaml.safe_dump(params, f)
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('nav2_bringup'), 'launch',
                                                   'navigation_launch.py')),
        launch_arguments={'use_sim_time': 'false', 'params_file': f.name, 'use_velocity_smoother': 'false'}.items())
    return [GroupAction([
        SetRemap(src='cmd_vel_nav', dst='cmd_vel'),        # as the vendor launch file
        SetRemap(src='cmd_vel_smoothed', dst='cmd_vel'),   # smoother -> base
        SetRemap(src='cmd_vel', dst='cmd_vel_nav'),        # controller and behaviours -> smoother
        SetParameter(name='bond_timeout', value=0.0),      # lifecycle manager: no bonds (see above)
        nav2,
    ])]

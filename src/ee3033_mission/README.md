# ee3033_mission

Final project package: map the sandbox, then autonomously search for the target,
report its (x, y) in the map frame and return to the start.
It uses the vendor packages on the car (`balance_robot_base`, `lslidar_driver_n10p`,
`balance_mapping`, `balance_navigation`) and does not modify them.

## Nodes

| Node | Inputs | Outputs |
|---|---|---|
| `target_detector` | `/camera/image_raw` | `/target_detector/detections` (`ee3033_interfaces/TargetDetectionArray`), `/target_detector/debug_image/compressed` |
| `target_localizer` | detections, `/scan_fixed`, `/camera_info`, TF | `/target/estimate` (`ee3033_interfaces/TargetEstimate`), `/target/markers` |
| `mission_manager` | `/map`, `/target/estimate`, TF, Nav2 actions `navigate_to_pose` and `spin` | `/initialpose`, `/mission/state`, `/mission/target` (`PointStamped`), `/mission/viewpoints`, result YAML |
| `cmd_vel_watchdog` | `/cmd_vel_teleop` | `/cmd_vel` (speed-limited; stop 0.6 s after the last command) |
| `auto_mapper` | `/map` (SLAM Toolbox), TF, Nav2 action `navigate_to_pose` | `/explore/state`, `/explore/frontiers`, map files |
| `odom_scaler` | `/odom_raw` (the base, with `odom_yaw_scale*` != 1) | `/odom`, TF `odom_combined -> base_footprint` with the base's left / right turns scaled |
| `heading_hold` | `/odom`, `/robot_euler`, `/explore/state`, `/mission/state` | `/cmd_vel` (turn in place back to the start heading until the exploration / mission starts; mapping and mission launch) |

Services: `/mission/start`, `/mission/stop` (`std_srvs/Trigger`), `/target_localizer/reset` (`std_srvs/Empty`),
`/explore/start`, `/explore/stop`, `/explore/save_map` (`std_srvs/Trigger`).

Tools:
- `ros2 run ee3033_mission setup_check [--computer]`: checks the course packages, devices, YOLO and network settings.
- `ros2 run ee3033_mission camera_lidar_check`: measures `camera_offset_yaw_deg` from camera vs LiDAR bearings of one object and saves an overlay image.
- `ros2 run ee3033_mission odom_turn_test [--ros-args -p go:=true]`: turns in place (by `/odom`) and aligns the LiDAR
  scans from before and after each turn: true / odometry turn per direction, i.e. `odom_yaw_scale` / `odom_yaw_scale_cw`
  (with `odom_scaler` on it should give 1.000). Dry run without `go:=true`; base and LiDAR only, no Nav2.
All tunable values are in `config/mission.yaml`.

## Autonomous mapping (auto_mapper)

Frontier exploration on the live SLAM Toolbox map (`ros2 launch ee3033_mission mapping.launch.py explore:=true`):
1. Frontiers: free cells next to unknown cells, grouped into clusters; clusters shorter than `min_frontier_size` are noise.
2. Traversable space: free cells at least `robot_radius` from occupied cells. A wavefront from the robot gives the
   path length to every reachable cell. A frontier's goal is the reachable cell closest to its centre (within `goal_reach`).
   A frontier shorter than `small_frontier_size` (0.35 m: a crack, a board joint, a corner, a pocket mouth) is
   looked at from a goal at least `small_goal_clearance` from walls and in sight of it, not driven into; with no
   such goal it is skipped (`small_fallback: true`: any safe goal in sight, as before). Lab maze 11-12: in a
   0.25 m pocket SLAM's heading slipped 5-6 deg and the far walls were drawn again rotated.
3. Cost = path length − `gain_weight` × frontier size. The robot drives to the cheapest frontier's goal along a
   route, like the way home: straight legs along corridor centres (≥ `route_clearance` from walls and unknown space),
   turning in place at each corner; Nav2 drives each leg. One long Nav2 goal cut the corner of a 0.45 m gap and the
   robot hit the wall. Without a route on the map, Nav2 gets the whole way.
4. The goal is kept until its frontier disappears (explored on the way), Nav2 fails or progress along the route stops; failed
   frontiers are blacklisted. After reaching a goal the node waits for a fresh map; a frontier that is still there
   cannot be resolved from there and is blacklisted. A frontier in the LiDAR's blind sector at the goal
   (`blind_from_deg`..`blind_to_deg`, behind the robot to the left, blocked by its body) is first faced by a turn.
5. No reachable frontier for `finish_wait` s, no `coverage_step` more free area, no resolved frontier and no route progress for
   `coverage_timeout` s (the area
   is complete; on site 11-06 it was after 61 s, and the next 104 s in two corners only thickened the walls from
   8.7 to 11.2 cm), or `max_duration` → save the map, drive back to the start pose (same heading, so the mission can
   use the map origin as initial pose). The map after the return goes to `<map_file>_after_return` only: the
   mission keeps the one from the end of the exploration. The map with the largest free area goes to
   `<map_file>_peak` (thinnest walls, but a wall seen only later is missing there).

Before `/explore/start` the node is a dry run: it shows the frontiers and the goal it would choose but does not move.
Offline tests on the VM: `scripts/sim_explore.py --demo` (no ROS) and `scripts/fake_world.py` (fake SLAM + Nav2).

## Target localization

1. YOLOv8n (imgsz 640, about 5 Hz on the Pi 5 with 2 threads) detects the target class (at 320 a bottle 0.7 m
   away scored 0.2, below the thresholds).
2. The box centre column gives the bearing in `camera_link` (`camera_info` K if calibrated, else `hfov_deg`).
3. TF at the image time turns the bearing into a ray in `map`.
4. The `/scan_fixed` scan closest in time, transformed to `map`, gives the range: the nearest cluster of hits around the ray. Hits on the walls of the saved map are ignored (`ignore_map_walls`): the target is never part of the map, so a beam passing above or beside the target must not give its range. If `target_height` > 0, the box height gives a fallback range.
5. Observations are fused: the largest cluster within `cluster_radius`, then the median. The target is confirmed after `min_observations` with spread ≤ `max_spread`.

## Search strategy

The start pose is the map origin, because mapping starts at the start position. The mission:
1. Sets the AMCL initial pose (heading uncertainty `initial_yaw_std_deg`) and lets AMCL settle with no-motion updates.
2. Localization check: at the pose AMCL reports, at least `min_scan_match` of the LiDAR points must lie on the map walls.
   If not, the robot does not move. The log lines starting with `[loc]` say how the robot really is placed, e.g.
   "turned 60 deg further LEFT, 15 cm off (+14 cm forward, +6 cm left)". If the scan clearly fits one nearby pose
   (at least `calibrate_min_fit` of the points on walls, and any other heading 20 deg or more away fits worse by
   `calibrate_margin`), that pose becomes AMCL's initial pose and the check runs again (auto-calibration, at most
   `max_calibrations` times). Otherwise the check repeats every 5 s until the robot is back on the start mark.
   AMCL only corrects itself after the odometry reports motion, so a robot that turns without the odometry
   noticing (balance drift, wheel slip, moved by hand) would otherwise stay mislocalized.
3. Spins once at the start.
4. Plans viewpoints on the saved map so that the whole free space is seen (the number is up to the planner):
   candidates are reachable cells at least `min_clearance` from walls on a `candidate_spacing` grid; a free cell
   counts as seen from a candidate if it is within `view_range` and in line of sight; greedy set cover picks
   candidates until no candidate adds more than 1 % (the start counts if the initial spin can turn there).
   The log says how much of the free space the viewpoints see (`[plan]` lines).
5. Visits the viewpoints in nearest-neighbour order and spins 360° at each one, repeating the localization check
   there (warning only).
6. Stops as soon as the target is confirmed, reports it with the last localization check (`scan_match` in the
   result file), and navigates home. At home it turns in place (Nav2 spin) back to the start heading
   (`home_heading`, within `home_yaw_tolerance_deg` 5°), so the robot need not be turned by hand for the next run
   (auto_mapper does the same).

Speed: `max_speed` of `mission.launch.py` and `mapping.launch.py` (0.125 m/s) sets Nav2 (DWB, velocity smoother)
and our own straight drives. Mapping ran at 0.10 m/s until 11-06; the spread of the wall hits did not depend on the
speed, and SLAM adds a scan every 0.1 m or 0.1 rad at any speed.

LiDAR-checked drive (both nodes): the map draws walls 2-3 cells thick, so the robot often stands in Nav2's
inscribed zone, or Nav2 stalls next to a wall, while the real way is clear. Before a Nav2 goal from closer than
`escape_trigger` to a map wall, and after a goal that failed or stalled, the LiDAR decides whether the robot is
really stuck: if the robot's rectangle can turn towards the leg's end and the strip its width sweeps is free of
LiDAR hits all the way (plus `drive_stop`), and the line crosses no map wall cell, the robot drives the leg itself
(`drive_speed`, steering towards the end, through the velocity smoother; a LiDAR hit within `drive_stop` stops it).
Otherwise, or if that drive stops short, it escapes as below.

Escape from walls (both nodes): a robot closer than `escape_trigger` to a map wall gets no help from Nav2 (Smac:
"Starting point in lethal space", spin and back-up: "Collision Ahead"). When the LiDAR-checked drive is not
possible, such a robot first moves (through the velocity smoother) to where the map shows `escape_target`
clearance: straight forward or backward, never through a wall, forward only as far as the LiDAR shows the robot's
rectangle (`body_half_length`, `body_half_width`) free, backward at most `escape_max_backward` (the LiDAR cannot
see behind the robot). If that does not get it clear (e.g. it stands along the wall), it first turns in place by
the smallest angle whose sweep of its corners the LiDAR shows free. On site the robot drifted during a viewpoint
spin into a wall that the map shows thicker than it is and stood there until the mission gave up.

Stuck goals (both nodes): a Nav2 goal during which the robot moves less than `stuck_move` for `stuck_time` s
ends, as reached if the robot is within `stuck_accept` of it, else as failed (drive or escape as above); a route
leg whose waypoint the robot is already within `leg_skip` of is skipped; within `leg_pass` of a waypoint that is
not the last one, the next leg is sent at once and Nav2 preempts the goal, so the robot does not stop at each
waypoint (it still stops for a turn of more than `turn_first_deg`); `nav_timeout` 30 s per goal. On site
the robot stood 0.14 m from two such goals while Nav2 kept replanning, until the 60 s timeout each time.
`stuck_move` is 0.08 m: with 0.05 the swaying robot (sway + SLAM steps, ~6 cm) reset the check all the time and it
never fired (site 11-06: 15 s at a leg end).

## Demo procedure

Quickest way, on the car inside the ROS 2 container (each step starts by itself, **the robot moves**):

```bash
ros2 run ee3033_mission demo.sh mapping   # robot at the start: explore, return, save map, stop the mapping nodes
# place the target
ros2 run ee3033_mission demo.sh mission   # robot at the start: search, report, return home
ros2 run ee3033_mission demo.sh stop      # stop the robot and all ROS nodes (also leftovers of earlier runs)
# --local after the step (demo.sh mapping --local): ROS traffic stays on the car (ROS_LOCALHOST_ONLY=1); the VM
#   and RViz see nothing. Big arena 11-12 with poor Wi-Fi: base and LiDAR nodes stalled up to 3 s, two falls.
```

Each step first stops ROS nodes left over from earlier runs. Ctrl-C stops the robot and all nodes.
Logs: `/tmp/ee3033_mapping.log`, `/tmp/ee3033_mission.log`. The manual steps below do the same.

demo.sh turns on two corrections by default (not yet run on site, only at home and in replays of site bags):
- odometry turn correction (`odom_yaw_scale:=0.976 odom_yaw_scale_cw:=0.956`, both steps): the base over-reports
  left turns by ~2.5 % and right turns by ~4.5 %; `odom_turn_test` measured 0.968 / 0.958 without, 0.999 / 1.004 with it.
- tuned AMCL (`amcl_tuned:=true`, mission only, `launch/localization.launch.py`): stronger laser model, an update every
  5 cm / 0.05 rad. Home mission: 3.6 cm mean position error (11.4 cm with the vendor parameters).

**Fallback** (the setup that ran on site in the morning of 11-11) if the robot gets lost, e.g. people in the arena
pull AMCL away (`[loc]` warnings at the viewpoints), or the turns look wrong:

```bash
ros2 run ee3033_mission demo.sh mapping odom_yaw_scale:=1.0 odom_yaw_scale_cw:=1.0
ros2 run ee3033_mission demo.sh mission odom_yaw_scale:=1.0 odom_yaw_scale_cw:=1.0 amcl_tuned:=false
```

Only the tuned AMCL: `demo.sh mission amcl_tuned:=false`. A map made with the turn correction works with or without
it in the mission (the map is drawn from the LiDAR). If Nav2 does not come up (`Waiting: ... nav2_active=False` for
more than a minute): `demo.sh stop`, then start the step again (demo.sh cleans Fast DDS shared memory first).

What the mission prints and what to do:
- `[loc] LOCALIZATION CHECK FAILED (at the start) ... turned 13 deg further RIGHT, 7 cm off (...)`: the robot is not
  where the program thinks. Followed by `AUTO-CALIBRATION 1/3 ...`: nothing to do, the program takes the pose that
  fits the scan and checks again 2 s later. Otherwise put the robot on the start mark facing the start direction;
  the check repeats every 5 s and the robot leaves by itself once it passes. If the arena changed since mapping and
  the check keeps failing, `ros2 service call /mission/start std_srvs/srv/Trigger` starts anyway.
- `[plan] N viewpoints ... see X% of the free space`: the planner chose N itself. A second `[plan]` line warns if
  part of the free space cannot be seen from anywhere the robot can reach.
- `[loc] ... (viewpoint k)` warnings: localization degraded while searching; the report then says POOR.
- Keep the start ≥ 20 cm from walls: otherwise Nav2 refuses the initial spin ("Collision Ahead").

Run the launch files on the car inside the ROS 2 container. Run teleop and RViz on the computer
(same `ROS_DOMAIN_ID`, `ROS_LOCALHOST_ONLY=0`).

```bash
# Step 1: mapping (robot at the start position)
ros2 launch ee3033_mission mapping.launch.py            # then drive from the VM (hold the key down):
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel_teleop
# cmd_vel_watchdog (on the car) relays /cmd_vel_teleop -> /cmd_vel and stops the robot
# 0.6 s after the last command, because the base keeps driving with the last /cmd_vel.
# or autonomous: ros2 launch ee3033_mission mapping.launch.py explore:=true
#   then ros2 service call /explore/start std_srvs/srv/Trigger: it explores, returns to the start
#   and saves /home/ros2_ws/maps/sandbox.{pgm,yaml} itself (skip the save step below)
mkdir -p /home/ros2_ws/maps                             # save the map (on the car):
ros2 service call /slam_toolbox/save_map slam_toolbox/srv/SaveMap "{name: {data: /home/ros2_ws/maps/sandbox}}"
# stop the mapping launch (Ctrl-C)

# Step 2: mission (robot back at the start, same heading as at the start of mapping)
ros2 launch ee3033_mission mission.launch.py map:=/home/ros2_ws/maps/sandbox.yaml
ros2 service call /mission/start std_srvs/srv/Trigger   # when /mission/state is IDLE
ros2 service call /mission/stop std_srvs/srv/Trigger    # emergency stop

# Watch from the VM
rviz2 -d $(ros2 pkg prefix ee3033_mission)/share/ee3033_mission/rviz/mission.rviz
```

The result is printed in the `mission_manager` log, published on `/mission/target` and written to `/home/ros2_ws/mission_result.yaml`.
A second line (and `x_cm`, `y_cm` in the YAML) gives it in cm from the mapping start, for checking with a tape:
x forward, y left of the robot's pose at the start of mapping (midpoint of the wheels), e.g.
`x = -4.6 cm (behind), y = -71.2 cm (right)`.

## Build

Copy `ee3033_interfaces` and `ee3033_mission` into the car's `/home/ros2_ws/src`, then:

```bash
cd /home/ros2_ws && colcon build --symlink-install --packages-select ee3033_interfaces ee3033_mission
```
YOLO weights are expected at `/home/ros2_ws/models/yolov8n.pt` (not in git; `yolov8n.pt` from Ultralytics).

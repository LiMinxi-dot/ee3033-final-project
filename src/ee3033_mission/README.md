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

Services: `/mission/start`, `/mission/stop` (`std_srvs/Trigger`), `/target_localizer/reset` (`std_srvs/Empty`),
`/explore/start`, `/explore/stop`, `/explore/save_map` (`std_srvs/Trigger`).

Tools:
- `ros2 run ee3033_mission setup_check [--computer]`: checks the course packages, devices, YOLO and network settings.
- `ros2 run ee3033_mission camera_lidar_check`: measures `camera_offset_yaw_deg` from camera vs LiDAR bearings of one object and saves an overlay image.
All tunable values are in `config/mission.yaml`.

## Autonomous mapping (auto_mapper)

Frontier exploration on the live SLAM Toolbox map (`ros2 launch ee3033_mission mapping.launch.py explore:=true`):
1. Frontiers: free cells next to unknown cells, grouped into clusters; clusters shorter than `min_frontier_size` are noise.
2. Traversable space: free cells at least `robot_radius` from occupied cells. A wavefront from the robot gives the
   path length to every reachable cell. A frontier's goal is the reachable cell closest to its centre (within `goal_reach`).
3. Cost = path length − `gain_weight` × frontier size. The cheapest frontier is sent to Nav2.
4. The goal is kept until its frontier disappears (explored on the way), Nav2 fails or progress stops; failed
   frontiers are blacklisted. After reaching a goal the node waits for a fresh map; a frontier that is still there
   cannot be resolved from there and is blacklisted.
5. No reachable frontier for `finish_wait` s (or `max_duration`) → save the map, drive back to the start pose
   (same heading, so the mission can use the map origin as initial pose), save again.

Before `/explore/start` the node is a dry run: it shows the frontiers and the goal it would choose but does not move.
Offline tests on the VM: `scripts/sim_explore.py --demo` (no ROS) and `scripts/fake_world.py` (fake SLAM + Nav2).

## Target localization

1. YOLOv8n (imgsz 320, about 9 Hz on the Pi 5) detects the target class.
2. The box centre column gives the bearing in `camera_link` (`camera_info` K if calibrated, else `hfov_deg`).
3. TF at the image time turns the bearing into a ray in `map`.
4. The `/scan_fixed` scan closest in time, transformed to `map`, gives the range: the nearest cluster of hits around the ray. Hits on the walls of the saved map are ignored (`ignore_map_walls`): the target is never part of the map, so a beam passing above or beside the target must not give its range. If `target_height` > 0, the box height gives a fallback range.
5. Observations are fused: the largest cluster within `cluster_radius`, then the median. The target is confirmed after `min_observations` with spread ≤ `max_spread`.

## Search strategy

The start pose is the map origin, because mapping starts at the start position. The mission:
1. Sets the AMCL initial pose (heading uncertainty `initial_yaw_std_deg`) and lets AMCL settle with no-motion updates.
2. Localization check: at the pose AMCL reports, at least `min_scan_match` of the LiDAR points must lie on the map walls.
   If not, the robot does not move. The log lines starting with `[loc]` say how the robot really is placed, e.g.
   "turned 60 deg further LEFT, 15 cm off (+14 cm forward, +6 cm left)", and the check repeats every 5 s until the
   robot is back on the start mark. AMCL only corrects itself after the odometry reports motion, so a robot that turns
   without the odometry noticing (balance drift, wheel slip, moved by hand) would otherwise stay mislocalized.
3. Spins once at the start.
4. Plans viewpoints on the saved map so that the whole free space is seen (the number is up to the planner):
   candidates are reachable cells at least `min_clearance` from walls on a `candidate_spacing` grid; a free cell
   counts as seen from a candidate if it is within `view_range` and in line of sight; greedy set cover picks
   candidates until no candidate adds more than 1 % (the start counts if the initial spin can turn there).
   The log says how much of the free space the viewpoints see (`[plan]` lines).
5. Visits the viewpoints in nearest-neighbour order and spins 360° at each one, repeating the localization check
   there (warning only).
6. Stops as soon as the target is confirmed, reports it with the last localization check (`scan_match` in the
   result file), and navigates home.

## Demo procedure

Quickest way, on the car inside the ROS 2 container (each step starts by itself, **the robot moves**):

```bash
ros2 run ee3033_mission demo.sh mapping   # robot at the start: explore, return, save map, stop the mapping nodes
# place the target
ros2 run ee3033_mission demo.sh mission   # robot at the start: search, report, return home
ros2 run ee3033_mission demo.sh stop      # stop the robot and all ROS nodes (also leftovers of earlier runs)
```

Each step first stops ROS nodes left over from earlier runs. Ctrl-C stops the robot and all nodes.
Logs: `/tmp/ee3033_mapping.log`, `/tmp/ee3033_mission.log`. The manual steps below do the same.

What the mission prints and what to do:
- `[loc] LOCALIZATION CHECK FAILED (at the start) ... turned 13 deg further RIGHT, 7 cm off (...)`: the robot is not
  where the program thinks. Put it on the start mark facing the start direction; the check repeats every 5 s and
  the robot leaves by itself once it passes. If the arena changed since mapping and the check keeps failing,
  `ros2 service call /mission/start std_srvs/srv/Trigger` starts anyway.
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

## Build

Copy `ee3033_interfaces` and `ee3033_mission` into the car's `/home/ros2_ws/src`, then:

```bash
cd /home/ros2_ws && colcon build --symlink-install --packages-select ee3033_interfaces ee3033_mission
```
YOLO weights are expected at `/home/ros2_ws/models/yolov8n.pt` (not in git; `yolov8n.pt` from Ultralytics).

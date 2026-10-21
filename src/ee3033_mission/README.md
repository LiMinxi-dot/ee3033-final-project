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

Services: `/mission/start`, `/mission/stop` (`std_srvs/Trigger`), `/target_localizer/reset` (`std_srvs/Empty`).

Tools:
- `ros2 run ee3033_mission setup_check [--computer]`: checks the course packages, devices, YOLO and network settings.
- `ros2 run ee3033_mission camera_lidar_check`: measures `camera_offset_yaw_deg` from camera vs LiDAR bearings of one object and saves an overlay image.
All tunable values are in `config/mission.yaml`.

## Target localization

1. YOLOv8n (imgsz 320, about 9 Hz on the Pi 5) detects the target class.
2. The box centre column gives the bearing in `camera_link` (`camera_info` K if calibrated, else `hfov_deg`).
3. TF at the image time turns the bearing into a ray in `map`.
4. The `/scan_fixed` scan closest in time, transformed to `map`, gives the range: the nearest cluster of hits around the ray. If `target_height` > 0, the box height gives a fallback range.
5. Observations are fused: the largest cluster within `cluster_radius`, then the median. The target is confirmed after `min_observations` with spread ≤ `max_spread`.

## Search strategy

The start pose is the map origin, because mapping starts at the start position. The mission:
1. Sets the AMCL initial pose and spins once at the start.
2. Builds viewpoints from the saved map: free space split into `viewpoint_spacing` blocks, the cell with the largest clearance in each block, keeping only cells reachable from the start.
3. Visits the viewpoints in nearest-neighbour order and spins 360° at each one.
4. Stops as soon as the target is confirmed, reports it, and navigates home.

## Demo procedure

Run the launch files on the car inside the ROS 2 container. Run teleop and RViz on the computer
(same `ROS_DOMAIN_ID`, `ROS_LOCALHOST_ONLY=0`).

```bash
# Step 1: mapping (robot at the start position)
ros2 launch ee3033_mission mapping.launch.py            # then drive from the VM (hold the key down):
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel_teleop
# cmd_vel_watchdog (on the car) relays /cmd_vel_teleop -> /cmd_vel and stops the robot
# 0.6 s after the last command, because the base keeps driving with the last /cmd_vel.
# or autonomous: ros2 launch ee3033_mission mapping.launch.py explore:=true
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

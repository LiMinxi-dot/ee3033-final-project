# EE3033 Final Project: Autonomous Mapping, Navigation and Target Localization

ROS 2 Humble software for the EE3033 balancing car (Raspberry Pi 5, N10P LiDAR, USB camera).
The car maps an unknown sandbox, then autonomously searches it for a target object, reports
the target position (x, y) in the map frame and returns to its starting position.

EE3033 Systems Integration and Design Lab, NUS. Group 6.

## Packages

| Package | Contents |
|---|---|
| [`ee3033_interfaces`](src/ee3033_interfaces) | Messages: `TargetDetection`, `TargetDetectionArray`, `TargetEstimate` |
| [`ee3033_mission`](src/ee3033_mission) | Nodes `target_detector`, `target_localizer`, `mission_manager`, `cmd_vel_watchdog`; tools `setup_check`, `camera_lidar_check`; launch files; parameters in `config/mission.yaml`; RViz config |

See [`src/ee3033_mission/README.md`](src/ee3033_mission/README.md) for nodes, topics and design details.

## System overview

```
Step 1 (mapping)   base + LiDAR + scan resampler + SLAM Toolbox  --(teleop via watchdog, or explore_lite)-->  saved map

Step 2 (mission)   base + LiDAR + scan resampler
                   map_server + AMCL  ->  map -> odom_combined TF
                   Nav2 (navigate_to_pose, spin)
                   camera -> target_detector (YOLOv8n) -> target_localizer (+ LiDAR, TF) -> /target/estimate
                   mission_manager: initial pose -> spin -> coverage viewpoints -> report (x, y) -> return home
```

- **Target localization:** the detection's bearing (camera model + calibrated mount offset) is cast as a ray in the map; the nearest LiDAR hits around the ray give the range; many observations are fused robustly (largest consistent cluster, median).
- **Search strategy:** viewpoints are generated from the saved map (free space split into blocks, the most open reachable cell per block), visited in nearest-neighbour order, with a 360° spin at each.
- **Safety:** `cmd_vel_watchdog` stops the robot if teleop commands stop arriving (the base keeps its last velocity), and `/mission/stop` cancels the mission at any time.

## Getting started on a car

### 1. Prerequisites

**Car:** the course Raspberry Pi OS image with the ROS 2 Humble container (`ros2-start`), whose
workspace `/home/ros2_ws` (= `~/ros2_ws` on the Pi) already contains the course packages
`balance_robot_base`, `lslidar_driver_n10p`, `balance_mapping`, `balance_navigation` and
`m-explore-ros2`, plus SLAM Toolbox, Nav2, `v4l2_camera` and the `ultralytics` Python package.
This repo does not modify any of them.

**Computer (VM):** ROS 2 Humble with `ros-humble-teleop-twist-keyboard`, `ros-humble-rviz2` and
`ros-humble-image-transport-plugins`, on the same network as the car (VMware: bridged adapter).
Both sides need the same `ROS_DOMAIN_ID` and `ROS_LOCALHOST_ONLY=0`.

### 2. Install on the car

```bash
# on the Pi (Raspberry Pi OS)
cd ~/ros2_ws/src && git clone https://github.com/LiMinxi-dot/ee3033-final-project.git

# inside the container
ros2-start
cd /home/ros2_ws && colcon build --symlink-install --packages-select ee3033_interfaces ee3033_mission
source install/setup.bash
mkdir -p /home/ros2_ws/maps
ros2 run ee3033_mission setup_check            # every line should be ✔
```

The YOLO weights (`/home/ros2_ws/models/yolov8n.pt`, 6.5 MB) are downloaded automatically on the
first start if the car has internet; otherwise copy `yolov8n.pt` there by hand.

On the computer, `ros2 run ee3033_mission setup_check --computer` checks the PC side (clone and
build the repo there too if you want the RViz config and to echo `/target/estimate`).

### 3. Camera calibration (already done for our car)

`config/mission.yaml` contains the camera mount calibration of our car (balance4): 1.5 cm left of
the centre line, optical axis 8.3° to the left of the course TF frame `camera_link`. Nothing needs
to be done unless the camera is moved or remounted. Then:

1. Measure by hand how far the camera is left (+) / right (−) of the centre line and put it in `camera_offset_y`.
2. Stand the car upright and start the sensors and the detector:
   ```bash
   ros2 launch ee3033_mission base_sensors.launch.py
   ros2 launch ee3033_mission perception.launch.py global_frame:=odom_combined
   ```
3. Put one object of a `target_classes` class about 1–1.5 m in front of the car, nothing else within
   ±15°, its middle at the LiDAR height (about 19 cm), and run
   ```bash
   ros2 run ee3033_mission camera_lidar_check
   ```
4. Put the suggested value in `camera_offset_yaw_deg` and check `/tmp/camera_lidar_check.png`: in the right
   panel the LiDAR points should sit on the objects.

Before the first mission, check the turning direction once: in teleop, `j` must turn the car
**left**. If it turns right, the base's angular sign is inverted and Nav2 will not work.

### 4. Run the demo

```bash
# Step 1 – mapping, car at the start position (car)
ros2 launch ee3033_mission mapping.launch.py
# drive from the computer, holding the keys down (i/,/j/l, k = stop)
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel_teleop -p speed:=0.15 -p turn:=0.6
# save the map (car), then stop the mapping launch
ros2 service call /slam_toolbox/save_map slam_toolbox/srv/SaveMap "{name: {data: /home/ros2_ws/maps/sandbox}}"

# Step 2 – mission, car back at the start with the same heading as at the start of mapping (car)
ros2 launch ee3033_mission mission.launch.py map:=/home/ros2_ws/maps/sandbox.yaml
ros2 service call /mission/start std_srvs/srv/Trigger    # when /mission/state is IDLE
ros2 service call /mission/stop std_srvs/srv/Trigger     # emergency stop

# watch from the computer
rviz2 -d $(ros2 pkg prefix ee3033_mission)/share/ee3033_mission/rviz/mission.rviz
```

The target position is printed in the `mission_manager` log, published on `/mission/target` and
written to `/home/ros2_ws/mission_result.yaml`.

## Known limitations

- The base has no command timeout and keeps driving after a fall; only the motor switch stops it.
- Transparent or shiny objects and objects lower than the LiDAR plane give no LiDAR range; set
  `target_height` to use the monocular fallback.
- The full mission (step 2) has not been tested end-to-end on the car yet.

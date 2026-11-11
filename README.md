# EE3033 Final Project: Autonomous Mapping, Navigation and Target Localization

ROS 2 Humble software for the EE3033 balancing car (Raspberry Pi 5, N10P LiDAR, USB camera).
The car maps an unknown sandbox, then autonomously searches it for a target object, reports
the target position (x, y) in the map frame and returns to its starting position.

EE3033 Systems Integration and Design Lab, NUS. Group 6.

## Packages

| Package | Contents |
|---|---|
| [`ee3033_interfaces`](src/ee3033_interfaces) | Messages: `TargetDetection`, `TargetDetectionArray`, `TargetEstimate` |
| [`ee3033_mission`](src/ee3033_mission) | Nodes `auto_mapper`, `target_detector`, `target_localizer`, `mission_manager`, `odom_scaler`, `cmd_vel_watchdog`; tools `setup_check`, `camera_lidar_check`, `odom_turn_test`; `scripts/demo.sh`; launch files; parameters in `config/mission.yaml`; RViz config |

See [`src/ee3033_mission/README.md`](src/ee3033_mission/README.md) for nodes, topics and design details.

## System overview

```
Both steps         base (+ odom_scaler: turn correction) + LiDAR + scan resampler

Step 1 (mapping)   SLAM Toolbox + Nav2
                   auto_mapper: frontier exploration -> return to the start (start heading) -> saved map

Step 2 (mission)   map_server + AMCL (tuned laser model)  ->  map -> odom_combined TF
                   Nav2 (navigate_to_pose, spin)
                   camera -> target_detector (YOLOv8n) -> target_localizer (+ LiDAR, TF) -> /target/estimate
                   mission_manager: localization check (auto-calibration) -> spin -> coverage viewpoints
                                    -> report (x, y) -> return home (start heading)
```

- **Autonomous mapping:** `auto_mapper` drives to the cheapest reachable frontier of the live SLAM map until no
  frontier is left or the free area stops growing, then returns to the start.
- **Routes:** both steps drive planned routes as short straight legs along corridor centres, turning in place at
  sharp corners and driving on through the other waypoints without stopping.
- **Target localization:** the detection's bearing (camera model + calibrated mount offset) is cast as a ray in the
  map; the nearest LiDAR hits around the ray give the range; many observations are fused robustly (largest
  consistent cluster, median).
- **Search strategy:** candidate viewpoints on a grid of the saved map; a greedy set cover picks the ones that see
  all free space within `view_range` (line of sight), visited in nearest-neighbour order with a 360° spin at each.
- **Localization:** the base over-reports turns (~2.5 % left, ~4.5 % right on our car); `odom_scaler` scales them
  out (`odom_turn_test` measures it). AMCL runs with a stronger laser model than the course defaults
  (`launch/localization.launch.py`). Before leaving and at every viewpoint the LiDAR scan is checked against the
  map walls.
- **Safety:** `/mission/stop` and `/explore/stop` stop the robot at any time; `cmd_vel_watchdog` stops it if teleop
  commands stop arriving (the base keeps its last velocity).

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
the centre line, optical axis 3.0° to the left (`camera_lidar_check` with our LiDAR TF,
`base_sensors.launch.py`). Nothing needs to be done unless the camera is moved or remounted. Then:

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

### 4. Odometry turn scale (already done for our car)

`scripts/demo.sh` passes `odom_yaw_scale:=0.976 odom_yaw_scale_cw:=0.956` (true / odometry turn, left / right) for
our car. For another car, measure it with only the base and the LiDAR running, the car at least 20 cm from anything
and nobody moving nearby:

```bash
ros2 launch ee3033_mission base_sensors.launch.py
ros2 run ee3033_mission odom_turn_test                          # dry run: checks only, the car does not move
ros2 run ee3033_mission odom_turn_test --ros-args -p go:=true   # THE CAR TURNS IN PLACE: 4 x 360 deg
```

and put the two ratios in `ODOM_ARGS` in `scripts/demo.sh`. Restarted with them
(`base_sensors.launch.py odom_yaw_scale:=... odom_yaw_scale_cw:=...`), the test should give about 1.000.

### 5. Run the demo

```bash
# on the car, inside the container; each step starts by itself (~45 s), THE CAR MOVES
ros2 run ee3033_mission demo.sh mapping   # car at the start: explore, return, save the map, stop
# place the target
ros2 run ee3033_mission demo.sh mission   # car at the start: search, report the target, return home
ros2 run ee3033_mission demo.sh stop      # stop the car and all ROS nodes
ros2 service call /mission/stop std_srvs/srv/Trigger   # emergency stop from the computer (or /explore/stop)

# watch from the computer
rviz2 -d $(ros2 pkg prefix ee3033_mission)/share/ee3033_mission/rviz/mission.rviz
```

The target position is printed in the `mission_manager` log, published on `/mission/target` and
written to `/home/ros2_ws/mission_result.yaml`. Each step also saves its log, maps and a bag to
`/home/ros2_ws/logs/<time>_<step>/`.

**Fallback** if the car gets lost (`[loc]` warnings at the viewpoints, e.g. people standing in the arena) or its
turns look wrong: the setup without the turn correction and the tuned AMCL,

```bash
ros2 run ee3033_mission demo.sh mapping odom_yaw_scale:=1.0 odom_yaw_scale_cw:=1.0
ros2 run ee3033_mission demo.sh mission odom_yaw_scale:=1.0 odom_yaw_scale_cw:=1.0 amcl_tuned:=false
```

If Nav2 does not come up (`Waiting: ... nav2_active=False` for more than a minute): `demo.sh stop`, then start the
step again. The manual launch commands are in [`src/ee3033_mission/README.md`](src/ee3033_mission/README.md).

## Known limitations

- The base has no command timeout and keeps driving after a fall; only the motor switch stops it.
- Transparent or shiny objects and objects lower than the LiDAR plane give no LiDAR range; set
  `target_height` to use the monocular fallback.
- Both steps ran end-to-end on the site sandbox and in a home arena (Nov 2026). The turn correction and the tuned
  AMCL were checked at home and in replays of site recordings, not yet live on site.
- People or objects in the arena that are not in the map can pull AMCL away; the viewpoint checks then warn.
- Below 11 V the motors stop (low-voltage protection): recharge below 11.5 V.

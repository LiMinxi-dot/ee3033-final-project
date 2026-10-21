# EE3033 Final Project: Autonomous Mapping, Navigation and Target Localization

ROS 2 Humble software for the EE3033 balancing car (Raspberry Pi 5, N10P LiDAR, USB camera).
The car maps an unknown sandbox, then autonomously searches it for a target object, reports
the target position (x, y) in the map frame and returns to its starting position.

EE3033 Systems Integration and Design Lab, NUS. Group 6.

## Packages

| Package | Contents |
|---|---|
| [`ee3033_interfaces`](src/ee3033_interfaces) | Messages: `TargetDetection`, `TargetDetectionArray`, `TargetEstimate` |
| [`ee3033_mission`](src/ee3033_mission) | Nodes `target_detector`, `target_localizer`, `mission_manager`, `cmd_vel_watchdog`; launch files for mapping and the mission; parameters in `config/mission.yaml`; RViz config |

The packages build on the course packages already installed on the car
(`balance_robot_base`, `lslidar_driver_n10p`, `balance_mapping`, `balance_navigation`) and on
SLAM Toolbox, Nav2 and Ultralytics YOLO. They do not modify the course packages.

## System overview

```
Step 1 (mapping)   base + LiDAR + scan resampler + SLAM Toolbox  --(teleop via watchdog, or explore_lite)-->  saved map

Step 2 (mission)   base + LiDAR + scan resampler
                   map_server + AMCL  ->  map -> odom_combined TF
                   Nav2 (navigate_to_pose, spin)
                   camera -> target_detector (YOLOv8n) -> target_localizer (+ LiDAR, TF) -> /target/estimate
                   mission_manager: initial pose -> spin -> coverage viewpoints -> report (x, y) -> return home
```

- **Target localization:** the detection's bearing (camera model + measured mount offset) is cast as a ray in the map; the nearest LiDAR hits around the ray give the range; many observations are fused robustly (largest consistent cluster, median).
- **Search strategy:** viewpoints are generated from the saved map (free space split into blocks, the most open reachable cell per block), visited in nearest-neighbour order, with a 360° spin at each.
- **Safety:** `cmd_vel_watchdog` stops the robot if teleop commands stop arriving (the base keeps its last velocity), and `/mission/stop` cancels the mission at any time.

See [`src/ee3033_mission/README.md`](src/ee3033_mission/README.md) for the nodes, topics and the demo procedure.

## Quick start

```bash
# on the car, in the ROS 2 container, after copying src/* to /home/ros2_ws/src and building
ros2 launch ee3033_mission mapping.launch.py                                   # step 1
ros2 launch ee3033_mission mission.launch.py map:=/home/ros2_ws/maps/sandbox.yaml   # step 2
ros2 service call /mission/start std_srvs/srv/Trigger
```

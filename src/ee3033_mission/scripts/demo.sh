#!/usr/bin/env bash
# The two demo steps, one command each. Run on the car inside the ROS 2 container:
#
#   ros2 run ee3033_mission demo.sh mapping   # step 1: robot at the start; explores, returns to the start,
#                                             #   saves the map, then stops the mapping nodes
#   (place the target)
#   ros2 run ee3033_mission demo.sh mission   # step 2: YOLO loads first; place the robot on the start mark and
#                                             #   press Enter: searches, reports, returns home
#   ros2 run ee3033_mission demo.sh stop      # stop the robot and all ROS nodes
#   ros2 run ee3033_mission demo.sh manual    # fallback for step 1: SLAM + driving with the keyboard in this
#                                             #   terminal; Ctrl-C (robot back on the start mark) saves the map
#   Further arguments go to the launch file, e.g. demo.sh mapping laser_yaw:=0.0 max_speed:=0.10
#   --local (or DEMO_LOCAL=1): all ROS traffic stays on the car (ROS_LOCALHOST_ONLY=1); the VM sees nothing (no
#   RViz, no ros2 CLI, no emergency stop from the VM: Ctrl-C here, or demo.sh stop on the car). 11-12, big arena with
#   poor Wi-Fi: the base and LiDAR nodes stalled for up to 3 s (header stamps too), SLAM dropped scans, two falls.
#   demo.sh stop (and the cleanup before each step) knows when the nodes were started with --local.
#   (they override ODOM_ARGS / MISSION_ARGS; fallback to the setup run on site on 11-11 morning, see the README:
#    demo.sh mapping|mission odom_yaw_scale:=1.0 odom_yaw_scale_cw:=1.0 amcl_tuned:=false)
#
# THE ROBOT MOVES BY ITSELF: each step starts automatically when Nav2 is ready (~45 s).
# Each step first stops every ROS node left over from earlier runs (a second base or
# LiDAR driver would block the serial ports). Ctrl-C stops the robot and everything.
# Full logs: /tmp/ee3033_mapping.log, /tmp/ee3033_mission.log (overwritten by the next run). At the end of each
# step (also after a failure or Ctrl-C) the log, the map and the result are copied to /home/ros2_ws/logs/<time>_<step>/,
# where each step also records a bag (mapping: to rerun SLAM offline; mission: falls, localization, speeds).
# Emergency stop from the VM (not with --local): ros2 service call /explore/stop (or /mission/stop) std_srvs/srv/Trigger

set -u
ulimit -c 0        # no core dumps: the LiDAR driver aborts on every stop and left a 14 MB core each time in the
                   # working directory (16 of them, 194 MB, filled the SD card on 11-12)
MAP=/home/ros2_ws/maps/sandbox
MAP_LOG=/tmp/ee3033_mapping.log
MISSION_LOG=/tmp/ee3033_mission.log
# Every ROS process started by our launch files (vendor drivers, SLAM, Nav2, our nodes)
PATTERN='bin/ros2 launch|bin/ros2 bag record|/home/ros2_ws/install/|/opt/ros/humble/lib/(nav2_|slam_toolbox|tf2_ros|v4l2_camera|image_transport)'
LAUNCH_PID=
TAIL_PID=
BAG_PID=
STEP=${1:-}
RUN_LOG=           # log of the step running now
RUN_DIR=/home/ros2_ws/logs/$(date +%m%d_%H%M%S)_${1:-}
# Odometry turn correction (odom_scaler via base_sensors.launch.py): the base over-reports left turns by ~2.5 % and
# right turns by ~4.5 %. odom_turn_test on 11-11: true / odometry 0.968 / 0.958 without, 0.999 / 1.004 with these.
ODOM_ARGS=(odom_yaw_scale:=0.976 odom_yaw_scale_cw:=0.956)
# AMCL laser-model overrides (localization.launch.py): home mission 11-11 with them 3.6 cm mean position error, 11.4 cm
# with the vendor parameters; replays of the two site missions 3.4 cm (4.5-5.4 cm).
MISSION_ARGS=(amcl_tuned:=true)

say() { echo -e "\033[1;36m[demo $(date +%H:%M:%S)] $*\033[0m"; }

nodes() {          # running ROS processes, except this script
  pgrep -af "$PATTERN" | grep -v -e demo.sh -e "^$$ "
}

start_bag() {      # start_bag <topics...>: record into the run folder
  mkdir -p "$RUN_DIR"
  setsid env --default-signal=INT ros2 bag record -o "$RUN_DIR/bag" "$@" > /tmp/ee3033_bag.log 2>&1 < /dev/null &
  BAG_PID=$!
  say "recording a bag to $RUN_DIR/bag"
}

stop_bag() {       # SIGINT so that rosbag2 writes its metadata
  [ -n "$BAG_PID" ] || return 0
  kill -INT -- "-$BAG_PID" 2>/dev/null
  for _ in $(seq 1 10); do kill -0 "$BAG_PID" 2>/dev/null || break; sleep 0.5; done
  BAG_PID=
}

stop_all() {
  stop_bag
  if [ -n "$TAIL_PID" ]; then
    kill "$TAIL_PID" 2>/dev/null
    pkill -f "[t]ail -n 0 -F /tmp/ee3033_"
  fi
  TAIL_PID=
  [ -n "$LAUNCH_PID" ] && kill -INT -- "-$LAUNCH_PID" 2>/dev/null
  LAUNCH_PID=
  for _ in $(seq 1 20); do
    nodes >/dev/null || return 0
    sleep 0.5
  done
  say "some nodes did not stop on SIGINT; terminating them"
  nodes | awk '{print $1}' | xargs -r kill -TERM 2>/dev/null; sleep 2
  nodes | awk '{print $1}' | xargs -r kill -KILL 2>/dev/null; sleep 1
  if nodes >/dev/null; then
    say "WARNING: still running:"; nodes
  fi
}

stop_robot() {     # cancel Nav2 goals and send zero velocity before the nodes go away:
                   # the base firmware keeps driving with the last /cmd_vel it received
  timeout 4 ros2 service call /explore/stop std_srvs/srv/Trigger >/dev/null 2>&1 &
  local a=$!
  timeout 4 ros2 service call /mission/stop std_srvs/srv/Trigger >/dev/null 2>&1 &
  local b=$!
  timeout 4 ros2 topic pub -r 10 -t 20 /cmd_vel geometry_msgs/msg/Twist "{}" >/dev/null 2>&1
  wait "$a" "$b" 2>/dev/null
}

save_run() {       # keep this run's log, maps and result (the next run overwrites /tmp/ee3033_*.log)
  [ -n "$RUN_LOG" ] || return 0
  mkdir -p "$RUN_DIR" && cp "$RUN_LOG" "$RUN_DIR"/
  # only the maps written by this run (sandbox, _peak, _after_return): older ones stay in maps/ from earlier runs
  find "$(dirname "$MAP")" -maxdepth 1 -name "$(basename "$MAP")*" -newermt "@$T0" -exec cp {} "$RUN_DIR"/ \; 2>/dev/null
  [ "${1:-}" = mission ] && cp /home/ros2_ws/mission_result.yaml "$RUN_DIR"/ 2>/dev/null
  say "log and map saved to $RUN_DIR"
}

save_manual_map() {  # manual mapping: SLAM's map now -> $MAP.pgm / .yaml (what the mission reads)
  say "saving the map to $MAP ..."
  # our saver, not /slam_toolbox/save_map: that one runs Nav2's map_saver, which waited only 2 s for /map and
  # failed on 11-12 (result 255), and writes free_thresh 0.25 (unknown cells read as free)
  timeout 25 ros2 run ee3033_mission save_map "$MAP" > /tmp/ee3033_save_map.log 2>&1
  if [ -f "$MAP.yaml" ] && [ "$(stat -c %Y "$MAP.yaml")" -ge "$T0" ]; then
    say "map saved: $MAP.yaml"
  else
    say "WARNING: the map was NOT saved (see /tmp/ee3033_save_map.log); the nodes are stopped now"
  fi
}

on_interrupt() {
  if [ "$STEP" = manual ] && [ -n "$LAUNCH_PID" ]; then
    say "Ctrl-C: stopping the robot, saving the map, stopping all nodes"
    stop_robot
    save_manual_map
  else
    say "interrupted: stopping the robot and all nodes"
    [ -n "$LAUNCH_PID" ] && stop_robot
  fi
  stop_all
  save_run "$STEP"
  exit 130
}
trap on_interrupt INT TERM

start_launch() {   # start_launch <log> <args...>: own process group, so it can be stopped as a whole
  local log=$1; shift
  # background jobs of a script ignore SIGINT; give the launch back the default so it can be stopped
  setsid env --default-signal=INT ros2 launch ee3033_mission "$@" > "$log" 2>&1 < /dev/null &
  LAUNCH_PID=$!
  # Shown: our nodes' progress lines and warnings, and errors of any node. worldToMap errors are the Nav2
  # planner's goal-tolerance search near the map edge (noise, many per second).
  tail -n 0 -F "$log" 2>/dev/null | grep --line-buffered -E \
    '\[state\]|\[loc\]|\[plan\]|Viewpoint [0-9]|Leg [0-9]|Goal [0-9]|Turning|Blacklisted|Exploration finished|Map saved|Home|Returning home|TARGET|Initial pose|System ready|Waiting:|no range|\] \[WARN\] \[[0-9.]+\] \[(auto_mapper|mission_manager|target_localizer)\]|ERROR|FATAL|died' \
    | grep --line-buffered -v 'worldToMap failed' &
  TAIL_PID=$!
}

wait_state() {     # wait_state <log>: sets RESULT to the final state (DONE / FAILED / STOPPED)
  RESULT=          # (not called as $(...), so Ctrl-C is handled between the short commands)
  while kill -0 "$LAUNCH_PID" 2>/dev/null; do
    RESULT=$(grep -oE '\[state\] [A-Z_]+ -> (DONE|FAILED|STOPPED)' "$1" | tail -1 | awk '{print $NF}')
    [ -n "$RESULT" ] && return
    sleep 2
  done
  RESULT=LAUNCH_EXITED
}

case "${1:-}" in
  mapping|mission|manual|stop) STEP=$1;;
  *) echo "usage: ros2 run ee3033_mission demo.sh mapping|mission|manual|stop [--local] [launch args]"; exit 2;;
esac
LOCAL=${DEMO_LOCAL:-0}
LAUNCH_ARGS=()
for a in "${@:2}"; do
  if [ "$a" = --local ]; then LOCAL=1; else LAUNCH_ARGS+=("$a"); fi
done
LOCAL_MARK=/tmp/ee3033_demo_local   # exists while the nodes run in local mode
# nodes left from a local run listen on the loopback only: stop them (zero /cmd_vel, service calls) the same way
[ -f "$LOCAL_MARK" ] && export ROS_LOCALHOST_ONLY=1

if nodes >/dev/null; then
  say "stopping ROS nodes left from earlier runs:"
  nodes | cut -c1-120
  stop_robot
  stop_all
fi
# Shared-memory segments of killed nodes (Fast DDS): on 11-11 Nav2's bringup hung twice in a row (lifecycle_manager
# waited for a lost change_state response, "failed to send response ... (timeout)") after nodes had been killed; after
# `fastdds shm clean` (3 zombie segments) it started. stop_all's kill -KILL leaves such segments (21 after one stop).
# Only segments no running process uses are removed.
fastdds shm clean 2>&1 | grep -E "[1-9][0-9]* zombie" | sed 's/^/[demo] /'
[ "$1" = stop ] && { rm -f "$LOCAL_MARK" /tmp/ee3033_go; say "all ROS nodes stopped"; exit 0; }
if [ "$LOCAL" = 1 ]; then
  export ROS_LOCALHOST_ONLY=1; touch "$LOCAL_MARK"
  say "LOCAL MODE: ROS traffic stays on the car (the VM and RViz see nothing; stop with Ctrl-C or demo.sh stop)"
else
  export ROS_LOCALHOST_ONLY=0; rm -f "$LOCAL_MARK"
fi

T0=$(date +%s)
if [ "$1" = mapping ]; then
  say "STEP 1: auto mapping. The robot starts exploring by itself when Nav2 is ready (~45 s)."
  RUN_LOG=$MAP_LOG
  start_launch "$MAP_LOG" mapping.launch.py explore:=true auto_start:=true "${ODOM_ARGS[@]}" "${LAUNCH_ARGS[@]}"
  # what SLAM got (~75 kB/s) and the speed commands and plans (11-06: the robot stood still at leg ends, no data why)
  start_bag /scan /scan_fixed /tf /tf_static /odom /robot_euler /map /cmd_vel /cmd_vel_nav /plan
  wait_state "$MAP_LOG"
  stop_bag           # also when nodes are left running for inspection (the SD card has ~0.5 GB free)
  save_run mapping
  if [ "$RESULT" != DONE ]; then
    say "mapping ended with '$RESULT' (see $MAP_LOG); nodes left running for inspection. Ctrl-C to stop."
    wait "$LAUNCH_PID"
    exit 1
  fi
  say "mapping done in $(( $(date +%s) - T0 )) s, map saved to $MAP.yaml. Stopping the mapping nodes."
  stop_all
  say "Next (robot at the start, target placed): ros2 run ee3033_mission demo.sh mission"
  exit 0
fi

if [ "$1" = manual ]; then
  [ -t 0 ] || { say "manual mapping needs a terminal (ssh -t, docker exec -it)"; exit 2; }
  say "STEP 1 (manual): SLAM + driving with the keyboard. The map origin is where the robot stands now."
  RUN_LOG=$MAP_LOG
  start_launch "$MAP_LOG" mapping.launch.py "${ODOM_ARGS[@]}" "${LAUNCH_ARGS[@]}"
  start_bag /scan /scan_fixed /tf /tf_static /odom /robot_euler /map /cmd_vel /cmd_vel_teleop
  say "waiting for SLAM Toolbox ..."
  for _ in $(seq 1 60); do
    grep -q "starting SLAM Toolbox" "$MAP_LOG" && timeout 5 ros2 service list 2>/dev/null | grep -q '^/slam_toolbox/save_map$' \
      && break
    sleep 1
  done
  say "SLAM is up. Drive slowly round the whole arena, back to the start mark facing the start direction, then Ctrl-C:"
  say "the robot stops, the map is saved to $MAP.yaml and all nodes stop."
  ros2 run ee3033_mission key_teleop    # Ctrl-C ends it and runs on_interrupt (stop, save the map, stop all)
  say "key_teleop ended: stopping the robot, saving the map, stopping all nodes"
  stop_robot
  save_manual_map
  stop_all
  save_run mapping
  say "Next (robot at the start, target placed): ros2 run ee3033_mission demo.sh mission"
  exit 0
fi

[ -f "$MAP.yaml" ] || { say "no map at $MAP.yaml: run demo.sh mapping first"; exit 1; }
say "STEP 2: mission on $MAP.yaml. The robot starts searching by itself when Nav2 is ready."
RUN_LOG=$MISSION_LOG
# In a terminal: camera + YOLO load first (mission.launch.py yolo_first), then Enter once the robot is placed
# starts the base driver (its odometry origin = where the robot stands then), AMCL and the mission.
GO=/tmp/ee3033_go
rm -f "$GO"
WAIT_ARGS=()
[ -t 0 ] && WAIT_ARGS=(start_file:="$GO")
start_launch "$MISSION_LOG" mission.launch.py map:="$MAP.yaml" auto_start:=true "${ODOM_ARGS[@]}" "${MISSION_ARGS[@]}" \
  "${WAIT_ARGS[@]}" "${LAUNCH_ARGS[@]}"
if [ -t 0 ]; then
  say "loading YOLO (the base is not started yet: place the robot meanwhile) ..."
  while kill -0 "$LAUNCH_PID" 2>/dev/null && ! grep -q "YOLO is ready: waiting for $GO" "$MISSION_LOG"; do sleep 0.5; done
  if kill -0 "$LAUNCH_PID" 2>/dev/null; then
    say "YOLO is ready. Place the robot on the start mark, facing the start direction, then press Enter to start."
    read -r _
    touch "$GO"
    say "starting the base, AMCL and the mission (the robot starts when Nav2 is ready)"
  fi
fi
# pitch (falls), battery, what AMCL got and the speed commands (11-06: a fall and a lost localization without data)
start_bag /scan_fixed /tf /tf_static /odom /robot_euler /power_voltage /amcl_pose /cmd_vel /cmd_vel_nav /plan \
  /mission/state /target/estimate
wait_state "$MISSION_LOG"
stop_bag
say "mission ended with '$RESULT' after $(( $(date +%s) - T0 )) s"
[ -f /home/ros2_ws/mission_result.yaml ] && cat /home/ros2_ws/mission_result.yaml
save_run mission
say "nodes left running (RViz still works). Ctrl-C to stop all."
wait "$LAUNCH_PID"

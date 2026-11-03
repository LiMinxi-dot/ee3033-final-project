#!/usr/bin/env bash
# The two demo steps, one command each. Run on the car inside the ROS 2 container:
#
#   ros2 run ee3033_mission demo.sh mapping   # step 1: robot at the start; explores, returns to the start,
#                                             #   saves the map, then stops the mapping nodes
#   (place the target)
#   ros2 run ee3033_mission demo.sh mission   # step 2: robot at the start; searches, reports, returns home
#   ros2 run ee3033_mission demo.sh stop      # stop the robot and all ROS nodes
#
# THE ROBOT MOVES BY ITSELF: each step starts automatically when Nav2 is ready (~45 s).
# Each step first stops every ROS node left over from earlier runs (a second base or
# LiDAR driver would block the serial ports). Ctrl-C stops the robot and everything.
# Full logs: /tmp/ee3033_mapping.log, /tmp/ee3033_mission.log
# Emergency stop from the VM: ros2 service call /explore/stop (or /mission/stop) std_srvs/srv/Trigger

set -u
MAP=/home/ros2_ws/maps/sandbox
MAP_LOG=/tmp/ee3033_mapping.log
MISSION_LOG=/tmp/ee3033_mission.log
# Every ROS process started by our launch files (vendor drivers, SLAM, Nav2, our nodes)
PATTERN='bin/ros2 launch|/home/ros2_ws/install/|/opt/ros/humble/lib/(nav2_|slam_toolbox|tf2_ros|v4l2_camera|image_transport)'
LAUNCH_PID=
TAIL_PID=

say() { echo -e "\033[1;36m[demo $(date +%H:%M:%S)] $*\033[0m"; }

nodes() {          # running ROS processes, except this script
  pgrep -af "$PATTERN" | grep -v -e demo.sh -e "^$$ "
}

stop_all() {
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

on_interrupt() {
  say "interrupted: stopping the robot and all nodes"
  [ -n "$LAUNCH_PID" ] && stop_robot
  stop_all
  exit 130
}
trap on_interrupt INT TERM

start_launch() {   # start_launch <log> <args...>: own process group, so it can be stopped as a whole
  local log=$1; shift
  # background jobs of a script ignore SIGINT; give the launch back the default so it can be stopped
  setsid env --default-signal=INT ros2 launch ee3033_mission "$@" > "$log" 2>&1 < /dev/null &
  LAUNCH_PID=$!
  # worldToMap errors: the Nav2 planner's goal-tolerance search near the map edge (noise, many per second)
  tail -n 0 -F "$log" 2>/dev/null | grep --line-buffered -E \
    '\[state\]|\[loc\]|\[plan\]|Viewpoint [0-9]|Goal [0-9]|Blacklisted|Exploration finished|Map saved|Home|TARGET|Initial pose|System ready|Waiting:|ERROR|FATAL|died' \
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
  mapping|mission|stop) ;;
  *) echo "usage: ros2 run ee3033_mission demo.sh mapping|mission|stop"; exit 2;;
esac

if nodes >/dev/null; then
  say "stopping ROS nodes left from earlier runs:"
  nodes | cut -c1-120
  stop_robot
  stop_all
fi
[ "$1" = stop ] && { say "all ROS nodes stopped"; exit 0; }

T0=$(date +%s)
if [ "$1" = mapping ]; then
  say "STEP 1: auto mapping. The robot starts exploring by itself when Nav2 is ready (~45 s)."
  start_launch "$MAP_LOG" mapping.launch.py explore:=true auto_start:=true
  wait_state "$MAP_LOG"
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

[ -f "$MAP.yaml" ] || { say "no map at $MAP.yaml: run demo.sh mapping first"; exit 1; }
say "STEP 2: mission on $MAP.yaml. The robot starts searching by itself when Nav2 is ready."
start_launch "$MISSION_LOG" mission.launch.py map:="$MAP.yaml" auto_start:=true
wait_state "$MISSION_LOG"
say "mission ended with '$RESULT' after $(( $(date +%s) - T0 )) s"
[ -f /home/ros2_ws/mission_result.yaml ] && cat /home/ros2_ws/mission_result.yaml
say "nodes left running (RViz still works). Ctrl-C to stop all."
wait "$LAUNCH_PID"

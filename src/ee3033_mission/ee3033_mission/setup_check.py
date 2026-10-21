"""Check that a car has everything this project needs before running it.

  ros2 run ee3033_mission setup_check [--video /dev/video0] [--computer]

Run it on the car inside the ROS 2 container (or with --computer on the PC,
which only checks what the PC needs: teleop, RViz, image plugins, network
settings). Prints one line per check and exits non-zero if something required
is missing.
"""

import argparse
import importlib
import os
import sys

import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_prefix, get_package_share_directory

OK, WARN, FAIL = '✔', '!', '✘'


class Checker:

    def __init__(self):
        self.failed = 0
        self.warned = 0

    def report(self, status, what, detail=''):
        if status == FAIL:
            self.failed += 1
        elif status == WARN:
            self.warned += 1
        print(f' {status} {what}' + (f'  ({detail})' if detail else ''))

    def package(self, name, why, required=True):
        try:
            self.report(OK, f'package {name}', get_package_prefix(name))
            return True
        except PackageNotFoundError:
            self.report(FAIL if required else WARN, f'package {name} missing', why)
            return False

    def path(self, path, what, required=True, hint=''):
        if os.path.exists(path):
            self.report(OK, what, path)
            return True
        self.report(FAIL if required else WARN, f'{what} missing: {path}', hint)
        return False

    def python(self, module, why, required=True):
        try:
            m = importlib.import_module(module)
            self.report(OK, f'python {module}', getattr(m, '__version__', ''))
            return True
        except Exception as e:  # noqa: BLE001
            self.report(FAIL if required else WARN, f'python {module} not importable', f'{why}: {e}')
            return False


def check_network(c):
    dom = os.environ.get('ROS_DOMAIN_ID')
    if dom is None:
        c.report(WARN, 'ROS_DOMAIN_ID not set', 'car and PC must use the same value, e.g. export ROS_DOMAIN_ID=4')
    else:
        c.report(OK, f'ROS_DOMAIN_ID={dom}', 'must be the same on the car and the PC')
    if os.environ.get('ROS_LOCALHOST_ONLY', '0') == '1':
        c.report(FAIL, 'ROS_LOCALHOST_ONLY=1', 'the car and the PC cannot see each other; set it to 0')
    else:
        c.report(OK, 'ROS_LOCALHOST_ONLY is not 1')


def check_car(c, video):
    print('ROS packages (course packages on the car image)')
    c.package('balance_robot_base', 'robot base driver')
    c.package('lslidar_driver_n10p', 'LiDAR driver')
    if c.package('balance_mapping', 'SLAM config + scan_resampler.py'):
        c.path(os.path.join(get_package_prefix('balance_mapping'), 'lib', 'balance_mapping', 'scan_resampler.py'),
               'scan_resampler.py')
    c.package('balance_navigation', 'Nav2 / AMCL config')
    c.package('slam_toolbox', 'apt: ros-humble-slam-toolbox')
    c.package('nav2_bringup', 'apt: ros-humble-navigation2 ros-humble-nav2-bringup')
    c.package('nav2_map_server', 'apt: ros-humble-nav2-map-server')
    c.package('explore_lite', 'm-explore-ros2, only for mapping.launch.py explore:=true', required=False)
    c.package('v4l2_camera', 'apt: ros-humble-v4l2-camera')
    c.package('ee3033_interfaces', 'build this repo: colcon build --packages-select ee3033_interfaces ee3033_mission')

    print('Devices')
    c.path('/dev/balance_controller', 'base serial port', hint='udev rule from the course image; is the STM32 USB connected?')
    c.path('/dev/balance_lidar', 'LiDAR serial port', hint='udev rule from the course image; is the LiDAR USB connected?')
    c.path(video, 'camera', hint='list cameras with: v4l2-ctl --list-devices; pass video_device:=/dev/videoN')

    print('Detection')
    c.python('cv2', 'OpenCV')
    c.python('numpy', 'NumPy')
    if c.python('ultralytics', 'pip install ultralytics (inside the container)'):
        cfg = os.path.join(get_package_share_directory('ee3033_mission'), 'config', 'mission.yaml')
        model = yaml.safe_load(open(cfg))['target_detector']['ros__parameters']['model_path']
        c.path(model, 'YOLO model', required=False,
               hint='downloaded automatically on first start if the car has internet')

    print('Files')
    maps = '/home/ros2_ws/maps'
    if os.path.isdir(maps) and os.access(maps, os.W_OK):
        c.report(OK, 'map folder writable', maps)
    else:
        c.report(WARN, f'map folder {maps} missing', f'create it with: mkdir -p {maps}')


def check_computer(c):
    print('Computer (teleop + RViz)')
    c.package('teleop_twist_keyboard', 'apt: ros-humble-teleop-twist-keyboard')
    c.package('rviz2', 'apt: ros-humble-rviz2')
    c.package('compressed_image_transport', 'apt: ros-humble-image-transport-plugins (camera view in RViz)',
              required=False)
    c.package('ee3033_interfaces', 'optional on the PC: needed to echo /target/estimate', required=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--video', default='/dev/video0')
    ap.add_argument('--computer', action='store_true', help='check the PC instead of the car')
    args, _ = ap.parse_known_args(argv)

    c = Checker()
    print('Network')
    check_network(c)
    if args.computer:
        check_computer(c)
    else:
        check_car(c, args.video)
    print(f'\n{c.failed} missing, {c.warned} warnings')
    sys.exit(1 if c.failed else 0)


if __name__ == '__main__':
    main()

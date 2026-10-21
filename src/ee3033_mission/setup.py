from glob import glob

from setuptools import setup

package_name = 'ee3033_mission'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='EE3033 Group 6',
    maintainer_email='LiMinxi-dot@users.noreply.github.com',
    description='EE3033 balancing car: autonomous target search and localization mission.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'target_detector = ee3033_mission.target_detector:main',
            'target_localizer = ee3033_mission.target_localizer:main',
            'mission_manager = ee3033_mission.mission_manager:main',
            'cmd_vel_watchdog = ee3033_mission.cmd_vel_watchdog:main',
        ],
    },
)

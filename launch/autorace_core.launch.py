#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("autorace_core_tank")
    default_params = os.path.join(pkg_share, "config", "autorace_core.yaml")

    params_file = LaunchConfiguration("params_file")

    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params,
            description="Path to a YAML file with autorace_core_tank parameters"
        ),
        Node(
            package="autorace_core_tank",
            executable="lane_pid_controller",
            name="autorace_core_tank",
            output="screen",
            parameters=[params_file],
        ),
    ])

# Copyright 2020 Yutaka Kondo <yutaka.kondo@youtalk.jp>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node

import xacro


def generate_launch_description():
    robot_name = "leap_hand_custom"
    package_name = robot_name + "_description"
    rviz_config = os.path.join(get_package_share_directory(
        package_name), "launch", robot_name + "_debug.rviz")
    robot_description = os.path.join(get_package_share_directory(
        package_name), "urdf", robot_name + "_dummy.urdf.xacro")
    robot_description_config = xacro.process_file(robot_description)

    controller_config = os.path.join(
        get_package_share_directory(
            package_name), "controllers", "controllers.yaml"
    )

    return LaunchDescription([
        # TODO(yongpeng): disable ros2 control (default: enabled)
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            parameters=[
                {"robot_description": robot_description_config.toxml()}, controller_config],
            output="screen",
        ),

        Node(
            package="controller_manager",
            executable="spawner.py",
            arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
            output="screen",
        ),

        # Node(
        #     package="controller_manager",
        #     executable="spawner.py",
        #     arguments=["velocity_controller", "-c", "/controller_manager"],
        #     output="screen",
        # ),

        # Node(
        #     package="controller_manager",
        #     executable="spawner.py",
        #     arguments=["joint_trajectory_controller", "-c", "/controller_manager"],
        #     output="screen",
        # ),

        Node(
            package="controller_manager",
            executable="spawner.py",
            arguments=["joint_position_controller", "-c", "/controller_manager"],
            output="screen",
        ),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            parameters=[
                {"robot_description": robot_description_config.toxml()}],
            output="screen",
        ),

        # TODO(yongpeng): joint state publisher gui (default: not included)
        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
            name="joint_state_publisher_gui",
            output="screen"),

        # Node(
        #     package="leap_hand_custom_description",
        #     executable="debug_publisher.py",
        #     parameters=[
        #         {"robot_description": robot_description_config.toxml()}],
        #     output="screen",
        # ),

        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            parameters=[
                {"robot_description": robot_description_config.toxml()}],
            arguments=["-d", rviz_config],
            output="screen",
        )

    ])

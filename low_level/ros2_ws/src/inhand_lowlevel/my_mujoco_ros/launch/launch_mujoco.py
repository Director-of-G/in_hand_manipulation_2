import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import xacro
import yaml

def generate_launch_description():
    low_level_config_dict = yaml.safe_load(open(os.path.join(
            get_package_share_directory('leap_ros2'),
            'config',
            'low_level_ctrl_params.yaml'
        ), 'r')
    )
    hand_type = low_level_config_dict['hand_type']
    algo_type = low_level_config_dict['algo_type']
    is_mjpc = (algo_type == 'mjpc')

    # Load URDF, MuJoCo and sensor config
    if hand_type == "leap":
        robot_name = "leap_hand_custom"
        package_name = robot_name + "_description"
        rviz_config = os.path.join(get_package_share_directory(
            # package_name), "launch", robot_name + "_debug.rviz")
            package_name), "launch", robot_name + "_debug.rviz")
        robot_description = os.path.join(get_package_share_directory(
            package_name), "urdf", robot_name + "_sim.urdf.xacro")
        robot_description_config = xacro.process_file(robot_description)
    elif hand_type == "allegro":
        robot_name = "allegro_hand"
        package_name = robot_name + "_description"
        rviz_config = os.path.join(get_package_share_directory(
            # package_name), "launch", robot_name + "_debug.rviz")
            package_name), "launch", robot_name + "_debug.rviz")
        robot_description = os.path.join(get_package_share_directory(
            package_name), "urdf", robot_name + ".urdf")
        robot_description_config = xacro.process_file(robot_description)

    # Specify the path to the MuJoCo model file
    config = os.path.join(
        get_package_share_directory('my_mujoco_ros'),
        'config', hand_type,
        'mujoco_config.yaml'  # Replace with the actual model filename
    )
    sensor_config = os.path.join(
        get_package_share_directory('my_mujoco_ros'),
        'config', hand_type,
        'sensor_config.yaml'
    )

    return LaunchDescription([
        Node(
            package='my_mujoco_ros',
            executable='mujoco_ros_core.py',
            name='mujoco_ros',
            output='screen',
            parameters=[config, sensor_config, {'is_mjpc': is_mjpc}],
        ),

        # Publish palm_lower and fake_world that are handled by task_specific_pose_estimator on hardware
        # ----------------------------------------
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            arguments=["0", "0", "0", "0", "3.1416", "0", "world", "palm_lower"],
            output="screen"
        ),

        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            arguments=["0", "0", "0", "0", "3.1416", "0", "palm_lower", "fake_world"],
            output="screen"
        ),
        # ----------------------------------------

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            parameters=[
                {"robot_description": robot_description_config.toxml()}],
            output="screen",
        ),

        Node(
            package="leap_hand_custom_description",
            executable="debug_publisher.py",
            # parameters=[
            #     {"hand_type": hand_type['hand_type']}],
            output="screen",
        ),

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

if __name__ == '__main__':
    generate_launch_description()

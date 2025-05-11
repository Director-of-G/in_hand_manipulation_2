import os
import yaml
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Specify the path to the MuJoCo model file
    config_url = os.path.join(
        get_package_share_directory('leap_ros2'),
        'config',
        'low_level_ctrl_params.yaml'  # Replace with the actual model filename
    )

    config_dict = yaml.safe_load(open(config_url, 'r'))

    node_list = []

    # leap hand hw node
    hw_type = config_dict['hw_type']
    launch_controller = True
    if hw_type == 'mujoco':
        launch_substitution = PathJoinSubstitution([
            FindPackageShare('my_mujoco_ros'),
            'launch',
            'launch_mujoco.py'
        ])
    else:
        raise NotImplementedError(f'Hardware type {hw_type} is not supported in this open source code!')

    node_list.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([launch_substitution])
        )
    )

    # hardware state node
    node_list.append(
        Node(
            package='leap_ros2',
            executable='hardware_state_manager.py',
            name='hardware_state_manager_node',
            output='screen',
        )
    )

    # low level controller node (launch for mujoco, launch separately for real hardware)
    if launch_controller:
        node_list.append(
            Node(
                package='leap_ros2',
                executable='low_controller_hw.py',
                name='low_level_node',
                output='screen',
                parameters=[{'config_url': config_url}],
            )
        )

    return LaunchDescription(node_list)

if __name__ == '__main__':
    generate_launch_description()

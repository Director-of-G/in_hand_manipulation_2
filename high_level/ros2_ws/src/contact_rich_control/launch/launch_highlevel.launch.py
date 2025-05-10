import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    config_url = os.path.join(
        get_package_share_directory('contact_rich_control'),
        'config',
        'high_level_ctrl_params.yaml'
    )
    highlevel_config = yaml.safe_load(open(config_url, 'r'))

    node_list = [
        Node(
            package='contact_rich_control',
            executable='test_highlevel.py',
            name='highlevel_controller_node',
            output='screen',
            parameters=[{'config_url': config_url}]
        )
    ]

    if highlevel_config['visualize']:
        node_list.append(Node(
            package='contact_rich_control',
            executable='meshcat_node.py',
            name='meshcat_node',
            output='screen'
        ))

    return LaunchDescription(node_list)

if __name__ == "__main__":
    generate_launch_description()

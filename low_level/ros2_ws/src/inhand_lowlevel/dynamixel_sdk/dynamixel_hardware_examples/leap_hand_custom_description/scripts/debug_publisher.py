#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy
import numpy as np
from ruamel.yaml import YAML
yaml = YAML()

from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Header
from rcl_interfaces.srv import GetParameters
from visualization_msgs.msg import Marker, MarkerArray
from moveit_msgs.msg import DisplayTrajectory

from urdf_parser_py.urdf import URDF
from leap_hand_custom_description.utils import get_force_vis_message, get_hfmc_vis_message, get_contact_point_vis_message
from common_msgs.msg import ContactState, JointTrajectoryWrapper, HybridFMCVis


def get_param(node_handle:Node, node_name:str, param_name:str, timeout=0.0):
    """ Block if timeout is None or negative. Don't wait if 0. """
    node_handle.get_logger().info(f'Wait for parameter /{node_name}/{param_name}...')
    client = node_handle.create_client(GetParameters, f'/{node_name}/get_parameters')
    client.wait_for_service(timeout_sec=timeout)
    request = GetParameters.Request()
    request.names = [param_name]
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node_handle, future, timeout_sec=timeout)
    if future.done():
        response = future.result()
        node_handle.get_logger().info(f'Parameter /{node_name}/{param_name} is set!')
        return response.values[0]
    else:
        node_handle.get_logger().error(f'Get parameter /{node_name}/{param_name} failed!')
        return

class DebugPublisherNode(Node):
    def __init__(self):
        super().__init__('debug_publisher')

        # self.declare_parameter('robot_description', '')
        # robot_description = self.get_parameter('robot_description').get_parameter_value().string_value
        # self.robot_description = URDF.from_xml_string(robot_description)

        # parameters
        self.declare_parameter('real_contact_force_scale', 0.1)
        self.declare_parameter('desired_contact_force_scale', 0.1)
        self.real_contact_force_scale = self.get_parameter('real_contact_force_scale').get_parameter_value().double_value
        self.desired_contact_force_scale = self.get_parameter('desired_contact_force_scale').get_parameter_value().double_value

        # Set the geom names via parameter
        # ----------------------------------------
        # self.highlevel_contact_geoms = [
        #     'leap_hand_right::thumb_fingertip_collision', \
        #     'leap_hand_right::fingertip_collision', \
        #     'leap_hand_right::fingertip_2_collision', \
        #     'leap_hand_right::fingertip_3_collision'
        # ]
        self.acquire_high_level_params()
        # ----------------------------------------

        self.desired_forces = np.zeros((4, 3))
        self.is_hfmc_published = False

        # hardware states
        # self.joint_state_subscriber_ = self.create_subscription(
        #     JointState,
        #     '/joint_states',
        #     self.joint_states_callback,
        #     10
        # )
        
        self.contact_force_vis_publisher_ = self.create_publisher(
            MarkerArray,
            '/real_contact_force_vis',
            10
        )
        self.desired_force_vis_publisher_ = self.create_publisher(
            MarkerArray,
            '/desired_contact_force_vis',
            10
        )
        self.contact_point_vis_publisher_ = self.create_publisher(
            MarkerArray,
            '/contact_point_vis',
            10
        )
        self.robot_traj_publisher = self.create_publisher(
            DisplayTrajectory,
            '/display_planned_path',
            10
        )
        self.hfmc_vis_publisher_ = self.create_publisher(
            MarkerArray,
            '/hfmc_vis',
            10
        )
        
        # acquire high-level trajectory
        self.highlevel_traj_subscriber_ = self.create_subscription(
            JointTrajectoryWrapper,
            '/high_level_traj',
            self.high_level_traj_callback,
            10
            # qos_profile=QoSProfile(
            #     depth=10,
            #     reliability=ReliabilityPolicy.BEST_EFFORT,
            #     history=HistoryPolicy.KEEP_LAST
            # ),
        )
        self.contact_state_subscriber_ = self.create_subscription(
            ContactState,
            '/contact_sensor_state',
            self.contact_state_callback,
            10
        )
        self.hfmc_debug_subscriber_ = self.create_subscription(
            HybridFMCVis, '/hfmc_debug',  self.hfmc_callback, 10
        )

        # 发布 WrenchStamped 消息的话题
        self.wrench_publisher_list = []

        self.get_logger().info('Debug publisher node initialized')

    def acquire_high_level_params(self):
        """ Get high level parameters through ROS service """
        params = get_param(self, 'highlevel_controller_node', 'params', timeout=None).string_value
        self.highlevel_params = yaml.load(params)
        self.highlevel_contact_geoms = list(self.highlevel_params['finger_to_geom_name_map'].values())

    def initialize_wrench_publishers(self, num):
        self.wrench_publisher_list.clear()
        for i in range(num):
            _publisher = self.create_publisher(WrenchStamped, f'joint_wrench/joint_{i}', 10)
            self.wrench_publisher_list.append(_publisher)

    # def joint_states_callback(self, msg):
    #     joint_names = msg.name

    #     if len(self.wrench_publisher_list) == 0:
    #         self.initialize_wrench_publishers(len(joint_names))

    #     # 假设joint_0到joint_15的effort都在-z方向，我们将其转换为WrenchStamped格式
    #     for i, joint in enumerate(joint_names):
    #         jid = int(joint.split("_")[-1])
    #         wrench_msg = WrenchStamped()

    #         # lookup child body frame
    #         wrench_frame = self.robot_description.joint_map.get(joint).child
            
    #         # 填充Header
    #         wrench_msg.header = Header()
    #         wrench_msg.header.stamp = self.get_clock().now().to_msg()
    #         wrench_msg.header.frame_id = wrench_frame

    #         wrench_msg.wrench.force.x = 0.0
    #         wrench_msg.wrench.force.y = 0.0
    #         wrench_msg.wrench.force.z = 0.0

    #         wrench_msg.wrench.torque.x = 0.0
    #         wrench_msg.wrench.torque.y = 0.0
    #         wrench_msg.wrench.torque.z = msg.effort[i] / 100

    #         # 发布WrenchStamped消息
    #         self.wrench_publisher_list[jid].publish(wrench_msg)

    def contact_state_callback(self, msg:ContactState):
        n_c = len(msg.names)
        contact_pt_markers, f_read_markers, f_desired_markers = MarkerArray(), MarkerArray(), MarkerArray()
        for i in range(n_c):
            geom_name = msg.names[i]
            j = self.highlevel_contact_geoms.index(geom_name)
            real_contact_force = self.real_contact_force_scale * \
                np.array([msg.wrenches[i].force.x, msg.wrenches[i].force.y, msg.wrenches[i].force.z])
            desired_contact_force = self.desired_contact_force_scale * self.desired_forces[j]

            contact_point = np.array([msg.points[i].x, msg.points[i].y, msg.points[i].z])
            # This is not provided by Tac3D official, and may not be accurate
            # We use the torque field to save normal
            real_contact_normal = np.array([msg.wrenches[i].torque.x, msg.wrenches[i].torque.y, msg.wrenches[i].torque.z])
            real_contact_normal = self.real_contact_force_scale * real_contact_normal
            marker = get_force_vis_message(
                force=real_contact_force,
                position=contact_point,
                marker_id=i,
                base_frame="fake_world"
            )
            normal = get_force_vis_message(
                force=real_contact_normal,
                position=contact_point,
                marker_id=i+8,
                rgba=(0.0, 0.0, 1.0, 1.0),
                base_frame="fake_world"
            )
            assert geom_name in self.highlevel_contact_geoms
            marker_desired = get_force_vis_message(
                force=desired_contact_force,
                position=contact_point,
                marker_id=i+4,
                rgba=(0.0, 1.0, 0.0, 1.0),
                base_frame="fake_world"
            )
            contact_pt = get_contact_point_vis_message(
                position=contact_point,
                marker_id=i+16,
                rgba=(1.0, 0.0, 0.0, 1.0),
                base_frame="fake_world"
            )
            f_read_markers.markers.append(marker)
            f_read_markers.markers.append(normal)
            f_desired_markers.markers.append(marker_desired)
            contact_pt_markers.markers.append(contact_pt)

        self.contact_force_vis_publisher_.publish(f_read_markers)
        self.desired_force_vis_publisher_.publish(f_desired_markers)
        self.contact_point_vis_publisher_.publish(contact_pt_markers)

    def high_level_traj_callback(self, msg:JointTrajectoryWrapper):
        # parse desired contact force
        traj = msg.traj
        desired_forces = np.asarray(traj.points[0].effort)
        self.desired_forces[:] = desired_forces.reshape(4, 3)

    def hfmc_callback(self, msg:HybridFMCVis):
        vis_msg = get_hfmc_vis_message(msg, stamp=self.get_clock().now().to_msg(), frame_id="palm_lower", action=Marker.MODIFY if self.is_hfmc_published else Marker.ADD)
        self.hfmc_vis_publisher_.publish(vis_msg)
        self.is_hfmc_published = True

def main(args=None):
    rclpy.init(args=args)
    node = DebugPublisherNode()
    # try:
    rclpy.spin(node)
    # except KeyboardInterrupt:
    #     pass
    # finally:
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
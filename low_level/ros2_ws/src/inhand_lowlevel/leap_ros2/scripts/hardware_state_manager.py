#!/usr/bin/env python3

# This script subscribes all hardware states, including
# - joint states
# - object states
# and publish useful states for the controllers, including
# - q: [qu; qa]
# - v: [vu; va]
# - w: contact forces

from copy import deepcopy
import numpy as np
import yaml

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped
from common_msgs.msg import HardwareStates, ContactState
from common_msgs.srv import ReadHardwareStates

from leap_ros2.utils import get_param, get_remapping, convert_object_dofs_dict_to_array, PoseEstimator

class HwStateManagerNode(Node):
    def __init__(self):
        super().__init__('hardware_state_manager_node')

        self.declare_parameter('predefined_high_level_params', '')
        predefined_high_level_params = self.get_parameter('predefined_high_level_params').get_parameter_value().string_value
        self.acquire_high_level_params(predefined_high_level_params)

        self.create_buffers()

        ## SUBSCRIBERS
        # ----------------------------------------
        # joint states
        self.joint_state_subscriber_ = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        # object states
        self.object_state_subscriber_ = self.create_subscription(
            PoseStamped,
            '/object_states',
            self.object_states_callback,
            10
        )
        # contact states
        self.contact_state_subscriber_ = self.create_subscription(
            ContactState,
            '/contact_sensor_state',
            self.contact_state_callback,
            10
        )
        # ----------------------------------------

        ## PUBLISHERS
        # ----------------------------------------
        self.hardware_state_publisher_ = self.create_publisher(
            HardwareStates,
            '/hardware_states',
            10
        )
        # ----------------------------------------

        ## SERVICE
        # ----------------------------------------
        self.hardware_state_service_ = self.create_service(
            ReadHardwareStates,
            'read_hardware_states',
            self.read_hardware_states_callback
        )
        # ----------------------------------------

        ## OTHERS
        # ----------------------------------------
        # pose estimator
        self.pose_estimator = PoseEstimator()
        # timer
        self.hardware_state_timer = self.create_timer(1 / 30, self.hardware_state_publish_callback)
        # ----------------------------------------

    def acquire_high_level_params(self, predefined_params: str):
        """ Get high level parameters through ROS service """
        if predefined_params == '':
            params = get_param(self, 'highlevel_controller_node', 'params', timeout=None).string_value
            params = yaml.safe_load(params)
            highlevel_joint_names = get_param(self, 'highlevel_controller_node', 'joint_names', timeout=None).string_array_value
        else:
            params = predefined_params
            params = yaml.safe_load(params)
            highlevel_joint_names = params['joint_names']

        self.get_logger().info(f'High level params: {params}')
        self.highlevel_params = params
        self.highlevel_contact_order = list(self.highlevel_params['finger_to_geom_name_map'].values())
        self.highlevel_joint_names = highlevel_joint_names

        object_dofs_dict = self.highlevel_params['object_dofs_dict']
        self.get_logger().info(f'Object dofs: {object_dofs_dict}')

        if 'quat' in object_dofs_dict:
            # must not have rpy if quat is used
            self.quat_as_se3_state = True
            assert len(object_dofs_dict['rpy']) == 0
        else:
            self.quat_as_se3_state = False

        self.object_dofs = convert_object_dofs_dict_to_array(object_dofs_dict)
        self.enable_pose_estimator = True

    def create_buffers(self):
        """ Create buffers for hardware states """
        nqa = len(self.highlevel_joint_names)
        self.joint_remapping = None
        self.remapped_joint_pos = np.zeros(nqa,)
        self.remapped_joint_vel = np.zeros(nqa,)
        self.nqa = nqa

        nc = len(self.highlevel_contact_order)
        self.contact_remapping = None
        self.remapped_contact_forces = np.zeros((nc, 3))
        self.remapped_contact_points = np.zeros((nc, 3))
        self.nc = nc

        self.object_pose = np.zeros(6,)     # [x, y, z, rx, ry, rz], pos+rvec
        self.object_vel = np.zeros(6,)      # [vx, vy, vz, wx, wy, wz], linvel+angvel

        self.latest_hardware_states = None
        self.latest_object_6d_pose = Pose()

        self.is_data_ready = {
            "joint_states": False,
            "contact_states": False
        }

    def check_data_ready(self):
        return all(self.is_data_ready.values())

    def joint_state_callback(self, msg:JointState):
        if self.joint_remapping is None:
            self.joint_remapping = get_remapping(
                source=msg.name, target=self.highlevel_joint_names
            )
        self.remapped_joint_pos[:] = np.array(msg.position)[self.joint_remapping]
        self.remapped_joint_vel[:] = np.array(msg.velocity)[self.joint_remapping]
        self.is_data_ready["joint_states"] = True

    def object_states_callback(self, msg:PoseStamped):
        """ Return all zeros if no object states are received """
        self.latest_object_6d_pose = msg.pose
        if self.enable_pose_estimator:
            self.pose_estimator.update(msg)
        pos, rvec = self.pose_estimator.get_pose_estimation()
        if pos is not None and rvec is not None:
            self.object_pose[:3] = pos.copy()
            self.object_pose[3:] = rvec.copy()
        linvel, angvel = self.pose_estimator.get_vel_estimation()
        self.object_vel[:3] = linvel.copy()
        self.object_vel[3:] = angvel.copy()

    def contact_state_callback(self, msg:ContactState):
        if self.contact_remapping is None:
            self.contact_remapping = get_remapping(
                source=msg.names, target=self.highlevel_contact_order
            )
        for i in range(self.nc):
            # contact force
            force = msg.wrenches[self.contact_remapping[i]].force
            self.remapped_contact_forces[i, :] = np.array([force.x, force.y, force.z])
            
            # contact point
            point = msg.points[self.contact_remapping[i]]
            self.remapped_contact_points[i, :] = np.array([point.x, point.y, point.z])
        self.is_data_ready["contact_states"] = True

    def read_hardware_states_callback(self, request, response):
        if self.latest_hardware_states is None:
            response.valid = False
        else:
            response.valid = True
            response.states = self.latest_hardware_states
        return response

    def hardware_state_publish_callback(self):
        if not self.check_data_ready():
            return
        
        # object states
        qu = self.object_pose[self.object_dofs]
        vu = self.object_vel[self.object_dofs]

        if self.quat_as_se3_state:
            qu = np.concatenate((qu, self.pose_estimator.get_unfiltered_quaternion()))
            vu = np.concatenate((vu, self.object_vel[3:]))

        # hand states
        qa = self.remapped_joint_pos
        va = self.remapped_joint_vel

        # concat states
        q = np.concatenate((qu, qa)).astype(np.float32)
        v = np.concatenate((vu, va)).astype(np.float32)

        # contact force
        w_meas = self.remapped_contact_forces.flatten().astype(np.float32)

        # contact point
        p_meas = self.remapped_contact_points.flatten().astype(np.float32)

        # publish
        msg = HardwareStates()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.q.data = q.tolist()
        msg.v.data = v.tolist()
        msg.w_meas.data = w_meas.tolist()
        msg.p_meas.data = p_meas.tolist()
        msg.obj_pose = self.latest_object_6d_pose
        self.latest_hardware_states = deepcopy(msg)
        self.hardware_state_publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    hardware_state_manager = HwStateManagerNode()
    rclpy.spin(hardware_state_manager)

    hardware_state_manager.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
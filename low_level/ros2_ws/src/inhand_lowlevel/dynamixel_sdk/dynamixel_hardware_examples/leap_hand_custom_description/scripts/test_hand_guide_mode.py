#!/usr/bin/env python3

# test Leap Hand joint control with dynamixel sdk

import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from common_msgs.msg import HardwareStates


class JointCommandPublisher(Node):
    def __init__(self):
        super().__init__('joint_command_publisher')

        self.joint_pos = np.zeros(16,)
        self.joint_commands = np.zeros(16,)
        self.contact_threshold = 3.0
        self.is_finger_guide_mode = np.zeros(4, dtype=bool)
        
        self.remapping_to_hw = []
        self.is_joint_pos_ready = False

        self.finger_to_jnt_map = {
            0: [12, 13, 14, 15],        # thumb
            1: [0, 1, 2, 3],            # index
            2: [4, 5, 6, 7],            # middle
            3: [8, 9, 10, 11]           # ring
        }

        self.publisher_ = self.create_publisher(Float64MultiArray, '/joint_position_controller/commands', 10)
        self.create_subscription(HardwareStates, '/hardware_states', self.hardware_state_callback, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

    def joint_state_callback(self, msg:JointState):
        if len(self.remapping_to_hw) == 0:
            self.remapping_to_hw = [msg.name.index(f'joint_{i}') for i in range(16)]
        self.joint_pos[:] = np.asarray(msg.position)[self.remapping_to_hw]
        if not self.is_joint_pos_ready:
            self.is_joint_pos_ready = True
            self.joint_commands[:] = self.joint_pos.copy()

    def hardware_state_callback(self, msg:HardwareStates):
        sensed_force = np.asarray(msg.w_meas.data).reshape((4, 3))
        self.is_finger_guide_mode[:] = np.linalg.norm(sensed_force, axis=-1) > self.contact_threshold
        print("is_finger_guide_mode: ", self.is_finger_guide_mode)

        if not self.is_joint_pos_ready:
            return

        for i in range(4):
            if self.is_finger_guide_mode[i]:
                jnt_idx = self.finger_to_jnt_map[i]
                self.joint_commands[jnt_idx] = self.joint_pos[jnt_idx].copy()

        self.publisher_.publish(Float64MultiArray(data=self.joint_commands.tolist()))


def main(args=None):
    rclpy.init(args=args)
    node = JointCommandPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3

# test Leap Hand joint control with dynamixel sdk

import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState


JOINT_POS_ALL_ZEROS = np.zeros(16,)
# joint order: from joint_0 to joint_15
# JOINT_POS_MF_CONTACT = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.66, 0.45, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# JOINT_POS_MF_CONTACT = np.array([0.0, 0.66, 0.45, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# JOINT_POS_GRASP = np.array([-1.05, 0.54, 1.35, 0.2, 0.0, 0.62, 0.73, 0.45, 1.05, 0.54, 1.35, 0.2, 1.39, 0.94, -0.37, 0.66])
JOINT_POS_SLIDE_WOOD_BOARD = np.array([-0.38, 0.78, 0.75, 0.26, 0.0, 0.73, 0.78, 0.39, 0.38, 0.78, 0.75, 0.26, -0.35, 0.0, -0.3, -0.4])

class JointCommandPublisher(Node):
    def __init__(self):
        super().__init__('joint_command_publisher')
        self.subscriber_ = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.publisher_ = self.create_publisher(Float64MultiArray, '/joint_position_controller/commands', 10)
        self.pub_frequency = 20
        self.timer_period = 1 / self.pub_frequency
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.joint_commands = None
        self.q_target = JOINT_POS_SLIDE_WOOD_BOARD

    def joint_state_callback(self, msg:JointState):
        num_joints = len(msg.name)
        received_names = msg.name
        received_joints = np.zeros(num_joints,)
        for i in range(num_joints):
            received_joints[i] = msg.position[received_names.index(f'joint_{i}')]

        # move linearly from received_joints to all zeros
        dq = self.q_target - received_joints
        dq_norm = np.linalg.norm(dq)
        dq_normalized = dq / dq_norm
        self.joint_commands = received_joints + dq_normalized * min(dq_norm, 0.3)

    def timer_callback(self):
        if self.joint_commands is None:
            return
        msg = Float64MultiArray()
        msg.data = self.joint_commands.tolist()
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published joint positions: {msg.data}')

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

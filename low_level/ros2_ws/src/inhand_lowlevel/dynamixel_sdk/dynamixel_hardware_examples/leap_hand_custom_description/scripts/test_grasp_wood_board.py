#!/usr/bin/env python3

# test Leap Hand joint control with dynamixel sdk

import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState


JOINT_POS_SLIDE_WOOD_BOARD = np.array([-0.38, 0.78, 0.75, 0.26, 0.0, 0.73, 0.78, 0.39, 0.38, 0.78, 0.75, 0.26, 1.61, 0.0, -1.11, -0.41])
JOINT_POS_GRASP_WOOD_BOARD = np.array([-0.38, 0.78, 0.75, 0.26, 0.0, 0.73, 0.78, 0.39, 0.38, 0.78, 0.75, 0.26, 1.82, 0.0, 0.15, 0.38])

class JointCommandPublisher(Node):
    def __init__(self):
        super().__init__('joint_command_publisher')
        self.publisher_ = self.create_publisher(Float64MultiArray, '/joint_position_controller/commands', 10)
        
        input("Press Enter to grasp the board harder!")
        self.grasp_board_harder()

        input("Press Enter to grasp the board to pick it up!")
        self.place_thumb()

    def grasp_board_harder(self):
        dq = JOINT_POS_SLIDE_WOOD_BOARD.copy()
        dq[[1, 2, 3, 5, 6, 7, 9, 10, 11]] += 0.05
        self.publisher_.publish(Float64MultiArray(data=dq.tolist()))

    def place_thumb(self):
        dq = JOINT_POS_GRASP_WOOD_BOARD.copy()
        self.publisher_.publish(Float64MultiArray(data=dq.tolist()))

def main(args=None):
    rclpy.init(args=args)
    node = JointCommandPublisher()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import time
import numpy as np
import pickle

from pydrake.all import PiecewisePolynomial

class TestTrajReplayNode(Node):

    def __init__(self):
        super().__init__('minimal_client_async')

        self.is_initialized = False
        self.x_traj = None
        
        self.x_command = []
        self.t_command = []
        self.x_read = []
        self.t_read = []

        self.sub_hand = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.pub_hand = self.create_publisher(Float64MultiArray, '/joint_position_controller/commands', 10) 

        self.timer = self.create_timer(1/100, self.timer_callback)

    def joint_state_callback(self, msg:JointState):
        received_joint_order = [int(jnt.split('_')[-1]) for jnt in msg.name]
        desired_joint_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        joint_remapping = [received_joint_order.index(j) for j in desired_joint_order]
        x_read = np.array(msg.position)[joint_remapping]

        if not self.is_initialized:
            data = pickle.load(open('/home/jyp/research/inhand_manipulation/dex_playground/data/20241206/running_logs.pkl', 'rb'))
            data_joint_order = [1, 0, 2, 3, 12, 13, 14, 15, 5, 4, 6, 7, 9, 8, 10, 11]
            desired_joint_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            joint_remapping = [data_joint_order.index(j) for j in desired_joint_order]
            x_traj = np.array(data['x_traj'])[:, 3:][:, joint_remapping]

            x_traj = np.concatenate(
                [np.linspace(x_read, x_traj[0], 20), x_traj], axis=0
            )
            print("x_traj.shape: ", x_traj.shape)

            dt = 0.1
            t_traj = dt * np.arange(x_traj.shape[0])

            self.x_traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(t_traj, x_traj.T)
            self.is_initialized = True
            self.start_time = time.time()

        else:
            self.x_read.append(x_read.tolist())
            self.t_read.append(time.time() - self.start_time)

    def timer_callback(self):
        if self.x_traj is None:
            return
        
        t_now = time.time() - self.start_time
        x_now = self.x_traj.value(t_now).flatten()
        print("x_now: ", x_now)

        msg = Float64MultiArray()
        msg.data = x_now.tolist()
        self.x_command.append(x_now.tolist())
        self.t_command.append(t_now)
        self.pub_hand.publish(msg)

        if t_now > 15:
            from matplotlib import pyplot as plt
            x_command = np.array(self.x_command)
            x_read = np.array(self.x_read)
            plt.plot(self.t_command, x_command[:, 1], label='cmd')
            plt.plot(self.t_read, x_read[:, 1], label='read')
            plt.legend()
            plt.show()
            breakpoint()

def main(args=None):
    rclpy.init(args=args)
    node = TestTrajReplayNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
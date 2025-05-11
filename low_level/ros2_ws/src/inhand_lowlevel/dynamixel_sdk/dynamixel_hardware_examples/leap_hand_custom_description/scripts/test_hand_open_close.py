#!/usr/bin/env python3

# test Leap Hand joint control with dynamixel sdk

import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float64MultiArray


def generate_joint_trajectory(t, min_angles, max_angles, period):
    """
    生成关节角正弦轨迹

    :param min_angles: numpy 数组，表示每个关节的最小角度
    :param max_angles: numpy 数组，表示每个关节的最大角度
    :param period: float，正弦波周期
    :return: numpy 数组，形状为 (len(min_angles),)，包含每个关节的轨迹
    """
    assert min_angles.shape == max_angles.shape, "min_angles 和 max_angles 的形状必须相同"
    assert period > 0, "周期必须为正数"

    # 计算正弦轨迹
    amplitude = (max_angles - min_angles) / 2  # 振幅
    offset = min_angles + amplitude  # 中心点
    omega = 2 * np.pi / period  # 角频率

    # 生成轨迹
    trajectory = offset + amplitude * np.sin(omega * t - np.pi / 2)

    return trajectory

class JointCommandPublisher(Node):
    def __init__(self):
        super().__init__('joint_command_publisher')
        self.publisher_ = self.create_publisher(Float64MultiArray, '/joint_position_controller/commands', 10)
        self.pub_frequency = 100
        self.timer_period = 1 / self.pub_frequency
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        
        # Define min and max angles for each joint
        self.min_angles = np.array([0.0] * 16)  # Minimum angles (default to 0 degrees for static joints)
        self.max_angles = np.array([0.0] * 16)  # Maximum angles (default to 45 degrees for oscillating joints)
        self.max_angles[[1, 5, 9]] = 35
        self.max_angles[[2, 6, 10]] = 35
        self.max_angles[[3, 7, 11]] = 35
        self.max_angles[[12, 13, 14, 15]] = 35

        # Convert degrees to rads
        self.max_angles = np.deg2rad(self.max_angles)
        self.min_angles = np.deg2rad(self.min_angles)

        self.joint_positions = self.min_angles.copy()  # Start with minimum angles

        self.start_time = self.get_clock().now()

    def timer_callback(self):
        current_time = self.get_clock().now()  # 获取当前时间
        time_diff = current_time - self.start_time  # 计算时间差
        seconds_diff = time_diff.nanoseconds / 1e9  # 将时间差转换为秒

        joint_commands = generate_joint_trajectory(
            t=seconds_diff,
            min_angles=self.min_angles,
            max_angles=self.max_angles,
            period=5.0
        )

        msg = Float64MultiArray()
        msg.data = joint_commands.tolist()
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

#!/usr/bin/env python3

# WARNING! Code for journal experiment.
# This code runs repetative experiments and resets the system periodically

import os
import time
import numpy as np
from scipy.spatial.transform import Rotation as SciR
import pickle
import yaml

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
from ament_index_python.packages import get_package_share_directory

from std_srvs.srv import Trigger
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from common_msgs.msg import HardwareStates, JointTrajectoryWrapper
from leap_ros2.utils import time_msg_to_float

class JournalExperimentNode(Node):
    def __init__(self):
        node_name = 'journal_experiment_node'
        super().__init__(node_name)

        test_method = yaml.safe_load(
            open(os.path.join(get_package_share_directory('leap_ros2'), 'config', 'low_level_ctrl_params.yaml'), 'r'),)['algo_type']
        self.test_method = test_method
        print(f"Test method {test_method}")

        # preparation
        self.rand_so3_data = np.load('./data/quat_targets-250318.npy')
        if test_method in ['mjpc']:
            # rotate the goal w.r.t. z-axis np.pi
            for i in range(len(self.rand_so3_data)):
                r = SciR.from_quat(self.rand_so3_data[i][[1, 2, 3, 0]])
                r = SciR.from_euler('z', np.pi) * r
                self.rand_so3_data[i] = r.as_quat()[[3, 0, 1, 2]]
        self.current_goal_id = -1
        self.is_manipulating = False
        self.is_resetting = False

        # parameters
        self.timeout = 60.0
        self.error_thres_deg = 5.0

        # buffers
        self.t_last_reset = -1
        self.reset_buffers()

        self.curr_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.curr_quat_target = np.array([1.0, 0.0, 0.0, 0.0])

        high_level_node_name = ''
        if test_method in ['ours']:
            high_level_node_name = 'highlevel_controller_node'
        elif test_method in ['mjpc']:
            high_level_node_name = 'mjpc_controller_node'

        low_level_node_name = 'low_level_node'
        sim_node_name = 'mujoco_ros'

        self.sub_cbg = MutuallyExclusiveCallbackGroup()
        self.srv_cbg = MutuallyExclusiveCallbackGroup()
        self.timer_cbg = MutuallyExclusiveCallbackGroup()

        # high level reset
        self.high_level_reset_client = self.create_client(Trigger, f'{high_level_node_name}/reset', callback_group=self.srv_cbg)
        if not self.high_level_reset_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f'Service {high_level_node_name}/reset not found!')

        if test_method in ['mjpc']:
            self.high_level_start_client = self.create_client(Trigger, f'{high_level_node_name}/start', callback_group=self.srv_cbg)
            if not self.high_level_start_client.wait_for_service(timeout_sec=5.0):
                self.get_logger().error(f'Service {high_level_node_name}/start not found!')

        if test_method in ['ours']:
            # low level reset
            self.low_level_reset_client = self.create_client(Trigger, f'{low_level_node_name}/reset', callback_group=self.srv_cbg)
            if not self.low_level_reset_client.wait_for_service(timeout_sec=5.0):
                self.get_logger().error(f'Service {low_level_node_name}/reset not found!')

        # sim reset
        self.sim_reset_client = self.create_client(Trigger, f'{sim_node_name}/reset', callback_group=self.srv_cbg)
        if not self.sim_reset_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f'Service {sim_node_name}/reset not found!')

        # high level param
        self.high_level_param_client = self.create_client(SetParameters, f'{high_level_node_name}/set_parameters', callback_group=self.srv_cbg)
        if not self.high_level_param_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('High level parameter service is not initialized!')

        # external control
        self.reset_srv = self.create_service(Trigger, f'{node_name}/call_reset', self.call_reset_task)
        self.start_task_srv = self.create_service(Trigger, f'{node_name}/call_start_task', self.call_start_task)

        # create a folder in data/ with current time
        dir_name = f'./data/log/{time.strftime("%Y%m%d-%H%M%S")}'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        self.dir_name = dir_name

        # subscriber
        if test_method in ['ours']:
            state_msg = HardwareStates
            state_topic = '/hardware_states'
        elif test_method in ['mjpc']:
            state_msg = Float64MultiArray
            state_topic = '/object_quat'

        self.state_sub = self.create_subscription(
            state_msg, state_topic,
            self.state_callback,
            10,
            callback_group=self.sub_cbg
        )
        
        self.cmd_sub = self.create_subscription(
            JointState,
            '/mujoco_ros/joint_commands',
            self.cmd_callback,
            10,
            callback_group=self.sub_cbg
        )

        if test_method in ['ours']:
            self.traj_sub = self.create_subscription(
                JointTrajectoryWrapper,
                '/high_level_traj',
                self.traj_callback,
                qos_profile=QoSProfile(
                    depth=10,
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    history=HistoryPolicy.KEEP_LAST
                ),
                callback_group=self.sub_cbg
            )

        self.get_logger().info(f'{node_name} ready!')

        self.start_time = self.get_ros_time()
        self.total_reset_times = 0
        self.last_reset_time = self.start_time
        self.reset_period = 10.0

        # start_task
        self.reset_task()
        self.start_task()

        self.timer = self.create_timer(1.0, callback=self.timer_callback, callback_group=self.timer_cbg)

    def get_ros_time(self):
        """ Return ROS time in float """
        return time_msg_to_float(self.get_clock().now().to_msg())
    
    def reset_buffers(self):
        self.t_last_rev_state = -1  # state
        self.dt_recv_state = []
        self.state_buf = []
        self.state_stamp = []

        self.t_last_rev_cmd = -1    # low level
        self.dt_recv_cmd = []
        self.cmd_buf = []
        self.cmd_stamp = []

        self.t_last_rev_traj = -1   # high level
        self.dt_recv_traj = []
    
    # ---------- callbacks ----------

    def state_callback(self, msg:HardwareStates):
        if not self.is_manipulating:
            return
        
        t_now = self.get_ros_time()

        if isinstance(msg, HardwareStates):
            self.curr_quat[:] = np.array(msg.q.data)[0:4]
            self.state_buf.append(np.array(msg.q.data).tolist())
            self.state_stamp.append(t_now - self.t_last_reset)
        elif isinstance(msg, Float64MultiArray):
            self.curr_quat[:] = np.array(msg.data)[0:4]
            self.state_buf.append(np.array(msg.data).tolist())
            self.state_stamp.append(t_now - self.t_last_reset)

    def cmd_callback(self, msg:JointState):
        if not self.is_manipulating:
            return
        
        t_now = self.get_ros_time()
        if self.t_last_rev_cmd < 0:
            self.t_last_rev_cmd = t_now
        else:
            self.dt_recv_cmd.append(t_now - self.t_last_rev_cmd)
            self.t_last_rev_cmd = t_now

        self.cmd_buf.append(np.array(msg.position).tolist())
        self.cmd_stamp.append(t_now - self.t_last_reset)

    def traj_callback(self, msg):
        if not self.is_manipulating:
            return
        
        t_now = self.get_ros_time()
        if self.t_last_rev_traj < 0:
            self.t_last_rev_traj = t_now
        else:
            self.dt_recv_traj.append(t_now - self.t_last_rev_traj)
            self.t_last_rev_traj = t_now

    # -------------------------------
    
    def call_reset_task(self, request, response):
        self.reset_task()
        response.success = True
        return response
    
    def call_start_task(self, request, response):
        self.start_task()
        response.success = True
        return response
    
    def reset_task(self):
        self.is_manipulating = False

        # reset goal
        new_target = self.rand_so3_data[self.current_goal_id + 1]
        self.reset_so3_target(new_target)
        self.get_logger().info(f'New SO(3) target: {new_target}')

        request = Trigger.Request()

        # reset high level
        future = self.high_level_reset_client.call_async(request)

        if self.test_method in ['ours']:
            # reset low level
            future = self.low_level_reset_client.call_async(request)

        # reset sim
        future = self.sim_reset_client.call_async(request)

        # wait until all reset are done
        time.sleep(2.0)
    
    def start_task(self):
        self.get_logger().info(f'Starting goal {self.current_goal_id + 1}...')

        if self.test_method in ['ours']:
            request_start_task = SetParameters.Request()
            request_start_task.parameters = [Parameter(name='start_task', value=ParameterValue(bool_value=True, type=ParameterType.PARAMETER_BOOL))]
            future = self.high_level_param_client.call_async(request_start_task)
            future.add_done_callback(self.start_task_callback)
        elif self.test_method in ['mjpc']:
            request = Trigger.Request()
            future = self.high_level_start_client.call_async(request)
            future.add_done_callback(self.start_task_callback)

    def start_task_callback(self, future):
        self.get_logger().info(f'Goal {self.current_goal_id + 1} started!')
        self.t_last_reset = self.get_ros_time()
        self.is_manipulating = True

    def reset_so3_target(self, quat_target):
        request = SetParameters.Request()
        request.parameters = [
            Parameter(name='rand_so3_target', value=ParameterValue(double_array_value=quat_target.tolist(), type=ParameterType.PARAMETER_DOUBLE_ARRAY))
        ]
        self.high_level_param_client.call_async(request)

    def post_task_process(self):
        high_freq = 1 / np.mean(self.dt_recv_traj)
        low_freq = 1 / np.mean(self.dt_recv_cmd)
        print(f'High level freq: {high_freq:.2f} Hz')
        print(f'Low level freq: {low_freq:.2f} Hz')
        print(f"Cmd buf shape: {np.array(self.cmd_buf).shape}")
        print(f"State buf shape: {np.array(self.state_buf).shape}")

        data = {
            'high_freq': high_freq,
            'low_freq': low_freq,
            'cmd_buf': self.cmd_buf,
            'cmd_stamp': self.cmd_stamp,
            'state_buf': self.state_buf,
            'state_stamp': self.state_stamp
        }

        pickle.dump(data, open(f'{self.dir_name}/goal_{self.current_goal_id}.pkl', 'wb'))

        self.reset_buffers()

    def timer_callback(self):
        # block timer callback when reseting
        if self.is_resetting:
            return
        
        self.is_resetting = True
        # ----------------------------------------
        t_now = self.get_ros_time()
        if self.is_manipulating and t_now - self.t_last_reset > self.timeout:
            self.current_goal_id += 1
            self.get_logger().info(f'Timeout for goal {self.current_goal_id}! Resetting...')
            
            if self.current_goal_id >= len(self.rand_so3_data) - 1:
                self.post_task_process()
                self.get_logger().info('All goals are done!')
                self.destroy_node()
                rclpy.shutdown()
            else:
                self.reset_task()
                self.post_task_process()
                self.start_task()
        # ----------------------------------------
        self.is_resetting = False
            

def main(args=None):
    try:
        rclpy.init(args=args)
        node = JournalExperimentNode()
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except:
        pass
    finally:
        executor.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
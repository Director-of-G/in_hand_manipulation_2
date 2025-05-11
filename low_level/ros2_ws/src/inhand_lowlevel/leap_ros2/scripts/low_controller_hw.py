#!/usr/bin/env python3
import os
import numpy as np
import time
from ruamel.yaml import YAML
yaml = YAML()

import threading
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy

from rcl_interfaces.srv import GetParameters
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from common_msgs.msg import HardwareStates

from ament_index_python.packages import get_package_share_directory

from leap_ros2.utils import (
    LowLevelOptions,
    convert_ros_traj_to_drake_traj,
    time_msg_to_float,
    get_param,
    shift_picewise_polynomial,
)
from leap_ros2.utils import \
    HardwareState, HardwareStatus, LowLevelCtrlMode, \
    convert_wrench_array_to_msg
from leap_ros2.contact_controller import LeapContactController
from leap_ros2.journal_logger import MyLogger
from common_msgs.msg import ContactState, JointTrajectoryWrapper


def get_command_dtype(type):
    if type == 'dummy' or type == 'real':
        return Float64MultiArray
    elif type == 'mujoco':
        return JointState
    else:
        raise NotImplementedError
    
def get_command_topic(type):
    if type == 'dummy' or type == 'real':
        return '/joint_position_controller/commands'
    elif type == 'mujoco':
        return '/mujoco_ros/joint_commands'
    else:
        raise NotImplementedError
    
def get_joint_command_msg(type, joint_names, joint_commands):
    if isinstance(joint_commands, np.ndarray):
        joint_commands = joint_commands.tolist()
    if type == 'dummy' or type == 'real':
        command_msg = Float64MultiArray()
        command_msg.data = joint_commands
        return command_msg
    elif type == 'mujoco':
        command_msg = JointState()
        command_msg.name = joint_names
        command_msg.position = joint_commands
        return command_msg
    else:
        raise NotImplementedError

class LowlevelControllerHw(Node):
    def __init__(self):
        node_name = 'low_level_node'
        super().__init__(node_name)  # 节点名称
        
        # get parameters
        self.declare_parameter('config_url', os.path.join(get_package_share_directory('leap_ros2'), 'config', 'low_level_ctrl_params.yaml'))
        config_url = self.get_parameter('config_url').get_parameter_value().string_value
        default_params = yaml.load(open(config_url, 'r'))

        hw_type = default_params['hw_type']
        enable_low_level_ctrl = default_params['enable_low_level_ctrl']

        self.acquire_high_level_params()
        options = self.make_options(default_params)
        self.options = options
        self.get_logger().info("Low level params: " + str(options))

        # buffers
        self.initialize_buffers()

        self.hw_joint_names = [f"joint_{id}" for id in range(16)]
        self.get_logger().info('Hardware takes joints in the following order: %s' % self.hw_joint_names)
        self.get_logger().info(f'Low level controller started with hardware {hw_type}!')

        print("Allocating resources...")
        contact_controller = LeapContactController(options)
        contact_controller.set_output_remapping(self.hw_joint_names)
        contact_controller.set_highlevel_remapping(self.highlevel_joint_names)
        self.declare_parameter('dq_scale', contact_controller.dq_scale) 
        self.declare_parameter('Qf_scale', contact_controller.Qf_scale)
        self.declare_parameter('Qp_scale', contact_controller.Qp_scale)
        self.declare_parameter('Qp_ori_scale', contact_controller.Qp_ori_scale)
        self.contact_controller = contact_controller

        self.declare_parameter('send_command', True)
        self.send_command = True
        self.is_resetting = False

        self.logger = MyLogger()

        self._high_level_traj_lock = threading.Lock()
        self.general_cbg = MutuallyExclusiveCallbackGroup()
        self.ctrl_cbg = MutuallyExclusiveCallbackGroup()
        self.interp_cbg = MutuallyExclusiveCallbackGroup()

        # SERVICES
        # ----------------------------------------
        self.reset_service = self.create_service(
            Trigger, f'{node_name}/reset', self.reset_handler
        )
        # ----------------------------------------

        # SUBSCRIBERS
        # ----------------------------------------
        self.high_level_traj_subscriber_ = self.create_subscription(
            JointTrajectoryWrapper,
            '/high_level_traj',
            self.high_level_traj_callback,
            qos_profile=QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST
            ),
            callback_group=self.general_cbg
        )
        self.hardware_state_subscriber_ = self.create_subscription(
            HardwareStates,
            '/hardware_states',
            self.hardware_states_callback,
            10,
            callback_group=self.ctrl_cbg
        )
        self.parameter_event_subscriber_ = self.add_on_set_parameters_callback(
            self.parameters_callback
        )

        # PUBLISHERS
        # ----------------------------------------
        self.hw_command_publisher_ = self.create_publisher(
            get_command_dtype(hw_type),
            get_command_topic(hw_type),
            10
        )
        self.contact_force_publisher_ = self.create_publisher(
            ContactState,
            '/contact_force_reference',
            10
        )
        if options.debug:
            from common_msgs.msg import HybridFMCVis
            self.hfmc_debug_publisher_ = self.create_publisher(
                HybridFMCVis, '/hfmc_debug', 10
            )

        if hw_type == "dummy" or hw_type == "real":
            _client = self.create_client(GetParameters, '/joint_position_controller/get_parameters')
            if not _client.wait_for_service(timeout_sec=5.0):
                self.get_logger().error('Parameter service not available')
                return
        self.hw_type = hw_type
        self.enable_low_level_ctrl = enable_low_level_ctrl

    def initialize_buffers(self):
        self.high_level_traj = None
        self.joint_names = None
        self.num_joints = 0
        self.remapping_to_hw = []
        self.command_q = None
        self.delta_q = None
        self.hardware_state = HardwareState(self.nq, self.nv, self.nc)

    def reset_handler(self, request, response):
        self.is_resetting = True        # block and wait for callback return
        time.sleep(0.5)
        self.reset()
        self.is_resetting = False

        self.get_logger().info('Low level controller has been reset.')
        response.success = True
        return response

    def reset(self):
        self.initialize_buffers()
        self.contact_controller.reset()
        self.logger.reset()

    def acquire_high_level_params(self):
        """ Get high level parameters through ROS service """
        params = get_param(self, 'highlevel_controller_node', 'params', timeout=None).string_value
        self.highlevel_params = yaml.load(params)
        self.ordered_contact_geoms = list(self.highlevel_params['finger_to_geom_name_map'].values())
        self.ordered_finger_links = list(self.highlevel_params['finger_to_link_name_map'].values())
        n_wrist_dofs = self.highlevel_params['n_wrist_dofs']
        self.nq = self.highlevel_params['nq'] - n_wrist_dofs
        self.nv = self.highlevel_params['nv'] - n_wrist_dofs
        self.nu = self.highlevel_params['nu'] - n_wrist_dofs
        self.nc = self.highlevel_params['nc']

        self.highlevel_joint_names = get_param(self, 'highlevel_controller_node', 'joint_names', timeout=None).string_array_value
        self.get_logger().info(f'Low level receive reference in the following order {self.highlevel_joint_names}!')

        model_url = get_param(self, 'highlevel_controller_node', 'model_url', timeout=None).string_value
        models_root = get_param(self, 'highlevel_controller_node', 'models_root', timeout=None).string_value
        self.model_url = model_url
        self.models_root = models_root

    def make_options(self, config):
        """ Load params from YAML file """
        options = LowLevelOptions()
        options.hw_type = config['hw_type']
        options.hand_type = config['hand_type']

        assert options.hw_type in ['real', 'mujoco']
        assert options.hand_type in ['leap', 'allegro']

        options.models_root = self.models_root
        options.model_url = self.model_url
        options.ordered_finger_geoms = self.ordered_contact_geoms
        options.ordered_finger_links = self.ordered_finger_links

        options.nq = self.nq; options.nqa = self.nu; options.nc = self.nc

        options.high_level_traj_shift_ratio = config['high_level_traj_shift_ratio']

        ctrl_params = config['ctrl_params']
        options.time_step = 1 / ctrl_params['low_level_frequency']
        options.mpc_horizon = ctrl_params['T_ctrl']

        options.mpic_params = ctrl_params['mpic_params']

        options.enable_coupling = ctrl_params['model_params']['enable_coupling']
        options.debug = config['debug']

        return options

    def get_ros_time(self):
        """ Return ROS time in float """
        return time_msg_to_float(self.get_clock().now().to_msg())
    
    def parameters_callback(self, params):
        for param in params:
            if param.name == 'dq_scale':
                self.contact_controller.set_dq_scale(param.value)
            elif param.name == 'Qf_scale':
                self.contact_controller.set_Qf_scale(param.value)
            elif param.name == 'Qp_scale':
                self.contact_controller.set_Qp_scale(param.value)
            elif param.name == 'Qp_ori_scale':
                self.contact_controller.set_Qp_ori_scale(param.value)
            elif param.name == 'send_command':
                self.send_command = param.value

        return SetParametersResult(successful=True)

    def hardware_states_callback(self, msg:HardwareStates):
        self.hardware_state.receive_time = time_msg_to_float(msg.header.stamp)
        self.hardware_state.receive_time_ros = msg.header.stamp
        self.hardware_state.q[:] = np.array(msg.q.data)
        self.hardware_state.v[:] = np.array(msg.v.data)
        self.hardware_state.f_ext[:] = np.array(msg.w_meas.data).reshape((4, 3))
        self.hardware_state.p_obj[:] = np.array(msg.p_meas.data).reshape((4, 3))

        obj_6d_pose = msg.obj_pose
        obj_pos = obj_6d_pose.position
        obj_quat = obj_6d_pose.orientation
        self.hardware_state.obj_pos[:] = np.array([obj_pos.x, obj_pos.y, obj_pos.z])
        self.hardware_state.obj_quat[:] = np.array([obj_quat.w, obj_quat.x, obj_quat.y, obj_quat.z])
        self.hardware_state.status = HardwareStatus.RECEIVED

        # run contact control once hardware states are received
        self.contact_controller_callback()

        if len(self.remapping_to_hw) == 16:
            t_now = time_msg_to_float(msg.header.stamp)
            self.logger.update_q_read(self.hardware_state.get_q_actuated(), t_now, self.remapping_to_hw)
            self.logger.update_obj_state(self.hardware_state.get_q_unactuated(), t_now)
            obj_norm = self.high_level_traj.n_object.value(t_now).reshape(self.nc, 3)
            self.logger.update_f_ext_and_norm(self.hardware_state.f_ext, obj_norm, t_now)

    def high_level_traj_callback(self, msg:JointTrajectoryWrapper):
        current_time = self.get_ros_time()

        ctrl_mode = int(msg.mode.data)
        self.contact_controller.set_ctrl_mode(ctrl_mode)
        self.get_logger().info(f'Received high level trajectory (mode:{ctrl_mode})!', throttle_duration_sec=1.0)

        if self.num_joints == 0:
            self.joint_names = msg.traj.joint_names
            self.num_joints = len(msg.traj.joint_names)
            for name in self.hw_joint_names:
                self.remapping_to_hw.append(self.joint_names.index(name))
            assert len(self.remapping_to_hw) == self.num_joints

        # to prevent discontinuity in high_level_traj, we shift the high level traj
        high_level_traj = convert_ros_traj_to_drake_traj(msg.traj)
        if self.high_level_traj is not None:
            old_high_level_q = self.high_level_traj.q.Clone()
            shift_ratio = self.options.high_level_traj_shift_ratio      # 0.75
            q_shift = shift_ratio * (old_high_level_q.value(current_time).flatten()[-16:] - \
                        high_level_traj.q.value(current_time).flatten()[-16:])
            high_level_traj.q = shift_picewise_polynomial(high_level_traj.q, q_shift)
        with self._high_level_traj_lock:
            self.high_level_traj = high_level_traj

        t_recv_state = time_msg_to_float(msg.header_recv_state.stamp)
        t_mpc_begin = time_msg_to_float(msg.header_begin_sol.stamp)
        t_mpc_finish = time_msg_to_float(msg.header_finish_sol.stamp)
        self.logger.set_start_time(current_time)
        self.logger.update_q_ref(high_level_traj.q.value(current_time), high_level_traj.q, [t_recv_state, t_mpc_begin, t_mpc_finish, current_time], self.remapping_to_hw)
        
        # publish desired contact force
        f_desired = self.high_level_traj.w.value(current_time).reshape(self.nc, 3)
        obj_norm_desired = self.high_level_traj.n_object.value(current_time).reshape(self.nc, 3)
        self.logger.update_f_des_and_norm(f_desired, obj_norm_desired, current_time)
        self.publish_contact_force_reference(f_desired)

    def publish_contact_force_reference(self, f_desired:np.ndarray):
        msg = ContactState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.names = self.ordered_contact_geoms
        assert len(msg.names) == f_desired.shape[0]
        for i in range(f_desired.shape[0]):
            msg.wrenches.append(convert_wrench_array_to_msg(f_desired[i]))
        self.contact_force_publisher_.publish(msg)

    def contact_controller_callback(self):
        """ run contact controller """
        if self.is_resetting:
            return
        
        if not self.enable_low_level_ctrl:
            return
        
        if self.joint_names is None or self.high_level_traj is None:
            return
        
        if len(self.remapping_to_hw) == 0:
            return
        
        if self.hardware_state.status is not HardwareStatus.RECEIVED:
            return
        
        current_time = self.get_ros_time()
        # set the first joint commands
        if self.command_q is None:
            with self._high_level_traj_lock:
                joint_commands = self.high_level_traj.q.value(current_time).flatten()[-16:]
            joint_commands = joint_commands[self.remapping_to_hw]
            self.command_q = joint_commands.copy()
            command_msg = get_joint_command_msg(self.hw_type, self.hw_joint_names, joint_commands)
            self.hw_command_publisher_.publish(command_msg)
        
        # set controller initial states
        # ------------------------------
        q = self.hardware_state.get_q_actuated().copy()        # the joint order is from high level model
        self.contact_controller.update_model(q, self.joint_names)
        
        qd = self.command_q.copy()              # the joint order is from hardware (0~16)
        self.contact_controller.set_desired_joints(qd, self.hw_joint_names)
        
        fext = self.hardware_state.f_ext.copy()
        self.contact_controller.set_measured_force(fext)
        
        p_obj = self.hardware_state.p_obj.copy()
        self.contact_controller.set_measured_contact_pts(p_obj)

        obj_pos = self.hardware_state.obj_pos.copy()
        self.contact_controller.set_measured_object_pose(obj_pos)
        # ------------------------------
        
        dq = np.zeros(self.nu)
        with self._high_level_traj_lock:
            dq[:] = self.contact_controller.solve(current_time, self.high_level_traj)
        joint_commands = self.command_q + dq
        self.command_q = joint_commands.copy()
        command_msg = get_joint_command_msg(self.hw_type, self.hw_joint_names, joint_commands)
        if self.send_command:
            self.hw_command_publisher_.publish(command_msg)

        t_now = self.get_ros_time()
        self.logger.update_q_cmd(joint_commands, t_now)
        if self.options.debug:
            msg = self.contact_controller.get_HFMC_debug_vis_msg()
            msg.header.stamp = self.get_clock().now().to_msg()
            self.hfmc_debug_publisher_.publish(msg)

    def interpolate_trajectory_callback(self):
        """ interpolate high level trajectory """
        joint_commands = None
        if self.high_level_traj is None:
            return
        if self.hardware_state.status is not HardwareStatus.RECEIVED:
            return
        else:
            # hardware states come at 30Hz, which is too low for interpolation
            current_time = self.get_ros_time()
            with self._high_level_traj_lock:
                joint_commands = self.high_level_traj.q.value(current_time).flatten()[-16:]
            joint_commands = joint_commands[self.remapping_to_hw]

        # the joint commands will be from joint_0 to joint_16 order
        joint_commands = joint_commands + self.delta_q
        self.command_q[:] = joint_commands.copy()
        command_msg = get_joint_command_msg(self.hw_type, self.hw_joint_names, joint_commands)
        self.hw_command_publisher_.publish(command_msg)

        t_now = self.get_ros_time()
        self.logger.update_q_cmd(joint_commands, t_now)

def main(args=None):
    try:
        rclpy.init(args=args)
        node = LowlevelControllerHw()
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

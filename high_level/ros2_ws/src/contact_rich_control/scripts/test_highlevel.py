#!/usr/bin/python
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import rclpy.parameter
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy
# from sensor_msgs.msg import JointState

import sys
import numpy as np
from ruamel.yaml import YAML
yaml = YAML()
import pickle
from io import StringIO
import threading    # for mutex

from contact_rich_control.common_hw import *
sys.path.append(DDP_SOLVER_DIR)

from std_srvs.srv import Trigger

from common.common_drake import HighLevelOptions, HandModel
from contact_rich_control.high_controller_hw import AllegroHighLevelControllerHw
from common_msgs.msg import HardwareStates, JointTrajectoryWrapper, MeshcatVis
from common.common_ddp import get_hand_joints_only, get_perturbed_quat

CONFIG_PATH = os.path.join(
    INHAND_HOME,
    'ros2_ws/src/contact_rich_control/config',
    'high_level_ctrl_params.yaml'
)


class HighlevelControl(Node):
    """A ROS2 Node that prints to the console periodically."""

    def __init__(self):
        node_name = 'highlevel_controller_node'
        super().__init__(node_name)

        self.declare_parameter('config_url', CONFIG_PATH)
        self.declare_parameter('start_task', False)
        config_url = self.get_parameter('config_url').get_parameter_value().string_value

        self.declare_parameter('rand_so3_target', [1.0, 0.0, 0.0, 0.0])

        default_params = yaml.load(open(config_url, 'r'))

        # self.num_contacts = len(default_params['finger_to_geom_name_map'])
        # without_lowlevel: if True, will run without low level
        # 1. update state from last solution
        # 2. skip hardware state check
        self.run_without_lowlevel = default_params['without_lowlevel']
        self.is_hand_initialized = False    # whether the hand has moved to initial pose

        # set options and broadcast parameters
        # options = self.make_test_options()
        options = self.make_options(default_params)
        self.options = options
        self.get_logger().info("High level params: " + str(options))

        default_params['finger_to_geom_name_map'] = options.finger_to_geom_name_map
        default_params['object_dofs_dict'] = options.object_dofs_dict
        default_params['object_link'] = options.object_link
        default_params['nq'] = options.nq
        default_params['n_wrist_dofs'] = options.n_wrist_dofs
        default_params['nv'] = options.nv
        default_params['nu'] = options.nu
        default_params['nc'] = options.nc
        default_params_str = StringIO()
        yaml.dump(default_params, default_params_str)
        default_params_str = default_params_str.getvalue()
        self.declare_parameter('params', default_params_str)
        self.declare_parameter('joint_names', [])
        self.declare_parameter('models_root', QSIM_MODEL_DIR)
        self.declare_parameter('model_url', options.model_url)

        self.high_level_time_step: float = 1 / self.options.high_level_frequency
        self.solver_iter: int = 0
        self.current_state_time = self.get_ros_time(return_float=True)
        self.hardware_states_ready = False

        print("Allocating resources...")
        controller_core = AllegroHighLevelControllerHw(options)
        self.controller_core = controller_core
        self.x0 = self.controller_core.x0()
        self.xa0 = self.controller_core.xa0()

        self._data_lock = threading.Lock()
        self.initialize_states()

        self.joint_names = self.controller_core.joint_names()
        self.hand_joint_names = get_hand_joints_only(self.joint_names)
        self.set_parameters([rclpy.parameter.Parameter(f'joint_names', \
            rclpy.Parameter.Type.STRING_ARRAY, self.hand_joint_names)])

        self.hardware_cbg = MutuallyExclusiveCallbackGroup()
        self.mpc_cbg = MutuallyExclusiveCallbackGroup()

        # reset service
        self.create_service(Trigger, f'{node_name}/reset', self.reset_handler)

        # purturb service
        self.create_service(Trigger, f'{node_name}/perturb', self.perturb_handler)

        # self.joint_state_subscriber_ = self.create_subscription(
        #     JointState,
        #     '/joint_states',
        #     self.joint_state_callback,
        #     10
        # )
        self.hardware_state_subscriber_ = self.create_subscription(
            HardwareStates,
            '/hardware_states',
            self.hardware_states_callback,
            10,
            callback_group=self.hardware_cbg
        )
        self.stored_traj_publisher_ = self.create_publisher(
            # JointTrajectory,
            JointTrajectoryWrapper,
            '/high_level_traj',
            qos_profile=QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
            ),
            # 10
        )
        self.visualize_publisher_ = self.create_publisher(
            MeshcatVis,
            '/meshcat/visualization',
            10
        )
        self.timer = self.create_timer(
            self.high_level_time_step,
            self.solver_callback,
            callback_group=self.mpc_cbg
        )

    def initialize_states(self):
        """ Set hand and object states """
        with self._data_lock:
            if self.run_without_lowlevel:
                self.ordered_joint_states = np.delete(self.x0, self.options.q_u_indices)
                self.unactuated_object_states = self.x0[self.options.q_u_indices]
            else:
                self.ordered_joint_states = np.zeros(self.options.nu,)
                self.unactuated_object_states = np.zeros(self.options.nq-self.options.nu,)

        if self.run_without_lowlevel:
            self.saved_solution_start_time = time.time()
            self.saved_solution_stamp = []
            self.saved_solution = []
            self.mpc_solve_time = []
            self.enable_perturb = False

    def reset_handler(self, request, response):
        # first set the options to block solver callback
        self.is_hand_initialized = False
        self.set_parameters([rclpy.parameter.Parameter('start_task', rclpy.Parameter.Type.BOOL, False)])

        self.solver_iter = 0
        self.current_state_time = self.get_ros_time(return_float=True)
        self.hardware_states_ready = False

        # visualize solution
        self.send_solution_for_visualize()

        # print statistics
        self.print_statistics()

        # wait until solver callback returns
        time.sleep(0.5)
        so3_target = self.get_parameter('rand_so3_target').get_parameter_value().double_array_value
        so3_target = np.array(so3_target)
        self.controller_core.set_so3_target(so3_target)
        self.reset()

        self.get_logger().info('High level controller has been reset.')
        response.success = True
        return response
    
    def perturb_handler(self, request, response):
        self.enable_perturb = True
        
        self.get_logger().info('Perturb the object state requested by user.')
        response.success = True
        return response

    def reset(self):
        self.initialize_states()

        # reset controller
        self.controller_core.reset()
        xu_goal = self.controller_core.x_goal()[:-16]
        self.get_logger().info(f"Generate new target: {xu_goal.tolist()}")

    def get_ros_time(self, return_float=False):
        """ Return ROS time in seconds for message """
        now = self.get_clock().now()
        if return_float:
            return ros_duration_to_seconds(now)
        else:
            return now.to_msg()

    def make_options(self, config):
        """ Make high level options from config """
        task_spec = yaml.load(
            open(os.path.join(MODEL_DIR, 'config', config['task_file']), 'r')
        )
        model_url = os.path.join(MODEL_DIR, 'yml', task_spec['model_url'])

        options = HighLevelOptions()
        options.model_url = model_url

        # choose hand type
        hand_type = config['hand_type']
        if hand_type == 'leap':
            options.hand_model = HandModel.kLeap
        elif hand_type == 'allegro':
            options.hand_model = HandModel.kAllegro
        else:
            raise NotImplementedError(f"Hand type {hand_type} not supported!")

        options.finger_to_geom_name_map = config['finger_to_geom_name_map'][hand_type]
        self.num_contacts = len(config['finger_to_geom_name_map'][hand_type])

        options.finger_to_joint_map = task_spec['jnames']
        options.finger_to_limits_map = task_spec['jlimits'] if 'jlimits' in task_spec else {}
        options.finger_to_x0_map = task_spec['x0']
        options.finger_to_xreg_map_list.append(task_spec['xa_reg'])
        if 'xa_reg2' in task_spec:
            options.finger_to_xreg_map_list.append(task_spec['xa_reg2'])
        options.object_dofs_dict = task_spec['object_dofs']

        # get wrist dofs
        if 'wrist' in task_spec['x0']:
            self.fixed_wrist_joints = task_spec['x0']['wrist']
        else:
            self.fixed_wrist_joints = None

        options.nq = task_spec['nq']
        options.n_wrist_dofs = len(self.fixed_wrist_joints) if self.fixed_wrist_joints is not None else 0
        options.nv = task_spec['nv']
        options.nu = task_spec['nu']
        options.nc = task_spec['nc']

        # object-centric task definition
        options.q_u_indices = task_spec['q_u_indices']
        if 'q_rot_indices' in task_spec:
            options.q_rot_indices = task_spec['q_rot_indices']
            options.dxu_so3 = task_spec['dxu_so3']
            options.init_xu_so3 = task_spec['init_xu_so3']
            options.target_xu_so3 = task_spec['target_xu_so3']
            options.random_target_xu_so3 = task_spec['random_target_xu_so3']
        options.dxu = task_spec['dxu']
        options.target_xu = task_spec['target_xu']
        options.use_target_xu = task_spec['use_target_xu']

        options.u_lb = task_spec['u_lb'] * np.ones(options.nu)
        options.u_ub = task_spec['u_ub'] * np.ones(options.nu)

        options.high_level_frequency = task_spec['high_level_frequency']
        options.ddp_execution_scale = task_spec['ddp_execution_scale']
        options.force_threshold = task_spec['force_threshold']
        options.desired_force_scale = task_spec['desired_force_scale']

        if 'force_thres_method' in task_spec:
            options.force_thres_method = task_spec['force_thres_method']
        if 'force_thres_params' in task_spec:
            options.force_thres_params = task_spec['force_thres_params']

        options.ddp_params = task_spec['ddp_params']
        options.object_link = options.ddp_params['object_link_name']
        options.T_mpc = task_spec['ddp_params']['T_ctrl']
        if 'model_url_sc' in options.ddp_params and \
            options.ddp_params['model_url_sc'] != '':
            options.model_url_sc = os.path.join(MODEL_DIR, 'sdf', options.ddp_params['model_url_sc'])
            options.mesh_url_sc = DEFAULT_MESH_DIR

        options.debug_mode = config['debug']
        options.visualize_in_meshcat = config['visualize']

        return options

    def move_hand_to_start(self, joint_states):
        # check if the hand has moved to initial pose
        if np.allclose(joint_states, self.xa0, atol=1e-1):
            start_task = self.get_parameter('start_task').get_parameter_value().bool_value
            self.is_hand_initialized = start_task
            self.get_logger().info('Hand has moved to initial pose.', throttle_duration_sec=2.0)
            if not self.is_hand_initialized:
                self.get_logger().warn('Please set /highlevel_controller_node/start_task to true!', throttle_duration_sec=2.0)
            return

        current_time = self.get_ros_time(return_float=True)
        # traj = get_ros_traj_set_point(
        #     self.options.q_init[1:], \
        #     current_time, self.options.T_mpc, self.options.high_level_time_step,
        #     n_c=self.num_contacts)
        traj = get_ros_traj_interp(
            start_q=joint_states, end_q=self.xa0,
            start_time=current_time, num_steps=self.options.T_mpc, time_step=self.high_level_time_step,
            n_c=self.num_contacts
        )
        traj.joint_names = self.hand_joint_names
        traj_wrapper = JointTrajectoryWrapper()
        current_ros_time = float_to_ros_time(current_time)
        traj_wrapper.header_recv_state.stamp = current_ros_time
        traj_wrapper.header_begin_sol.stamp = current_ros_time
        traj_wrapper.header_finish_sol.stamp = current_ros_time
        traj_wrapper.mode.data = 0       # joint space
        traj_wrapper.traj = traj
        self.stored_traj_publisher_.publish(traj_wrapper)

    def solver_callback(self):
        """Method that is periodically called by the timer."""
        # wait till hardware ready
        if (not self.run_without_lowlevel) and (not self.hardware_states_ready):
            self.get_logger().info('Waiting for hardware states...', throttle_duration_sec=2.0)
            return

        with self._data_lock:
            latest_joint_states = self.ordered_joint_states.copy()
            latest_hand_joint_states = latest_joint_states[-16:].copy()
            latest_object_states = self.unactuated_object_states.copy()

        if not self.is_hand_initialized:
            self.get_logger().info(f'Moving the hand to initial pose...', throttle_duration_sec=2.0)
            # self.get_logger().warn(f'We disable messages from this node to debug!')
            # return

            self.move_hand_to_start(latest_hand_joint_states)
            # self.publish_visualization_msg()
            return

        if self.solver_iter == 0:
            state = self.x0.copy()
        else:
            qu_now = latest_object_states.copy()
            qa_now = latest_joint_states
            q_now = np.insert(qa_now, 0, qu_now)
            state = q_now.copy()

        if self.run_without_lowlevel:
            self.current_state_time = self.get_ros_time(return_float=True)

        # mpc start time is the latest joint states time
        current_state_time = self.current_state_time

        mpc_start_time = self.get_ros_time()
        sol = self.controller_core.SolveMpc(current_state_time, state)
        mpc_end_time = self.get_ros_time()
        if self.run_without_lowlevel:
            self.mpc_solve_time.append(
                ros_time_to_seconds(mpc_end_time) - ros_time_to_seconds(mpc_start_time))

        self.publish_visualization_msg()

        if self.run_without_lowlevel:
            self.update_states_bootstrap()
            if self.enable_perturb:
                self.add_perturb_to_xu()
                self.enable_perturb = False

        # FIXME: after solution, latest_joint_states is updated,
        # we shift the trajectory so its value as mpc_end_time,
        # which is mpc_start_time + mpc_solve_time, matches
        # latest_joint_states, this is to avoid jerks
        traj = convert_drake_traj_to_ros_traj(
            sol, self.high_level_time_step,
            # q_now=latest_joint_states, t_solve=mpc_solve_time
        )
        traj.joint_names = self.hand_joint_names

        traj_wrapper = JointTrajectoryWrapper()
        traj_wrapper.header_recv_state.stamp = float_to_ros_time(current_state_time)
        traj_wrapper.header_begin_sol.stamp = mpc_start_time
        traj_wrapper.header_finish_sol.stamp = mpc_end_time
        traj_wrapper.mode.data = 1       # cartesian space
        traj_wrapper.traj = traj
        self.stored_traj_publisher_.publish(traj_wrapper)

        self.solver_iter = self.solver_iter + 1
        if self.options.debug_mode:
            self.get_logger().info(f"Solver iteration: {self.solver_iter}")

    # def joint_state_callback(self, msg:JointState):
    #     """ Callback function for the joint states """
    #     # with self.lock:
    #     rec_j_order = msg.name
    #     joint_mapping = [rec_j_order.index(name) for name in self.joint_names]
    #     self.current_state_time = ros_time_to_seconds(msg.header.stamp)
    #     self.ordered_joint_states = np.array(msg.position)[joint_mapping]
        
    def publish_visualization_msg(self):
        # if not self.options.visualize_in_meshcat:
        #     return
        
        msg = self.controller_core.get_visualize_msg()
        self.visualize_publisher_.publish(msg)

    def send_solution_for_visualize(self):
        """ Send the solution trajectory for MeshCat visualization """
        # if not self.options.visualize_in_meshcat:
        #     return
        
        if not self.run_without_lowlevel:
            return
        
        msg = self.controller_core.get_visualization_traj_msg(self.saved_solution)
        self.visualize_publisher_.publish(msg)

    def print_statistics(self):
        """ Print statistics when reset """
        if self.run_without_lowlevel:
            if len(self.mpc_solve_time) > 0:
                avg_t_mpc = np.mean(self.mpc_solve_time)
                max_t_mpc, min_t_mpc  = np.max(self.mpc_solve_time), np.min(self.mpc_solve_time)
                self.get_logger().info(f"MPC solve time: avg= {avg_t_mpc:.3f} s, max= {max_t_mpc:.3f} s, min= {min_t_mpc:.3f} s")

    def hardware_states_callback(self, msg:HardwareStates):
        """
        Callback function for the hardware states.
        The received joints are in order (remapping was done in the publisher)
        """
        self.current_state_time = ros_time_to_seconds(msg.header.stamp)
        q = np.array(msg.q.data)
        with self._data_lock:
            self.unactuated_object_states[:] = np.atleast_1d(q[:-16])
            if self.fixed_wrist_joints is not None:
                self.ordered_joint_states[:] = np.concatenate((self.fixed_wrist_joints, q[-16:]))
            else:
                self.ordered_joint_states[:] = q[-16:]
        self.hardware_states_ready = True

    def update_states_bootstrap(self):
        """
            Update the state from the last solution (in a bootstrap manner)
            The state includes unactuated_object_states and ordered_joint_states
        """
        self.saved_solution_stamp.append(time.time() - self.saved_solution_start_time)
        self.saved_solution.append(self.controller_core.x_obs().tolist())
        with self._data_lock:
            xu_next = self.controller_core.x_next()[self.options.q_u_indices]
            xa_next = self.controller_core.xa_next()
            self.unactuated_object_states[:] = xu_next
            if self.fixed_wrist_joints is not None:
                self.ordered_joint_states[:] = np.concatenate((self.fixed_wrist_joints, xa_next))
            else:
                self.ordered_joint_states[:] = xa_next

    def add_perturb_to_xu(self):
        """ add perturbation to object state """
        # perturb so3
        if self.options.q_rot_indices is not None:
            quat_now = self.unactuated_object_states[self.options.q_rot_indices]
            quat_new = get_perturbed_quat(quat_now, ang_limit=np.deg2rad(35))
            self.unactuated_object_states[self.options.q_rot_indices] = quat_new


def main(args=None):
    """
    The main function.
    :param args: Not used directly by the user, but used by ROS2 to configure
    certain aspects of the Node.
    """
    # try:
    rclpy.init(args=args)
    node = HighlevelControl()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    # except KeyboardInterrupt:
    #     pass
    # except Exception as e:
    #     print(f"Error: {e}")
    # finally:
    #     print("Shutting down...")
    executor.shutdown()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

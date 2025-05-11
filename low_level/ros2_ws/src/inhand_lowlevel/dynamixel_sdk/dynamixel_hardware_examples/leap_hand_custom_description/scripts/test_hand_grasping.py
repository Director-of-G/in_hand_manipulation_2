#!/usr/bin/env python3

# test Leap Hand joint control with dynamixel sdk

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy
from enum import Enum
import numpy as np
from scipy.optimize import minimize
import time
from builtin_interfaces.msg import Duration
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from common_msgs.msg import ContactState, JointTrajectoryWrapper
from leap_ros2.utils import cross_product_matrix


class MyObjective(Enum):
    FORCE_BALANCE = 0
    WRENCH_BALANCE = 1

CALC_FORCE_MAGNITUDE_OBJECTIVE = MyObjective.WRENCH_BALANCE

HW_JOINT_ORDER = [f'joint_{i}' for i in list(range(16))]

# leap hand
HIGHLEVEL_JOINT_ORDER = [f'joint_{i}' for i in [1, 0, 2, 3, 12, 13, 14, 15, 5, 4, 6, 7, 9, 8, 10, 11]]
# allegro hand
# HIGHLEVEL_JOINT_ORDER = [f'joint_{i}' for i in range(16)]

CONTACT_ORDER = [
    'leap_hand_right::thumb_fingertip_collision',       # thumb
    'leap_hand_right::fingertip_collision',             # index
    'leap_hand_right::fingertip_2_collision',           # middle
    'leap_hand_right::fingertip_3_collision'            # ring
]

# ONLY in real world
# pre-saved hand configurations (joint_0~joint_16)
Q0_BANANA = np.array([-2.91456357e-02,  1.04464090e+00,  3.16000044e-01,  2.10155368e-01,
        1.18116520e-01,  1.00322342e+00,  4.12640840e-01,  2.05553427e-01,
        1.76407799e-01,  4.01902974e-01,  5.16951561e-01, -7.05631152e-02,
        1.63062167e+00, -1.53398083e-03,  4.47922409e-01, -2.13223338e-01])
Q0_PEAR = np.array([0.05675729, 1.56619442, 0.10891264, -0.25310683, \
                    0.03681554, 0.38963112, -0.48320395, -0.22856314, \
                    0.07516506,  0.05215535, -0.18254372,  0.11351458, \
                    1.51250505,  0.11351458, -0.40497094, 0.36201948])
Q0_PERSIMMON = np.array([-0.11504856, 0.73477679, 0.59058261, 0.2638447, \
                         0.02300971, 0.85596132, 0.3850292, 0.33287385, \
                         0.07056312, 0.02914564, 0.0398835, -0.00920388, \
                         1.52170897, 0.08743691, 0.01994175, 0.33594179])
Q0_MANGO = np.array([-0.10124274, 0.9664079, 0.45252433, 0.12732041, \
                     0.03528156, 1.06151474, 0.27918452, 0.21015537, \
                     0.10737866, 0.13499032, 0.09817477, 0.02914564, \
                     1.58000028, -0.01687379, -0.02607767, 0.21935926])
Q0_PEACH = np.array([-0.02454369,  1.01396132,  0.06902914,  0.320602  ,  0.04448544,
       -0.02454369, -0.10124274, -0.07516506, -0.03834952, -0.06135923,
        0.0398835 , -0.22549519,  1.43273807, -0.07823303,  0.10584468,
        0.04601942])
Q0_PEPPER = np.array([-3.06796171e-02,  1.24866045e+00,  1.22718466e-02,  7.05631152e-02,
        1.53398083e-03,  1.02623320e+00,  3.40543747e-01,  6.90291375e-02,
       -3.52815576e-02,  8.13009813e-02,  8.43689442e-02, -1.97883531e-01,
        1.62448573e+00, -4.14174832e-02,  1.22718468e-01, -4.60194238e-03])
Q0_ONION = np.array([-0.01687379,  0.76699042,  0.43565056,  0.36815539,  0.01380583,
        0.02914564, -0.01533981,  0.03374758, -0.02147573,  0.03374758,
       -0.03834952, -0.05522331,  1.44500995, -0.14879614, -0.27458256,
        0.62893212])

# ONLY in simulation
QO_YCB_BOX = np.array([0.0, 0.4345, 0.8090, 0.6935, 0.0, 0.4345, 0.8090, 0.6935, 0.0, 
                        0.4345, 0.8090, 0.6935, 1.79, 0.0, -0.626, 1.22])
QO_YCB_PEAR = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.6583, 0.7731, 0.28, 0.0, 
                        0.0, 0.0, 0.0, 1.2624, 0.8115, 0.195, 0.6886])

Q0_ALLEGRO_SPHERE = np.array([0.2, 0.95, 1.0, 1.0, 0.0, 0.6, 1.0, 1.0, -0.2, 0.95, 1.0, 1.0, 0.6, 1.95, 1.0, 1.0])

def compute_balanced_force_magnitude(N, Gmat, magnitude, weight=1.0):
    """
        Compute the force magnitude along each contact normal
        :param N: (nc, 3) contact normal
        :param Gmat: (6, 3*nc) grasping matrix
        :param magnitude: positive, desired force magnitude
        :param weight: positive, to penalize ||f||-manitude
    """
    def get_augmented_N(N):
        nc = N.shape[0]
        N_aug = np.zeros((nc, 3*nc))
        for i in range(nc):
            N_aug[i, i*3:(i+1)*3] = N[i]
        
        return N_aug

    # ex. in_contact=[True, False, True] --> 
    # np.repeat(in_contact, 3)= [True, True, True, False, False, False, True, True, True]
    in_contact = np.linalg.norm(N, axis=-1) > 1e-5
    N_reduced = N[in_contact, :]                        # (nc, 3)
    N_augmented = get_augmented_N(N_reduced)            # (nc, 3*nc)
    G_reduced = Gmat[:, np.repeat(in_contact, 3)]       # (6*3nc)
    nc = N_reduced.shape[0]
    
    def objective_force_balance(x):
        return x.T @ (N_reduced @ N_reduced.T + weight * np.eye(nc)) @ x \
                - 2 * weight * magnitude * np.ones(nc).T @ x
    
    def objective_wrench_balance(x):
        return x.T @ ((N_augmented @ G_reduced.T @ G_reduced @ N_augmented.T) + weight * np.eye(nc)) @ x \
                - 2 * weight * magnitude * np.ones(nc).T @ x
    
    if CALC_FORCE_MAGNITUDE_OBJECTIVE == MyObjective.FORCE_BALANCE:
        res = minimize(objective_force_balance, magnitude*np.ones(nc), method='BFGS')
    elif CALC_FORCE_MAGNITUDE_OBJECTIVE == MyObjective.WRENCH_BALANCE:
        res = minimize(objective_wrench_balance, magnitude*np.ones(nc), method='BFGS')

    force = np.zeros(4,)
    force[in_contact] = res.x.copy()

    return force


def float_to_ros_duration(duration):
    return Duration(sec=int(duration), nanosec=int((duration - int(duration)) * 1e9))

def get_ros_traj_set_point(q, force, normal, start_time, num_steps, time_step, n_c=4):
    """ Create high-level traj to a set point """
    assert force.shape == (n_c, 3)
    assert normal.shape == (n_c, 3)

    ros_traj = JointTrajectory()
    ros_traj.header.stamp = start_time
    for i in range(num_steps):
        _time_from_start = i*time_step
        point = JointTrajectoryPoint()
        point.positions = q.tolist()
        # fill other fields
        point.velocities = [0.0]*len(q)
        point.accelerations = normal.flatten().tolist()
        point.effort = force.flatten().tolist()
        point.time_from_start = float_to_ros_duration(_time_from_start)
        ros_traj.points.append(point)
    
    return ros_traj

def wait_for_message(node, topic, msg_type, timeout=None):
    future = rclpy.task.Future()

    def callback(msg):
        if not future.done():
            future.set_result(msg)

    sub = node.create_subscription(msg_type, topic, callback, 10)

    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)

    node.destroy_subscription(sub)

    if future.done():
        return future.result()
    else:
        return None

class JointCommandPublisher(Node):
    def __init__(self):
        super().__init__('joint_command_publisher')

        self.joint_pos = np.zeros(16,)
        self.joint_commands = np.zeros(16,)
        self.contact_threshold = 1.0
        self.is_finger_guide_mode = np.zeros(4, dtype=bool)
        
        self.remapping_to_hw = []
        self.remapping_to_highlevel = []
        self.is_joint_pos_ready = False
        self.get_remapped_joints(wait_for_message(self, '/joint_states', JointState))       # initialize remapping

        self.finger_to_jnt_map = {
            0: [12, 13, 14, 15],        # thumb
            1: [0, 1, 2, 3],            # index
            2: [4, 5, 6, 7],            # middle
            3: [8, 9, 10, 11]           # ring
        }
        self.preset_grasping_pose = QO_YCB_BOX       # will be sent to hand if not None

        # self.publisher_ = self.create_publisher(Float64MultiArray, '/joint_position_controller/commands', 10)
        self.publisher_ = self.create_publisher(
            JointTrajectoryWrapper,
            '/high_level_traj',
            # 10
            qos_profile=QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST
            ),
        )

        # reach pre-grasping pose
        self.q0 = self.get_joints()
        if self.preset_grasping_pose is not None:
            self.goto_joints(self.preset_grasping_pose, last_time=2.0)
        
        init_joint_pos = self.get_remapped_joints(wait_for_message(self, '/joint_states', JointState))
        self.q0 = init_joint_pos.copy()

        input('Please insert the object in hand and press Enter to continue...')
        self.get_logger().info('Will grasp harder to detect object normal...')
        self.grasp_harder(dq=0.03)

        self.object_normal = np.zeros((4, 3))       # (nc, 3)
        self.force_magnitude = 1.0                    # 2N

        self.get_object_normal()

        input('Press Enter to start Hybrid-Force-Motion-Control...')
        self.get_logger().info('Will start Hybrid-Force-Motion-Control...')
        self.create_timer(1/10, callback=self.timer_callback)

    def get_joints(self):
        """ Return the current joint pos """
        joint_pos = self.get_remapped_joints(wait_for_message(self, '/joint_states', JointState))
        return joint_pos.copy()

    def goto_joints(self, q, last_time=0.0):
        """ move to joint pos q """
        t_start = time.time()
        while True:
            if last_time == 0.0:
                qd = q
            else:
                qd = self.q0 + (q - self.q0) * (time.time() - t_start) / last_time
            traj = get_ros_traj_set_point(
                q=qd.copy()[self.remapping_to_highlevel],
                force=np.zeros((4, 3)),
                normal=np.zeros((4, 3)),
                start_time=self.get_clock().now().to_msg(),
                num_steps=10, time_step=0.1, n_c=4
            )
            traj.joint_names = HIGHLEVEL_JOINT_ORDER
            traj_msg = JointTrajectoryWrapper()
            traj_msg.traj = traj
            traj_msg.mode.data = 0
            self.publisher_.publish(traj_msg)

            if time.time() - t_start > last_time:
                break
            else:
                rclpy.spin_once(self, timeout_sec=0.1)

    def get_remapped_joints(self, msg:JointState):
        if len(self.remapping_to_hw) == 0:
            self.remapping_to_hw = [msg.name.index(f'joint_{i}') for i in range(16)]
            self.remapping_to_highlevel = [HW_JOINT_ORDER.index(i) for i in HIGHLEVEL_JOINT_ORDER]
        remapped_joints = np.asarray(msg.position)[self.remapping_to_hw]
        return remapped_joints
    
    def grasp_harder(self, dq=0.025, last_time=5.0):
        """
            Will interpolate each joint from q0 to q0-dq
        """
        t_start = time.time()
        while time.time() - t_start < last_time:
            t = time.time() - t_start
            self.joint_commands = self.q0 + dq * t / last_time
            self.joint_commands[[0, 4, 8]] = self.q0[[0, 4, 8]].copy()
            self.goto_joints(self.joint_commands)

    # obj_contact_pts = self.get_object_contact_pts()
    # Gmat = np.zeros((6, 3*self.nc))
    # for i in range(self.nc):
    #     Gmat[0:3, 3*i:3*(i+1)] = np.eye(3,)
    #     rel_contact_p = obj_contact_pts[i] - self.measured_obj_pos
    #     Gmat[3:, 3*i:3*(i+1)] = cross_product_matrix(rel_contact_p)

    def get_object_normal(self):
        """
            Get object normal of the contact point
        """
        contact_state = wait_for_message(self, '/contact_sensor_state', ContactState)
        object_states = wait_for_message(self, '/object_states', PoseStamped)

        # get object normal and contact points
        obj_contact_pts = np.zeros((4, 3))
        for i in range(4):
            idx_of_msg = contact_state.names.index(CONTACT_ORDER[i])
            # use the torque field to store object normal
            torque_as_normal = contact_state.wrenches[idx_of_msg].torque
            self.object_normal[i] = np.array([torque_as_normal.x, torque_as_normal.y, torque_as_normal.z])
            # contact point on object
            contact_pt = contact_state.points[idx_of_msg]
            obj_contact_pts[i] = np.array([contact_pt.x, contact_pt.y, contact_pt.z])

        # compute grasping matrix
        obj_pos = object_states.pose.position
        measured_obj_pos = np.array([obj_pos.x, obj_pos.y, obj_pos.z])
        Gmat = np.zeros((6, 3*4))
        for i in range(4):
            Gmat[0:3, 3*i:3*(i+1)] = np.eye(3,)
            rel_contact_p = obj_contact_pts[i] - measured_obj_pos
            Gmat[3:, 3*i:3*(i+1)] = cross_product_matrix(rel_contact_p)
        
        self.balanced_force_magnitude = compute_balanced_force_magnitude(self.object_normal, Gmat, self.force_magnitude, weight=0.2)
        print(f"Get object normal from initial contact: {self.object_normal} \nand balanced force magnitude: {self.balanced_force_magnitude}")

    def timer_callback(self):
        """
            Will publish high-level trajectory in this callback
        """
        force = self.balanced_force_magnitude.reshape(-1, 1) * self.object_normal.copy()
        normal = self.object_normal.copy()
        traj = get_ros_traj_set_point(
            q=self.q0.copy()[self.remapping_to_highlevel],
            force=force,
            normal=normal,
            start_time=self.get_clock().now().to_msg(),
            num_steps=10, time_step=0.1, n_c=4
        )
        traj.joint_names = HIGHLEVEL_JOINT_ORDER
        traj_msg = JointTrajectoryWrapper()
        traj_msg.traj = traj
        traj_msg.mode.data = 1
        self.publisher_.publish(traj_msg)


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

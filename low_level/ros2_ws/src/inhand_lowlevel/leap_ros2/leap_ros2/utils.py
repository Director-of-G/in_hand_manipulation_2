from enum import Enum
import re
import numpy as np
from scipy.spatial.transform import Rotation as SciR
from pydrake.all import PiecewisePolynomial

import rclpy
from rclpy.node import Node
from rclpy.client import Client
from rcl_interfaces.srv import GetParameters

from trajectory_msgs.msg import JointTrajectory
from geometry_msgs.msg import PoseStamped, Wrench
from builtin_interfaces.msg import Time


def time_msg_to_float(time:Time):
    return time.sec + time.nanosec * 1e-9


def duration_msg_to_float(duration):
    return duration.sec + duration.nanosec * 1e-9


def calc_delta_time(t1:Time, t2:Time):
    """ Return t2-t1 in seconds """
    return time_msg_to_float(t2) - time_msg_to_float(t1)


def make_rpy_continuous(rpy_old, rpy_new):
    """ Make sequential rpy continuous """
    def angle_difference(angle1, angle2):
        """ Angles are in [-pi, pi] """
        diff = (angle2 - angle1 + np.pi) % (2 * np.pi) - np.pi
        return diff
    
    # FIXME: the second euler angle is discontinuous, so leave it unchanged
    # however, we need ry rotations in the open box task
    rpy = np.zeros_like(rpy_new)
    # rpy[[0, 2]] = rpy_old[[0, 2]] + angle_difference(rpy_old[[0, 2]], rpy_new[[0, 2]])
    rpy[0:3] = rpy_old[0:3] + angle_difference(rpy_old[0:3], rpy_new[0:3])

    return rpy


def convert_wrench_array_to_msg(wrench_array):
    wrench_msg = Wrench()
    wrench_msg.force.x, wrench_msg.force.y, wrench_msg.force.z = wrench_array[:3]
    if wrench_array.shape[0] == 6:
        wrench_msg.torque.x, wrench_msg.torque.y, wrench_msg.torque.z = wrench_array[3:]
    return wrench_msg


def get_param(node_handle:Node, node_name:str, param_name:str, timeout=0.0):
    """ Block if timeout is None or negative. Don't wait if 0. """
    node_handle.get_logger().info(f'Wait for parameter /{node_name}/{param_name}...')
    client = node_handle.create_client(GetParameters, f'/{node_name}/get_parameters')
    client.wait_for_service(timeout_sec=timeout)
    request = GetParameters.Request()
    request.names = [param_name]
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node_handle, future, timeout_sec=timeout)
    if future.done():
        response = future.result()
        node_handle.get_logger().info(f'Parameter /{node_name}/{param_name} is set!')
        return response.values[0]
    else:
        node_handle.get_logger().error(f'Get parameter /{node_name}/{param_name} failed!')
        return
    

def compute_pose_error(t1, R1, t2, R2):
    """
        Return a 6d vector [p_e, r_e] including
        - position error (p_e): t2 - t1
        - orientation error (r_e): log(R2 * R1.T)
        (t1, R1) is the reference pose
    """
    p_e = t2 - t1
    R_e = SciR.from_matrix(np.dot(R2, R1.T)).as_rotvec()
    return np.concatenate([p_e, R_e])


def compute_orientation_error_jacobian_inv_lie_algebra(r):
    """
        Compute J_l(r)^{-1} acccording to the linear 
        approximation of the Baker-Campbell-Hausdorff (BCH)
        formula.
        :param r: axis-angle
    """
    angle = np.linalg.norm(r)
    if angle < 1e-5:
        return np.eye(3)
    halfangle = angle / 2
    axis = r / angle

    Jinv = (halfangle / np.tan(halfangle)) * np.eye(3) + \
            (1 - halfangle / np.tan(halfangle)) * np.outer(axis, axis) - \
            halfangle * np.cross(np.eye(3), axis)
    
    return Jinv


def cross_product_matrix(v):
    """
        Get the cross product matrix M of a 3d vector
        such that for any vector x, Mx = cross(v,x)
    """
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


class ReferenceTrajectory(object):
    def __init__(self) -> None:
        self.start_time = 0.0
        self.start_time_ros = Time()
        self.q = PiecewisePolynomial()
        self.v = PiecewisePolynomial()
        self.w = PiecewisePolynomial()
        self.n_object = PiecewisePolynomial()


class HardwareStatus(Enum):
    INITIALIED = 0
    RECEIVED = 1


class LowLevelCtrlMode(Enum):
    JOINTSPACE = 0
    CARTESIANSPACE = 1


class LowLevelContactPtsSource(Enum):
    MEASURED = 0
    HIGHLEVEL = 1


class HardwareState(object):
    """ q and v must be [qu,qa] and [vu,va] """
    def __init__(self, nq, nv, nc) -> None:
        self.status = HardwareStatus.INITIALIED
        self.receive_time = 0.0
        self.receive_time_ros = Time()
        self.q = np.zeros(nq,)
        self.v = np.zeros(nv,)
        self.f_ext = np.zeros((nc, 3))
        self.p_obj = np.zeros((nc, 3))
        
        # object 6d pose
        self.obj_pos = np.zeros(3,)
        self.obj_quat = np.array([1, 0, 0, 0])  # wxyz

    def get_q_actuated(self):
        """ Get hand dofs """
        return self.q[-16:]
    
    def get_q_unactuated(self):
        """ Get object dofs """
        return self.q[:-16]


class LowLevelOptions(object):
    def __init__(self) -> None:
        self.hw_type = ''
        self.hand_type = ''
        self.models_root = ''
        self.model_url = ''
        self.ordered_finger_geoms = []
        self.ordered_finger_links = []
        
        self.nc = 4
        self.nqa = 16
        self.nq = 17
        self.time_step = 0.01
        self.mpc_horizon = 5

        self.high_level_traj_shift_ratio = 0.75

        # deprecated
        # self.lqr_weights = {}
        # self.model_params = {}

        self.mpic_params = {}

        self.enable_coupling = False
        self.debug = False


class PoseEstimator(object):
    """
    Pose estimator for single rigid body
    """
    def __init__(self) -> None:
        self.last_time = None
        self.last_position = np.zeros(3,)           # (3,)
        self.last_rotation = np.zeros((3, 3))       # (3, 3) rotmat
        self.last_euler_continuous = np.zeros(3,)   # (3,)

        self.unfiltered_rotation = np.zeros((3, 3))

        self.estimated_linvel = np.zeros(3,)
        self.estimated_angvel = np.zeros(3,)

    def reset(self):
        self.last_time = None
        self.last_position = np.zeros(3,)
        self.last_rotation = np.zeros((3, 3))
        self.last_euler_continuous = np.zeros(3,)

    def update(self, pose:PoseStamped):
        position = pose.pose.position
        current_position = np.array([position.x, position.y, position.z])

        orientation = pose.pose.orientation
        _quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        current_rotation = SciR.from_quat(_quaternion).as_matrix()
        current_euler = SciR.from_quat(_quaternion).as_euler('xyz')

        self.unfiltered_rotation = current_rotation

        if self.last_time is not None:
            dt = calc_delta_time(self.last_time, pose.header.stamp)
            delta_position = current_position - self.last_position
            delta_rotation = np.dot(current_rotation, self.last_rotation.T)
            self.estimated_linvel = delta_position / dt
            self.estimated_angvel = SciR.from_matrix(delta_rotation).as_rotvec() / dt

        self.last_position = current_position
        self.last_rotation = current_rotation
        self.last_euler_continuous = make_rpy_continuous(self.last_euler_continuous, current_euler)
        self.last_time = pose.header.stamp

    def get_pose_estimation(self):
        if self.last_time is None:
            return None, None
        else:
            # return self.last_position, SciR.from_matrix(self.last_rotation).as_rotvec()
            return self.last_position.copy(), self.last_euler_continuous.copy()
        
    def get_quaternion(self):
        if self.last_time is None:
            return np.array([1, 0, 0, 0])
        else:
            return SciR.from_matrix(self.last_rotation).as_quat()[[3, 0, 1, 2]]       # xyzw --> wxyz

    def get_unfiltered_quaternion(self):
        return SciR.from_matrix(self.unfiltered_rotation).as_quat()[[3, 0, 1, 2]]

    def get_vel_estimation(self):
        if self.last_time is None:
            self.estimated_linvel = np.zeros(3,)
            self.estimated_angvel = np.zeros(3,)
        
        return self.estimated_linvel.copy(), self.estimated_angvel.copy()


def convert_ros_traj_to_drake_traj(traj:JointTrajectory):
    num_steps = len(traj.points)
    assert num_steps > 0
    nq = len(traj.points[0].positions)
    nv = len(traj.points[0].velocities)
    nc = len(traj.points[0].effort) // 3

    q_knots = np.zeros((num_steps, nq))
    v_knots = np.zeros((num_steps, nv))
    w_knots = np.zeros((num_steps, nc * 3))
    n_obj_knots = np.zeros((num_steps, nc * 3))
    for i in range(num_steps):
        q_knots[i] = traj.points[i].positions
        v_knots[i] = traj.points[i].velocities
        w_knots[i] = traj.points[i].effort
        n_obj_knots[i] = traj.points[i].accelerations

    start_time = time_msg_to_float(traj.header.stamp)
    t_knots = start_time + \
        np.array([duration_msg_to_float(traj.points[i].time_from_start) for i in range(num_steps)])
    
    q_polynomial = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(t_knots, q_knots.T)
    v_polynomial = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(t_knots, v_knots.T)
    w_polynomial = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(t_knots, w_knots.T)
    n_obj_polynomial = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(t_knots, n_obj_knots.T)

    ref_traj = ReferenceTrajectory()
    ref_traj.start_time = start_time
    ref_traj.start_time_ros = traj.header.stamp
    ref_traj.q = q_polynomial
    ref_traj.v = v_polynomial
    ref_traj.w = w_polynomial
    ref_traj.n_object = n_obj_polynomial

    return ref_traj


def shift_picewise_polynomial(poly:PiecewisePolynomial.CubicWithContinuousSecondDerivatives, shift_amout):
    assert len(shift_amout) == poly.rows()
    t_knots = poly.get_segment_times()
    q_knots = poly.vector_values(t_knots).T
    q_knots += shift_amout
    new_poly = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(t_knots, q_knots.T)
    return new_poly


def update_picewise_polynomial(poly:PiecewisePolynomial.CubicWithContinuousSecondDerivatives, q_new):
    assert len(q_new) == poly.rows()
    t_knots = poly.get_segment_times()
    new_poly = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(t_knots, q_new.T)
    return new_poly


def convert_object_dofs_dict_to_array(dofs_dict):
    dof_map = {
        'x': 0,
        'y': 1,
        'z': 2,
        'rx': 3,
        'ry': 4,
        'rz': 5
    }
    dofs = []
    for trans_key in dofs_dict['xyz']:
        dofs.append(dof_map[trans_key])
    for rot_key in dofs_dict['rpy']:
        dofs.append(dof_map[rot_key])

    return dofs


def get_remapping(source, target):
    remapping = []
    for name in target:
        remapping.append(source.index(name))
    return remapping


class LowLevelMPCProblemData(object):
    def __init__(self, n_c, n_qa, h, horizon):
        self.discrete_step = h
        self.horizon_length = horizon

        self.n_c = n_c
        self.n_qa = n_qa
        self.n_x = 2*self.n_qa+3*self.n_c
        self.n_u = self.n_qa

        self.J_list = np.zeros((self.n_c, 3, self.n_qa))
        self.Kbar_list = np.zeros((self.n_c, 3, 3))

        self.x0 = np.zeros((self.n_x, 1))
        self.x_ref = np.zeros((self.n_x, self.horizon_length+1))
        self.u_ref = np.zeros((self.n_u, self.horizon_length))
        self.fext_ff = np.zeros((3*self.n_c, self.horizon_length))

        self.object_normal = np.zeros((self.horizon_length+1, self.n_c, 3))

    def set_x_ref(self, x_ref):
        assert x_ref.shape == (self.n_x, self.horizon_length+1)
        self.x_ref = x_ref.copy()

    def set_x0(self, x0):
        if len(x0.shape) == 1:
            x0 = x0.reshape(-1, 1)
        assert x0.shape == (self.n_x, 1)
        self.x0 = x0.copy()

    def set_J(self, J_list):
        assert J_list.shape == (self.n_c, 3, self.n_qa)
        self.J_list = J_list.copy()

    def set_Kbar(self, Kbar_list):
        assert Kbar_list.shape == ((self.n_c, 3, 3))
        self.Kbar_list = Kbar_list.copy()

    def set_Ac(self, A):
        assert A.shape == (self.n_x, self.n_x)
        self.Ac = A.copy()

    def set_Bc(self, B):
        assert B.shape == (self.n_x, self.n_u)
        self.Bc = B.copy()

    def set_Ad(self, A):
        assert A.shape == (self.n_x, self.n_x)
        self.Ad = A.copy()

    def set_Bd(self, B):
        assert B.shape == (self.n_x, self.n_u)
        self.Bd = B.copy()

    def set_Q(self, Q):
        assert Q.shape == (self.n_x, self.n_x)
        self.Q = Q.copy()

    def set_R(self, R):
        assert R.shape == (self.n_u, self.n_u)
        self.R = R.copy()

    def set_delR(self, delR):
        assert delR.shape == (self.n_u, self.n_u)
        self.delR = delR


class EMASmoother(object):
    """
    Smoothing using EMA (Exponential Moving Average)
    """
    def __init__(self, alpha, dim=16) -> None:
        """
        :param alpha: 0~1, 0 for no smoothing, 1 for full smoothing (constant)
        """
        self.alpha = np.clip(alpha, 0.0, 1.0)
        if abs(self.alpha - alpha) > 1e-6:
            print(f'Alpha value clipped to {self.alpha}!')
        self.dim = dim
        self.prev_data = np.zeros(self.dim)

    def reset(self):
        self.prev_data = np.zeros_like(self.prev_data)

    def update(self, curr_data):
        if not isinstance(curr_data, np.ndarray):
            curr_data = np.array(curr_data)
        assert curr_data.shape == (self.dim,)

        updated_data = self.alpha * self.prev_data + (1-self.alpha) * curr_data
        self.prev_data = updated_data
        return updated_data
    
    def get_latest(self):
        return self.prev_data.copy()

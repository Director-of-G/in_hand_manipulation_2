import os
import numpy as np
from pydrake.all import PiecewisePolynomial
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration, Time


INHAND_HOME = os.environ.get('INHAND_HOME')
DRAKE_HOME = os.environ.get("DRAKE_HOME")
QSIM_HOME = os.environ.get("QSIM_HOME")

MODEL_DIR = os.path.join(INHAND_HOME, 'models')
DDP_SOLVER_DIR = os.path.join(INHAND_HOME, 'ddp')
OPTIMIZER_DIR = os.path.join(DDP_SOLVER_DIR, 'optimizer')

DEFAULT_MESH_DIR = os.path.join(DRAKE_HOME, "manipulation/models", "leap_hand_description/urdf/leap_hand/mesh/simple_obj")

QSIM_MODEL_DIR = os.path.join(QSIM_HOME, 'models')


class StoredTrajectory(object):
    def __init__(self):
        self.start_time = -1.0
        self.q = PiecewisePolynomial()  # joint position
        self.u = PiecewisePolynomial()  # desired (delta) joint position
        self.w = PiecewisePolynomial()  # contact wrench
        self.n_object = PiecewisePolynomial()  # contact normal

def ros_time_to_seconds(time):
    return time.sec + time.nanosec * 1e-9

def ros_duration_to_seconds(duration):
    return duration.nanoseconds * 1e-9

def float_to_ros_time(time):
    return Time(sec=int(time), nanosec=int((time - int(time)) * 1e9))

def float_to_ros_duration(duration):
    return Duration(sec=int(duration), nanosec=int((duration - int(duration)) * 1e9))

def get_ros_traj_set_point(q, start_time, num_steps, time_step, n_c=4):
    """ Create high-level traj to a set point """
    ros_traj = JointTrajectory()
    ros_traj.header.stamp = float_to_ros_time(start_time)
    for i in range(num_steps):
        _time_from_start = i*time_step
        point = JointTrajectoryPoint()
        point.positions = q.tolist()
        # fill other fields
        point.velocities = [0.0]*len(q)
        point.accelerations = [0.0]*n_c*3
        point.effort = [0.0]*n_c*3
        point.time_from_start = float_to_ros_duration(_time_from_start)
        ros_traj.points.append(point)
    
    return ros_traj

def get_ros_traj_interp(start_q, end_q, start_time, num_steps, time_step, n_c=4):
    """ Create high-level traj to a set point through interpolation and velocity saturation """
    ros_traj = JointTrajectory()
    ros_traj.header.stamp = float_to_ros_time(start_time)

    max_vel = 5.0
    max_dq = max_vel*time_step
    q = start_q + np.clip(1.5*(end_q-start_q), -max_dq, max_dq)
    q_linear = np.linspace(start_q, q, num_steps)
    for i in range(num_steps):
        _time_from_start = i*time_step
        point = JointTrajectoryPoint()
        point.positions = q_linear[i].tolist()
        # fill other fields
        point.velocities = [0.0]*len(q)
        point.accelerations = [0.0]*n_c*3
        point.effort = [0.0]*n_c*3
        point.time_from_start = float_to_ros_duration(_time_from_start)
        ros_traj.points.append(point)
    
    return ros_traj

def convert_drake_traj_to_ros_traj(drake_traj:StoredTrajectory, time_step, q_now=None, t_solve=None):
    """
        Convert Drake's StoredTrajectory format to ROS's JointTrajectory Fromat
        If q_now is given, will shift all q[t] values by q0 - q_now
    """
    ros_traj = JointTrajectory()
    start_time = drake_traj.start_time
    ros_traj.header.stamp = float_to_ros_time(start_time)

    num_steps = drake_traj.q.get_number_of_segments() + 1

    if q_now is not None:
        assert t_solve is not None
        dq_shift = q_now - drake_traj.q.value(t_solve).flatten()
    else:
        dq_shift = np.zeros_like(drake_traj.q.value(0.0).flatten())

    # t_knots of the points start from 0.0
    for i in range(num_steps):
        _time_from_start = i*time_step
        point = JointTrajectoryPoint()

        # joint pos
        point.positions = (drake_traj.q.value(_time_from_start).flatten() + dq_shift).tolist()

        # joint vel
        _delta_q = drake_traj.u.value(_time_from_start).flatten()
        point.velocities = (_delta_q / time_step).tolist()

        # store object normal in acceleration field
        point.accelerations = drake_traj.n_object.value(_time_from_start).flatten().tolist()

        # contact force (n_c*6 stored here for convenience)
        point.effort = drake_traj.w.value(_time_from_start).flatten().tolist()

        point.time_from_start = float_to_ros_duration(_time_from_start)
        ros_traj.points.append(point)

    return ros_traj

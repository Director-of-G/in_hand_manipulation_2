############################################################
# This script is modified from inhand_ddp_full_hand.py,    #
# but demonstrates a simplified case. The manipuland could #
# only rotates w.r.t. the z-axis.                          #
############################################################

import os
import sys


import copy
import time
import pickle
import numpy as np
# import quaternion
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt

import crocoddyl
import pinocchio as pin

from irs_mpc2.irs_mpc_params import (
    SmoothingMode,
    kSmoothingMode2ForwardDynamicsModeMap,
)
from irs_mpc2.quasistatic_visualizer import QuasistaticVisualizer

from qsim.parser import QuasistaticParser
from qsim_cpp import QuasistaticSimulatorCpp
from qsim_cpp import GradientMode, ForwardDynamicsMode

from manipulation.meshcat_utils import AddMeshcatTriad



from pydrake.all import (
    AngleAxis,
    PiecewisePolynomial,
    Quaternion,
    RigidTransform
)

from pydrake.geometry import (
    Rgba,
    Cylinder,
)

from pydrake.math import (
    RollPitchYaw,
    RotationMatrix,
)

from meshcat import transformations as tf
from common.common_ddp import *
from residual_model import ResidualModelFrameRotation
from common.inhand_ddp_helper import InhandReferenceGenerator, ContactTrackReference
from common.ddp_logger import *
from common.get_model_path import *

QSIM_MODEL_PATH = os.path.join(YML_PATH, "allegro_3d_sphere.yml")


# convert 2d array to list of 1d array
def convert_array_to_list(array_2d):
    return [array_2d[i, :] for i in range(array_2d.shape[0])]

def make_skew_symmetric_from_vec(v):
    return np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])

def CalcNW2Qdot(q):
    # mat_w2dq = np.array([[-q[1], -q[2], -q[3]],
    #                      [q[0], q[3], -q[2]],
    #                      [-q[3], q[0], q[1]],
    #                      [q[2], -q[1], q[0]]]) * 0.5

    E = np.zeros((4, 3))
    E[0, :] = -q[-3:]
    bottom_rows_E = -make_skew_symmetric_from_vec(q[-3:])
    bottom_rows_E[0, 0] = q[0]; bottom_rows_E[1, 1] = q[0]; bottom_rows_E[2, 2] = q[0]
    E[-3:, :] = bottom_rows_E.copy()
    E *= 0.5

    return E

def CalcNQdot2W(q):
    E = np.zeros((3, 4))
    E[:, 0] = -q[-3:]
    right_rows_E = make_skew_symmetric_from_vec(q[-3:])
    right_rows_E[0, 0] = q[0]; right_rows_E[1, 1] = q[0]; right_rows_E[2, 2] = q[0]
    E[:, -3:] = right_rows_E.copy()
    E *= 2

    return E

def convert_rpy_to_quat(rpy):
    quat = RollPitchYaw(rpy).ToQuaternion().wxyz()
    return quat

def convert_quat_to_rpy(quat):
    rpy = RotationMatrix(Quaternion(quat)).ToRollPitchYaw().vector()
    return rpy

def x_rpy_to_x_quat(x):
    x_rpy = np.atleast_2d(x)
    nx = x_rpy.shape[0]
    x_quat = np.zeros((nx, 20))
    for i in range(nx):
        x_quat[i] = np.concatenate((
            convert_rpy_to_quat(x_rpy[i, 0:3]),
            x_rpy[i, 3:]
        ))

    return x_quat.squeeze()

def x_quat_to_x_rpy(x):
    x_quat = np.atleast_2d(x)
    nx = x_quat.shape[0]
    x_rpy = np.zeros((nx, 19))
    for i in range(nx):
        x_rpy[i] = np.concatenate((
            convert_quat_to_rpy(x_quat[i, 0:4]),
            x_quat[i, 4:]
        ))

    return x_rpy.squeeze()

def normalize_array(arr):
    return arr / np.linalg.norm(arr)

def normalize_quaternions(x, options):
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    q_u_quat_indices = options.q_u_indices[:4]
    x[:, q_u_quat_indices] = \
        x[:, q_u_quat_indices] / np.linalg.norm(x[:, q_u_quat_indices], axis=1).reshape(-1, 1)

class QSimDDPParams(object):
    def __init__(self):
        # model
        self.ball_joint_type = BallJointType.kRPY
        self.mount_xyz = np.array([-0.06, 0.0, 0.072])

        self.h = 0.1            # 0.025
        self.kappa = 100        # smoothing
        self.kappa_exact = 10000    # smoothing for exact rollout
        self.nx = 20
        self.nu = 16
        self.q_a_indices = [4, 5, 6, 7, 8, 9, 10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19]
        self.q_u_indices = [0, 1, 2, 3]

        self.xrot_slices = slice(0, 4)

        self.enable_exact_rollout = False

        self.T_trajopt = 40
        self.T_ctrl = 10            # 20
        
        self.x0 = np.array([
            1, 0, 0, 0,                 # ball
            # 0.1368093, 0.58294937, 0.21369098, -0.77187396,
            0.2, 0.95, 1.0, 1.0,                # index finger
            0.0, 0.6, 1.0, 1.0,                 # middle finger
            -0.2, 0.95, 1.0, 1.0,               # ring finger
            0.6, 1.95, 1.0, 1.0
        ])
        self.xa_reg = np.array([
            0.1, 1.0, 1.0, 1.0,                 # index finger
            0.0, 0.7, 1.0, 1.0,                 # middle finger
            -0.1, 1.0, 1.0, 1.0,                # ring finger
            0.6, 1.9, 1.0, 1.0                  # thumb
        ])
        # observation
        self.x_obs = self.x0.copy()
        # reference (for trajopt)
        self.target_rpy = np.array([0.25*np.pi, 0.75*np.pi, -0.5*np.pi])
        self.target = \
            np.insert(
            self.xa_reg,
            0,
            RollPitchYaw(self.target_rpy).ToQuaternion().wxyz()
        )
        # reference (for MPC)
        self.x_ref = None
        # warm start (for MPC)
        self.x_init = None
        self.u_init = None
        # self.dyaw = 0.15

        self.u_lb = -0.1 * np.ones(self.nu)             # -0.05
        self.u_ub = 0.1 * np.ones(self.nu)              # 0.05
        # self.execution_scale = 1.0                       # 1.0, positive scalar, scale u0 when execution (i.e., rollout new x0)

        self.ddp_solver = crocoddyl.SolverBoxDDP         # SolverBoxDDP, SolverDDP
        self.ddp_verbose = False

        self.resolve_traj_opt = False
        self.visualize_while_control = True

        # viz helper
        self.viz_helper = InhandReferenceGenerator()
        self.contact_request = ContactTrackReferenceRequest()
        self.contact_response = ContactTrackReferenceResponse()

        # trajopt params
        self.TO_w_u = 1e1
        self.TO_w_a = 1e-2
        self.TO_w_u_T = 1e2
        self.TO_w_a_T = 1e0

        # mpc control params
        # self.w_u = 5e1
        # self.w_a = (1e-2, 1e0)          # (1e-2, 1e0)
        # self.w_u_T = 5e3
        # self.w_a_T = 1e3                # 1e3
        
        self.w_xreg = 1
        self.w_ureg = 0.04


class ReferenceGeneratorRPY(object):
    """
        This class generates the reference for the RPY rotation task
    """
    def __init__(self):
        # the targets
        self.target = Quaternion([1, 0, 0, 0])
        self.target_axis = np.array([0, 0, 1])
        self.target_angle = 0

        # the lut
        self.quat_lut = np.zeros((0, 4))        # quaternion mid-points
        self.dyaw_lut = 1.0 * np.pi / 200
        
        self.error_thresh = 5e-6                # ~3 degrees
        self.dyaw_ref = 0.025       # step length
        self.T_ref = 11                         # reference length

    def set_start_and_target_rpy(self, start_rpy, target_rpy):
        # clear lut
        self.quat_lut = np.zeros((0, 4))

        start_quat = Quaternion(convert_rpy_to_quat(start_rpy))
        target_quat = Quaternion(convert_rpy_to_quat(target_rpy))
        start_rot = RotationMatrix(start_quat)
        self.target = target_quat

        dq = start_quat.inverse().multiply(target_quat)
        angle_axis = RotationMatrix(dq).ToAngleAxis()

        self.target_angle = angle_axis.angle()
        self.target_axis = angle_axis.axis()

        # make lut
        n_lut = int(abs(self.target_angle) // self.dyaw_lut)
        # TODO(yongpeng): apply rotation matrix, not sure if any bugs
        for i in range(n_lut):
            rmat_i = RotationMatrix(
                AngleAxis(i * np.sign(self.target_angle) * self.dyaw_lut, self.target_axis)
            )
            quat_i = start_rot.multiply(rmat_i).ToQuaternion().wxyz()
            self.quat_lut = np.append(self.quat_lut, np.array([quat_i]), axis=0)

        # TODO(yongpeng): interpolate wxyz
        # # interpolate wxyz
        # quat_lut_unnormalized = np.linspace(start_quat.wxyz(), target_quat.wxyz(), n_lut)
        # # extend ahead
        # for i in range(1, 5*self.T_ref):
        #     quat_i = quat_lut_unnormalized[-1] + i * (quat_lut_unnormalized[-1] - quat_lut_unnormalized[-1])
        #     quat_lut_unnormalized = np.append(quat_lut_unnormalized, np.array([quat_i]), axis=0)
        # self.quat_lut = quat_lut_unnormalized / np.linalg.norm(quat_lut_unnormalized, axis=1).reshape(-1, 1)

        # TODO(yongpeng): random rotation (continuous)
        self.rand_axis = normalize_array(np.random.uniform(-1, 1, size=(3,)))

    def generate_reference_from_x0(self, x0):
        """
            x0: quat
        """
        quat_ref = np.zeros((self.T_ref, 4))

        # TODO(yongpeng): interpolate from nearesr neighbor
        # # find nearest
        # qdiff = self.quat_lut.dot(x0)
        # near_idx = np.argmax(np.abs(qdiff))
        # assert (abs(qdiff[near_idx]) <= 1.0)
        # q_near = self.quat_lut[near_idx]
        # rot_near = RotationMatrix(Quaternion(q_near))

        # # make reference
        # for i in range(self.T_ref):
        #     rmat_i = RotationMatrix(
        #         AngleAxis(i * np.sign(self.target_angle) * self.dyaw_ref, self.target_axis)
        #     )
        #     quat_i = rot_near.multiply(rmat_i).ToQuaternion().wxyz()

        #     if np.dot(quat_i, x0) < 0:
        #         quat_i = -quat_i

        #     quat_ref[i] = quat_i

        # TODO(yongpeng): interpolate wxyz from nearest neighbor
        # # find nearest
        # qdiff = self.quat_lut.dot(x0)
        # near_idx = np.argmax(np.abs(qdiff))
        # assert (abs(qdiff[near_idx]) <= 1.0)

        # # make reference
        # for i in range(self.T_ref):
        #     if near_idx + i >= len(self.quat_lut):
        #         quat_i = self.quat_lut[-1]
        #     else:
        #         quat_i = self.quat_lut[near_idx + i]

        #     if np.dot(quat_i, x0) < 0:
        #         quat_i = -quat_i

        #     quat_ref[i] = quat_i

        # TODO(yongpeng): interpolate from x0
        # calc difference
        start_quat = Quaternion(x0)
        target_quat = self.target
        start_rot = RotationMatrix(start_quat)

        dq = start_quat.inverse().multiply(target_quat)
        angle_axis = RotationMatrix(dq).ToAngleAxis()

        angle_ = angle_axis.angle()
        axis_ = angle_axis.axis()

        interpolate_steps = int(abs(angle_) / self.dyaw_ref)
        print(f"error {angle_}deg, interp steps {interpolate_steps}")

        for i in range(min(interpolate_steps, self.T_ref)):
            rmat_i = RotationMatrix(
                AngleAxis(i * np.sign(angle_) * self.dyaw_ref, axis_)
            )
            quat_i = start_rot.multiply(rmat_i).ToQuaternion().wxyz()

            # invert the sign if needed
            if np.dot(quat_i, x0) < 0:
                quat_i = -quat_i

            quat_ref[i] = quat_i

        for i in range(interpolate_steps, self.T_ref):
            quat_ref[i] = self.target.wxyz() if np.dot(self.target.wxyz(), x0) >= 0 else -self.target.wxyz()

        # TODO(yongpeng): set reference to target
        # for i in range(self.T_ref):
        #     quat_ref[i] = self.target.wxyz() if np.dot(self.target.wxyz(), x0) >= 0 else -self.target.wxyz()

        # TODO(yongpeng): continuous rotation
        # start_rot = RotationMatrix(Quaternion(x0))
        # for i in range(self.T_ref):
        #     rmat_i = RotationMatrix(
        #         AngleAxis(i * self.dyaw_ref, self.rand_axis)
        #     )
        #     quat_i = start_rot.multiply(rmat_i).ToQuaternion().wxyz()

        #     # invert the sign if needed
        #     if np.dot(quat_i, x0) < 0:
        #         quat_i = -quat_i

        #     quat_ref[i] = quat_i

        # TODO(debug)
        print(f"ref[0] <--> x0: {self.compute_error_between(quat_ref[0], x0)}, \
                ref[0] <--> target: {self.compute_error_between(quat_ref[0], self.target)}")
        
        print(f"ref[1] <--> x0: {self.compute_error_between(quat_ref[1], x0)}, \
                ref[1] <--> target: {self.compute_error_between(quat_ref[1], self.target)}")
        
        # error_to_x0_list, error_to_target_list = [], []
        # for i in range(2, self.T_ref):
        #     error_to_x0_list.append(self.compute_error_between(quat_ref[i], x0))
        #     error_to_target_list.append(self.compute_error_between(quat_ref[i], self.target))
        # print("error to x0: ", error_to_x0_list)
        # print("error to target: ", error_to_target_list)

        return quat_ref

    def compute_error(self, x0):
        """
            x0: quat
        """
        quat0 = Quaternion(x0)
        dq = quat0.inverse().multiply(self.target)

        return abs(RotationMatrix(dq).ToAngleAxis().angle())
    
    def compute_error_between(self, q0, q1):
        """
            q0, q1: quaternions
        """
        q0, q1 = Quaternion(q0), Quaternion(q1)
        dq = q0.inverse().multiply(q1)

        return abs(RotationMatrix(dq).ToAngleAxis().angle())


## ----------------------------------------


def allocate_resources_for_DDP(options:QSimDDPParams, num_envs=21):
    parser = QuasistaticParser(QSIM_MODEL_PATH)

    # qsims
    default_q_sims = []
    for i in range(num_envs):
        q_sim = parser.make_simulator_cpp()
        default_q_sims.append(q_sim)

    # sim params
    default_sim_params = copy.deepcopy(default_q_sims[0].get_sim_params())
    default_sim_params.gradient_mode = GradientMode.kAB
    
    default_sim_params.h = options.h
    default_sim_params.log_barrier_weight = options.kappa
    default_sim_params.forward_mode = ForwardDynamicsMode.kLogIcecream
    # default_sim_params.calc_contact_forces = False
    default_sim_params.calc_contact_forces = True
    default_sim_params.use_free_solvers = True

    # quasi-static model definition
    state = crocoddyl.StateVector(options.nx)
    state.nq = state.nx; state.nv = 0       # quasi-static
    actuation = crocoddyl.ActuationModelAbstract(state, nu=options.nu)

    return default_q_sims, default_sim_params, state, actuation


class QuasistaticActionModel(crocoddyl.ActionModelAbstract):
    def __init__(self, q_sim, sim_params, state, actuation, defaultCostModel, options:QSimDDPParams):
        crocoddyl.ActionModelAbstract.__init__(self, state, actuation.nu, defaultCostModel.nr)
        
        self.default_cost = defaultCostModel
        self.default_cost_data = self.default_cost.createData(crocoddyl.DataCollectorAbstract())

        self.q_sim = q_sim
        self.sim_params = sim_params

        # rpy
        self.nx_qsim_ = len(self.q_sim.get_mbp_positions_as_vec())
        self.nu_qsim_ = q_sim.num_actuated_dofs()
        # quaternion
        self.nx_ = options.nx
        self.nu_ = options.nu

        self.q_a_indices_ = options.q_a_indices
        self.q_u_indices_ = options.q_u_indices

    def calc(self, data, x, u=None):
        """
            x: includes quaternion
        """
        if u is None:
            u = np.zeros(self.nu_)
        u_incre = u.copy()

        u = u + x[self.q_a_indices_]

        # quat --> rpy
        x_rpy = np.zeros(self.nx_qsim_)
        x_rpy[0:3] = convert_quat_to_rpy(x[0:4])
        x_rpy[3:] = x[4:].copy()

        # call q_sim's forward dynamics
        xnext_rpy = self.q_sim.calc_dynamics_forward(x_rpy, u, self.sim_params)

        # rpy --> quat
        data.xnext = np.zeros_like(x)
        quat_next = convert_rpy_to_quat(xnext_rpy[0:3])
        # TODO(yongpeng): make the quaternion value continuous
        # if (np.dot(quat_next, x[0:4]) < 0):
        #     quat_next = -quat_next
        data.xnext[0:4] = quat_next
        data.xnext[4:] = xnext_rpy[3:].copy()

        # compute default cost (use quat)
        self.default_cost.calc(self.default_cost_data, x, u_incre)
        default_cost_value = sum([c.cost for c in self.default_cost_data.costs.todict().values()])

        data.cost = default_cost_value

    def calcDiff(self, data, x, u=None):
        """
            x: includes quaternion
        """
        if u is None:
            u = np.zeros(self.nu_)
        u_incre = u.copy()

        # call q_sim's backward dynamics
        self.q_sim.calc_dynamics_backward(self.sim_params)

        partial_u_partial_q = np.zeros((self.nu_, self.nx_))
        partial_u_partial_q[:, self.q_a_indices_] = np.eye(self.nu_)

        # calc omega <--> qdot projection matrices
        left_mat = np.zeros((self.nx_, self.nx_qsim_))
        right_mat = np.zeros((self.nx_qsim_, self.nx_))

        left_mat[0:4, 0:3] = CalcNW2Qdot(x[0:4]); left_mat[4:, 3:] = np.eye(self.nu_)
        right_mat[0:3, 0:4] = CalcNQdot2W(x[0:4]); right_mat[3:, 4:] = np.eye(self.nu_)

        Dq_nextDq_rpy = self.q_sim.get_Dq_nextDq()      # rpy, (nx_qsim, nx_qsim)
        Dq_nextDqa_cmd_rpy = self.q_sim.get_Dq_nextDqa_cmd()    # rpy, (nx_qsim, n_u)

        Dq_nextDq = left_mat @ Dq_nextDq_rpy @ right_mat
        Dq_nextDqa_cmd = left_mat @ Dq_nextDqa_cmd_rpy

        data.Fx = Dq_nextDq + Dq_nextDqa_cmd @ partial_u_partial_q
        data.Fu = Dq_nextDqa_cmd

        # differentiate default cost (use quat)
        self.default_cost.calcDiff(self.default_cost_data, x, u_incre)

        data.Lx = self.default_cost_data.Lx
        data.Lu = self.default_cost_data.Lu
        data.Lxx = self.default_cost_data.Lxx
        data.Lxu = self.default_cost_data.Lxu
        data.Luu = self.default_cost_data.Luu


## TODO(yongpeng): these codes are for solving inhand trajectory optimization with DDP
def solve_ddp_trajopt_problem(
    default_q_sims,
    default_sim_params,
    state,
    actuation,
    options: QSimDDPParams,
    q_vis: QuasistaticVisualizer=None
):
    # residual for state and cost
    xResidual = crocoddyl.ResidualModelState(state, options.target, actuation.nu)
    xActivation = crocoddyl.ActivationModelWeightedQuad(
        np.array([1e1] * (options.nx - options.nu) + [1e-2]*options.nu)
    )
    xTActivation = crocoddyl.ActivationModelWeightedQuad(
        np.array([1e2] * (options.nx - options.nu) + [1e0]*options.nu)
    )
    uResidual = crocoddyl.ResidualModelControl(state, actuation.nu)

    # cost term
    xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
    uRegCost = crocoddyl.CostModelResidual(state, uResidual)
    xRegTermCost = crocoddyl.CostModelResidual(state, xTActivation, xResidual)

    # cost model
    runningCostModel = crocoddyl.CostModelSum(state, actuation.nu)
    terminalCostModel = crocoddyl.CostModelSum(state, actuation.nu)

    runningCostModel.addCost("goalTracking", xRegCost, 1)
    runningCostModel.addCost("controlReg", uRegCost, 0.04)

    terminalCostModel.addCost("goalTracking", xRegTermCost, 1)

    # pushing action model
    T = options.T_trajopt
    runningModels = []
    for i in range(T):
        runningModel = QuasistaticActionModel(
            default_q_sims[i], default_sim_params, state, actuation, runningCostModel, options
        )
        runningModel.u_lb = options.u_lb; runningModel.u_ub = options.u_ub
        runningModels.append(runningModel)

    terminalModel = QuasistaticActionModel(
        default_q_sims[T], default_sim_params, state, actuation, terminalCostModel, options
    )
    terminalModel.u_lb = options.u_lb; terminalModel.u_ub = options.u_ub

    problem = crocoddyl.ShootingProblem(options.x0, runningModels, terminalModel)
    solver = options.ddp_solver(problem)

    # create feasible trajectory
    rollout_start = time.time()
    xs = [options.x0]
    us = [0.0 * np.ones(options.nu)] * T
    for i in range(T):
        data = runningModels[i].createData()
        runningModels[i].calc(data, xs[-1], us[i])
        xs.append(data.xnext)
    print("rollout time: ", time.time() - rollout_start)

    print("xT of init state-control trajectory: ", xs[-1])

    solver.setCallbacks(
        [
            crocoddyl.CallbackVerbose(),
        ]
    )

    # solve optimization
    solver.solve(
        init_xs=xs,
        init_us=us,
        maxiter=50,
        is_feasible=True,
        init_reg=1e0
    )

    # visualization
    if q_vis is not None:
        print("visualizing trajopt result...")
        xs_arr = np.array(solver.xs)
        # q_vis.draw_configuration(x_quat_to_x_rpy(options.x0))
        # q_vis.publish_trajectory(x_quat_to_x_rpy(xs_arr), h=options.h)
        
        xT = xs_arr[-1]
        print("target rpy: ", convert_quat_to_rpy(options.target[0:4]))
        print("reached rpy: ", convert_quat_to_rpy(xT[0:4]))

    xs_arr, us_arr = np.array(solver.xs), np.array(solver.us)
    np.save("../optimizer/warmstart/xs_trival_allegro_rpy.npy", xs_arr)
    np.save("../optimizer/warmstart/us_trival_allegro_rpy.npy", us_arr)
    print("continue?")
    import pdb; pdb.set_trace()

    return xs_arr, us_arr


def visualize_one_step_mpc_in_meshcat(meshcat, x0, options:QSimDDPParams):
    # draw system
    if q_vis is not None:
        q_vis.draw_configuration(x_quat_to_x_rpy(x0))

    # draw object frame
    X_WO = RigidTransform(
        quaternion=Quaternion(x0[0:4]),
        p=options.mount_xyz
    )
    AddMeshcatTriad(
        meshcat=meshcat,
        path="drake/frames/object",
        length=0.1,
        radius=0.004,
        opacity=1.0,
        X_PT=X_WO
    )

    # draw goal frame
    X_WG = RigidTransform(
        quaternion=RollPitchYaw(options.target_rpy).ToQuaternion(),
        p=options.mount_xyz
    )
    AddMeshcatTriad(
        meshcat=meshcat,
        path="drake/frames/goal",
        length=0.1,
        radius=0.004,
        opacity=0.5,
        X_PT=X_WG
    )


def generate_one_step_cost_model(
    options:QSimDDPParams, state, actuation, x_ref, w_u, w_a, is_terminal=False
):
    # cost models
    defaultCostModel = crocoddyl.CostModelSum(state, actuation.nu)

    # residual model (only consider finger pos regulation with ResidualModelState)
    xResidual = crocoddyl.ResidualModelState(state, x_ref, actuation.nu)
    xActivation = crocoddyl.ActivationModelWeightedQuad(
        # np.array([w_u]*(options.nx-options.nu) + [w_a]*options.nu)
        np.array([0.0]*(options.nx-options.nu) + [w_a]*options.nu)
    )
    # cost term
    xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)

    # frame rotation model (consider object tracking here)
    Rref = convert_quat_to_matrix(x_ref[options.xrot_slices])
    objOriResidual = ResidualModelFrameRotation(state, actuation.nu, Rref, options.xrot_slices)
    objOriCost = crocoddyl.CostModelResidual(
        state,
        crocoddyl.ActivationModelWeightedQuad(np.array([w_u]*3)),
        objOriResidual
    )

    # add cost
    defaultCostModel.addCost("goalTracking", xRegCost, 1)                       # 'goalTracking' - 1
    defaultCostModel.addCost("objOriCost", objOriCost, 1)                       # 'objOriCost' - 1
    if not is_terminal:
        uResidual = crocoddyl.ResidualModelControl(state, actuation.nu)
        uRegCost = crocoddyl.CostModelResidual(state, uResidual)
        defaultCostModel.addCost("controlReg", uRegCost, 0.01)                         # 'controlReg' - 1e-1       

    return defaultCostModel


def generate_one_step_mpc_problem(
    options:QSimDDPParams,
    q_sims, sim_params,
    state, actuation,
    x0, x_ref, xs, us,
    maxiter=3, iter=0,
    ref_gen:ReferenceGeneratorRPY=None,
    logger=None,
    viz_helper=None
):
    T = options.T_ctrl

    assert x_ref.shape[0] == T+1
    x_ref_T = x_ref[-1]

    assert xs.shape[0] == T+1 and us.shape[0] == T
    
    runningModels = []
    for i in range(T):
        w_a = 5e-1 if (i > 0.75*T) else 5e-3            # 100
        # w_a = 1e-1
        defaulCostModel = generate_one_step_cost_model(
            options, state, actuation,
            x_ref[i], 5e1, w_a, is_terminal=False
        )
        actionModel = QuasistaticActionModel(
            q_sims[i], sim_params, state, actuation, defaulCostModel, options
        )
        actionModel.u_lb = options.u_lb; actionModel.u_ub = options.u_ub
        runningModels.append(actionModel)

    w_a_T = 5e1                                     # 100
    defaulCostModel = generate_one_step_cost_model(
        options, state, actuation,
        x_ref_T, 5e2, w_a_T, is_terminal=True
    )
    terminalModel = QuasistaticActionModel(
        q_sims[T], sim_params, state, actuation, defaulCostModel, options
    )
    terminalModel.u_lb = options.u_lb; terminalModel.u_ub = options.u_ub

    # shooting problem
    problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)
    solver = options.ddp_solver(problem)

    # make feasible init state-action trajectory
    if options.ddp_verbose:
        print("u_ddp[0]: ", us[0])
    us_ = convert_array_to_list(us.copy())
    xs_ = problem.rollout(us_)

    solver.setCallbacks(
        [
            crocoddyl.CallbackVerbose(),
        ]
    )

    solver.solve(
        init_xs=xs_,
        init_us=us_,
        maxiter=maxiter,
        is_feasible=True,
        init_reg=1e0
    )

    print("||r||: ", np.linalg.norm(problem.runningModels[2].default_cost_data.costs['objOriCost'].r))
    print("||Rx||: ", np.linalg.norm(problem.runningModels[2].default_cost_data.costs['objOriCost'].Rx[:, 0:4], axis=-1))

    xs_arr = np.array(solver.xs)
    us_arr = np.array(solver.us)

    error_to_x0_list, error_to_target_list = [], []
    for i in range(xs_arr.shape[0]):
        error_to_x0_list.append(compute_error_between(xs_arr[i, 0:4], x0[0:4]))
        error_to_target_list.append(compute_error_between(xs_arr[i, 0:4], convert_rpy_to_quat(options.target_rpy)))
    print("error to x0: ", error_to_x0_list)
    print("error to target: ", error_to_target_list)

    if options.enable_exact_rollout:
        problem.runningModels[0].sim_params.log_barrier_weight = options.kappa_exact
        data = runningModels[0].createData()
        runningModels[0].calc(data, xs_arr[0], us_arr[0])
        xs_arr[1] = data.xnext
        problem.runningModels[0].sim_params.log_barrier_weight = options.kappa

    # log data
    if logger is not None:
        # TODO(yongpeng): this needs to be changed to 3d rotation case
        # assert viz_helper is not None
        # contact_sdists = q_sims[0].get_sdists()
        # viz_helper.parse_finger_order(q_sims[0].get_geom_names_Bc())
        # contact_sdists = viz_helper.get_ranked_data_1d(contact_sdists)
        
        logger.log_data(
            index=iter,
            ## for analysis
            ## --------------------
            # ddp_cost=solver.cost,
            # ddp_solved_u0=us_arr[0],
            # ddp_obj_yaw=x0[0],
            # ddp_sdists=contact_sdists
            ddp_error=ref_gen.compute_error(xs_arr[0, 0:4]) if ref_gen is not None else 0.0
            ## --------------------
        )
        
    return xs_arr, us_arr


def solve_ddp_mpc_problem(
    N_steps,
    options: QSimDDPParams,
    q_sims, sim_params,
    state, actuation,
    ref_gen: ReferenceGeneratorRPY,
    logger: DDPLogger=None,
    q_vis: QuasistaticVisualizer=None,
    viz_helper: InhandReferenceGenerator=None
):
    """
        options should contain
            - T_ctrl
            - x0, x_ref
            - x_init, u_init
    """
    T = options.T_ctrl
    x0 = options.x0

    # trival xs, us
    xs0, us0 = options.x_init.copy()[:T+1], options.u_init.copy()[:T]

    xs, us = xs0, us0

    N = N_steps
    x0 = x0.copy()
    x0_last_iter = x0.copy()
    x_traj, u_traj = [x0], []
    for i in range(N-T):
        # TODO(debug)
        if np.dot(x0[0:4], x0_last_iter[0:4]) < 0:
            x0[0:4] = -x0[0:4]
        x0_last_iter = x0.copy()
        
        x_ref_i = np.zeros((T+1, options.nx))
        x_ref_i[:, 0:4] = ref_gen.generate_reference_from_x0(x0[0:4])
        x_ref_i[:, 4:] = options.xa_reg

        # # debug
        # if i > 350:
        #     q_vis.publish_trajectory(
        #         x_quat_to_x_rpy(x_ref_i),
        #         h=options.h
        #     )
        #     import pdb; pdb.set_trace()

        print("------ MPC iter {} ------".format(i))
        print("x0 (wxyz): ", x0[0:4])
        print("xref (wxyz): ", x_ref_i[1, 0:4])
        if logger is not None:
            logger.log_data(
                index=i,
                ddp_obj_x=x0[0:4],
                ddp_ref_x=x_ref_i[1, 0:4]
            )

        xs, us = generate_one_step_mpc_problem(
            options,
            q_sims, sim_params,
            state, actuation,
            x0, x_ref_i, xs, us,
            maxiter=1, iter=i,
            ref_gen=ref_gen,
            logger=logger,
            viz_helper=viz_helper
        )

        if options.visualize_while_control:
            visualize_one_step_mpc_in_meshcat(q_vis.q_sim_py.meshcat, x0, options)

        # rollout 1 step, and shift old xs-us 1 step
        x0, u0 = xs[1], us[0]
        xs = np.append(xs[1:], np.expand_dims(xs[-1], axis=0), axis=0)
        us = np.append(us[1:], np.expand_dims(np.zeros(options.nu,), axis=0), axis=0)

        # # disturbance
        # x0[0:4] += 0.002 * np.random.normal(size=4,)
        # x0[0:4] = x0[0:4] / np.linalg.norm(x0[0:4])

        x_traj.append(x0)

        if viz_helper:
            p_W = default_q_sims[0].get_points_Ac()
            f_W = default_q_sims[0].get_f_Ac()[:, :3]
            viz_helper.parse_finger_order(default_q_sims[0].get_geom_names_Bc())
            viz_helper.plot_contact_points(p_W)
            viz_helper.plot_contact_forces(p_W, f_W)

    if q_vis is not None:
        q_vis.publish_trajectory(
            x_quat_to_x_rpy(x_traj),
            h=options.h
        )

    return x_traj


def get_tracking_objectives(
    options: QSimDDPParams,
    q_sims
):
    """
        Get the tracking objectives for low level controller
        - contact force
        - contact point
    """
    n_c = options.viz_helper.n_c
    options.contact_response.p_ACa_A = np.empty((options.T_ctrl, n_c, 3))
    options.contact_response.f_BA_W = np.empty((options.T_ctrl, n_c, 6))

    for i in range(options.T_ctrl):
        p_ACa_A = q_sims[i].get_points_Ac()
        f_BA_W = q_sims[i].get_f_Ac()
        geom_names = q_sims[i].get_geom_names_Bc()
        options.viz_helper.parse_finger_order(geom_names)

        p_ACa_A = options.viz_helper.get_ranked_contact_data(p_ACa_A)
        f_BA_W = options.viz_helper.get_ranked_contact_data(f_BA_W)

        options.contact_response.p_ACa_A[i] = p_ACa_A
        options.contact_response.f_BA_W[i] = f_BA_W    


def run_ddp_trajopt(
    options: QSimDDPParams,
    q_sims, sim_params,
    state, actuation,
    maxiter=50
):
    """
        NOTE: x should have [xa; xu] order
        Run one step MPC
        Problem data:
            - x0
            - x_ref
            - xs_init
            - us_init
        Parameters:
            - T
            - max_iter
        Return:
            - xs
            - us
    """
    T = options.T_trajopt
    x0 = options.x0
    x_target = options.target

    # residual for state and cost
    xResidual = crocoddyl.ResidualModelState(state, x_target, actuation.nu)
    xActivation = crocoddyl.ActivationModelWeightedQuad(
        np.array([options.TO_w_u] * (options.nx - options.nu) + [options.TO_w_a]*options.nu)
    )
    xTActivation = crocoddyl.ActivationModelWeightedQuad(
        np.array([options.TO_w_u_T] * (options.nx - options.nu) + [options.TO_w_a_T]*options.nu)
    )
    uResidual = crocoddyl.ResidualModelControl(state, actuation.nu)

    # cost term
    xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
    uRegCost = crocoddyl.CostModelResidual(state, uResidual)
    xRegTermCost = crocoddyl.CostModelResidual(state, xTActivation, xResidual)

    # cost model
    runningCostModel = crocoddyl.CostModelSum(state, actuation.nu)
    terminalCostModel = crocoddyl.CostModelSum(state, actuation.nu)

    runningCostModel.addCost("goalTracking", xRegCost, options.w_xreg)
    runningCostModel.addCost("controlReg", uRegCost, options.w_ureg)

    terminalCostModel.addCost("goalTracking", xRegTermCost, options.w_xreg)

    # pushing action model
    runningModels = []
    for i in range(T):
        runningModel = QuasistaticActionModel(
            q_sims[i], sim_params, state, actuation, runningCostModel, options
        )
        runningModel.u_lb = options.u_lb; runningModel.u_ub = options.u_ub
        runningModels.append(runningModel)

    terminalModel = QuasistaticActionModel(
        q_sims[T], sim_params, state, actuation, terminalCostModel, options
    )
    terminalModel.u_lb = options.u_lb; terminalModel.u_ub = options.u_ub

    problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)
    solver = options.ddp_solver(problem)

    # create feasible trajectory
    xs = [x0]
    us = [0.0 * np.ones(options.nu)] * T
    for i in range(T):
        data = runningModels[i].createData()
        runningModels[i].calc(data, xs[-1], us[i])
        xs.append(data.xnext)

    solver.setCallbacks(
        [
            crocoddyl.CallbackVerbose(),
        ]
    )

    # solve optimization
    solver.solve(
        init_xs=xs,
        init_us=us,
        maxiter=maxiter,
        is_feasible=True,
        init_reg=1e0
    )

    xs_arr, us_arr = np.array(solver.xs), np.array(solver.us)

    # return x with quaternion
    xs_arr = np.roll(xs_arr, -4, axis=1)

    return xs_arr, us_arr


def run_ddp_mpc(
    options: QSimDDPParams,
    q_sims, sim_params,
    state, actuation,
    maxiter=1,
    logger: DDPLogger=None,
    q_vis: QuasistaticVisualizer=None
):
    """
        NOTE: x should have [xa; xu] order
        Run one step MPC
        Problem data:
            - x0
            - x_ref
            - xs_init
            - us_init
        Parameters:
            - T
            - max_iter
        Return:
            - xs
            - us
    """
    T = options.T_ctrl
    x0 = options.x_obs
    x_ref = options.x_ref

    # trival xs, us
    xs0, us0 = options.x_init.copy()[:T+1], options.u_init.copy()[:T]

    # reverse xa, xu order
    # x0 = np.roll(x0, 4)                   # x0 is already in [xu, xa] order
    # x_ref = np.roll(x_ref, 4, axis=1)     # x_ref is already in [xu, xa] order
    xs0 = np.roll(xs0, 4, axis=1)

    # solve trajectory optimization first
    xs, us = generate_one_step_mpc_problem(
        options,
        q_sims, sim_params,
        state, actuation,
        x0, x_ref, xs0, us0,
        maxiter=maxiter, iter=-1, ref_gen=None, logger=None
    )
    get_tracking_objectives(options, q_sims)

    xs = np.array(xs)
    us = np.array(us)

    xs = np.roll(xs, -4, axis=1)

    return xs, us


def plot_logger_data(logger: DDPLogger):
    _, ddp_error = logger.get_data_in_array("ddp_error")
    _, ddp_obj_x = logger.get_data_in_array("ddp_obj_x")
    _, ddp_ref_x = logger.get_data_in_array("ddp_ref_x")
    quat_lut = logger.get_data("quat_lut")
    t_lut = np.linspace(0, len(ddp_error)-1, len(quat_lut))

    plt.figure()
    plt.plot(ddp_error)
    plt.grid("on")

    plt.figure()
    for i in range(4):
        plt.subplot(4, 1, i+1)
        plt.plot(ddp_obj_x[:, i], linestyle='-', label="x")
        plt.plot(ddp_ref_x[:, i], linestyle='--', label='ref')
        plt.plot(t_lut, quat_lut[:, i], linestyle='-.', label='lut')
        plt.grid("on"); plt.legend()

    plt.show()


def save_logger_data(logger: DDPLogger, path, plot=False):
    _, ddp_cost = logger.get_data_in_array("ddp_cost")
    _, ddp_solved_u0 = logger.get_data_in_array("ddp_solved_u0")
    _, ddp_obj_yaw = logger.get_data_in_array("ddp_obj_yaw")
    _, ddp_sdists = logger.get_data_in_array("ddp_sdists")
    
    running_logs = {
        "ddp_cost": ddp_cost,
        "ddp_solved_u0": ddp_solved_u0,
        "ddp_obj_yaw": ddp_obj_yaw,
        "ddp_sdists": ddp_sdists
    }

    if plot:
        plt.figure()
        for i in range(4):
            plt.subplot(4, 1, i+1)
            plt.plot(ddp_sdists[:, i])
            plt.grid("on"); plt.legend()
        plt.show()

    with open(path, "wb") as f:
        pickle.dump(running_logs, f)


if __name__ == "__main__":
    params = QSimDDPParams()

    params.enable_exact_rollout = True
    default_q_sims, default_sim_params, state, actuation = \
        allocate_resources_for_DDP(params, num_envs=41)

    q_parser = QuasistaticParser(QSIM_MODEL_PATH)
    q_vis = QuasistaticVisualizer.make_visualizer(q_parser)

    q_vis.draw_configuration(x_quat_to_x_rpy(params.x0))
    meshcat = q_vis.q_sim_py.meshcat

    # set visualization
    viz_helper = InhandReferenceGenerator(meshcat)
    viz_helper.set_elements_in_meshcat()
    viz_helper.parse_external_contacts([])

    # set random target
    params.target_rpy = np.array([0.6, -0.6, -0.6])
    print("target rpy: ", params.target_rpy)

    ## test MPC control
    ## --------------------
    # solve trajopt
    ref_gen = ReferenceGeneratorRPY()
    ref_gen.set_start_and_target_rpy(
        convert_quat_to_rpy(params.x0[0:4]),
        params.target_rpy
    )
    quat_target_trajopt = ref_gen.generate_reference_from_x0(params.x0[0:4])[-1]
    params.target[0:4] = quat_target_trajopt

    # xs_init, us_init = solve_ddp_trajopt_problem(
    #     default_q_sims, default_sim_params,
    #     state, actuation, params,
    #     q_vis=q_vis
    # )
    xs_init, us_init = np.load("../optimizer/warmstart/xs_trival_allegro_rpy.npy"), \
                        np.load("../optimizer/warmstart/us_trival_allegro_rpy.npy")
    
    # solve MPC
    logger = DDPLogger(20, 16, 0, 0)

    params.x_init = xs_init
    params.u_init = us_init

    N_steps = 200

    solve_ddp_mpc_problem(
        N_steps,
        params,
        default_q_sims, default_sim_params,
        state, actuation,
        ref_gen,
        logger=logger,
        q_vis=q_vis,
        viz_helper=viz_helper
    )

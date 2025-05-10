############################################################
# This script is modified from inhand_ddp_full_hand.py,    #
# but demonstrates a simplified case. The manipuland could #
# only rotates w.r.t. the z-axis.                          #
# This version (v2) is modified from                       #
# inhand_ddp_full_hand_zrot.py                             #
# This script provides a interface for any object,         #
# that is specified in the yml file                        #
############################################################

import os
import sys


import copy
import time
import pickle
import numpy as np
import yaml
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



from pydrake.all import (
    JointIndex,
    PiecewisePolynomial,
    Quaternion,
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
from common.inhand_ddp_helper import InhandDDPVizHelper, ContactTrackReference
from common.ddp_logger import *
from common.get_model_path import *


# get model path from file name
def get_model_path_from_file_name(file_name):
    qsim_model_path = os.path.join(YML_PATH, file_name+".yml")
    model_param_path = os.path.join(CONFIG_PATH, file_name+".yaml")
    return qsim_model_path, model_param_path

# convert 2d array to list of 1d array
def convert_array_to_list(array_2d):
    return [array_2d[i, :] for i in range(array_2d.shape[0])]

# convert Drake configurations to Pinocchio configurations
# Drake: [hand joints; sphere pos, sphere quat{w, x, y, z}]
# Pinocchio: [hand joints; sphere pos, sphere quat{x, y, z, w}]
def convert_drake_to_pinocchio_configurations(x):
    x = x.copy()
    x = np.concatenate((x[:16], x[-3:], x[-6:-3], np.atleast_1d(x[-7])))
    return x

def convert_pinocchio_to_drake_configurations(x):
    x = x.copy()
    x = np.concatenate((x[:16], np.atleast_1d(x[-1]), x[-4:-1], x[-7:-4]))
    return x

def convert_drake_to_pinocchio_grad(DX):
    DX = DX.copy()
    DX = np.concatenate((DX[:, :16], DX[:, -3:], DX[:, -6:-3], DX[:, -7].reshape(-1, 1)), axis=1)
    return DX

def create_se3_from_posquat(x):
    quat = x[[1, 2, 3, 0]]
    se3 = pin.SE3(
        R.from_quat(quat).as_matrix(),
        x[4:]
    )
    return se3

def get_omega_to_qdot_map(q):
    mat_w2dq = np.array([[-q[1], -q[2], -q[3]],
                         [q[0], q[3], -q[2]],
                         [-q[3], q[0], q[1]],
                         [q[2], -q[1], q[0]]]) * 0.5
    return mat_w2dq

def convert_rpy_to_quat(rpy):
    quat = RollPitchYaw(rpy).ToQuaternion().wxyz()
    return quat

def convert_quat_to_rpy(quat):
    rpy = RotationMatrix(Quaternion(quat)).ToRollPitchYaw()
    return rpy

def normalize_quaternions(x, options):
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    q_u_quat_indices = options.q_u_indices[:4]
    x[:, q_u_quat_indices] = \
        x[:, q_u_quat_indices] / np.linalg.norm(x[:, q_u_quat_indices], axis=1).reshape(-1, 1)

class QSimDDPParams(object):
    def __init__(self):
        # model
        self.model_name = "allegro_3d_full_zrot_valve"
        self.mount_xyz = [-0.08, 0.0, 0.11]

        self.h = 0.1            # 0.025
        self.kappa = 100        # smoothing
        self.kappa_exact = 10000    # smoothing for exact rollout
        self.nx = 17
        self.nu = 16
        # self.q_u_indices = [16]
        self.q_u_indices = [0]
        self.q_a_indices = [1, 2, 3, 4, 5, 6, 7, 8,
                            9, 10, 11, 12, 13, 14, 15, 16]
        self.normalize_quaternions = True
        self.enable_exact_rollout = False

        self.T_trajopt = 40
        self.T_ctrl = 10            # 20
        self.T_xa_reg_interp = 100  # period of xa interpolation
        
        self.x0 = np.array([
            0.0,     # ball
            0.2, 0.95, 1.0, 1.0,                 # index finger
            0.0, 0.6, 1.0, 1.0,                 # middle finger
            -0.2, 0.95, 1.0, 1.0,                # ring finger
            0.6, 1.95, 1.0, 1.0
        ])
        self.xa_reg = np.array([
            0.1, 1.0, 1.0, 1.0,                 # index finger
            0.0, 0.7, 1.0, 1.0,                 # middle finger
            -0.1, 1.0, 1.0, 1.0,                # ring finger
            0.6, 1.9, 1.0, 1.0                 # thumb
        ])
        self.xa_reg2 = np.array([
            0.1, 1.0, 1.0, 1.0,                 # index finger
            0.0, 0.7, 1.0, 1.0,                 # middle finger
            -0.1, 1.0, 1.0, 1.0,                # ring finger
            0.6, 1.9, 1.0, 1.0                 # thumb
        ])
        # observation
        self.x_obs = self.x0.copy()
        # reference (for trajopt)
        self.target = \
            np.insert(
            self.xa_reg,
            0,
            np.pi/6
        )
        # reference (for MPC)
        self.x_ref = None
        # warm start (for MPC)
        self.x_init = None
        self.u_init = None
        self.dyaw = 5.0 * np.pi/200

        self.u_lb = -0.08 * np.ones(self.nu)             # -0.05
        self.u_ub = 0.08 * np.ones(self.nu)              # 0.05
        self.execution_scale = 1.0                       # 1.0, positive scalar, scale u0 when execution (i.e., rollout new x0)

        self.ddp_solver = crocoddyl.SolverBoxDDP         # SolverBoxDDP, SolverDDP
        self.ddp_verbose = False

        self.resolve_traj_opt = False
        self.visualize_while_control = True

        # viz helper
        self.viz_helper = InhandDDPVizHelper()
        self.contact_request = ContactTrackReferenceRequest()
        self.contact_response = ContactTrackReferenceResponse()

        # trajopt params
        self.TO_w_u = 1e1
        self.TO_w_a = 1e-2
        self.TO_w_u_T = 1e2
        self.TO_w_a_T = 1e0

        # mpc control params
        self.w_u = 1e0
        self.w_a = (1e-2, 1e-1)          # (1e-2, 1e0)
        self.w_u_T = 1e2
        self.w_a_T = 5e2                # 1e3
        
        self.w_xreg = 1
        self.w_ureg = 0.04


## ----------------------------------------


def allocate_resources_for_DDP(options:QSimDDPParams, num_envs=21):
    parser = QuasistaticParser(
        get_model_path_from_file_name(options.model_name)[0]
    )

    # qsims
    default_q_sims = []
    for i in range(num_envs):
        q_sim = parser.make_simulator_cpp()
        default_q_sims.append(q_sim)

    # qa and qu indices
    options.q_a_indices = default_q_sims[0].get_q_a_indices_into_q()
    options.q_u_indices = default_q_sims[0].get_q_u_indices_into_q()

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
    def __init__(self, q_sim, sim_params, state, actuation, defaultCostModel):
        crocoddyl.ActionModelAbstract.__init__(self, state, actuation.nu, defaultCostModel.nr)
        
        self.default_cost = defaultCostModel
        self.default_cost_data = self.default_cost.createData(crocoddyl.DataCollectorAbstract())

        self.q_sim = q_sim
        self.sim_params = sim_params

        self.nx_ = len(self.q_sim.get_mbp_positions_as_vec())
        self.nu_ = q_sim.num_actuated_dofs()
        self.q_a_indices_ = q_sim.get_q_a_indices_into_q()
        self.q_u_indices_ = q_sim.get_q_u_indices_into_q()

        self.time_calc_ = 0.0
        self.time_calc_diff_ = 0.0

    def calc(self, data, x, u=None):
        start_time = time.time()

        if u is None:
            u = np.zeros(self.nu_)
        u_incre = u.copy()

        u = u + x[self.q_a_indices_]

        # call q_sim's forward dynamics
        data.xnext = self.q_sim.calc_dynamics_forward(x, u, self.sim_params)

        # compute default cost
        self.default_cost.calc(self.default_cost_data, x, u_incre)
        default_cost_value = sum([c.cost for c in self.default_cost_data.costs.todict().values()])

        data.cost = default_cost_value

        self.time_calc_ += time.time() - start_time

    def calcDiff(self, data, x, u=None):
        start_time = time.time()

        if u is None:
            u = np.zeros(self.nu_)
        u_incre = u.copy()

        # call q_sim's backward dynamics
        self.q_sim.calc_dynamics_backward(self.sim_params)

        partial_u_partial_q = np.zeros((self.nu_, self.nx_))
        partial_u_partial_q[:, self.q_a_indices_] = np.eye(self.nu_)

        data.Fx = self.q_sim.get_Dq_nextDq() + self.q_sim.get_Dq_nextDqa_cmd() @ partial_u_partial_q
        data.Fu = self.q_sim.get_Dq_nextDqa_cmd()

        self.default_cost.calcDiff(self.default_cost_data, x, u_incre)

        data.Lx = self.default_cost_data.Lx
        data.Lu = self.default_cost_data.Lu
        data.Lxx = self.default_cost_data.Lxx
        data.Lxu = self.default_cost_data.Lxu
        data.Luu = self.default_cost_data.Luu

        self.time_calc_diff_ += time.time() - start_time


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
    # xResidual = crocoddyl.ResidualModelState(state, options.target, actuation.nu)
    # xActivation = crocoddyl.ActivationModelWeightedQuad(
    #     np.array([1e1] * (options.nx - options.nu) + [1e-2]*options.nu)
    # )
    # xTActivation = crocoddyl.ActivationModelWeightedQuad(
    #     np.array([1e2] * (options.nx - options.nu) + [1e0]*options.nu)
    # )

    # indices
    q_a_indices_ = options.q_a_indices
    q_u_indices_ = options.q_u_indices

    # assign weights
    weights_linear = np.zeros(options.nx)
    weights_linear_T = np.zeros(options.nx)
    weights_linear[q_a_indices_] = options.TO_w_a; weights_linear[q_u_indices_] = options.TO_w_u
    weights_linear_T[q_a_indices_] = options.TO_w_a_T; weights_linear_T[q_u_indices_] = options.TO_w_u_T

    # residual for state and cost
    xResidual = crocoddyl.ResidualModelState(state, options.target, actuation.nu)
    xActivation = crocoddyl.ActivationModelWeightedQuad(weights_linear)
    xTActivation = crocoddyl.ActivationModelWeightedQuad(weights_linear_T)
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
    T = options.T_trajopt
    runningModels = []
    for i in range(T):
        runningModel = QuasistaticActionModel(
            default_q_sims[i], default_sim_params, state, actuation, runningCostModel
        )
        runningModel.u_lb = options.u_lb; runningModel.u_ub = options.u_ub
        runningModels.append(runningModel)

    terminalModel = QuasistaticActionModel(
        default_q_sims[T], default_sim_params, state, actuation, terminalCostModel
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
    import pdb; pdb.set_trace()

    # visualization
    if q_vis is not None:
        q_vis.draw_configuration(options.x0)
        q_vis.publish_trajectory(solver.xs, h=options.h)
    np.save("../optimizer/warmstart/xs_trival_valve.npy", solver.xs)
    np.save("../optimizer/warmstart/us_trival_valve.npy", solver.us)
    import pdb; pdb.set_trace()


def visualize_one_step_mpc_in_meshcat(x0, x_ref_T, x_T, q_vis=None):
    if q_vis is not None:
        q_vis.draw_configuration(x0)


def generate_one_step_cost_model(
    options:QSimDDPParams, state, actuation, x_ref, w_u, w_a, is_terminal=False
):
    # indices
    q_a_indices_ = options.q_a_indices
    q_u_indices_ = options.q_u_indices

    # weights
    weights_linear = np.zeros(options.nx)
    weights_linear[q_a_indices_] = w_a
    weights_linear[q_u_indices_] = w_u

    # cost models
    defaultCostModel = crocoddyl.CostModelSum(state, actuation.nu)

    # residual model
    xResidual = crocoddyl.ResidualModelState(state, x_ref, actuation.nu)
    xActivation = crocoddyl.ActivationModelWeightedQuad(
        np.array([w_u]*(options.nx-options.nu) + [w_a]*options.nu)
    )

    # cost term
    xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)

    # add cost
    defaultCostModel.addCost("goalTracking", xRegCost, options.w_xreg)                       # 'goalTracking' - 1
    if not is_terminal:
        uResidual = crocoddyl.ResidualModelControl(state, actuation.nu)
        uRegCost = crocoddyl.CostModelResidual(state, uResidual)
        defaultCostModel.addCost("controlReg", uRegCost, options.w_ureg)                         # 'controlReg' - 1e-1       

    return defaultCostModel


def generate_one_step_mpc_problem(
    options:QSimDDPParams,
    q_sims, sim_params,
    state, actuation,
    x0, x_ref, xs, us,
    maxiter=3, iter=0, logger=None,
    viz_helper=None
):
    def recalc_warm_start(xs, us, time_step=0.1, offset=0):
        num_steps = len(us)
        time_steps = np.arange(0, time_step*(num_steps+1), time_step)

        q_knots = xs.copy()
        u_knots = us.copy()
        u_knots = np.concatenate((u_knots, u_knots[-1].reshape(1, -1)), axis=0)

        q_traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            time_steps,
            q_knots.T
        )

        u_traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            time_steps,
            u_knots.T
        )

        xs_new = np.zeros_like(xs)
        us_new = np.zeros_like(us)

        start_time = offset
        for i in range(num_steps+1):
            t = start_time + i * time_step
            xs_new[i] = q_traj.value(t).flatten()
            if i < num_steps:
                us_new[i] = u_traj.value(t).flatten()

        us_new[-1] = np.zeros_like(us_new[-1])

        return xs_new, us_new

    T = options.T_ctrl

    assert x_ref.shape[0] == T+1
    x_ref_T = x_ref[-1]

    assert xs.shape[0] == T+1 and us.shape[0] == T
    
    runningModels = []
    for i in range(T):
        # w_a = 1e0 if (i > 0.75*T) else 1e-2            # (1e-2, 1e0)
        # w_a = 1e-1
        w_a = options.w_a[1] if (i > 0.75*T) else options.w_a[0]
        defaulCostModel = generate_one_step_cost_model(
            options, state, actuation,
            # x_ref[i], 1e1, w_a, is_terminal=False
            x_ref[i], options.w_u, w_a, is_terminal=False
        )
        actionModel = QuasistaticActionModel(
            q_sims[i], sim_params, state, actuation, defaulCostModel
        )
        actionModel.u_lb = options.u_lb; actionModel.u_ub = options.u_ub
        runningModels.append(actionModel)

    # w_a_T = 1e3                                     # 1e3
    defaulCostModel = generate_one_step_cost_model(
        options, state, actuation,
        # x_ref_T, 1e2, w_a_T, is_terminal=True
        x_ref_T, options.w_u_T, options.w_a_T, is_terminal=True
    )
    terminalModel = QuasistaticActionModel(
        q_sims[T], sim_params, state, actuation, defaulCostModel
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

    xs_arr = np.array(solver.xs)
    us_arr = np.array(solver.us)

    if options.enable_exact_rollout:
        problem.runningModels[0].sim_params.log_barrier_weight = options.kappa_exact
        data = runningModels[0].createData()
        runningModels[0].calc(data, xs_arr[0], us_arr[0])
        xs_arr[1] = data.xnext
        problem.runningModels[0].sim_params.log_barrier_weight = options.kappa

    # log data
    if logger is not None:
        assert viz_helper is not None
        contact_sdists = q_sims[0].get_sdists()
        viz_helper.parse_finger_order(q_sims[0].get_geom_names_Bc())
        contact_sdists = viz_helper.get_ranked_data_1d(contact_sdists)
        
        logger.log_data(
            index=iter,
            ## for analysis
            ## --------------------
            ddp_cost=solver.cost,
            ddp_solved_u0=us_arr[0],
            ddp_obj_yaw=x0[0],
            ddp_sdists=contact_sdists
            ## --------------------
        )
        
    return xs_arr, us_arr


def solve_ddp_mpc_problem(
    options: QSimDDPParams,
    q_sims, sim_params,
    state, actuation,
    logger: DDPLogger=None,
    q_vis: QuasistaticVisualizer=None,
    viz_helper: InhandDDPVizHelper=None
):
    """
        options should contain
            - T_ctrl
            - x0, x_ref
            - x_init, u_init
    """
    T = options.T_ctrl
    x0 = options.x0
    x_ref = options.x_ref

    # trival xs, us
    xs0, us0 = options.x_init.copy()[:T+1], options.u_init.copy()[:T]

    xs, us = xs0, us0

    N = x_ref.shape[0]
    x0 = x0.copy()
    x_traj, u_traj = [x0], []
    for i in range(N-T):
        # x_ref_i = x_ref[i:i+T+1]
        x_ref_i = generate_reference_from_x0(i, x0, options)

        print("------ MPC iter {} ------".format(i))
        print("q0 (rpy): ", np.array([0.0, 0.0, x0[0]]))
        print("q_ref (rpy): ", np.array([0.0, 0.0, x_ref_i[0][0]]))

        xs, us = generate_one_step_mpc_problem(
            options,
            q_sims, sim_params,
            state, actuation,
            x0, x_ref_i, xs, us,
            maxiter=1, iter=i, logger=logger,
            viz_helper=viz_helper
        )

        if options.visualize_while_control:
            visualize_one_step_mpc_in_meshcat(
                x0=x0, x_ref_T=x_ref_i[-1], x_T=xs[-1], q_vis=q_vis
            )

        # rollout 1 step, and shift old xs-us 1 step
        x0, u0 = xs[1], us[0]
        xs = np.append(xs[1:], np.expand_dims(xs[-1], axis=0), axis=0)
        us = np.append(us[1:], np.expand_dims(np.zeros(options.nu,), axis=0), axis=0)

        x_traj.append(x0)

        if viz_helper:
            p_W = default_q_sims[0].get_points_Ac()
            f_W = default_q_sims[0].get_f_Ac()[:, :3]
            viz_helper.parse_finger_order(default_q_sims[0].get_geom_names_Bc())
            viz_helper.plot_contact_points(p_W)
            viz_helper.plot_contact_forces(p_W, f_W)

    if q_vis is not None:
        q_vis.publish_trajectory(x_traj, h=options.h)

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
    x0 = np.roll(x0, 1)
    x_ref = np.roll(x_ref, 1, axis=1)
    xs0 = np.roll(xs0, 1, axis=1)

    # solve trajectory optimization first
    xs, us = generate_one_step_mpc_problem(
        options,
        q_sims, sim_params,
        state, actuation,
        x0, x_ref, xs0, us0,
        maxiter=maxiter, iter=-1, logger=None
    )
    get_tracking_objectives(options, q_sims)

    xs = np.array(xs)
    us = np.array(us)

    xs = np.roll(xs, -1, axis=1)

    return xs, us


def generate_sine_interp_between_end_points(i, x1, x2, T):
    """
        Generate the sine interpolation between two end points, x1 and x2
    """
    phase_diff = np.repeat([0.0, np.pi/2, np.pi, 3*np.pi/2], 4)
    x_i = (x1 + x2) / 2 + (x2 - x1) / 2 * np.sin(2 * np.pi * i / T + phase_diff)
    return x_i

def generate_reference_from_x0(iter, x0, params:QSimDDPParams):
    yaw0 = x0[params.q_u_indices[0]]
    yaw_ref = yaw0 + np.arange(0, params.T_ctrl+2) * params.dyaw

    x_ref = np.tile(params.target, (params.T_ctrl+1, 1))
    x_ref[:, params.q_u_indices[0]] = yaw_ref[1:]
    x_ref[:, params.q_a_indices] = generate_sine_interp_between_end_points(
        i=iter,
        x1=params.xa_reg,
        x2=params.xa_reg2,
        T=params.T_xa_reg_interp
    )

    return x_ref


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

def AddJointSlidersToMeshcat(meshcat, plant):
    sliders = {}
    slider_num = 0
    positions = []
    lower_limit, upper_limit = -10.0 * np.ones(plant.num_joints()), 10.0 * np.ones(plant.num_joints())
    resolution = 0.01 * np.ones(plant.num_joints())
    for i in range(plant.num_joints()):
        joint = plant.get_joint(JointIndex(i))
        low = joint.position_lower_limits()
        upp = joint.position_upper_limits()
        for j in range(joint.num_positions()):
            index = joint.position_start() + j
            description = joint.name()
            if joint.num_positions() > 1:
                description += "_" + joint.position_suffix(j)
            lower_limit[slider_num] = max(low[j], lower_limit[slider_num])
            upper_limit[slider_num] = min(upp[j], upper_limit[slider_num])
            value = (lower_limit[slider_num] + upper_limit[slider_num]) / 2
            positions.append(value)
            meshcat.AddSlider(
                value=value,
                min=lower_limit[slider_num],
                max=upper_limit[slider_num],
                step=resolution[slider_num],
                name=description,
            )
            sliders[index] = description
            slider_num += 1


if __name__ == "__main__":
    params = QSimDDPParams()

    # set model dependent params
    params.model_name = "allegro_3d_full_zrot_valve"
    params.mount_xyz = [-0.08, 0.0, 0.11]
    params.enable_exact_rollout = False

    default_q_sims, default_sim_params, state, actuation = \
        allocate_resources_for_DDP(params, num_envs=41)

    q_parser = QuasistaticParser(
        get_model_path_from_file_name(params.model_name)[0]
    )

    # load model params
    with open(get_model_path_from_file_name(params.model_name)[1], "r") as f:
        model_params = yaml.safe_load(f)
        # x0
        params.x0[1:5] = model_params["x0"]["index"]
        params.x0[5:9] = model_params["x0"]["middle"]
        params.x0[9:13] = model_params["x0"]["ring"]
        params.x0[13:17] = model_params["x0"]["thumb"]
        # xa_reg
        params.xa_reg[0:4] = model_params["xa_reg"]["index"]
        params.xa_reg[4:8] = model_params["xa_reg"]["middle"]
        params.xa_reg[8:12] = model_params["xa_reg"]["ring"]
        params.xa_reg[12:16] = model_params["xa_reg"]["thumb"]
        # xa_reg2
        params.xa_reg2[0:4] = model_params["xa_reg2"]["index"]
        params.xa_reg2[4:8] = model_params["xa_reg2"]["middle"]
        params.xa_reg2[8:12] = model_params["xa_reg2"]["ring"]
        params.xa_reg2[12:16] = model_params["xa_reg2"]["thumb"]
        # target
        params.target = np.insert(params.xa_reg, 0, np.pi/6)
        params.dyaw = model_params["dyaw"]
        params.u_lb = model_params["u_lb"] * np.ones_like(params.u_lb)
        params.u_ub = model_params["u_ub"] * np.ones_like(params.u_ub)

    q_vis = QuasistaticVisualizer.make_visualizer(q_parser)
    q_vis.draw_configuration(params.x0)
    meshcat = q_vis.q_sim_py.meshcat

    viz_helper = InhandDDPVizHelper(meshcat)
    viz_helper.set_elements_in_meshcat()

    # solve_ddp_trajopt_problem(
    #     default_q_sims, default_sim_params,
    #     state, actuation, params,
    #     q_vis=q_vis
    # )
    # exit(0)

    xs_init, us_init = np.load("../optimizer/warmstart/xs_trival_valve.npy"), \
                        np.load("../optimizer/warmstart/us_trival_valve.npy")
    
    # T = 10
    # x0 = x0

    logger = DDPLogger(17, 16, 0, 0)

    dtheta = 1.25 * np.pi/200
    q_ref = np.zeros((410, params.nx))
    q_ref[:, 1:] = params.x0[1:]
    for i in range(q_ref.shape[0]):
        q_ref[i, 0] = i*dtheta

    params.x_ref = q_ref
    params.x_init = xs_init
    params.u_init = us_init

    solve_ddp_mpc_problem(
        params,
        default_q_sims, default_sim_params,
        state, actuation,
        logger=logger,
        q_vis=q_vis,
        viz_helper=viz_helper
    )

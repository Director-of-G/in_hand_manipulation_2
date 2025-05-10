############################################################
# This script is modified from inhand_ddp_full_hand.py,    #
# but demonstrates a simplified case. The manipuland could #
# only rotates w.r.t. the z-axis.                          #
############################################################

import os
import sys


import copy
import time
import numpy as np
import pickle
import yaml
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
from qsim_cpp import GradientMode, ForwardDynamicsMode, DfDxMode



from pydrake.all import (
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
from ddp.ddp_solver import solve_ddp_trajopt_problem as solve_ddp_trajopt_unified
from ddp.ddp_solver import ddp_runner as ddp_runner_unified
from common.inhand_ddp_helper import InhandDDPVizHelper, ContactTrackReference
from common.ddp_logger import *
from common.get_model_path import *


X_AXIS_LENGTH=0.08
QSIM_MODEL_PATH = os.path.join(YML_PATH, "leap_3d_open_box.yml")
MODEL_PARAM_PATH = os.path.join(CONFIG_PATH, "leap_3d_open_box.yaml")


# convert 2d array to list of 1d array
def convert_array_to_list(array_2d):
    return [array_2d[i, :] for i in range(array_2d.shape[0])]

class QSimDDPParams(object):
    def __init__(self):
        # model
        self.finger_link_names = []
        self.object_link_name = ""

        self.h = 0.1            # 0.025
        self.kappa = 100        # smoothing
        self.kappa_exact = 10000    # smoothing for exact rollout

        self.nx = 17
        self.nu = 16
        self.q_u_indices = [0]

        self.T_trajopt = 40
        self.T_ctrl = 10            # 20
        
        self.x0 = np.zeros(self.nx)
        self.xa_reg = np.zeros(self.nu)
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

        self.u_lb = -0.12 * np.ones(self.nu)             # -0.05
        self.u_ub = 0.12 * np.ones(self.nu)              # 0.05
        self.execution_scale = 0.5                       # 1.0, positive scalar, scale u0 when execution (i.e., rollout new x0)

        self.ddp_solver = crocoddyl.SolverBoxDDP         # SolverBoxDDP, SolverDDP
        self.ddp_verbose = False

        self.resolve_traj_opt = False
        self.visualize_while_control = True

        self.enable_exact_rollout = False               # use exact rollout for xs, not for solving us

        # viz helper
        self.viz_helper = InhandDDPVizHelper()
        self.contact_request = ContactTrackReferenceRequest()
        self.contact_response = ContactTrackReferenceResponse()


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
    # default_sim_params.gradient_dfdx_mode = DfDxMode.kAutoDiff
    default_sim_params.gradient_dfdx_mode = DfDxMode.kAnalyticWithFiniteDiff
    
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

        self.nx_ = len(self.q_sim.get_mbp_positions_as_vec())
        self.nu_ = q_sim.num_actuated_dofs()
        self.num_wrist_joints = options.num_wrist_joints
        self.q_a_indices_ = q_sim.get_q_a_indices_into_q()
        self.q_u_indices_ = q_sim.get_q_u_indices_into_q()

        self.q_wrist = np.zeros(self.num_wrist_joints,)

        self.total_runs_calc_ = 0
        self.total_runs_calc_diff_ = 0
        self.time_calc_ = 0.0
        self.time_calc_diff_ = 0.0

    def set_wrist_pose(self, q_wrist):
        assert q_wrist is not None
        if not isinstance(q_wrist, np.ndarray):
            q_wrist = np.atleast_1d(q_wrist)
        assert q_wrist.shape[0] == self.num_wrist_joints
        self.q_wrist = q_wrist.copy()

    def calc(self, data, x, u=None):
        start_time = time.time()

        if u is None:
            u = np.zeros(self.nu_)
        u_incre = u.copy()

        u = u + x[self.q_a_indices_]
        # FIXME: manually set the wrist dofs
        u[:self.num_wrist_joints] = self.q_wrist.copy()

        # call q_sim's forward dynamics
        data.xnext = self.q_sim.calc_dynamics_forward(x, u, self.sim_params)

        # compute default cost
        self.default_cost.calc(self.default_cost_data, x, u_incre)
        default_cost_value = sum([c.cost for c in self.default_cost_data.costs.todict().values()])

        data.cost = default_cost_value

        self.time_calc_ += time.time() - start_time
        self.total_runs_calc_ += 1

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

        # FIXME: manually disable gradients for wrist dofs
        Dq_nextDq = self.q_sim.get_Dq_nextDq().copy()
        Dq_nextDqa_cmd = self.q_sim.get_Dq_nextDqa_cmd().copy()
        Dq_nextDq[1:1+self.num_wrist_joints, :] = 0
        Dq_nextDq[:, 1:1+self.num_wrist_joints] = 0
        Dq_nextDqa_cmd[1:1+self.num_wrist_joints, :] = 0
        Dq_nextDqa_cmd[:, 0:self.num_wrist_joints] = 0
        data.Fx = Dq_nextDq + Dq_nextDqa_cmd @ partial_u_partial_q
        data.Fu = Dq_nextDqa_cmd

        self.default_cost.calcDiff(self.default_cost_data, x, u_incre)

        data.Lx = self.default_cost_data.Lx
        data.Lu = self.default_cost_data.Lu
        data.Lxx = self.default_cost_data.Lxx
        data.Lxu = self.default_cost_data.Lxu
        data.Luu = self.default_cost_data.Luu

        self.time_calc_diff_ += time.time() - start_time
        self.total_runs_calc_diff_ += 1


# The class that simulates a hand with wrist.
# class HandWristSimulator(object):
#     def __init__(self, q_sim, sim_params, options:QSimDDPParams) -> None:
#         """
#             :param q_sim: the QuasistaticSimulator of the whole hand-wrist system
#             :param sim_params: parameters for running q_sim
#             :param options: including the dimensions of wrist and hand only
#         """
#         self.q_sim = q_sim
#         self.sim_params = sim_params

#         self.x = np.zeros(options.nx,)      # current state
#         self.u = np.zeros(options.nu,)      # current control input

#     def step(self, u):



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
    # FIXME: add large weights to wrist dofs
    xActivation = crocoddyl.ActivationModelWeightedQuad(
        np.array([1e1] * (options.nx - options.nu) + [1e-2]*(options.nu))
        # np.array([1e1] * (options.nx - options.nu) + [1e5]*6+[1e-2]*(options.nu-6))
    )
    # FIXME: add large weights to wrist dofs
    xTActivation = crocoddyl.ActivationModelWeightedQuad(
        np.array([1e2] * (options.nx - options.nu) + [1e0]*(options.nu))
        # np.array([1e2] * (options.nx - options.nu) + [1e5]*6+[1e0]*(options.nu-6))
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

    # FIXME: (CANNOT SOLVE)add tight limits to wrist dofs
    # x_lb = np.concatenate((-np.inf*np.ones(options.nx-options.nu), -0.001*np.ones(6,), -np.inf*np.ones(options.nu-6)))
    # x_ub = np.concatenate((np.inf*np.ones(options.nx-options.nu), 0.001*np.ones(6,), np.inf*np.ones(options.nu-6)))
    # xBounds = crocoddyl.ActivationModelQuadraticBarrier(
    #     crocoddyl.ActivationBounds(x_lb, x_ub)
    # )
    # xBoundsCost = crocoddyl.CostModelResidual(
    #     state, xBounds, crocoddyl.ResidualModelState(state, actuation.nu)
    # )
    # runningCostModel.addCost("jointLimits", xBoundsCost, 1e3)
    # terminalCostModel.addCost("jointLimits", xBoundsCost, 1e5)

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
        q_vis.draw_configuration(options.x0)
        q_vis.publish_trajectory(solver.xs, h=options.h)
    np.save("../optimizer/warmstart/xs_trival_leap_open_box.npy", solver.xs)
    np.save("../optimizer/warmstart/us_trival_leap_open_box.npy", solver.us)
    import pdb; pdb.set_trace()


def visualize_one_step_mpc_in_meshcat(x0, x_ref_T, x_T, q_vis=None):
    if q_vis is not None:
        q_vis.draw_configuration(x0)


def generate_one_step_cost_model(
    options:QSimDDPParams, state, actuation, x_ref, w_u, w_a, is_terminal=False
):
    if isinstance(w_u, list):
        assert(len(w_u) == options.nx - options.nu)
    else:
        w_u = [w_u] * (options.nx - options.nu)

    if isinstance(w_a, list):
        assert(len(w_a) == options.nu)
    else:
        w_a = [w_a] * options.nu

    # cost models
    defaultCostModel = crocoddyl.CostModelSum(state, actuation.nu)

    # residual model
    xResidual = crocoddyl.ResidualModelState(state, x_ref, actuation.nu)
    xActivation = crocoddyl.ActivationModelWeightedQuad(
        np.array(w_u+ w_a)
    )

    # cost term
    xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)

    # add cost
    defaultCostModel.addCost("goalTracking", xRegCost, 1)                       # 'goalTracking' - 1
    if not is_terminal:
        uResidual = crocoddyl.ResidualModelControl(state, actuation.nu)
        uRegCost = crocoddyl.CostModelResidual(state, uResidual)
        defaultCostModel.addCost("controlReg", uRegCost, 0.04)                         # 'controlReg' - 1e-1       

    return defaultCostModel


def generate_one_step_mpc_problem(
    options:QSimDDPParams,
    q_sims, sim_params,
    state, actuation,
    x0, x_ref, xs, us,
    maxiter=3, wrist_pose=None,
    iter=0, logger=None
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
        # w_a = 1e0 if (i > 0.75*T) else 1e-2            # 100
        # FIXME: add large weights to wrist dofs
        w_a = 5e-2          # 1e-1
        # w_a = [1e5]*6+[1e-1]*(options.nu-6)
        defaulCostModel = generate_one_step_cost_model(
            options, state, actuation,
            x_ref[i], 1e1, w_a, is_terminal=False       # 1e1
        )
        actionModel = QuasistaticActionModel(
            q_sims[i], sim_params, state, actuation, defaulCostModel, options
        )
        actionModel.u_lb = options.u_lb; actionModel.u_ub = options.u_ub
        actionModel.set_wrist_pose(wrist_pose)
        runningModels.append(actionModel)

    w_a_T = 1e2                                     # 5e2
    # w_a_T = [1e6]*6+[5e2]*(options.nu-6)
    defaulCostModel = generate_one_step_cost_model(
        options, state, actuation,
        x_ref_T, 1e2, w_a_T, is_terminal=True       # 1e2
    )
    terminalModel = QuasistaticActionModel(
        q_sims[T], sim_params, state, actuation, defaulCostModel, options
    )
    terminalModel.u_lb = options.u_lb; terminalModel.u_ub = options.u_ub
    terminalModel.set_wrist_pose(wrist_pose)

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

    # timing report
    print("-------- MPC iter time report --------")
    calc_times = [runningModels[i].time_calc_ for i in range(T)]
    calc_times.append(terminalModel.time_calc_)
    calc_diff_times = [runningModels[i].time_calc_diff_ for i in range(T)]
    calc_diff_times.append(terminalModel.time_calc_diff_)
    print("avg. calc time: ", np.mean(calc_times))
    print("avg. calc runs: ", np.mean([runningModels[i].total_runs_calc_ for i in range(T)] + [terminalModel.total_runs_calc_]))
    print("avg. calcDiff time: ", np.mean(calc_diff_times))
    print("avg. calcDiff runs: ", np.mean([runningModels[i].total_runs_calc_diff_ for i in range(T)] + [terminalModel.total_runs_calc_diff_]))

    xs_arr = np.array(solver.xs)
    us_arr = np.array(solver.us)

    # scale and re-calc us here
    # --------------------
    xs0_ = xs_arr[0].copy()
    us0_ = us_arr[0].copy()

    xs_arr, us_arr = recalc_warm_start(
        xs_arr, us_arr,
        time_step=options.h,
        offset=options.execution_scale*options.h
    )

    # if options.enable_exact_rollout:
    #     problem.runningModels[0].sim_params.log_barrier_weight = options.kappa_exact
    #     data = runningModels[0].createData()
    #     runningModels[0].calc(data, xs_arr[0], us_arr[0])
    #     xs_arr[1] = data.xnext
    #     problem.runningModels[0].sim_params.log_barrier_weight = options.kappa

    # problem.runningModels[0].sim_params.log_barrier_weight = options.kappa_exact
    # data = runningModels[0].createData()
    # us0_scaled_ = options.execution_scale * us0_
    # runningModels[0].calc(data, xs0_, us0_scaled_)
    # xs_arr[0] = data.xnext
    # problem.runningModels[0].sim_params.log_barrier_weight = options.kappa

    # log data
    if logger is not None:
        # first_step_cost = problem.runningModels[0].calcCost(x0, us_[0])
        # residual_x_reg = first_step_cost["goalTracking"]
        # residual_x_u = np.linalg.norm(residual_x_reg.r[0])
        # residual_x_a = np.linalg.norm(residual_x_reg.r[1:])

        # phi_list = problem.runningModels[0].getSortedData()
        # yaw_list = [x0[0], x_ref[-1][0], xs_arr[-1][0]]     # x_real, x_ref, x_plan
        # residual_list = [residual_x_u, residual_x_a]        # r_u, r_a
        # cost_list = [solver.cost]                          # current_cost, opt_cost

        assert viz_helper is not None
        contact_sdists = q_sims[0].get_sdists()
        viz_helper.parse_finger_order(q_sims[0].get_geom_names_Bc())
        contact_sdists = viz_helper.get_ranked_data_1d(contact_sdists)

        logger.log_data(
            index=iter,
            # phi=phi_list,
            # yaw=yaw_list,
            # residual=residual_list,
            # cost=cost_list,

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

    # FIXME: create schedule for wrist pose
    # wrist_poses = np.linspace(0, np.pi/2, N-T)
    wrist_poses = -np.pi/6 * np.ones(N-T)

    for i in range(N-T):
        # x_ref_i = x_ref[i:i+T+1]
        x_ref_i = generate_reference_from_x0(x0, options)

        print("------ MPC iter {} ------".format(i))
        print("q0 (rpy): ", np.array([0.0, 0.0, x0[0]]))
        print("q_ref (rpy): ", np.array([0.0, 0.0, x_ref_i[0][0]]))

        xs, us = generate_one_step_mpc_problem(
            options,
            q_sims, sim_params,
            state, actuation,
            x0, x_ref_i, xs, us,
            maxiter=1, wrist_pose=wrist_poses[i],
            iter=i, logger=logger
        )

        # q_vis.publish_trajectory(xs, h=options.h)
        # import pdb; pdb.set_trace()

        if options.visualize_while_control:
            visualize_one_step_mpc_in_meshcat(
                x0=x0, x_ref_T=x_ref_i[-1], x_T=xs[-1], q_vis=q_vis
            )

        # rollout 1 step, and shift old xs-us 1 step
        # x0, u0 = xs[1], us[0]
        # xs = np.append(xs[1:], np.expand_dims(xs[-1], axis=0), axis=0)
        # us = np.append(us[1:], np.expand_dims(np.zeros(options.nu,), axis=0), axis=0)
        x0 = xs[0]

        x_traj.append(x0)
        # u_traj.append(u0)

        if viz_helper:
            p_W = default_q_sims[0].get_points_Ac()
            f_W = default_q_sims[0].get_f_Ac()[:, :3]
            viz_helper.parse_finger_order(default_q_sims[0].get_geom_names_Bc())
            viz_helper.plot_contact_points(p_W)
            viz_helper.plot_contact_forces(p_W, f_W)

    if q_vis is not None:
        q_vis.publish_trajectory(x_traj, h=options.h)

    return x_traj
    # return x_traj, u_traj


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


def generate_reference_from_x0(x0, params:QSimDDPParams):
    yaw0 = x0[params.q_u_indices[0]]
    yaw_ref = yaw0 + np.arange(0, params.T_ctrl+2) * params.dyaw

    x_ref = np.tile(params.target, (params.T_ctrl+1, 1))
    x_ref[:, params.q_u_indices[0]] = yaw_ref[1:]

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

# Utility functions
def make_ddp_params(ddp_params, extras):
    """ Configure DDP params """
    param = DDPSolverParams()
    param.h = ddp_params['h']
    param.kappa = ddp_params['kappa']
    param.kappa_exact = ddp_params['kappa_exact']
    param.auto_diff = ddp_params['auto_diff']
    param.nx = extras['nq']
    param.nu = extras['nu']
    param.q_u_indices = [0]

    param.T_trajopt = ddp_params['T_trajopt']
    param.T_ctrl = ddp_params['T_ctrl']

    # mpc weights
    param.w_u = ddp_params['w_u']
    param.w_a = ddp_params['w_a']
    param.w_uT = ddp_params['w_uT']
    param.w_aT = ddp_params['w_aT']

    # trajopt weights
    param.TO_w_u = ddp_params['TO_w_u']
    param.TO_w_a = ddp_params['TO_w_a']
    param.TO_w_uT = ddp_params['TO_w_uT']
    param.TO_w_aT = ddp_params['TO_w_aT']
    param.TO_W_x = ddp_params['TO_W_x']
    param.TO_W_u = ddp_params['TO_W_u']
    param.TO_W_xT = ddp_params['TO_W_xT']

    # cost sum weights
    param.W_X = ddp_params['W_X']
    param.W_U = ddp_params['W_U']
    param.W_SC = ddp_params['W_SC']
    param.W_J = ddp_params['W_J']

    param.execution_scale = extras['ddp_execution_scale']
    param.dxu = extras['dxu']
    param.target_xu = extras['target_xu']
    param.use_target_xu = extras['use_target_xu']
    param.u_lb = extras['u_lb'] * np.ones(param.nu)
    param.u_ub = extras['u_ub'] * np.ones(param.nu)

    param.contact_request.finger_to_geom_name_map = {
        'thumb': 'leap_hand_right::thumb_fingertip_collision',
        'index': 'leap_hand_right::fingertip_collision',
        'middle': 'leap_hand_right::fingertip_2_collision',
        'ring': 'leap_hand_right::fingertip_3_collision'
    }
    param.contact_request.object_link_name = ddp_params['object_link_name']
    param.contact_request.finger_link_names = ddp_params['finger_link_names']

    return param

def set_init_and_reg_states(param:DDPSolverParams, q_sim, extras):
    """ Set x0 and xa_reg """
    joint_names = parse_joint_names_from_plant(q_sim.get_plant())

    x0 = fill_jpos_arr_with_dic(joint_names, extras['jnames'], extras['x0'])
    xa_reg = fill_jpos_arr_with_dic(joint_names, extras['jnames'], extras['xa_reg'])
    
    # initialize the object part of x0 to be all zeros
    param.x0 = np.insert(x0.copy(), 0, np.zeros_like(extras['q_u_indices']))
    param.xa_reg = xa_reg.copy()

    # set trajopt target to be multiplies of dxu
    target = np.insert(xa_reg.copy(), 0, 5 * np.atleast_1d(extras['dxu']))
    param.target = target.copy()

    # set state reference
    param.x_ref = np.tile(param.x0, (param.T_ctrl+1, 1))

    param.x_obs = param.x0.copy()

    # set state limits
    j_lb, j_ub = load_joint_limits(joint_names, param.x0, extras['jlimits'])
    if not (np.isinf(-j_lb).all() and np.isinf(j_ub).all()):
        param.enable_j_cost = True
        param.x_lb = np.insert(j_lb, 0, -np.inf*np.ones_like(extras['q_u_indices']))
        param.x_ub = np.insert(j_ub, 0, np.inf*np.ones_like(extras['q_u_indices']))
        print(f"Enable joint limits cost with state lower bounds: {param.x_lb} and upper bounds {param.x_ub}!")

    return param

def load_warm_start(param:DDPSolverParams, extras):
    ws_suffix = extras['warmstart_suffix']
    param.x_init = np.load(os.path.join('../optimizer/warmstart', f'xs_trival_{ws_suffix}.npy'))
    param.u_init = np.load(os.path.join('../optimizer/warmstart', f'us_trival_{ws_suffix}.npy'))
    return param

def prepare_everything_for_ROS():
    """
        Return everything required by a ROS node
    """
    params = QSimDDPParams()
    params.enable_exact_rollout = True
    default_q_sims, default_sim_params, state, actuation = \
        allocate_resources_for_DDP(params, num_envs=41)

    q_parser = QuasistaticParser(QSIM_MODEL_PATH)
    q_vis = QuasistaticVisualizer.make_visualizer(q_parser)
    meshcat = q_vis.q_sim_py.meshcat
    plant_ = default_q_sims[0].get_plant()
    ordered_slider_names = AddJointSlidersToMeshcat(meshcat, plant_, apply_add=False)

    with open(MODEL_PARAM_PATH, "r") as f:
        model_params = yaml.safe_load(f)
        # x0
        params.x0[1:] = fill_jpos_arr_with_dic(ordered_slider_names, model_params['jnames'], model_params['x0'])[1:]
        # xa_reg
        params.xa_reg = fill_jpos_arr_with_dic(ordered_slider_names, model_params['jnames'], model_params['xa_reg'])[1:]
        params.target = np.insert(params.xa_reg, 0, np.pi/6)

    q_vis.draw_configuration(params.x0)
    viz_helper = InhandDDPVizHelper(meshcat)
    viz_helper.set_elements_in_meshcat()
    viz_helper.finger_to_geom_name_map = {
        "thumb": "leap_hand_right::thumb_fingertip_collision",
        "index": "leap_hand_right::fingertip_collision",
        "middle": "leap_hand_right::fingertip_2_collision",
        "ring": "leap_hand_right::fingertip_3_collision"
    }

    # set collision body names for q_sims used for mpc
    params.finger_link_names = ['fingertip', 'fingertip_2', 'fingertip_3', 'thumb_fingertip']
    params.object_link_name = "box_link"
    for i in range(len(default_q_sims)):
        default_q_sims[i].set_collision_body_names(params.finger_link_names, params.object_link_name)

    # TODO: Will use the unified parameter parsing, instead of hard-code in single file
    from ruamel.yaml import YAML
    loaded_params = YAML().load(open(MODEL_PARAM_PATH, "r"))
    loaded_ddp_params = make_ddp_params(loaded_params['ddp_params'], loaded_params)
    loaded_ddp_params = set_init_and_reg_states(loaded_ddp_params, default_q_sims[0], loaded_params)
    loaded_ddp_params = load_warm_start(loaded_ddp_params, loaded_params['ddp_params'])
    q_vis.draw_configuration(loaded_ddp_params.x0)
    # xs_init, us_init = solve_ddp_trajopt_unified(
    #     default_q_sims, default_sim_params,
    #     state, actuation, loaded_ddp_params
    # )
    # np.save("../optimizer/warmstart/xs_trival_leap_open_box.npy", xs_init)
    # np.save("../optimizer/warmstart/us_trival_leap_open_box.npy", us_init)
    # q_vis.publish_trajectory(xs_init, h=loaded_ddp_params.h)

    # solve_ddp_trajopt_problem(
    #     default_q_sims, default_sim_params,
    #     state, actuation, params,
    #     q_vis=q_vis
    # )
    # exit(0)

    xs_init, us_init = np.load("../optimizer/warmstart/xs_trival_leap_open_box.npy"), \
                        np.load("../optimizer/warmstart/us_trival_leap_open_box.npy")

    logger = DDPLogger(params.nx, params.nu, 0, 0)

    dtheta = 1.25 * np.pi/200
    q_ref = np.zeros((35, params.nx))
    q_ref[:, 1:] = params.x0[1:]
    for i in range(q_ref.shape[0]):
        q_ref[i, 0] = i*dtheta

    params.x_ref = q_ref
    params.x_init = xs_init
    params.u_init = us_init
    loaded_ddp_params.x_init = xs_init
    loaded_ddp_params.u_init = us_init

    return params, default_q_sims, default_sim_params, state, actuation, logger, q_vis, viz_helper, loaded_ddp_params

def update_reference(params:DDPSolverParams):
    xu_now = np.atleast_1d(params.x_obs[params.q_u_indices])
    
    dxu = params.dxu
    if params.use_target_xu:
        dxu = np.clip((params.target_xu - xu_now) / params.T_ctrl, -np.abs(dxu), np.abs(dxu))
    xu_ref = xu_now + np.arange(1, params.T_ctrl+2)[:, np.newaxis] * dxu
    params.x_ref[:, params.q_u_indices] = xu_ref

    return params

def solve_ddp_mpc_unified(default_q_sims, default_sim_params, state, actuation, q_vis, viz_helper, params:DDPSolverParams, niters=500):
    x_traj = []
    for i in range(niters):
        print(f"------ MPC iter {i} ------")
        params = update_reference(params)
        xs, us = ddp_runner_unified(
            params,
            default_q_sims, default_sim_params,
            state, actuation,
            maxiter=1
        )

        # TODO: set x0 and xs, us
        params.x_obs = xs[0].copy()

        params.x_init = xs.copy()
        params.u_init = us.copy()

        visualize_one_step_mpc_in_meshcat(
            x0=params.x_obs, x_ref_T=None, x_T=None, q_vis=q_vis
        )

        p_W = default_q_sims[0].get_points_Ac()
        f_W = default_q_sims[0].get_f_Ac()[:, :3]
        viz_helper.parse_finger_order(default_q_sims[0].get_geom_names_Bc())
        viz_helper.plot_contact_points(p_W)
        viz_helper.plot_contact_forces(p_W, f_W)

        x_traj.append(params.x_obs.copy().tolist())

    return x_traj

if __name__ == "__main__":
    params, default_q_sims, default_sim_params, state, actuation, \
            logger, q_vis, viz_helper, loaded_params = prepare_everything_for_ROS()

    # solve_ddp_mpc_problem(
    #     params,
    #     default_q_sims, default_sim_params,
    #     state, actuation,
    #     logger=logger,
    #     q_vis=q_vis,
    #     viz_helper=viz_helper
    # )

    solved_x_traj = solve_ddp_mpc_unified(
        default_q_sims, default_sim_params, state, actuation, q_vis, viz_helper,
        loaded_params, niters=35
    )
    q_vis.publish_trajectory(solved_x_traj, loaded_params.h)

############################################################
# This script is to manipulate planar objects in hand.     #
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
from common.inhand_ddp_helper import InhandDDPVizHelper, ContactTrackReference
from common.ddp_logger import *
from common.get_model_path import *


X_AXIS_LENGTH=0.08
QSIM_MODEL_PATH = os.path.join(YML_PATH, "leap_3d_slider.yml")
MODEL_PARAM_PATH = os.path.join(CONFIG_PATH, "leap_3d_slider.yaml")


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
        self.h = 0.1            # 0.025
        self.kappa = 100        # smoothing (default: 10)
        self.kappa_exact = 10000    # smoothing for exact rollout
        self.unactuated_mass_scale = 1.0
        self.nx = 19
        self.nu = 16
        self.q_u_indices = [0, 1, 2]
        self.normalize_quaternions = True

        self.T_trajopt = 40
        self.T_ctrl = 10            # 20
        
        self.x0 = np.array([
            0.0, 0.0, 0.0,                          # ball
            0.05,  1.15,  1.  ,  0.9,               # index finger
            -0.1 ,  0.9 ,  1.  ,  1.,               # middle finger
            -0.1 ,  1.2 ,  1.  ,  1.,               # ring finger
            0.5 ,  1.7 ,  1.1 ,  1.1,               # thumb
        ])
        self.xa_reg = np.array([
            0.05,  1.15,  1.  ,  0.9,               # index finger
            -0.1 ,  0.9 ,  1.  ,  1.,               # middle finger
            -0.1 ,  1.2 ,  1.  ,  1.,               # ring finger
            0.5 ,  1.7 ,  1.1 ,  1.1,               # thumb
        ])
        # observation
        self.x_obs = self.x0.copy()
        # reference (for trajopt)
        self.target = \
            np.insert(
            self.xa_reg,
            0,
            0.01
        )
        # reference (for MPC)
        self.x_ref = None
        # warm start (for MPC)
        self.x_init = None
        self.u_init = None
        # self.dyaw = 0.0 * np.pi/200
        self.dtrans = 0.005   # translation step size in y
        self.target_trans = 0.27   # set fixed target

        # weight
        self.w_u = [1e1, 1e1, 1e2]                       # double or ndarray
        self.w_u_T = [1e2, 1e2, 1e3]

        self.u_lb = -0.12 * np.ones(self.nu)             # -0.05
        self.u_ub = 0.12 * np.ones(self.nu)              # 0.05
        self.execution_scale = 0.5                       # 1.0, positive scalar, scale u0 when execution (i.e., rollout new x0)

        self.ddp_solver = crocoddyl.SolverBoxDDP         # SolverBoxDDP, SolverDDP
        self.ddp_verbose = False

        self.resolve_traj_opt = False
        self.visualize_while_control = True

        self.enable_exact_rollout = True               # use exact rollout for xs, not for solving us

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
    default_sim_params.gradient_dfdx_mode = DfDxMode.kAutoDiff
    # default_sim_params.gradient_dfdx_mode = DfDxMode.kAnalyticWithFiniteDiff
    
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
        self.nv_ = sum([len(v) for k, v in self.q_sim.get_velocity_indices().items()])
        self.nu_ = q_sim.num_actuated_dofs()
        self.q_a_indices_ = q_sim.get_q_a_indices_into_q()
        self.q_u_indices_ = q_sim.get_q_u_indices_into_q()

        self.time_calc_ = 0.0
        self.time_calc_diff_ = 0.0

    def calc(self, data, x, u=None, tau_ext=None):
        start_time = time.time()

        if u is None:
            u = np.zeros(self.nu_)
        u_incre = u.copy()

        u = u + x[self.q_a_indices_]

        # handle external torques
        if tau_ext is not None:
            assert len(tau_ext) == self.nv_
        else:
            tau_ext = np.zeros(self.nv_)

        # call q_sim's forward dynamics
        data.xnext = self.q_sim.calc_dynamics_forward(x, u, tau_ext, self.sim_params)

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


class ResidualModelPairCollision(crocoddyl.ResidualModelAbstract):
    """
        Note that pin_model and geom_model only include the hand.
        While state include both the hand and the object.
        Thus state_slice is the hand dofs in state.
    """
    def __init__(self, state, nu, pin_model, geom_model, pair_indices, state_slice):
        # TODO: IMPORTANT! According to 
        # https://github.com/loco-3d/crocoddyl/blob/da92f67394c07c987458a8cb24bc0e33edd64227/include/crocoddyl/core/residual-base.hxx#L88-L105,
        # setting v_dependent to False for quasi-static model (nv=0) will leave residual.Lx=0,
        # which is incorrect. So we set v_dependent to True.

        # TODO: IMPORTANT! Please set joint_index to parentJoint of geom1,
        # as geom2 is considered as the (fixed) environment.
        crocoddyl.ResidualModelAbstract.__init__(self, state, 3, nu, True, True, False)

        joint_ids = []
        for pair_index in pair_indices:
            assert len(geom_model.collisionPairs) > pair_index
            geom_id1 = geom_model.collisionPairs[pair_index].first
            geom_id2 = geom_model.collisionPairs[pair_index].second
            parent_joint_id1 = geom_model.geometryObjects[geom_id1].parentJoint
            parent_joint_id2 = geom_model.geometryObjects[geom_id2].parentJoint
            joint_ids.append((parent_joint_id1, parent_joint_id2))

        self.num_pairs = len(pair_indices)

        self.pin_model = pin_model
        self.pin_data = self.pin_model.createData()
        self.geom_model = geom_model
        self.geom_data = self.geom_model.createData()

        self.pair_ids = pair_indices
        self.joint_ids = joint_ids
        self.state_slice = state_slice

        self.q = None

    def calc(self, data, x, u):
        q = x[self.state_slice]
        self.q = q
        pin.updateGeometryPlacements(self.pin_model, self.pin_data, self.geom_model, self.geom_data, q)
        
        for i in range(self.num_pairs):
            pair_id = self.pair_ids[i]
            pin.computeDistance(self.geom_model, self.geom_data, pair_id)
            data.r[:] += self.geom_data.distanceResults[pair_id].getNearestPoint1() - \
                            self.geom_data.distanceResults[pair_id].getNearestPoint2()

    def calcDiff(self, data, x, u):
        q = self.q

        pin.computeJointJacobians(self.pin_model, self.pin_data, q)
        for i in range(self.num_pairs):
            pair_id = self.pair_ids[i]
            joint_id1, joint_id2 = self.joint_ids[i]
            
            d1 = self.geom_data.distanceResults[pair_id].getNearestPoint1() - \
                        self.pin_data.oMi[joint_id1].translation
            J1 = pin.getJointJacobian(self.pin_model, self.pin_data, joint_id1, pin.LOCAL_WORLD_ALIGNED)

            J1[:3] += np.matmul(pin.skew(d1).T, J1[-3:])

            d2 = self.geom_data.distanceResults[pair_id].getNearestPoint2() - \
                        self.pin_data.oMi[joint_id2].translation
            J2 = pin.getJointJacobian(self.pin_model, self.pin_data, joint_id2, pin.LOCAL_WORLD_ALIGNED)

            J2[:3] += np.matmul(pin.skew(d2).T, J2[-3:])

            data.Rx[:3, self.state_slice] += (J1[:3] - J2[:3])

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
    # default: x_u(1e1), x_a(1e-2)
    xActivation = crocoddyl.ActivationModelWeightedQuad(
        np.array([1e1, 1e-1, 1e-1] + [5e-2] * options.nu)
    )
    # default: x_u_T(1e2), x_a_T(1e0)
    xTActivation = crocoddyl.ActivationModelWeightedQuad(
        np.array([1e0, 1e1, 1e1] + [1e0] * options.nu)
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
    runningCostModel.addCost("controlReg", uRegCost, 0.01)

    terminalCostModel.addCost("goalTracking", xRegTermCost, 10)

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

    # visualization
    if q_vis is not None:
        q_vis.draw_configuration(options.x0)
        q_vis.publish_trajectory(solver.xs, h=options.h)
    np.save("../optimizer/warmstart/xs_trival_slider_cube_leap.npy", solver.xs)
    np.save("../optimizer/warmstart/us_trival_slider_cube_leap.npy", solver.us)
    import pdb; pdb.set_trace()


def visualize_one_step_mpc_in_meshcat(x0, x_ref_T, x_T, q_vis=None):
    if q_vis is not None:
        q_vis.draw_configuration(x0)


def generate_one_step_cost_model(
    options:QSimDDPParams, state, actuation, x_ref, w_u, w_a, models_sc, is_terminal=False
):
    if isinstance(w_u, list):
        assert(len(w_u) == options.nx - options.nu)
    else:
        w_u = [w_u] * (options.nx - options.nu)

    if isinstance(w_a, list):
        assert(len(w_a) == options.nu)
    else:
        # FIXME(yongpeng): disable thumb motion
        w_a = [w_a] * options.nu
        # w_a = [w_a] * 4 + [w_a * 100] * 4 + [w_a] * 8

    # print("w_u: ", w_u)
    # print("w_a: ", w_a)
        
    # parse self-collision models
    pin_model = models_sc[0]
    geom_model = models_sc[1]

    # cost models
    defaultCostModel = crocoddyl.CostModelSum(state, actuation.nu)

    # residual model
    xResidual = crocoddyl.ResidualModelState(state, x_ref, actuation.nu)
    xActivation = crocoddyl.ActivationModelWeightedQuad(np.array(w_u + w_a))

    # self.collision cost
    xSelfCollideCost = crocoddyl.CostModelResidual(
        state,
        crocoddyl.ActivationModel2NormBarrier(3, alpha=0.1),
        ResidualModelPairCollision(
            state, actuation.nu, pin_model, geom_model,
            pair_indices=[0], state_slice=range(3, 19)
        )
    )    

    # cost term
    xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)

    # add cost
    defaultCostModel.addCost("goalTracking", xRegCost, 1)                       # 'goalTracking' - 1
    defaultCostModel.addCost("selfCollide", xSelfCollideCost, 0)             # 'selfCollide' - 1e-2
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
    models_sc,
    maxiter=3, iter=0, logger=None
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
        w_u = options.w_u if hasattr(options, "w_u") else 1e1
        w_a = 1e-2                      # 5e-2
        defaulCostModel = generate_one_step_cost_model(
            options, state, actuation,
            x_ref[i], w_u, w_a, models_sc, is_terminal=False       # 1e1
        )
        actionModel = QuasistaticActionModel(
            q_sims[i], sim_params, state, actuation, defaulCostModel
        )
        actionModel.u_lb = options.u_lb; actionModel.u_ub = options.u_ub
        runningModels.append(actionModel)

    w_u_T = options.w_u_T if hasattr(options, "w_u_T") else 1e2
    w_a_T = 5e1                                     # 5e1
    defaulCostModel = generate_one_step_cost_model(
        options, state, actuation,
        x_ref_T, w_u_T, w_a_T, models_sc, is_terminal=True       # 1e2
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

    # examine self collision cost
    sc_cost_term = runningModels[0].default_cost_data.costs.todict()['selfCollide']

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
    # normalize_quaternions(xs_arr, options=options)

    # problem.runningModels[0].sim_params.log_barrier_weight = options.kappa_exact
    # data = runningModels[0].createData()
    # us0_scaled_ = options.execution_scale * us0_
    # runningModels[0].calc(data, xs0_, us0_scaled_)
    # xs_arr[0] = data.xnext
    # problem.runningModels[0].sim_params.log_barrier_weight = options.kappa

    # problem.runningModels[0].sim_params.log_barrier_weight = options.kappa_exact
    # problem.runningModels[0].sim_params.forward_mode = ForwardDynamicsMode.kSocpMp
    # problem.runningModels[0].sim_params.unactuated_mass_scale = 1.0

    # data = runningModels[0].createData()
    # us0_scaled_ = options.execution_scale * us0_
    # runningModels[0].calc(data, xs0_, us0_scaled_)
    # xs_arr[0] = data.xnext
    # us_arr[0] = us0_scaled_
    # problem.runningModels[0].sim_params.log_barrier_weight = options.kappa
    # problem.runningModels[0].sim_params.forward_mode = ForwardDynamicsMode.kLogIcecream
    # problem.runningModels[0].sim_params.unactuated_mass_scale = options.unactuated_mass_scale

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

        if viz_helper is not None:
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
            ddp_solved_x0=xs_arr[0],
            ddp_solved_x0_obj=xs_arr[0][:3],
            ddp_solved_u0=us_arr[0],
            ddp_obj_yaw=x0[0],
            # ddp_sdists=contact_sdists
            ## --------------------
        )
        
    return xs_arr, us_arr


def solve_ddp_mpc_problem(
    options: QSimDDPParams,
    q_sims, sim_params,
    state, actuation,
    models_sc,
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
        x_ref_i = generate_reference_from_x0(x0, options)

        print("------ MPC iter {} ------".format(i))
        print("xu (yaw, x, y): ", x0[:3])
        print("xu_ref (yaw, x, y): ", x_ref_i[0][:3])

        xs, us = generate_one_step_mpc_problem(
            options,
            q_sims, sim_params,
            state, actuation,
            x0, x_ref_i, xs, us,
            models_sc,
            maxiter=1, iter=i, logger=logger
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
        u0 = us[0]

        # FIXME: manually disturbe x0 for debug
        # if i == 50:
        #     x0[1] += 0.04

        x_traj.append(x0)
        u_traj.append(u0)

        if viz_helper:
            p_W = default_q_sims[0].get_points_Ac()
            f_W = default_q_sims[0].get_f_Ac()[:, :3] * 10
            viz_helper.parse_finger_order(default_q_sims[0].get_geom_names_Bc())
            viz_helper.plot_contact_points(p_W)
            viz_helper.plot_contact_forces(p_W, f_W)
            if logger:
                f_W_ordered = viz_helper.get_ranked_contact_data(f_W)
                logger.log_data(index=i, f_W_norm=np.linalg.norm(f_W_ordered, axis=-1))

    if q_vis is not None:
        q_vis.publish_trajectory(x_traj, h=options.h)

    return x_traj, u_traj


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
    # make sure the revolute joint is at index 0
    trans0 = x0[params.q_u_indices[0]]
    dtrans = params.dtrans
    dtrans = np.clip((params.target_trans - trans0) / params.T_ctrl, -abs(dtrans), abs(dtrans))
    trans_ref = trans0 + np.arange(0, params.T_ctrl+2) * dtrans

    nxu_ = len(params.q_u_indices)
    xu_ref = np.zeros((params.T_ctrl+2, nxu_))
    xu_ref[:, params.q_u_indices[0]] = trans_ref

    x_ref = np.tile(params.target, (params.T_ctrl+1, 1))
    x_ref[:, params.q_u_indices] = xu_ref[1:]

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


def prepare_everything_for_ROS():
    """
        Return everything required by a ROS node
    """
    params = QSimDDPParams()
    params.enable_exact_rollout = True
    default_q_sims, default_sim_params, state, actuation = \
        allocate_resources_for_DDP(params, num_envs=41)

    # TODO(yongpeng): will tune the mass scale help?
    default_sim_params.unactuated_mass_scale = params.unactuated_mass_scale

    q_parser = QuasistaticParser(QSIM_MODEL_PATH)
    q_vis = QuasistaticVisualizer.make_visualizer(q_parser)
    meshcat = q_vis.q_sim_py.meshcat
    plant_ = default_q_sims[0].get_plant()
    ordered_slider_names = AddJointSlidersToMeshcat(meshcat, plant_, apply_add=False)

    num_obj_joints = len(params.q_u_indices)

    with open(MODEL_PARAM_PATH, "r") as f:
        model_params = yaml.safe_load(f)
        # x0
        params.x0[num_obj_joints:] = fill_jpos_arr_with_dic(ordered_slider_names, model_params['jnames'], model_params['x0'])[num_obj_joints:]
        # xa_reg
        params.xa_reg = fill_jpos_arr_with_dic(ordered_slider_names, model_params['jnames'], model_params['xa_reg'])[num_obj_joints:]
        xu_reg = params.x0[:num_obj_joints].copy()
        xu_reg[0] = 0.1
        params.target = np.insert(params.xa_reg, 0, xu_reg)

    logger = DDPLogger(params.nx, params.nu, 0, 0)
    # logger = None

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
    params.object_link_name = "cube_link"
    for i in range(len(default_q_sims)):
        default_q_sims[i].set_collision_body_names(params.finger_link_names, params.object_link_name)

    # TODO(yongpeng): solve trajopt for warm start
    # solve_ddp_trajopt_problem(
    #     default_q_sims, default_sim_params,
    #     state, actuation, params,
    #     q_vis=q_vis
    # )
    # exit(0)

    xs_init, us_init = np.load("../optimizer/warmstart/xs_trival_slider_cube_leap.npy"), \
                        np.load("../optimizer/warmstart/us_trival_slider_cube_leap.npy")

    dtheta = 1.25 * np.pi/200
    q_ref = np.zeros((150, params.nx))
    q_ref[:, num_obj_joints:] = params.x0[num_obj_joints:]
    for i in range(q_ref.shape[0]):
        q_ref[i, params.q_u_indices[0]] = i*dtheta

    params.x_ref = q_ref
    params.x_init = xs_init
    params.u_init = us_init

    return params, default_q_sims, default_sim_params, state, actuation, logger, q_vis, viz_helper


def load_pinocchio_model():
    """ Hardcode for testing """
    sdf_url = os.path.join(SDF_PATH, "leap_3d_4finger_tac3d_simple_pinocchio.sdf")
    pin_model, _, collision_model, _ = pin.buildModelsFromSdf(
        filename=sdf_url,
        package_dirs=os.path.join(DRAKE_HOME, "manipulation/models", "leap_hand_description/urdf/leap_hand/mesh/simple_obj")
    )
    # collision_model.addCollisionPair(pin.CollisionPair(0, 2))
    collision_model.addCollisionPair(pin.CollisionPair(3, 2))
    return pin_model, collision_model


if __name__ == "__main__":
    # models for self collision (pin_model and geom_model)
    models_sc = load_pinocchio_model()

    params, default_q_sims, default_sim_params, state, actuation, \
            logger, q_vis, viz_helper = prepare_everything_for_ROS()

    x_traj, u_traj = solve_ddp_mpc_problem(
        params,
        default_q_sims, default_sim_params,
        state, actuation,
        models_sc,
        logger=logger,
        q_vis=q_vis,
        viz_helper=viz_helper
    )

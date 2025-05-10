import numpy as np
import crocoddyl
from pydrake.all import (
    PiecewisePolynomial
)
from qsim_cpp import QuasistaticSimulatorCpp
from qsim_cpp import ForwardDynamicsMode
from action_model import QuasistaticActionModel
from residual_model import ResidualModelPairCollision, ResidualModelFrameRotation
from common.common_ddp import DDPSolverParams, convert_array_to_list, convert_quat_to_matrix, normalize_array


def get_tracking_objectives(
    options: DDPSolverParams,
    q_sims
):
    """
        Get the tracking objectives for low level controller
        - contact force
        - contact point
    """
    n_c = options.contact_request.n_c()
    options.contact_response.p_ACa_A = np.empty((options.T_ctrl, n_c, 3))
    options.contact_response.f_BA_W = np.empty((options.T_ctrl, n_c, 3))
    options.contact_response.n_obj_W = np.empty((options.T_ctrl, n_c, 3))

    for i in range(options.T_ctrl):
        p_ACa_A = q_sims[i].get_points_Ac()
        f_BA_W = q_sims[i].get_f_Ac()[:, :3]
        n_obj_W = q_sims[i].get_Nhat()
        geom_names = q_sims[i].get_geom_names_Bc()
        options.contact_request.parse_finger_order(geom_names)

        p_ACa_A = options.contact_request.get_ranked_contact_data(p_ACa_A)
        f_BA_W = options.contact_request.get_ranked_contact_data(f_BA_W)
        n_obj_W = options.contact_request.get_ranked_contact_data(n_obj_W)

        options.contact_response.p_ACa_A[i] = p_ACa_A
        options.contact_response.f_BA_W[i] = f_BA_W
        options.contact_response.n_obj_W[i] = n_obj_W


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


def generate_one_step_cost_model(
    options:DDPSolverParams,
    state, actuation,
    x_ref, w_u, w_a, w_u_so3,
    is_terminal=False
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
    costModel = crocoddyl.CostModelSum(state, actuation.nu)

    # state regulation cost
    costState = crocoddyl.CostModelResidual(
        state,
        crocoddyl.ActivationModelWeightedQuad(np.array(w_u + w_a)),
        crocoddyl.ResidualModelState(state, x_ref, actuation.nu)
    )
    costModel.addCost("stateReg", costState, options.W_X)

    # SO(3) rotation tracking
    if options.q_rot_indices is not None:
        # lie group
        Rref = convert_quat_to_matrix(x_ref[options.q_rot_indices])
        costObjRotationSO3 = crocoddyl.CostModelResidual(
            state,
            crocoddyl.ActivationModelWeightedQuad(np.array([w_u_so3]*3)),
            ResidualModelFrameRotation(state, actuation.nu, Rref, options.q_rot_indices)
        )

        # quaternion subtraction
        costObjRotationL2Norm = crocoddyl.CostModelResidual(
            state,
            crocoddyl.ActivationModelWeightedQuad(np.array([w_u_so3]*4 + [0] * options.nu)),
            crocoddyl.ResidualModelState(state, x_ref, actuation.nu)
        )

        costModel.addCost("objRotationSO3", costObjRotationSO3, options.W_U_SO3)
        costModel.addCost("objRotationL2Norm", costObjRotationL2Norm, options.W_U_SO3)

    # joint limits cost
    if options.enable_j_cost:
        x_lb = options.x_lb; x_ub = options.x_ub
        costJointLimits = crocoddyl.CostModelResidual(
            state,
            crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(x_lb, x_ub)),
            crocoddyl.ResidualModelState(state, actuation.nu)
        )
        costModel.addCost("jointLimits", costJointLimits, options.W_J)

    # self collision cost
    if options.enable_sc_cost:
        pin_model = options.models_sc[0]
        geom_model = options.models_sc[1]
        costSelfCollide = crocoddyl.CostModelResidual(
            state,
            crocoddyl.ActivationModel2NormBarrier(3, alpha=0.1),
            ResidualModelPairCollision(
                state, actuation.nu, pin_model, geom_model,
                pair_indices=options.pair_index_sc, state_slice=options.state_slice_sc
            )
        )
        costModel.addCost("selfCollide", costSelfCollide, options.W_SC)

    if not is_terminal:
        costControl = crocoddyl.CostModelResidual(
            state, 
            crocoddyl.ResidualModelControl(state, actuation.nu)
        )
        costModel.addCost("controlReg", costControl, options.W_U)

    return costModel


def generate_one_step_mpc_problem(
    options:DDPSolverParams,
    q_sims, sim_params,
    state, actuation,
    x0, x_ref, xs, us,
    maxiter=3
):
    if options.n_wrist > 0 or options.q_rot_indices is not None:
        extras = options
    else:
        extras = None

    T = options.T_ctrl

    assert x_ref.shape[0] == T+1
    x_ref_T = x_ref[-1]

    assert xs.shape[0] == T+1 and us.shape[0] == T

    w_u = options.w_u
    w_u_so3 = options.w_u_so3
    w_aT = options.w_aT
    w_uT = options.w_uT
    w_uT_so3 = options.w_uT_so3
    
    runningModels = []
    for i in range(T):
        w_a = options.w_a[0] if (i > 0.75*T) else options.w_a[1]
        defaulCostModel = generate_one_step_cost_model(
            options, state, actuation,
            x_ref[i], w_u, w_a, w_u_so3, is_terminal=False
        )
        actionModel = QuasistaticActionModel(
            q_sims[i], sim_params, state, actuation, defaulCostModel, extras
        )
        actionModel.u_lb = options.u_lb; actionModel.u_ub = options.u_ub
        
        if options.n_wrist > 0:
            actionModel.set_wrist_pose(options.x0[options.q_wrist_indices])

        runningModels.append(actionModel)

    defaulCostModel = generate_one_step_cost_model(
        options, state, actuation,
        x_ref_T, w_uT, w_aT, w_uT_so3, is_terminal=True
    )
    terminalModel = QuasistaticActionModel(
        q_sims[T], sim_params, state, actuation, defaulCostModel, extras
    )
    terminalModel.u_lb = options.u_lb; terminalModel.u_ub = options.u_ub

    if options.n_wrist > 0:
        terminalModel.set_wrist_pose(options.x0[options.q_wrist_indices])
    
    # shooting problem
    problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)
    solver = options.ddp_solver(problem)

    # make feasible init state-action trajectory
    if options.ddp_verbose:
        print("u_ddp[0]: ", us[0])
        solver.setCallbacks([crocoddyl.CallbackVerbose()])
    
    us_ = convert_array_to_list(us.copy())
    xs_ = problem.rollout(us_)

    solver.solve(
        init_xs=xs_,
        init_us=us_,
        maxiter=maxiter,
        is_feasible=True,
        init_reg=1e0
    )
    # print("xs0(solved): ", xs_[0])

    # record cost data
    if options.ddp_verbose:
        opt_cost_dict = {'Total': []}
        for i in range(T):
            opt_cost_dict['Total'].append(problem.runningDatas[i].cost)
            step_cost = problem.runningModels[0].default_cost_data.costs.todict()
            for key, cost in step_cost.items():
                if key not in opt_cost_dict:
                    opt_cost_dict[key] = []
                opt_cost_dict[key].append(cost.cost)
        for key, cost in opt_cost_dict.items():
            print(f"{key}:\n {cost}")

    xs_arr = np.array(solver.xs)
    us_arr = np.array(solver.us)

    # calc exact rollout
    # --------------------
    xs0_ = xs_arr[0].copy()
    us0_ = us_arr[0].copy()

    xs_arr, us_arr = recalc_warm_start(
        xs_arr, us_arr,
        time_step=options.h,
        offset=options.execution_scale*options.h
    )

    # normalize quaternions
    if options.q_rot_indices is not None:
        xs_arr[:, options.q_rot_indices] = normalize_array(xs_arr[:, options.q_rot_indices])

    # print("xs0(recalc): ", xs_arr[0])

    # problem.runningModels[0].sim_params.log_barrier_weight = options.kappa_exact
    # data = runningModels[0].createData()
    # us0_scaled_ = options.execution_scale * us0_
    # runningModels[0].calc(data, xs0_, us0_scaled_)
    # xs_arr[0] = data.xnext
    # problem.runningModels[0].sim_params.log_barrier_weight = options.kappa
        
    return xs_arr, us_arr


def ddp_runner(
    options: DDPSolverParams,
    q_sims, sim_params,
    state, actuation,
    maxiter=1
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

    # solve trajectory optimization first
    xs, us = generate_one_step_mpc_problem(
        options,
        q_sims, sim_params,
        state, actuation,
        x0, x_ref, xs0, us0,
        maxiter=maxiter
    )

    get_tracking_objectives(options, q_sims)

    xs = np.array(xs)
    us = np.array(us)

    return xs, us


def solve_ddp_trajopt_problem(
    default_q_sims,
    default_sim_params,
    state,
    actuation,
    options: DDPSolverParams,
):
    if options.n_wrist > 0 or options.q_rot_indices is not None:
        extras = options
    else:
        extras = None

    # parse weights
    w_u = options.TO_w_u
    w_a = options.TO_w_a
    w_uT = options.TO_w_uT
    w_aT = options.TO_w_aT
    W_x = options.TO_W_x
    W_u = options.TO_W_u
    W_xT = options.TO_W_xT

    # check dimensions
    if isinstance(w_u, list):
        assert(len(w_u) == options.nx - options.nu)
    else:
        w_u = [w_u] * (options.nx - options.nu)

    if isinstance(w_a, list):
        assert(len(w_a) == options.nu)
    else:
        w_a = [w_a] * options.nu

    if isinstance(w_uT, list):
        assert(len(w_uT) == options.nx - options.nu)
    else:
        w_uT = [w_uT] * (options.nx - options.nu)

    if isinstance(w_aT, list):
        assert(len(w_aT) == options.nu)
    else:
        w_aT = [w_aT] * options.nu

    # residual for state and cost
    xResidual = crocoddyl.ResidualModelState(state, options.target_TO, actuation.nu)
    xActivation = crocoddyl.ActivationModelWeightedQuad(np.array(w_u+w_a))
    xTActivation = crocoddyl.ActivationModelWeightedQuad(np.array(w_uT+w_aT))
    uResidual = crocoddyl.ResidualModelControl(state, actuation.nu)

    # cost term
    xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
    uRegCost = crocoddyl.CostModelResidual(state, uResidual)
    xRegTermCost = crocoddyl.CostModelResidual(state, xTActivation, xResidual)

    # cost model
    runningCostModel = crocoddyl.CostModelSum(state, actuation.nu)
    terminalCostModel = crocoddyl.CostModelSum(state, actuation.nu)

    runningCostModel.addCost("stateReg", xRegCost, W_x)
    runningCostModel.addCost("controlReg", uRegCost, W_u)

    terminalCostModel.addCost("stateReg", xRegTermCost, W_xT)

    # pushing action model
    T = options.T_trajopt
    runningModels = []
    for i in range(T):
        runningModel = QuasistaticActionModel(
            default_q_sims[i], default_sim_params, state, actuation, runningCostModel, extras
        )
        runningModel.u_lb = options.u_lb; runningModel.u_ub = options.u_ub
        runningModels.append(runningModel)

    terminalModel = QuasistaticActionModel(
        default_q_sims[T], default_sim_params, state, actuation, terminalCostModel, extras
    )
    terminalModel.u_lb = options.u_lb; terminalModel.u_ub = options.u_ub

    problem = crocoddyl.ShootingProblem(options.x0, runningModels, terminalModel)
    solver = options.ddp_solver(problem)

    # create feasible trajectory
    xs = [options.x0]
    us = [0.0 * np.ones(options.nu)] * T
    for i in range(T):
        data = runningModels[i].createData()
        runningModels[i].calc(data, xs[-1], us[i])
        xs.append(data.xnext)

    solver.setCallbacks([crocoddyl.CallbackVerbose(),])

    # solve optimization
    solver.solve(
        init_xs=xs,
        init_us=us,
        maxiter=50,
        is_feasible=True,
        init_reg=1e0
    )
    
    xs_init = np.array(solver.xs)
    us_init = np.array(solver.us)

    return xs_init, us_init

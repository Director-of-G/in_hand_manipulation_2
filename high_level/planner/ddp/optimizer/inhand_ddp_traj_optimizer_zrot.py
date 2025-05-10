############################################################
# This script is modified from                             #
# inhand_ddp_traj_optimizer.py. This script serves as an   #
# agent connecting ddp algorithm and drake simulation.     #
############################################################

import os
import sys
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
import numpy as np

from ddp.ddp_solver import solve_ddp_trajopt_problem, ddp_runner
from common.common_drake import *
from common.common_ddp import (
    DDPSolverParams,
    parse_joint_names_from_plant,
    fill_jpos_arr_with_dic,
    allocate_resources_for_DDP,
    load_pinocchio_model,
    load_joint_limits
)

from ddp.reference_generator import ReferenceGeneratorRPY

OPTIMIZER_DIR = os.path.dirname(os.path.abspath(__file__))
WARMSTART_DIR = os.path.join(OPTIMIZER_DIR, 'warmstart')


class AllegroDDPTrajOptimizer(object):
    # ! TO AVOID CONFLICTS
    # ! states should be in [xu; xa] order anywhere in trajopt
    # ! different order (such as Drake) should be converted outside trajopt
    def __init__(self, options:HighLevelOptions):
        # sim_params
        self.high_level_options = options
        self.make_ddp_params(options)
        T = self.params().T_ctrl
        self.T = T

        # allocate resources
        default_q_sims, default_sim_params, state, actuation = \
            allocate_resources_for_DDP(
                options.model_url, self.params(), num_envs=T+1
            )
        self.q_sims = default_q_sims
        self.sim_params = default_sim_params
        self.state = state
        self.actuation = actuation

        # set x0 and xa_reg
        self.ref_gen_so3 = None         # reference generator for so3
        self.set_init_and_reg_states(options)

        # get warm start
        ws_status = self.load_warm_start(options)
        if ws_status is False:
            self.compute_warm_start()

        # x_obs
        self.params_.x_obs = self.params_.x0.copy()

        # x_ref
        x_ref = np.tile(self.params_.x0, (T+1, 1))
        self.params_.x_ref = x_ref

        # solution
        self.xs_ = self.params_.x_init.copy()
        self.us_ = self.params_.u_init.copy()
        self.p_ACa_A_ = np.zeros((T, options.nc, 3))            # contact points on object, expressed in world frame
        self.f_BA_W_ = np.zeros((T, options.nc, 3))             # contact force on object, expressed in world frame
        self.n_obj_W_ = np.zeros((T, options.nc, 3))            # contact normal on object, expressed in world frame

    def make_ddp_params(self, options:HighLevelOptions):
        """ Configure DDP params """
        ddp_params = options.ddp_params
        self.params_ = DDPSolverParams()
        self.params_.h = ddp_params['h']
        self.params_.kappa = ddp_params['kappa']
        self.params_.kappa_exact = ddp_params['kappa_exact']
        self.params_.auto_diff = ddp_params['auto_diff']
        self.params_.nx = options.nq
        self.params_.nu = options.nu
        self.params_.q_u_indices = options.q_u_indices
        self.params_.q_rot_indices = options.q_rot_indices

        if options.q_rot_indices is not None:
            self.params_.has_ori = True
            self.params_.ori_start = options.q_rot_indices[0]

        if 'wrist' in options.finger_to_joint_map:
            self.params_.n_wrist = len(options.finger_to_joint_map['wrist'])
            self.params_.q_wrist_indices = [self.params_.q_u_indices[-1]+1+i for i in range(self.params_.n_wrist)]

        self.params_.T_trajopt = ddp_params['T_trajopt']
        self.params_.T_ctrl = ddp_params['T_ctrl']

        # mpc weights
        self.params_.w_u = ddp_params['w_u']
        self.params_.w_a = ddp_params['w_a']
        if 'w_u_so3' in ddp_params:
            self.params_.w_u_so3 = ddp_params['w_u_so3']
        self.params_.w_uT = ddp_params['w_uT']
        self.params_.w_aT = ddp_params['w_aT']
        if 'w_uT_so3' in ddp_params:
            self.params_.w_uT_so3 = ddp_params['w_uT_so3']

        # trajopt weights
        self.params_.TO_w_u = ddp_params['TO_w_u']
        self.params_.TO_w_a = ddp_params['TO_w_a']
        self.params_.TO_w_uT = ddp_params['TO_w_uT']
        self.params_.TO_w_aT = ddp_params['TO_w_aT']
        self.params_.TO_W_x = ddp_params['TO_W_x']
        self.params_.TO_W_u = ddp_params['TO_W_u']
        self.params_.TO_W_xT = ddp_params['TO_W_xT']

        # cost sum weights
        self.params_.W_X = ddp_params['W_X']
        self.params_.W_U = ddp_params['W_U']
        if 'W_U_SO3' in ddp_params:
            self.params_.W_U_SO3 = ddp_params['W_U_SO3']
        self.params_.W_SC = ddp_params['W_SC']
        self.params_.W_J = ddp_params['W_J']

        if options.model_url_sc != '' and len(ddp_params['pairs_sc']) > 0:
            self.params_.enable_sc_cost = True
            self.params_.models_sc, self.params_.pair_index_sc = load_pinocchio_model(
                options.model_url_sc, options.mesh_url_sc, ddp_params['pairs_sc']
            )
            self.params_.state_slice_sc = range(options.nq-options.nu, options.nq)
            print("Enable self-collision cost with pairs: ", ddp_params['pairs_sc'])

        self.params_.execution_scale = options.ddp_execution_scale
        self.params_.dxu = options.dxu
        self.params_.target_xu = np.atleast_1d(options.target_xu)
        self.params_.use_target_xu = options.use_target_xu
        self.params_.u_lb = options.u_lb
        self.params_.u_ub = options.u_ub

        self.params_.contact_request.finger_to_geom_name_map = \
            options.finger_to_geom_name_map
        self.params_.contact_request.object_link_name = ddp_params['object_link_name']
        self.params_.contact_request.finger_link_names = ddp_params['finger_link_names']

        self.params_.ddp_verbose = options.debug_mode

        self.force_threshold = options.force_threshold
        self.force_scale = options.desired_force_scale

    def reset(self):
        # reset warm start, solution and reference
        self.load_warm_start(self.high_level_options)
        self.xs_ = self.params_.x_init.copy()
        self.us_ = self.params_.u_init.copy()
        self.params_.x_obs = self.params_.x0.copy()
        self.params_.x_ref = np.tile(self.params_.x0, (self.T+1, 1))

        self.p_ACa_A_[:] = 0
        self.f_BA_W_[:] = 0
        self.n_obj_W_[:] = 0

        # generate new target
        q_rot_indices = self.params().q_rot_indices
        if q_rot_indices is not None:
            self.reset_so3_target()

    def reset_so3_target(self):
        q_rot_indices = self.params().q_rot_indices 
        options = self.high_level_options
        if self.ref_gen_so3 is not None:
            ref_gen_so3 = self.ref_gen_so3
        else:
            ref_gen_so3 = ReferenceGeneratorRPY(step_len=options.dxu_so3, num_steps=options.T_mpc)
            self.ref_gen_so3 = ref_gen_so3

        if self.params().preset_so3_target is not None:
            self.params_.target[q_rot_indices] = self.params().preset_so3_target
            ref_gen_so3.set_start_and_target_quat(
                start_quat=self.params().x0[q_rot_indices],
                target_quat=self.params().preset_so3_target
            )
            return
 
        if options.random_target_xu_so3:
            ref_gen_so3.generate_random_so3_target()
            options.target_xu_so3 = ref_gen_so3.target.wxyz()
        else:
            ref_gen_so3.set_start_and_target_quat(
                start_quat=self.params().x0[q_rot_indices],
                target_quat=options.target_xu_so3
            )

        self.params_.target[q_rot_indices] = options.target_xu_so3

    def set_init_and_reg_states(self, options:HighLevelOptions):
        """ Set x0 and xa_reg """
        joint_names = parse_joint_names_from_plant(self.q_sims[0].get_plant())
        self.joint_names_ = joint_names

        x0 = fill_jpos_arr_with_dic(joint_names, options.finger_to_joint_map, options.finger_to_x0_map)
        xa_reg = fill_jpos_arr_with_dic(joint_names, options.finger_to_joint_map, options.finger_to_xreg_map_list[0])
        
        # initialize the object part of x0 to be all zeros
        self.params_.x0 = np.insert(x0.copy(), 0, np.zeros_like(options.q_u_indices))
        self.params_.xa_reg = xa_reg.copy()

        # set trajopt target to be multiplies of dxu
        target_TO = np.insert(xa_reg.copy(), 0, 5 * np.atleast_1d(options.dxu))
        self.params_.target_TO = target_TO.copy()

        # set target for manipulation
        if options.use_target_xu:
            target = np.insert(xa_reg.copy(), 0, np.atleast_1d(options.target_xu))
        else:
            target = np.insert(xa_reg.copy(), 0, np.zeros_like(options.q_u_indices))
        self.params_.target = target.copy()

        # set state limits
        j_lb, j_ub = load_joint_limits(joint_names, x0, options.finger_to_limits_map)
        if not (np.isinf(-j_lb).all() and np.isinf(j_ub).all()):
            self.params_.enable_j_cost = True
            self.params_.x_lb = np.insert(j_lb, 0, -np.inf*np.ones_like(options.q_u_indices))
            self.params_.x_ub = np.insert(j_ub, 0, np.inf*np.ones_like(options.q_u_indices))
            print(f"Enable joint limits cost with state lower bounds: {self.params_.x_lb} and upper bounds {self.params_.x_ub}!")

        # reference generator for SO(3)
        q_rot_indices = self.params_.q_rot_indices
        if q_rot_indices is not None:
            # reset x0
            self.params_.x0[q_rot_indices] = options.init_xu_so3

            self.reset_so3_target()

            # reset target
            target_xu_so3_trajopt = self.ref_gen_so3.generate_reference_from_x0(self.params_.x0[q_rot_indices])[-1]
            self.params_.target_TO[q_rot_indices] = target_xu_so3_trajopt

    def set_so3_target(self, quat_target):
        self.params_.preset_so3_target = quat_target.copy()

    def compute_warm_start(self):
        """ compute xs_init and us_init """
        xs_init, us_init = solve_ddp_trajopt_problem(
            self.q_sims, self.sim_params,
            self.state, self.actuation, self.params(),
        )
        self.params_.x_init = xs_init[:self.T+1].copy()
        self.params_.u_init = xs_init[:self.T].copy()

    def set_warm_start(self, xs_init, us_init):
        assert xs_init.shape[0] == self.params_.T_ctrl+1
        assert us_init.shape[0] == self.params_.T_ctrl
        assert xs_init.shape[-1] == self.params_.nx
        assert us_init.shape[-1] == self.params_.nu
        self.params_.x_init = xs_init.copy()
        self.params_.u_init = us_init.copy()

    def params(self):
        return self.params_
    
    def joint_names(self):
        return self.joint_names_
    
    def x0(self):
        return self.params_.x0

    def xa0(self):
        return self.params_.x0[-16:]
    
    def x_goal(self):
        return self.params_.target

    def set_observation(self, x_obs):
        x_obs = x_obs.copy()
        assert x_obs.shape[-1] == self.params_.nx
        self.params_.x_obs = x_obs.copy()

        delta_x = x_obs[-16:] - self.params_.x0[-16:]
        # print(f"max joint violation (min, max)=({min(delta_x)}, {max(delta_x)})")

    def get_observation(self):
        return self.params_.x_obs.copy()

    def load_warm_start(self, options:HighLevelOptions):
        """ Return true if load successful """
        try:
            ws_suffix = options.ddp_params['warmstart_suffix']
            xs_init = np.load(os.path.join(WARMSTART_DIR, f'xs_trival_{ws_suffix}.npy'))
            us_init = np.load(os.path.join(WARMSTART_DIR, f'us_trival_{ws_suffix}.npy'))
            xs_init = xs_init[:self.params_.T_ctrl+1]
            us_init = us_init[:self.params_.T_ctrl]
            assert xs_init.shape[-1] == self.params_.nx
            assert us_init.shape[-1] == self.params_.nu
            self.params_.x_init = xs_init.copy()
            self.params_.u_init = us_init.copy()
            return True
        except:
            print(f'Warm start solution not found: {ws_suffix}!')
            return False

    def get_warm_start(self):
        return self.params_.x_init, self.params_.u_init

    def get_solution(self):
        return self.xs_, self.us_
    
    def get_observation(self):
        return self.params_.x_obs
    
    def x_next(self):
        """
            Get x0 on the predicted trajectory
        """
        return self.xs_[0].copy()
    
    def xa_next(self):
        """
            Get x0 on the predicted trajectory
        """
        return self.xs_[0, -16:].copy()

    def get_contact_tracking_reference(self):
        def apply_hard_threshold(vec, thres=0.1):
            vec = np.minimum(vec, thres)
            return
        
        def apply_soft_threshold(vec, k=0.04, thres=2):
            L = 2 * thres
            vec = L / (1 + np.exp(-k*vec)) - L / 2
            return vec
        
        def apply_scale(vec, scale=0.04):
            vec = scale * vec
            return vec

        def threshold_force(force, method, params):
            magnitude = np.linalg.norm(force, axis=-1, keepdims=True)
            direction = force / np.maximum(magnitude, 1e-6)
            
            if method == 'soft_thres':
                magnitude = apply_soft_threshold(magnitude, params[0], params[1])
            elif method == 'scale':
                magnitude = apply_scale(magnitude, params[0])

            return magnitude * direction
        
        def scale_force(force, scale=1.0):
            return scale * force
        
        p_ACa_A = self.p_ACa_A_.copy()
        f_BA_W = self.f_BA_W_.copy()
        f_BA_W = threshold_force(
            f_BA_W,
            self.high_level_options.force_thres_method,
            self.high_level_options.force_thres_params)
        f_BA_W = scale_force(f_BA_W, self.force_scale)
        n_obj_W = self.n_obj_W_.copy()
        return p_ACa_A, f_BA_W, n_obj_W

    def update_reference(self):
        xu_now = np.atleast_1d(self.params_.x_obs[self.params_.q_u_indices])
        mutable_dims = np.abs(np.atleast_1d(self.params_.dxu)) > 0
        xu_mutable_now = xu_now[mutable_dims]
        
        dxu = np.atleast_1d(self.params_.dxu).copy()
        if self.params_.use_target_xu:
            dxu[:] = np.clip((self.params_.target_xu - xu_now) / self.params_.T_ctrl, -np.abs(dxu), np.abs(dxu))

        xu_ref = np.atleast_1d(self.params_.target_xu)[np.newaxis, :].repeat(self.params_.T_ctrl+1, axis=0)
        xu_ref[:, mutable_dims] = xu_mutable_now + np.arange(1, self.params_.T_ctrl+2)[:, np.newaxis] * dxu[mutable_dims]

        self.params_.x_ref[:, self.params_.q_u_indices] = xu_ref

        # handle so3
        q_rot_indices = self.params_.q_rot_indices
        if q_rot_indices is not None:
            so3_now = self.params_.x_obs[q_rot_indices] # quat (wxyz)
            so3_ref = self.ref_gen_so3.generate_reference_from_x0(so3_now)  # quat (wxyz)

            # q and -q are equal (revert q since we use ||q-qref|| as cost)
            for i in range(so3_ref.shape[0]):
                if np.dot(so3_ref[i], so3_now) < 0:
                    so3_ref[i] = -so3_ref[i]

            # normalize
            so3_ref = so3_ref / np.linalg.norm(so3_ref, axis=-1, keepdims=True)

            self.params_.x_ref[:, q_rot_indices] = so3_ref

    def run_mpc(self):
        """
            Run one step MPC, and set warm start
        """
        self.update_reference()
        # print("x0(x_obs): ", self.params_.x_obs)
        # print("xs0: ", self.params_.x_init[0])
        xs, us = ddp_runner(
            self.params_,
            self.q_sims, self.sim_params,
            self.state, self.actuation,
            maxiter=2
        )
        self.xs_ = xs.copy()
        self.us_ = us.copy()

        self.p_ACa_A_[:] = self.params_.contact_response.p_ACa_A.copy()
        self.f_BA_W_[:] = self.params_.contact_response.f_BA_W.copy()
        self.n_obj_W_[:] = self.params_.contact_response.n_obj_W.copy()

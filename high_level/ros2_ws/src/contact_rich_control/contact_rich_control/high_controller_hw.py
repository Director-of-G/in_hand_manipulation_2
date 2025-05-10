"""
This file is modified from drake_playground/high_controller.py,
which runs in simulation.
This file is designed to be integrated into a ROS node for
hardware evaluations.
"""
import sys
from std_msgs.msg import Bool, Float32MultiArray
from common_msgs.msg import MeshcatVis

from contact_rich_control.common_hw import *
sys.path.append(DDP_SOLVER_DIR)

from optimizer.inhand_ddp_traj_optimizer_zrot import AllegroDDPTrajOptimizer as TrajOptimizer
from common.common_drake import *
from common.common_ddp import convert_quat_to_rpy


class AllegroHighLevelControllerHw(object):
    def __init__(self, options:HighLevelOptions):
        self.time_step_ = (1 / options.high_level_frequency)
        self.nq_ = options.nq
        self.nv_ = options.nv
        self.nu_ = options.nu
        self.nc_ = options.nc
        self.q_u_indices_ = options.q_u_indices
        self.q_rot_indices_ = options.q_rot_indices

        # DDP trajectory optimizer
        self.optimizer_ = TrajOptimizer(options)

        self.num_steps_ = self.optimizer().params().T_ctrl
        self.ddp_execution_scale = options.ddp_execution_scale

        # warm-start
        self.initialize_warmstart()
        
        self.total_mpc_iters = 0

    def initialize_warmstart(self):
        warm_start_solution = OptimizerSolution()
        x_init_, u_init_ = self.optimizer_.get_warm_start()
        warm_start_solution.q = x_init_
        warm_start_solution.u = u_init_
        warm_start_solution.w = np.zeros((self.num_steps_, self.nc_*6))
        warm_start_solution.n_obj = np.zeros((self.num_steps_, self.nc_*3))
        self.warm_start_solution_ = warm_start_solution
        initial_guess_traj = StoredTrajectory()
        self.StoreOptimizerSolution(warm_start_solution, 0.0, initial_guess_traj)
        self.stored_trajectory_ = initial_guess_traj

    def reset(self):
        self.optimizer_.reset()
        self.initialize_warmstart()
        self.total_mpc_iters = 0

    def optimizer(self):
        return self.optimizer_

    def warm_start_solution(self):
        return self.warm_start_solution_

    def joint_names(self):
        return self.optimizer().joint_names()
    
    def x0(self):
        return self.optimizer().x0()
    
    def xa0(self):
        return self.optimizer().xa0()
    
    def x_next(self):
        return self.optimizer().x_next()
    
    def xa_next(self):
        return self.optimizer().xa_next()
    
    def x_obs(self):
        return self.optimizer().get_observation()

    def x_goal(self):
        return self.optimizer().x_goal()
    
    def set_so3_target(self, quat_target):
        self.optimizer_.set_so3_target(quat_target)
    
    def get_visualize_msg(self):
        msg = MeshcatVis()
        msg.vis_trajectory = Bool(data=False)
        
        # xs, _ = self.optimizer_.get_solution()
        xs = self.optimizer_.get_observation()
        p_W, f_W, _ = self.optimizer_.get_contact_tracking_reference()

        # system_state = xs[0].flatten()
        system_state = xs.flatten()
        if self.q_rot_indices_ is not None:
            system_state_rpy = np.zeros(self.nv_)
            system_state_rpy[-16:] = system_state[-16:]
            system_state_rpy[:-16] = convert_quat_to_rpy(system_state[self.q_rot_indices_])
            system_state = system_state_rpy.copy()

        system_goal_state = self.optimizer().x_goal().flatten()
        if self.q_rot_indices_ is not None:
            system_goal_state_rpy = np.zeros(self.nv_)
            system_goal_state_rpy[-16:] = system_goal_state[-16:]
            system_goal_state_rpy[:-16] = convert_quat_to_rpy(system_goal_state[self.q_rot_indices_])
            system_goal_state = system_goal_state_rpy.copy()

        msg.q.data = system_state.tolist()
        msg.q_goal.data = system_goal_state.tolist()
        msg.p_world.data = p_W[0].flatten().tolist()
        msg.f_world.data = f_W[0, :, :3].flatten().tolist()
        return msg
    
    def get_visualization_traj_msg(self, q_traj):
        """ Convert quat to rpy (if needed) """
        msg = MeshcatVis()
        msg.vis_trajectory = Bool(data=True)

        if self.q_rot_indices_ is not None:
            q_traj = np.array(q_traj)
            for i in range(q_traj.shape[0]):
                old_q = q_traj[i]
                new_q = np.zeros(self.nv_)
                new_q[-16:] = old_q[-16:]
                new_q[:-16] = convert_quat_to_rpy(old_q[self.q_rot_indices_])
                msg.q_traj.append(
                    Float32MultiArray(data=new_q.tolist())
                )

        return msg

    def SolveMpc(self, ros_time, state):
        """
            :param ros_time: the time that state is acquired (since controller starts)
        """
        # SET OBS
        current_time = ros_time
        # print("real time: ", current_time)
        assert state.shape[0] == self.nq_, f"state dimension mismatch: {state.shape[0]} vs {self.nq_}"
        state = np.concatenate((state, np.zeros(self.nv_)))
        x_now = state.copy()
        q_now = x_now[:self.nq_]
        # v_now = x_now[self.nq_:]
        self.optimizer_.set_observation(q_now)

        # INITIAL GUESS
        q_guess = np.zeros((self.num_steps_+1, self.nq_))
        u_guess = np.zeros((self.num_steps_, self.nu_))
        stored_trajectory = self.stored_trajectory_
        self.UpdateInitialGuess(stored_trajectory, current_time, q_guess, u_guess)
        q_guess[0] = q_now.copy()
        # u_guess[0] = np.zeros(self.nu_)

        # WARM START
        self.optimizer_.set_warm_start(q_guess, u_guess)

        # TRAJOPT
        self.optimizer_.run_mpc()

        # optimizer solution (xs has [xu;xa] format)
        xs_, us_ = self.optimizer_.get_solution()
        xs_a_ = np.delete(xs_, self.q_u_indices_, axis=-1)    # get only actuated joints
        p_ACa_A, f_BA_W, n_obj_W = self.optimizer_.get_contact_tracking_reference()
        
        solution_ = OptimizerSolution()
        solution_.q = xs_
        solution_.u = us_
        solution_.p = p_ACa_A
        solution_.w = f_BA_W    # B is assumed to be fingertip, use the applied force (from fingertip to object)
        solution_.n_obj = n_obj_W
        self.StoreOptimizerSolution(solution_, current_time, stored_trajectory)

        # store only the actuated hand dofs
        solution_.q = xs_a_[:, -16:]
        solution_.u = us_[:, -16:]
        stored_trajectory_actuated = StoredTrajectory()
        self.StoreOptimizerSolution(solution_, current_time, stored_trajectory_actuated)

        self.total_mpc_iters = self.total_mpc_iters + 1

        return stored_trajectory_actuated

    def get_fake_object_state(self, current_time):
        dt = current_time - self.stored_trajectory_.start_time
        return self.stored_trajectory_.q.value(dt).flatten()[self.q_u_indices_]

    def UpdateInitialGuess(
        self,
        stored_trajectory:StoredTrajectory,
        current_time,
        q_guess,
        u_guess
    ):
        """
            TODO(yongpeng): due to previous bad design, stored_trajectory.start_time
            is the actual time of birth. But stored_trajectort.q/u/w all start from 0!
            Plan to fix this in the future
        """
        # delta_time = current_time - stored_trajectory.start_time
        # print("dt: ", delta_time)

        # # first iter, use warm start
        # if self.total_mpc_iters == 0:
        #     start_time = 0.0
        # else:
        #     if self.ddp_execution_scale == 1.0:
        #         start_time = current_time - stored_trajectory.start_time
        #     else:
        #         start_time = self.time_step_ * self.ddp_execution_scale
        start_time = 0.0
        for i in range(self.num_steps_+1):
            t = start_time + i * self.time_step_
            q_guess[i] = stored_trajectory.q.value(t).flatten()
            if i < self.num_steps_:
                u_guess[i] = stored_trajectory.u.value(t).flatten()

    def StoreOptimizerSolution(
        self,
        solution:OptimizerSolution,
        start_time,
        stored_trajectory:StoredTrajectory
    ):
        time_steps = self.time_step_*np.arange(0, self.num_steps_+1)

        q_knots = solution.q.copy()
        u_knots = solution.u.copy()
        w_knots = solution.w.copy()
        n_obj_knots = solution.n_obj.copy()

        if len(w_knots.shape) == 3:
            w_knots = w_knots.reshape(w_knots.shape[0], -1)

        if len(n_obj_knots.shape) == 3:
            n_obj_knots = n_obj_knots.reshape(n_obj_knots.shape[0], -1)

        u_knots = np.concatenate((u_knots, u_knots[-1].reshape(1, -1)), axis=0)
        w_knots = np.concatenate((w_knots, w_knots[-1].reshape(1, -1)), axis=0)
        n_obj_knots = np.concatenate((n_obj_knots, n_obj_knots[-1].reshape(1, -1)), axis=0)

        stored_trajectory.start_time = start_time

        stored_trajectory.q = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            time_steps,
            q_knots.T
        )
        
        stored_trajectory.u = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            time_steps,
            u_knots.T
        )
        
        stored_trajectory.w = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            time_steps,
            w_knots.T
        )

        stored_trajectory.n_object = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            time_steps,
            n_obj_knots.T
        )

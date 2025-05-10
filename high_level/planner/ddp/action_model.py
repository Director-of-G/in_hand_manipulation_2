import numpy as np
import crocoddyl

from common.common_ddp import (
    convert_quat_to_rpy,
    convert_rpy_to_quat,
    CalcNW2Qdot,
    CalcNQdot2W,
    delete_listb_from_lista,
)


class QuasistaticActionModel(crocoddyl.ActionModelAbstract):
    def __init__(self, q_sim, sim_params, state, actuation, defaultCostModel, extras=None):
        crocoddyl.ActionModelAbstract.__init__(self, state, actuation.nu, defaultCostModel.nr)
        
        self.default_cost = defaultCostModel
        self.default_cost_data = self.default_cost.createData(crocoddyl.DataCollectorAbstract())

        self.q_sim = q_sim
        self.sim_params = sim_params

        self.nx_cqdc_ = len(self.q_sim.get_mbp_positions_as_vec())
        self.nu_ = q_sim.num_actuated_dofs()
        self.q_a_indices_ = q_sim.get_q_a_indices_into_q()
        self.q_u_indices_ = q_sim.get_q_u_indices_into_q()
        
        if extras is not None:
            self.num_wrist_joints = extras.n_wrist
            self.q_wrist = np.zeros(self.num_wrist_joints,)
        else:
            self.num_wrist_joints = 0

        # if x includes orientation dims
        if extras is not None and extras.has_ori:
            self.has_ori = True
            self.nx_ddp_ = extras.nx
            self.ori_slc_cqdc = slice(extras.ori_start, extras.ori_start+3)     # CQDC represents ori as RPY
            self.ori_slc_ddp = slice(extras.ori_start, extras.ori_start+4)    # DDP represents ori as Quat
            self.q_u_indices_ = extras.q_u_indices
            self.q_a_indices_ = delete_listb_from_lista(list(range(self.nx_ddp_)), self.q_u_indices_)
        else:
            self.has_ori = False
            self.nx_ddp_ = self.nx_cqdc_

    def set_wrist_pose(self, q_wrist):
        if self.num_wrist_joints == 0:
            return
        assert q_wrist is not None
        if not isinstance(q_wrist, np.ndarray):
            q_wrist = np.atleast_1d(q_wrist)
        assert q_wrist.shape[0] == self.num_wrist_joints
        self.q_wrist = q_wrist.copy()

    def calc(self, data, x, u=None):
        if u is None:
            u = np.zeros(self.nu_)
        u_incre = u.copy()

        u = u + x[self.q_a_indices_]
        if self.num_wrist_joints > 0:
            u[:self.num_wrist_joints] = self.q_wrist.copy()

        # call q_sim's forward dynamics
        if self.has_ori:
            # quat --> rpy
            x_rpy = np.zeros(self.nx_cqdc_)
            x_rpy[self.ori_slc_cqdc] = convert_quat_to_rpy(x[self.ori_slc_ddp])
            x_rpy[:self.ori_slc_cqdc.start] = x[:self.ori_slc_ddp.start]
            x_rpy[self.ori_slc_cqdc.stop:] = x[self.ori_slc_ddp.stop:]

            # step
            xnext_rpy = self.q_sim.calc_dynamics_forward(x_rpy, u, self.sim_params)
            
            # rpy --> quat
            data.xnext = np.zeros_like(x)
            data.xnext[self.ori_slc_ddp] = convert_rpy_to_quat(xnext_rpy[self.ori_slc_cqdc])
            data.xnext[:self.ori_slc_ddp.start] = xnext_rpy[:self.ori_slc_cqdc.start]
            data.xnext[self.ori_slc_ddp.stop:] = xnext_rpy[self.ori_slc_cqdc.stop:]
        else:
            data.xnext = self.q_sim.calc_dynamics_forward(x, u, self.sim_params)

        # compute default cost
        self.default_cost.calc(self.default_cost_data, x, u_incre)
        default_cost_value = sum([c.cost for c in self.default_cost_data.costs.todict().values()])

        data.cost = default_cost_value

    def calcDiff(self, data, x, u=None):
        if u is None:
            u = np.zeros(self.nu_)
        u_incre = u.copy()

        # call q_sim's backward dynamics
        self.q_sim.calc_dynamics_backward(self.sim_params)

        partial_u_partial_q = np.zeros((self.nu_, self.nx_ddp_))
        partial_u_partial_q[:, self.q_a_indices_] = np.eye(self.nu_)

        # data.Fx = self.q_sim.get_Dq_nextDq() + self.q_sim.get_Dq_nextDqa_cmd() @ partial_u_partial_q
        # data.Fu = self.q_sim.get_Dq_nextDqa_cmd()

        Dq_nextDq = self.q_sim.get_Dq_nextDq().copy()
        Dq_nextDqa_cmd = self.q_sim.get_Dq_nextDqa_cmd().copy()


        if self.has_ori:
            ndofs_before_ori = self.ori_slc_ddp.start
            n_dofs_after_ori = self.nx_cqdc_ - self.ori_slc_cqdc.stop

            # calc omega <--> qdot projection matrices
            left_mat = np.zeros((self.nx_ddp_, self.nx_cqdc_))
            right_mat = np.zeros((self.nx_cqdc_, self.nx_ddp_))

            left_mat[self.ori_slc_ddp, self.ori_slc_cqdc] = CalcNW2Qdot(x[self.ori_slc_ddp])
            left_mat[:self.ori_slc_ddp.start, :self.ori_slc_cqdc.start] = np.eye(ndofs_before_ori)
            left_mat[self.ori_slc_ddp.stop:, self.ori_slc_cqdc.stop:] = np.eye(n_dofs_after_ori)

            right_mat[self.ori_slc_cqdc, self.ori_slc_ddp] = CalcNQdot2W(x[self.ori_slc_ddp])
            right_mat[:self.ori_slc_cqdc.start, :self.ori_slc_ddp.start] = np.eye(ndofs_before_ori)
            right_mat[self.ori_slc_cqdc.stop:, self.ori_slc_ddp.stop:] = np.eye(n_dofs_after_ori)

            Dq_nextDq_rpy = self.q_sim.get_Dq_nextDq()      # rpy, (nx_qsim, nx_qsim)
            Dq_nextDqa_cmd_rpy = self.q_sim.get_Dq_nextDqa_cmd()    # rpy, (nx_qsim, n_u)

            Dq_nextDq = left_mat @ Dq_nextDq_rpy @ right_mat
            Dq_nextDqa_cmd = left_mat @ Dq_nextDqa_cmd_rpy

        if self.num_wrist_joints > 0:
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

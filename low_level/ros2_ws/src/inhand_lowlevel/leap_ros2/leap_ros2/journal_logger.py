"""
    The logger class for IROS24 -> journal experiments.
"""
import time
import numpy as np
import os
import pickle
from pydrake.all import PiecewisePolynomial

class MyLogger(object):
    def __init__(self) -> None:
        self.initialize_logger()

    def initialize_logger(self):
        self.start_time = 0.0
        self.is_initialized = False

        self.q_ref = []         # reference joint pos at each t_ref sampling step
        self.q_ref_look_ahead = []
        self.t_knots_ref = []
        self.t_ref = {"recv_state": [], "mpc_begin": [], "mpc_finish": [], "recv_traj": []}

        self.q_read = []
        self.t_read = []

        self.q_cmd = []
        self.t_cmd = []

        self.obj_state = []
        self.t_obj_state = []

        self.fext_read = [] # sensed contact force
        self.n_read = []     # (interpolated) object normal
        self.t_fext = []

        self.fdes = []      # (desired) contact force
        self.ndes = []      # (desired) object normal
        self.t_fdes = []

    def reset(self):
        self.initialize_logger()

    def set_start_time(self, time:float):
        if self.is_initialized:
            return
        self.is_initialized = True
        self.start_time = time

    def update_q_ref(self, q_ref, q_ref_look_ahead:PiecewisePolynomial.CubicWithContinuousSecondDerivatives, time, remapping=None):
        if not self.is_initialized:
            return
        
        t_recv_state, t_mpc_begin, t_mpc_end, t_recv_traj = time
        self.t_ref['recv_state'].append(t_recv_state - self.start_time)
        self.t_ref['mpc_begin'].append(t_mpc_begin - self.start_time)
        self.t_ref['mpc_finish'].append(t_mpc_end - self.start_time)
        self.t_ref['recv_traj'].append(t_recv_traj - self.start_time)
        
        t_knots = q_ref_look_ahead.get_segment_times()
        t_knots = np.linspace(t_knots[0], t_knots[-1], 100)
        q_knots = q_ref_look_ahead.vector_values(t_knots).T
        t_knots = np.array(t_knots) - self.start_time
        q_ref = np.array(q_ref)

        if remapping:
            q_knots = q_knots[:, remapping]
            q_ref = q_ref[remapping]
        
        self.t_knots_ref.append(t_knots)
        self.q_ref.append(q_ref)
        self.q_ref_look_ahead.append(np.array(q_knots))

    def update_q_cmd(self, q_cmd:np.ndarray, time:float, remapping=None):
        if not self.is_initialized:
            return
        if remapping:
            q_cmd = q_cmd[remapping]
        self.q_cmd.append(q_cmd)
        self.t_cmd.append(time - self.start_time)

    def update_q_read(self, q_read:np.ndarray, time:float, remapping=None):
        if not self.is_initialized:
            return
        if remapping:
            q_read = q_read[remapping]
        self.q_read.append(q_read)
        self.t_read.append(time - self.start_time)

    def update_obj_state(self, obj_state:np.ndarray, time:float):
        if not self.is_initialized:
            return
        self.obj_state.append(obj_state.copy())
        self.t_obj_state.append(time - self.start_time)

    def update_f_ext_and_norm(self, f_ext:np.ndarray, n_read:np.ndarray, time:float):
        if not self.is_initialized:
            return
        self.fext_read.append(f_ext.copy())
        self.n_read.append(n_read.copy())
        self.t_fext.append(time - self.start_time)

    def update_f_des_and_norm(self, f_des:np.ndarray, n_des:np.ndarray, time:float):
        if not self.is_initialized:
            return
        self.fdes.append(f_des)
        self.ndes.append(n_des)
        self.t_fdes.append(time - self.start_time)

    def log_data(self, rootdir=''):
        data = {
            'start_time': self.start_time,
            't_ref': self.t_ref,
            'q_ref': self.q_ref,
            'q_ref_look_ahead': self.q_ref_look_ahead,
            't_knots_ref': self.t_knots_ref,
            't_cmd': self.t_cmd,
            'q_cmd': self.q_cmd,
            't_read': self.t_read,
            'q_read': self.q_read,
            't_obj_state': self.t_obj_state,
            'obj_state': self.obj_state,
            't_fext': self.t_fext,
            'fext_read': self.fext_read,
            'n_read': self.n_read,
            't_fdes': self.t_fdes,
            'fdes': self.fdes,
            'ndes': self.ndes
        }
        # create a folder of today if not exist
        today = time.strftime('%Y-%m-%d', time.localtime())
        timestamp = time.strftime('%H-%M-%S', time.localtime())
        os.mkdir(os.path.join(rootdir, today)) if not os.path.exists(os.path.join(rootdir, today)) else None
        filepath = os.path.join(rootdir, today, f'logs_at_{timestamp}.pkl')
        pickle.dump(data, open(filepath, 'wb'))

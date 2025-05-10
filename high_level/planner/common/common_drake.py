from enum import Enum
import numpy as np
from pydrake.all import (
    PiecewisePolynomial,
    RigidTransform,
)


class OutputMode(Enum):
    kNone = 0
    kFextFeedbackPy = 1
    kFextFeedbackCpp = 2
    kFextFeedforwardTauCpp = 3
    kFextFeedforwardPosCpp = 4

class InHandTask(Enum):
    kSphereZrot = 0
    kValveZrot = 1
    kSphereRPY = 2
    kSphereRolling = 3

class HandModel(Enum):
    kAllegro = 0
    kLeap = 1

class FeedforwardMode(Enum):
    kNone = 0
    kTau = 1
    kPos = 2

CPP_CONTROL_MODES = [OutputMode.kFextFeedbackCpp, OutputMode.kFextFeedforwardTauCpp, OutputMode.kFextFeedforwardPosCpp]
FEED_BACK_MODES = [OutputMode.kFextFeedbackPy, OutputMode.kFextFeedbackCpp]
FEED_FORWARD_MODES = [OutputMode.kFextFeedforwardTauCpp, OutputMode.kFextFeedforwardPosCpp]

class ContactTrackReferenceRequest(object):
    def __init__(self):
        self.object_link_name = ''
        self.finger_link_names = []
        self.finger_to_geom_name_map = {}
        self.finger_to_index_map = {}
        self.external_contact_geom_names = []

    def n_c(self):
        return len(self.finger_to_geom_name_map)

    def parse_finger_order(self, geom_names):
        for key in list(self.finger_to_geom_name_map.keys()):
            try:
                self.finger_to_index_map[key] = \
                    geom_names.index(self.finger_to_geom_name_map[key])
            except:
                # lost contact
                self.finger_to_index_map[key] = -1

    def get_ranked_contact_data(self, data):
        ranked_data = np.zeros((0, data.shape[1]))
        for key, _ in self.finger_to_geom_name_map.items():
            index_ = self.finger_to_index_map[key]
            if index_ == -1:
                # lost contact
                ranked_data = np.concatenate(
                    (ranked_data, np.zeros((1, data.shape[1]))), axis=0
                )
            else:
                ranked_data = np.concatenate(
                    (ranked_data, data[index_].reshape(1, -1)), axis=0
                )

        return ranked_data


class ContactTrackReferenceResponse(object):
    def __init__(self):
        self.p_ACa_A = np.zeros((0, 4, 3))
        self.f_BA_W = np.zeros((0, 4, 3))
        self.n_obj_W = np.zeros((0, 4, 3))


class TrajOptParams(object):
    """
        Tunable parameters of (DDP) trajopt
    """
    def __init__(self) -> None:
        self.w_u = 1e1              # weight of running unactuated dims
        self.w_a = (1e-2, 1e0)      # weight of running actuated dims (<3/4H, >=3/4H)
        self.w_u_T = 1e2            # weight of terminal unactuated dims
        self.w_a_T = 1e3            # weight of terminal actuated dims
        self.w_xreg = 1             # weight of state regulation
        self.w_ureg = 0.04          # weight of control regulation

        self.dyaw = -1.0 * np.pi / 200
        self.u_lb = -0.05
        self.u_ub = 0.05

        # xa_reg and xa_reg2 are used for long-horizon rotation
        self.xa_reg = np.zeros(16,)
        self.xa_reg2 = np.zeros(16,)
        self.T_xa_reg_interp = 100


class LowLevelMPCProblemData(object):
    def __init__(self, n_c, n_qa, h, horizon):
        self.discrete_step = h
        self.horizon_length = horizon

        self.n_c = n_c
        self.n_qa = n_qa
        self.n_x = 2*self.n_qa+3*self.n_c
        self.n_u = self.n_qa

        self.J_list = np.zeros((self.n_c, 3, self.n_qa))
        self.Kbar_list = np.zeros((self.n_c, 3, 3))

        self.x0 = np.zeros((self.n_x, 1))
        self.x_ref = np.zeros((self.n_x, self.horizon_length+1))
        self.u_ref = np.zeros((self.n_u, self.horizon_length))
        self.fext_ff = np.zeros((3*self.n_c, self.horizon_length))

    def set_x_ref(self, x_ref):
        assert x_ref.shape == (self.n_x, self.horizon_length+1)
        self.x_ref = x_ref.copy()

    def set_x0(self, x0):
        if len(x0.shape) == 1:
            x0 = x0.reshape(-1, 1)
        assert x0.shape == (self.n_x, 1)
        self.x0 = x0.copy()

    def set_J(self, J_list):
        assert J_list.shape == (self.n_c, 3, self.n_qa)
        self.J_list = J_list.copy()

    def set_Kbar(self, Kbar_list):
        assert Kbar_list.shape == ((self.n_c, 3, 3))
        self.Kbar_list = Kbar_list.copy()

    def set_Ac(self, A):
        assert A.shape == (self.n_x, self.n_x)
        self.Ac = A.copy()

    def set_Bc(self, B):
        assert B.shape == (self.n_x, self.n_u)
        self.Bc = B.copy()

    def set_Ad(self, A):
        assert A.shape == (self.n_x, self.n_x)
        self.Ad = A.copy()

    def set_Bd(self, B):
        assert B.shape == (self.n_x, self.n_u)
        self.Bd = B.copy()

    def set_Q(self, Q):
        assert Q.shape == (self.n_x, self.n_x)
        self.Q = Q.copy()

    def set_R(self, R):
        assert R.shape == (self.n_u, self.n_u)
        self.R = R.copy()

    def set_delR(self, delR):
        assert delR.shape == (self.n_u, self.n_u)
        self.delR = delR


class DrakeSimulatorOptions(object):
    def __init__(self):
        # ----- simulation settings -----
        self.sim_time_step = 0.001              # timestep for simulation
        self.sim_realtime_rate = 1.0
        self.sim_time = 3.0
        # self.gravity_field = [-6.17, 1.54, -7.72]         # this fails
        self.gravity_field = [0.0, 0.0, -9.81]
        self.is_3d_floating = False             # if the sphere (manipuland) is free-floating or rotation-only

        # ----- hand settings -----
        self.hand_offset = {
            "trans": [0., 0., 0.],
            "rpy": [0., -np.pi/2, 0.]
        }
        self.gravity_compensation = True
        
        # ----- manipuland settings -----
        self.object_offset = {
            "trans": [-0.06, 0.0, 0.072],
            "rpy": [0., 0., 0.]
        }
        self.sphere_radius = 0.06
        self.sphere_mass = 0.01
        
        # ----- hardware-level PD controller -----
        self.Kp = np.concatenate(
            (0.5*np.ones(16,), np.zeros(7,))
        )
        self.Kd = np.concatenate(
            (1e-3*np.ones(16,), np.zeros(6,))
        )

        # ----- problem definition -----
        self.inhand_task = InHandTask.kSphereZrot
        self.q_init = np.array([
            # 0.4, 0.95, 0.9, 1.0,
            # 0.0, 0.45, 1.0, 1.0,
            # -0.25, 0.85, 1.0, 1.0,
            # 0.45, 1.85, 1., 1.2,
            # 1, 0, 0, 0, -0.065, 0.005, 0.07
            0.2, 0.95, 1.0, 1.0,                 # index finger
            0.0, 0.6, 1.0, 1.0,                 # middle finger
            -0.2, 0.95, 1.0, 1.0,                # ring finger
            0.5, 1.85, 1.0, 1.0,                 # thumb
            1, 0, 0, 0, -0.06, 0.0, 0.07     # ball
        ])
        self.v_init = np.array([
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ])
        self.q_a_indices = [0, 1, 2, 3, 4, 5, 6, 7,
                            8, 9, 10, 11, 12, 13, 14, 15]
        self.q_u_indices = [16, 17, 18, 19, 20, 21, 22]
        
        self.finger_to_link_map = {
            "index": "link_3",
            "middle": "link_7",
            "ring": "link11",
            "thumb": "link_15",
        }

        # ----- contact modeling -----
        self.contact_threshold = 0.1            # distance threshold for contact
        self.sigma = 0.001
        self.k = 500
        self.vd = 0.1
        self.vs = 0.1
        self.mu = 1.0

        # ----- low level reference -----
        self.contact_request = ContactTrackReferenceRequest()
        self.contact_request.finger_to_geom_name_map = {
            "thumb": "allegro_hand_right::link_15_tip_collision_2",
            "index": "allegro_hand_right::link_3_tip_collision_1",
            "middle": "allegro_hand_right::link_7_tip_collision_1",
            "ring": "allegro_hand_right::link_11_tip_collision_1"
        }
        self.contact_request.external_contact_geom_names = [
            "table::collision"
        ]

        # ----- low level mpc setings -----
        self.low_level_time_step = 0.01         # discretization time step for low level controller
        self.low_level_frequency = 100
        self.low_level_mpc_horizon = 5          # low level mpc horizon
        self.weight_q = 1e-3                       # regulation weight of joint positions
        self.weight_dq = 0.5                      # regulation weight of joint velocities
        self.weight_f = 1.0                       # regulation weight of contact forces
        self.weight_u = 0.5                       # regularization weight of control input
        self.weight_du = 0.0                      # regularization weight of input rate (0.5)
        self.k_gain_tau_feedforward = 1 / 50      # gain for feedforward control
        self.k_wref_decay = 1                    # decay ratio of the reference wrench (wrench=wrench/ratio)
        self.low_level_output_mode = OutputMode.kFextFeedbackCpp
        self.enable_multi_contact = False

        # ----- high level mpc settings -----
        self.use_replay_traj = False            # whether to replay the saved trajectory
        self.normalize_quaternions = True
        self.high_level_time_step = 0.05         # discretization time step for high level controller
        self.high_level_frequency = 20
        self.T_trajopt = 20                     # time horizon for trajopt
        self.T_mpc = 10                         # high level mpc horizon
        self.dyaw = 5 * np.pi / 200
        self.ddp_execution_scale = 1.0          # traj scaling for DDP MPC (default: 1.0, no scaling)
        self.trajopt_params = TrajOptParams()

        # ----- visualization settings -----
        self.h_logger = 0
        self.h_logger_high = 0
        self.viz_high_level_output_time_steps = 5   # viz how many high level discrete steps
        self.viz_high_level_output_per_iters = 2    # viz high level output after how many iters
        self.visualize_contact_reference = True # plot the contact points and forces (references)
        self.viz_jpos = False

        # ----- apply external force -----
        self.enable_external_force = False

        # ----- auxiliary options -----
        self.nq = 17                        # number of joint positions
        self.nv = 17                        # number of joint velocities
        self.nu = 16                        # number of actuated joints
        self.hand_model = HandModel.kAllegro    # hand model

    def __str__(self):
        attributes = vars(self)
        return '\n'.join(f'{key}: {value}' for key, value in attributes.items())


class HighLevelOptions(object):
    """ Options to initialize the high level trajectory optimizer """
    def __init__(self) -> None:
        self.model_url = ''
        self.model_url_sc = ''
        self.mesh_url_sc = ''
        self.hand_model = HandModel.kLeap
        self.finger_to_geom_name_map = None
        self.finger_to_joint_map = None
        self.finger_to_x0_map = None
        self.finger_to_limits_map = None
        self.finger_to_xreg_map_list = []
        self.object_dofs_dict = None        # configure the low level state manager
        self.object_link = ''               # name of the object link (for meshcat visualize)
        self.nq = 17
        self.n_wrist_dofs = 0
        self.nv = 17
        self.nu = 16
        self.nc = 4

        self.q_init = None
        self.q_u_indices = None
        self.q_rot_indices = None           # pure rotation dimensions in q (SO(3))
        self.dxu = 0.0                      # xu step size for generating reference
        self.dxu_so3 = 0.0                  # xu step size for so3 (delta angle)
        self.init_xu_so3 = [1.0, 0.0, 0.0, 0.0]     # init xu for so3
        self.target_xu = 0.0                # xu target for generating reference
        self.target_xu_so3 = [1.0, 0.0, 0.0, 0.0]   # xu target for so3 (quaternion wxyz)
        self.random_target_xu_so3 = False
        self.use_target_xu = False
        self.u_lb = 0.0
        self.u_ub = 0.0

        self.T_mpc = 10
        self.high_level_frequency = 10
        self.ddp_execution_scale = 1.0
        self.force_threshold = 3.0
        self.desired_force_scale = 1.0
        self.force_thres_method = 'soft_thres'
        self.force_thres_params = [0.04, 2.0]
        
        self.ddp_params = None              # dict
        self.debug_mode = False
        self.visualize_in_meshcat = False


class StoredTrajectory(object):
    def __init__(self):
        self.start_time = -1.0
        self.q = PiecewisePolynomial()  # joint position
        self.u = PiecewisePolynomial()  # desired (delta) joint position
        self.w = PiecewisePolynomial()  # contact wrench
        # TODO(yongpeng): add contact point


class OptimizerSolution(object):
    def __init__(self):
        self.q = None       # joint position
        self.u = None       # desired (delta) joint position
        self.p = None       # contact point
        self.w = None       # contact wrench
        self.n_obj = None   # object normal


def normalize(v):
    return v / np.linalg.norm(v)

def create_frame_from_axis(p, Xx=None, Yx=None, Zx=None):
    if Xx is None:
        assert Yx is not None and Zx is not None
        Xx = np.cross(normalize(Yx), normalize(Zx))
    elif Yx is None:
        assert Xx is not None and Zx is not None
        Yx = np.cross(normalize(Zx), normalize(Xx))
    elif Zx is None:
        assert Xx is not None and Yx is not None
        Zx = np.cross(normalize(Xx), normalize(Yx))

    frame = RigidTransform(
        np.c_[Xx, Yx, Zx], p
    )

    return frame

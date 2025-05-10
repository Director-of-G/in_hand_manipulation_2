import re
from enum import Enum
import numpy as np
from scipy.spatial.transform import Rotation as SciR
import copy
import crocoddyl
import pinocchio as pin
from qsim.parser import QuasistaticParser
from qsim_cpp import GradientMode, ForwardDynamicsMode, DfDxMode
from common.common_drake import (
    ContactTrackReferenceRequest,
    ContactTrackReferenceResponse
)

from pydrake.all import (
    JointIndex,
    Quaternion,
)

from pydrake.math import (
    RollPitchYaw,
    RotationMatrix,
)

class BallJointType(Enum):
    kRPY = 0
    kQuat = 1

class DDPSolverParams(object):
    """ Options to configure the DDP solver """
    def __init__(self):
        # ------ load from YAML ------
        self.h = 0.1                                    # 0.025
        self.kappa = 100                                # smoothing
        self.kappa_exact = 10000                        # smoothing for exact rollout
        self.auto_diff = True                           # use autodiff to compute gradients
        self.nx = 17
        self.nu = 16
        self.n_wrist = 0                                # number of wrist joints
        self.q_u_indices = [0]
        self.q_rot_indices = None
        
        # ----- for action model -----
        self.has_ori = False
        self.ori_start = 0
        # ----------------------------
        
        self.q_wrist_indices = []                       # indices of wrist joints in x0

        self.T_trajopt = 40                             # trajopt horizon
        self.T_ctrl = 10                                # MPC horizon

        self.execution_scale = 0.5                      # 1.0, positive scalar, scale u0 when execution (i.e., rollout new x0)
        self.dxu = 5.0 * np.pi/200                      # desired object motion, represented as the increment of xu
        self.target_xu = 0.0
        self.use_target_xu = False
        self.preset_so3_target = None                   # can be used for journal experiments as given quat target
        self.u_lb = -0.08 * np.ones(self.nu)            # -0.05
        self.u_ub = 0.08 * np.ones(self.nu)             # 0.05

        self.contact_request = ContactTrackReferenceRequest()
        self.contact_response = ContactTrackReferenceResponse()
        
        # weights (mpc)
        self.w_a = [1, 1]                               # weights for finger regulation (before 0.75T; after 0.75T)
        self.w_u = 1
        self.w_u_so3 = 0
        self.w_aT = 1
        self.w_uT = 1
        self.w_uT_so3 = 0
        
        # weights (trajopt)
        self.TO_w_u = 1
        self.TO_w_a = 1
        self.TO_w_uT = 1
        self.TO_w_aT = 1
        self.TO_W_x = 1
        self.TO_W_u = 1
        self.TO_W_xT = 1

        # weights (cost sum)
        self.W_X = 1
        self.W_U = 1
        self.W_U_SO3 = 0
        self.W_SC = 0
        self.W_J = 0
        # ----------------------------

        # ------ set after qsim initialized ------
        self.x0 = np.zeros(17,)
        self.xa_reg = np.zeros(16,)
        self.target = np.zeros(17,)                     # target system state (if use_target_xu=True)
        self.target_TO = np.zeros(17,)                  # target state for trajopt
        # ----------------------------

        # ------ set at runtime ------
        self.x_obs = self.x0.copy()                     # observation
        self.x_ref = None                               # reference
        self.x_init = None                              # state warmstart
        self.u_init = None                              # control warmstart

        self.enable_j_cost = False                      # enable joint limits cost
        self.x_lb = None                                # state lower bound
        self.x_ub = None                                # state upper bound

        self.ddp_solver = crocoddyl.SolverBoxDDP        # SolverBoxDDP, SolverDDP
        self.ddp_verbose = False

        self.enable_sc_cost = False                     # enable self-collision cost
        self.models_sc = None                           # for selfCollide cost, (pin_model, geom_model)
        self.pair_index_sc = []                         # for selfCollide cost, indices of considered pairs
        self.state_slice_sc = None                      # for selfCollide cost, range of pin_model dofs in state

        self.visualize_while_control = True
        # ----------------------------

#####################
# Utility Functions #
#####################
        
# convert 2d array to list of 1d array
def convert_array_to_list(array_2d):
    return [array_2d[i, :] for i in range(array_2d.shape[0])]

def fill_jpos_arr_with_dic(model_j_names, j_names, jpos_dic):
    if isinstance(model_j_names, dict):
        model_j_names = model_j_names.values()
    jpos_arr = np.zeros(len(model_j_names),)
    for name, jvalues in jpos_dic.items(): # 'thumb', 'index', 'middle', 'ring'
        jnames = j_names[name]
        for jid, jval in zip(jnames, jvalues):
            idx = list(model_j_names).index(f"joint_{jid}")
            jpos_arr[idx] = jval

    return jpos_arr

def AddJointSlidersToMeshcat(meshcat, plant, apply_add=True):
    """
        :return sliders: ordered joint(slider) names in Drake
    """
    
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
            if apply_add:
                meshcat.AddSlider(
                    value=value,
                    min=lower_limit[slider_num],
                    max=upper_limit[slider_num],
                    step=resolution[slider_num],
                    name=description,
                )
            sliders[index] = description
            slider_num += 1

    return sliders

def parse_joint_names_from_plant(plant):
    """
        parse hand joint names from model
        this is used to set x0 and xa_reg afterwards
    """
    
    pattern = r'joint_([0-9]|1[0-5])'
    joint_names = []
    for i in range(plant.num_joints()):
        _name = plant.get_joint(JointIndex(i)).name()
        if re.match(pattern, _name) or _name == 'joint_wrist':
            joint_names.append(_name)
    return joint_names

def get_hand_joints_only(joint_names):
    """ get hand joint names, excluding wrist joints """
    pattern = r'joint_([0-9]|1[0-5])'
    hand_joint_names = []
    for name in joint_names:
        if re.match(pattern, name):
            hand_joint_names.append(name)
    return hand_joint_names

def allocate_resources_for_DDP(model_url, options:DDPSolverParams, num_envs=21):
    parser = QuasistaticParser(model_url)

    # qsims
    default_q_sims = []
    for i in range(num_envs):
        q_sim = parser.make_simulator_cpp()
        default_q_sims.append(q_sim)

    # sim params
    default_sim_params = copy.deepcopy(default_q_sims[0].get_sim_params())
    default_sim_params.gradient_mode = GradientMode.kAB
    if options.auto_diff:
        default_sim_params.gradient_dfdx_mode = DfDxMode.kAutoDiff
    else:
        default_sim_params.gradient_dfdx_mode = DfDxMode.kAnalyticWithFiniteDiff
    
    default_sim_params.h = options.h
    default_sim_params.log_barrier_weight = options.kappa
    default_sim_params.forward_mode = ForwardDynamicsMode.kLogIcecream
    default_sim_params.calc_contact_forces = True
    default_sim_params.use_free_solvers = True

    # quasi-static model definition
    state = crocoddyl.StateVector(options.nx)
    state.nq = state.nx; state.nv = 0       # quasi-static
    actuation = crocoddyl.ActuationModelAbstract(state, nu=options.nu)

    # set collision body names: this is used by autodiff
    finger_link_names = options.contact_request.finger_link_names
    object_link_name = options.contact_request.object_link_name
    for i in range(len(default_q_sims)):
        default_q_sims[i].set_collision_body_names(finger_link_names, object_link_name)

    return default_q_sims, default_sim_params, state, actuation


def load_pinocchio_model(sdf_url, mesh_url, collision_pairs=[]):
    """
    :param sdf_url: sdf model path
    :param mesh_url: mesh path to use only file name in sdf
    :param collision_pairs: collision pair indices
    """
    pin_model, _, collision_model, _ = pin.buildModelsFromSdf(filename=sdf_url, package_dirs=mesh_url)
    pair_index = []
    for pair in collision_pairs:
        pair_index.append(len(collision_model.collisionPairs))
        geom_id1 = collision_model.getGeometryId(pair[0])
        geom_id2 = collision_model.getGeometryId(pair[1])
        collision_model.addCollisionPair(pin.CollisionPair(geom_id1, geom_id2))
    return (pin_model, collision_model), pair_index


def load_joint_limits(joint_names, neutral_joint, joint_limits_dict):
    """
    :param joint_names: ordered joint names
    :param neutral_joint: the neutral position for all joints (x0 or xa_reg)
    :param joint_limits: dict of joint limits
    """
    njoints = len(joint_names)
    j_lb, j_ub = -np.inf*np.ones(njoints,), np.inf*np.ones(njoints,)
    if joint_limits_dict is not None:
        if 'delta' in joint_limits_dict:
            delta_lb, delta_ub = joint_limits_dict['delta']
            assert len(neutral_joint) == njoints
            j_lb[:] = neutral_joint + delta_lb
            j_ub[:] = neutral_joint + delta_ub
        else:
            for jid, jlimits in joint_limits_dict.items():
                idx = joint_names.index(f"joint_{jid}")
                j_lb[idx] = jlimits[0]
                j_ub[idx] = jlimits[1]

    return j_lb, j_ub


##################################
#           Math Utils           #
##################################

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

def convert_rpy_to_matrix(rpy):
    matrix = RotationMatrix(RollPitchYaw(rpy)).matrix()
    return matrix

def convert_matrix_to_rpy(matrix):
    rpy = RotationMatrix(matrix).ToRollPitchYaw().vector()
    return rpy

def convert_quat_to_matrix(quat):
    matrix = RotationMatrix(Quaternion(quat)).matrix()
    return matrix

def get_angvel_over_euler_derivative(euler, seq='RPY'):
    """
        Euler kinematics equation
            \dot{\omega} = T(\theta) \dot{\theta}
        Return T(\theta)

        Note that RPY is a special case that rotates w.r.t. fixed frame, thus T=I
    """
    if seq == 'RPY':
        return np.eye(3)
    else:
        raise NotImplementedError(f'Euler sequence {seq} is not supported!')

def compute_error_between(q0, q1):
    """
        Compute the angle between two quaternions
        q0, q1: quaternions
    """
    q0, q1 = Quaternion(q0), Quaternion(q1)
    dq = q0.inverse().multiply(q1)

    return abs(RotationMatrix(dq).ToAngleAxis().angle())

def normalize_array(arr):
    return arr / np.linalg.norm(arr, axis=-1, keepdims=True)

###################################
#           Other Utils           #
###################################

def delete_listb_from_lista(lista, listb):
    ret = lista.copy()
    for item in listb:
        ret.remove(item)

    return ret

def get_bounded_random_rotation(ang_limit=0):
    """
        Generate a random rotation matrix.
        The rotation has bounded angle when converted to angle-axis form.
    """
    assert ang_limit >= 0

    rvec = SciR.random().as_rotvec()
    ang = np.linalg.norm(rvec)
    axis = rvec / ang
    ang = np.clip(ang, -ang_limit, ang_limit)
    rot = SciR.from_rotvec(ang * axis).as_matrix()

    return rot

def get_perturbed_quat(quat_wxyz, ang_limit=0):
    """
        Get the perturbed quaternion.
    """
    drot = get_bounded_random_rotation(ang_limit)
    rot = SciR.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_matrix()
    new_quat_wxyz = SciR.from_matrix(drot @ rot).as_quat()[[3, 0, 1, 2]]

    return new_quat_wxyz


if __name__ == '__main__':
    q0 = np.array([0.1368093, 0.58294937, 0.21369098, -0.77187396])
    rpy0 = convert_quat_to_rpy(q0)
    print("rpy0: ", rpy0)

    q0 = Quaternion(q0)
    q1 = Quaternion(convert_rpy_to_quat(np.array([0.6096838, 2.65064825, -0.85382545])))
    dq = q0.inverse().multiply(q1)
    print("dq(rpy): ", convert_quat_to_rpy(dq.wxyz()))

    q0 = Quaternion()
    q1 = Quaternion(convert_rpy_to_quat(np.array([-1.25445047, -0.05434401, -0.87374735])))
    dq = q0.inverse().multiply(q1)
    print("dq(rpy): ", convert_quat_to_rpy(dq.wxyz()))

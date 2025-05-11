import os
import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory

import pinocchio as pin

from pygrampc import Grampc
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Vector3
from leap_ros2.contact_model_grampc import SoftContactV3
from leap_ros2.utils import (
    LowLevelMPCProblemData, ReferenceTrajectory, LowLevelOptions, LowLevelCtrlMode, LowLevelContactPtsSource,
    cross_product_matrix
)
from common_msgs.msg import HybridFMCVis


def initialize_pin_model(config:LowLevelOptions):
    if config.hand_type == "leap":
        package_name = "leap_hand_custom_description"
        mesh_url = "meshes"
        if config.hw_type == "real":
            pin_model_name = "leap_hand_custom_tac3d.urdf"
        elif config.hw_type == "mujoco":
            pin_model_name = "leap_hand_custom.urdf"
    elif config.hand_type == "allegro":
        package_name = "allegro_hand_description"
        mesh_url = "assets/allegro"
        pin_model_name = "allegro_hand.urdf"

    # initialize hand model
    hand_urdf_url = os.path.join(get_package_share_directory(package_name), "urdf", pin_model_name)
    hand_mesh_url = os.path.join(get_package_share_directory(package_name), mesh_url)
    pin_model, collision_model, visual_model = pin.buildModelsFromUrdf(hand_urdf_url, hand_mesh_url)

    return pin_model, collision_model, visual_model

class LeapContactController(object):
    def __init__(self, config:LowLevelOptions) -> None:
        self.config = config

        self.initialize_hand_model(config)          # cost a few seconds
        self.build_controller(config)
        self.initialize_solver()

    def initialize_hand_model(self, config:LowLevelOptions):
        pin_model, _, _ = initialize_pin_model(config)

        self.pin_model = pin_model
        self.pin_data = pin_model.createData()
        self.ftip_frame_ids = [self.pin_model.getFrameId(link) for link in config.ordered_finger_links]

    def build_controller(self, config:LowLevelOptions):
        # dims
        # ----------------------------------------
        self.nc = len(self.ftip_frame_ids)
        self.nq = self.pin_model.nq
        self.q = np.zeros(self.nq,)             # actual joint pos (pinocchio order)
        self.qd = np.zeros(self.nq,)            # desired joint pos (pinocchio order)
        self.update_model(self.get_joints())    # initialize with zero pos
        # ----------------------------------------

        # set model parameters
        # ----------------------------------------
        # transform from 'fake_world' to pinocchio's 'world'
        self.T_fake2pinworld = np.diag([-1., 1., -1.])

        # robot
        Kp, Kd = config.mpic_params['Kp'], config.mpic_params['Kd']
        self.Kp = Kp * np.eye(self.nq,)
        self.Kd = Kd * np.eye(self.nq)
        
        # scale
        self.dq_scale = config.mpic_params['dq_scale']                  # 1.0

        # scale for debug
        self.Qf_scale = config.mpic_params['Qf_scale']                  # (open_door: 10.0, open_door_hand_left: 5.0, planar_slide: 10.0, planar_rotate: 5.0)
        self.Qp_scale = config.mpic_params['Qp_scale']                  # (open_door: 10.0, planar_slide: 10.0,  planar_rotate: 30.0)
        self.Qp_ori_scale = config.mpic_params['Qp_ori_scale']          # 1.0

        # environment to be set
        Ke_scalar = config.mpic_params['Ke_scalar']                     # 200
        Ke_thumb = np.diag(Ke_scalar * np.array([1, 0, 0], dtype=float))
        Ke_index = np.diag(Ke_scalar * np.array([1, 0, 0], dtype=float))
        Ke_middle = np.diag(Ke_scalar * np.array([1, 0, 0], dtype=float))
        Ke_ring = np.diag(Ke_scalar * np.array([1, 0, 0], dtype=float))
        self.Ke = [Ke_thumb, Ke_index, Ke_middle, Ke_ring]
        self.Ke_scalar = Ke_scalar
        # ----------------------------------------

        # set controller parameters
        # ----------------------------------------
        # force weights
        Qf_scalar = config.mpic_params['Qf_scalar']
        Qf_thumb = [Qf_scalar, 0, 0]
        Qf_index = [Qf_scalar, 0, 0]
        Qf_middle = [Qf_scalar, 0, 0]
        Qf_ring = [Qf_scalar, 0, 0]
        Qf = np.array(Qf_thumb + Qf_index + Qf_middle + Qf_ring, dtype=float)
        self.Qf = np.diag(Qf)
        self.Qf_scalar = Qf_scalar

        # position weights
        Qp_scalar = config.mpic_params['Qp_scalar']
        Qp_ori_scalar = config.mpic_params['Qp_ori_scalar']
        Qp_thumb = [Qp_scalar] * 3 + [Qp_ori_scalar] * 3
        Qp_index = [Qp_scalar] * 3 + [Qp_ori_scalar] * 3
        Qp_middle = [Qp_scalar] * 3 + [Qp_ori_scalar] * 3
        Qp_ring = [Qp_scalar] * 3 + [Qp_ori_scalar] * 3
        self.Qp_cart_mode = [
            np.diag(Qp_thumb).astype(float), np.diag(Qp_index).astype(float), \
            np.diag(Qp_middle).astype(float), np.diag(Qp_ring).astype(float)]
        self.Qp_joint_mode = [np.zeros((6, 6), dtype=float) for i in range(4)]
        self.Qp = self.Qp_joint_mode
        self.Qp_scalar = Qp_scalar

        # joint weights
        Qq_scalar = config.mpic_params['Qq_scalar']
        self.Qq_cart_mode = np.diag([0.0]*self.nq)
        self.Qq_joint_mode = np.diag([Qq_scalar]*self.nq)
        self.Qq = self.Qq_joint_mode

        # control regulation
        R_scalar = config.mpic_params['R_scalar']
        self.R = R_scalar*np.eye(self.nq,)
        # ----------------------------------------

        # other parameters
        # ----------------------------------------
        # the forces are applied to the environment
        self.measured_force = np.zeros((self.nc, 3))
        self.force_lower_bound = config.mpic_params['force_lower_bound']            # threshold for contact detection
        self.force_upper_bound = config.mpic_params['force_upper_bound']              # 2N to avoid damage

        self.measured_obj_contact_pts = np.zeros((self.nc, 3))                      # from tactile sensor
        self.highlevel_obj_contact_pts = np.zeros((self.nc, 3))                     # the first step
        self.contact_pts_src = LowLevelContactPtsSource.MEASURED

        self.measured_obj_pos = np.zeros(3,)
        self.measured_obj_quat = np.array([1, 0, 0, 0])

        self.ctrl_mode = LowLevelCtrlMode.JOINTSPACE
        self.set_ctrl_mode_params()

        self.enable_coupling = config.enable_coupling
        # ----------------------------------------

    def initialize_solver(self):
        # use GRAMPC solver
        options = os.path.join(get_package_share_directory('leap_ros2'), 'config', 'SoftContactLeap.json')
        options_dict = yaml.safe_load(open(options, 'r'))
        self.time_step = options_dict['Parameters']['dt']
        self.mpc_horizon = options_dict['Options']['Nhor']

        # problem = SoftContactV2(
        problem = SoftContactV3(
            self.Qq, self.Qp, self.Qf, self.R, self.Kd, self.Kp,
            self.compute_jacobian, self.compute_stiffness, self.compute_fk,
            self.nc, self.mpc_horizon, self.enable_coupling)
        grampc = Grampc(problem, options, plot_prediction=False)
        self.problem = problem
        self.solver = grampc
        self.u0 = np.zeros(self.nq,)
        
        self.n_iters = 0
        self.print_every = 30

        # mpc data
        self.mpc_data = LowLevelMPCProblemData(self.nc, self.nq, self.time_step, self.mpc_horizon)

    def reset(self):
        self.build_controller(self.config)
        self.initialize_solver()

    def get_joint_names(self):
        joint_names = []
        for joint_id in range(1, self.pin_model.njoints):
            joint_names.append(self.pin_model.names[joint_id])
        return joint_names

    def remap_joints(self, joint_values, joint_names=None):
        """ remap joints to pinocchio order """
        new_joint_values = np.zeros(self.nq,)
        if joint_names is None:
            new_joint_values[:] = joint_values
        else:
            for name, value in zip(joint_names, joint_values):
                # skip index(0), the universe joint
                joint_id = self.pin_model.getJointId(name)
                new_joint_values[joint_id-1] = value

        return new_joint_values
    
    def set_output_remapping(self, joint_order):
        """ remap output joints from pinocchio order to given order (i.e., hardware, 0~16) """
        pin_joint_names = self.get_joint_names()
        self.joint_remapping = [pin_joint_names.index(name) for name in joint_order]

    def set_highlevel_remapping(self, joint_order):
        """ remap high level reference from given order (i.e., sdf order) to pinocchio order """
        pin_joint_names = self.get_joint_names()
        self.remap_high2pin = [joint_order.index(name) for name in pin_joint_names]

    def update_model(self, joint_values, joint_names=None):
        joint_values_remapped = self.remap_joints(joint_values, joint_names)
        self.q[:] = joint_values_remapped
        pin.forwardKinematics(self.pin_model, self.pin_data, self.get_joints())

    def get_joints(self):
        """ Get q """
        return self.q.copy()
    
    def get_desired_joints(self):
        """ Get qd """
        return self.qd.copy()
    
    def get_concat_states(self):
        """ Get concat states [q, qd, fext] """
        return np.concatenate((self.get_joints(), self.get_desired_joints(), self.measured_force.flatten()))
    
    def get_object_contact_pts(self):
        """ Get contact points on the object """
        if self.contact_pts_src == LowLevelContactPtsSource.MEASURED:
            return self.measured_obj_contact_pts
        elif self.contact_pts_src == LowLevelContactPtsSource.HIGHLEVEL:
            return self.highlevel_obj_contact_pts
        else:
            raise NotImplementedError
    
    def set_desired_joints(self, joint_values, joint_names=None):
        """ Set qd """
        joint_values_remapped = self.remap_joints(joint_values, joint_names)
        self.qd[:] = joint_values_remapped

    def set_measured_force(self, force):
        self.measured_force[:] = force @ self.T_fake2pinworld.T

    def set_measured_contact_pts(self, contact_pts):
        """ Contact points on the object """
        self.measured_obj_contact_pts[:] = contact_pts @ self.T_fake2pinworld.T

    def set_measured_object_pose(self, pos):
        """ Set measured object pose """
        self.measured_obj_pos[:] = pos @ self.T_fake2pinworld.T

    def set_dq_scale(self, scale):
        self.dq_scale = max(scale, 0.0)

    def set_Qf_scale(self, scale):
        self.Qf_scale = max(scale, 0.0)

    def set_Qp_scale(self, scale):
        self.Qp_scale = max(scale, 0.0)

    def set_Qp_ori_scale(self, scale):
        self.Qp_ori_scale = max(scale, 0.0)
    
    def set_ctrl_mode(self, mode:int):
        self.ctrl_mode = LowLevelCtrlMode(mode)
        self.set_ctrl_mode_params()
    
    def set_ctrl_mode_params(self):
        if self.ctrl_mode == LowLevelCtrlMode.JOINTSPACE:
            self.Qp = self.Qp_joint_mode
            self.Qq = self.Qq_joint_mode
        elif self.ctrl_mode == LowLevelCtrlMode.CARTESIANSPACE:
            self.Qp = self.Qp_cart_mode
            self.Qq = self.Qq_cart_mode
        else:
            raise NotImplementedError
    
    def enable_print(self):
        if not self.config.debug:
            return False
        
        if self.n_iters % self.print_every == 0:
            return True
        else:
            return False

    def compute_jacobian(self, q=None):
        """ Compute the jacobians of each fingertip """
        if q is None:
            q = self.get_joints()
        J_ftips = [pin.computeFrameJacobian(self.pin_model, self.pin_data, q, id, pin.LOCAL_WORLD_ALIGNED) for id in self.ftip_frame_ids]
        return J_ftips

    def compute_stiffness(self, J_all):
        """ Compute the resulting stiffness """
        K_cart_inv = [J @ np.linalg.inv(self.Kp) @ J.T for J in J_all]
        K_result = [np.linalg.inv(np.eye(3,) + Ke @ Kinv) @ Ke for Ke, Kinv in zip(self.Ke, K_cart_inv)]
        
        return K_result

    def compute_fk(self, q=None):
        """ Compute the forward kinematics of each fingertip """
        if q is None:
            q = self.get_joints()
        pin.framesForwardKinematics(self.pin_model, self.pin_data, q)
        fk_ftips = [self.pin_data.oMf[id].copy() for id in self.ftip_frame_ids]

        return fk_ftips
    
    def compute_grasping_matrix(self):
        """
            Compute the grasping matrix, note that:
            1. Only compute once for efficiency
            2. All finger contacts are considered to avoid singularities (active and inactive contacts, in order)
        """
        obj_contact_pts = self.get_object_contact_pts()
        Gmat = np.zeros((6, 3*self.nc))
        for i in range(self.nc):
            Gmat[0:3, 3*i:3*(i+1)] = np.eye(3,)
            rel_contact_p = obj_contact_pts[i] - self.measured_obj_pos
            Gmat[3:, 3*i:3*(i+1)] = cross_product_matrix(rel_contact_p)
        
        return Gmat

    def calc_reference(self, current_time:float, ref:ReferenceTrajectory):
        """ This is the simplified version of AllegroLowLevelController:CalcStateReference """
        def threshold_force(force, threshold=0.1):
            magnitude = np.linalg.norm(force, axis=1, keepdims=True)
            direction = np.zeros_like(force)
            for j in range(len(force)):
                if magnitude[j] > 1e-5:
                    direction[j] = force[j] / magnitude[j]
            magnitude = np.minimum(magnitude, threshold)

            return magnitude * direction

        for i in range(self.mpc_horizon + 1):
            t_i = current_time + i * self.time_step

            q_i = ref.q.value(t_i)[-self.nq:].flatten()
            qd_i = ref.q.value(t_i)[-self.nq:].flatten()    # not really, but Q=0 for now
            w_i = ref.w.value(t_i).reshape(self.nc, 3)      # in sdf's world (fake_world in tf2)
            w_i = w_i @ self.T_fake2pinworld.T              # transform into pinocchio's world
            w_i = threshold_force(w_i, self.force_upper_bound)
            w_i = w_i.flatten()

            # remap from high level to pinocchio
            q_i = q_i[self.remap_high2pin]
            qd_i = qd_i[self.remap_high2pin]

            n_obj_i = ref.n_object.value(t_i).reshape(self.nc, 3)
            n_obj_i = n_obj_i @ self.T_fake2pinworld.T

            x_ref = np.concatenate((q_i, qd_i, w_i))
            self.mpc_data.x_ref[:, i] = x_ref
            self.mpc_data.object_normal[i, ...] = n_obj_i

        desired_force = ref.w.value(current_time).reshape(self.nc, 3)
        in_contact = np.linalg.norm(desired_force, axis=1) > self.force_lower_bound

        self.calc_HFMC_matrices(in_contact)

    def calc_HFMC_matrices(self, in_contact):
        """
            Calc Hybrid-Force-Motion-Control (HFMC) matrices, including
            - Environment stiffness (Ke)
            - Force tracking weight Qf
            - Position tracking weight Qp
            The computation is based on object normal
        """
        def get_ortho_vec_3x3(v):
            """ get orthogonal vecs in R(3) """
            A = np.random.rand(3, 3)
            A[:, 0] = v
            ortho_vec = np.linalg.qr(A)[0][:, 1:]
            return ortho_vec

        avg_normal = np.mean(self.mpc_data.object_normal, axis=0)   # (4, 3)
        for i in range(self.nc):
            # track position only if desired force is small or object normal not provided
            if not in_contact[i] or np.linalg.norm(avg_normal[i]) < 1e-5:
                Qf_i = np.zeros((3, 3))
                Qp_i = np.eye(3,)
            else:
                n_i = avg_normal[i] / np.linalg.norm(avg_normal[i])
                T_i = get_ortho_vec_3x3(n_i)        # (3, 2)
                Qf_i = np.outer(n_i, n_i)           # (3, 3), r(Qf)=1
                Qp_i = np.matmul(T_i, T_i.T)        # (3, 3), r(Qp)=2

            self.Ke[i] = self.Ke_scalar * Qf_i
            self.Qf[i*3:(i+1)*3, i*3:(i+1)*3] = self.Qf_scalar * Qf_i
            self.Qp_cart_mode[i][:3, :3] = self.Qp_scalar * Qp_i

            if self.enable_print():
                print(f"Qf[{i}]={Qf_i},\n Qp[{i}]={Qp_i}")

    def get_HFMC_debug_vis_msg(self):
        """ Get vis msg for debug HFMC """
        def get_PCA(M):
            eigval, eigvec = np.linalg.eig(M)
            return eigvec[:, np.abs(eigval)>1e-5].reshape(3, -1)
        
        def orthogonalize_matrix(M):
            """ Make matrix an orthogonal one """
            M = np.linalg.qr(M)[0]
            assert np.allclose(M.T @ M, np.eye(3))
            return M
        
        def make_right_hand_matrix(M):
            """ Make rotation matrix right-handed """
            if np.linalg.det(M) < 0:
                M[:, 2] = -M[:, 2]
            return M

        msg = HybridFMCVis()
        fk_ftips = self.compute_fk()
        Qf_ftips = [self.Qf[0:3, 0:3].copy(), self.Qf[3:6, 3:6].copy(),
              self.Qf[6:9, 6:9].copy(), self.Qf[9:12, 9:12].copy()]
        Qp_ftips = [Qp[:3, :3].copy() for Qp in self.Qp_cart_mode]
        for i in range(self.nc):
            fk, Qf, Qp = fk_ftips[i].translation, Qf_ftips[i]/self.Qf_scalar, Qp_ftips[i]/self.Qp_scalar
            msg.points.append(Vector3(x=fk[0], y=fk[1], z=fk[2]))

            # orthogonalize
            direction_f, direction_p = get_PCA(Qf), get_PCA(Qp)
            direction = np.concatenate((direction_f, direction_p), axis=1)
            direction = orthogonalize_matrix(direction)
            direction = make_right_hand_matrix(direction)
            msg.directions.append(Float32MultiArray(data=direction.flatten()))

            w_f, w_p = np.zeros(3,), np.ones(3,)
            w_f[:direction_f.shape[1]] = 1.0
            w_p[:direction_f.shape[1]] = 0.0
            msg.w_position.append(Vector3(x=w_p[0], y=w_p[1], z=w_p[2]))
            msg.w_force.append(Vector3(x=w_f[0], y=w_f[1], z=w_f[2]))

        return msg

    def get_dq(self):
        """ Get dimensions of dq (in contact fingers) """
        xdes = self.mpc_data.x_ref[:, :self.mpc_horizon].T

        if self.enable_print():
            print("Qq=", self.problem.Qq)
            print("Qp=", self.problem.Qp)
            print("Qf=", self.problem.Qf)

        # reset weights
        Qq_new = self.Qq.copy()
        self.problem.set_Qq(Qq_new)
        
        Qp_new = []
        for i in range(len(self.Qp)):
            Qp_scaled = self.Qp[i].copy()
            Qp_scaled[:3, :3] *= self.Qp_scale
            Qp_scaled[3:, 3:] *= self.Qp_ori_scale
            Qp_new.append(Qp_scaled)
        self.problem.set_Qp(Qp_new)

        Qf_new = self.Qf_scale * self.Qf.copy()
        self.problem.set_Qf(Qf_new)

        # set grasping matrix (for coupling term computation)
        Gmat = self.compute_grasping_matrix()
        self.problem.set_grasping_matrix(Gmat)

        # run contact controller
        x0 = self.get_concat_states()
        self.problem.reset()
        self.problem.initialize_xdes_spline(self.time_step * np.arange(self.mpc_horizon), xdes)
        self.solver.set_param({
            'x0': x0,
            'u0': self.u0,
            't0': 0.0,
        })
        self.solver.run()
        u0 = self.solver.sol.unext

        # warm start next iter
        self.u0 = self.solver.rws.u[:, 1]
        
        dq = self.dq_scale * u0 * self.time_step
        
        J_all = [J[:3] for J in self.compute_jacobian()]
        if self.enable_print():
            print(f"[thumb] x0={x0[4:8]}, f0={x0[35:38]}")
            print(f"[thumb] dv={J_all[0] @ dq}")
            print(f"[middle] dv={J_all[2] @ dq}")
        
        return dq

    def solve(self, current_time:float, ref:ReferenceTrajectory):
        """
        :param current_time: current (ROS) time in seconds
        :return dq: delta desired joint pos
        """
        self.calc_reference(current_time, ref)
        
        dq = self.get_dq()
        dq = dq[self.joint_remapping]

        self.n_iters += 1

        return dq

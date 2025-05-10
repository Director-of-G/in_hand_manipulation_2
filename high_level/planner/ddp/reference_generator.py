import numpy as np
from scipy.spatial.transform import Rotation as SciR

from pydrake.all import (
    AngleAxis,
    Quaternion,
)

from pydrake.math import (
    RollPitchYaw,
    RotationMatrix,
)

from common.common_ddp import convert_rpy_to_quat, convert_quat_to_rpy


class ReferenceGeneratorRPY(object):
    """
        This class generates the reference for the RPY rotation task
    """
    def __init__(self, step_len=1.0*np.pi/200, num_steps=11):
        assert isinstance(step_len, float)

        # the targets
        self.start = Quaternion([1, 0, 0, 0])
        self.target = Quaternion([1, 0, 0, 0])
        self.target_axis = np.array([0, 0, 1])
        self.target_angle = 0

        # the lut
        self.quat_lut = np.zeros((0, 4))        # quaternion mid-points
        self.dyaw_lut = step_len
        
        self.dyaw_ref = step_len                # step length
        self.T_ref = num_steps+1                  # reference length

    def generate_random_so3_target(self):
        # while True:
        #     rand_so3 = SciR.random()
        #     if rand_so3.as_matrix()[2, 2] > 0:
        #         self.target = Quaternion(rand_so3.as_quat()[[3, 0, 1, 2]])      # xyzw --> wxyz
        #         self.set_start_and_target_quat(self.start.wxyz(), self.target.wxyz())
        #         break

        deg_limit = 135
        rand_so3 = SciR.random()
        rvec = rand_so3.as_rotvec()
        axis = rvec / np.linalg.norm(rvec)
        ang = np.random.uniform(-np.deg2rad(deg_limit), np.deg2rad(deg_limit))
        rot = SciR.from_rotvec(ang * axis)
        self.target = Quaternion(rot.as_quat()[[3, 0, 1, 2]])      # xyzw --> wxyz
        self.set_start_and_target_quat(self.start.wxyz(), self.target.wxyz())

    def set_start_and_target_quat(self, start_quat, target_quat):
        start_rpy = convert_quat_to_rpy(start_quat)
        target_rpy = convert_quat_to_rpy(target_quat)
        self.set_start_and_target_rpy(start_rpy, target_rpy)

    def set_start_and_target_rpy(self, start_rpy, target_rpy):
        # clear lut
        self.quat_lut = np.zeros((0, 4))

        start_quat = Quaternion(convert_rpy_to_quat(start_rpy))
        target_quat = Quaternion(convert_rpy_to_quat(target_rpy))
        start_rot = RotationMatrix(start_quat)
        self.start = start_quat
        self.target = target_quat

        dq = start_quat.inverse().multiply(target_quat)
        angle_axis = RotationMatrix(dq).ToAngleAxis()

        self.target_angle = angle_axis.angle()
        self.target_axis = angle_axis.axis()

        # make lut
        n_lut = int(abs(self.target_angle) // self.dyaw_lut)
        # TODO(yongpeng): apply rotation matrix, not sure if any bugs
        for i in range(n_lut):
            rmat_i = RotationMatrix(
                AngleAxis(i * np.sign(self.target_angle) * self.dyaw_lut, self.target_axis)
            )
            quat_i = start_rot.multiply(rmat_i).ToQuaternion().wxyz()
            self.quat_lut = np.append(self.quat_lut, np.array([quat_i]), axis=0)

        # TODO(yongpeng): interpolate wxyz
        # # interpolate wxyz
        # quat_lut_unnormalized = np.linspace(start_quat.wxyz(), target_quat.wxyz(), n_lut)
        # # extend ahead
        # for i in range(1, 5*self.T_ref):
        #     quat_i = quat_lut_unnormalized[-1] + i * (quat_lut_unnormalized[-1] - quat_lut_unnormalized[-1])
        #     quat_lut_unnormalized = np.append(quat_lut_unnormalized, np.array([quat_i]), axis=0)
        # self.quat_lut = quat_lut_unnormalized / np.linalg.norm(quat_lut_unnormalized, axis=1).reshape(-1, 1)

        # TODO(yongpeng): random rotation (continuous)
        # self.rand_axis = normalize_array(np.random.uniform(-1, 1, size=(3,)))

    def generate_reference_from_x0(self, x0):
        """
            x0: quat
        """
        quat_ref = np.zeros((self.T_ref, 4))

        # TODO(yongpeng): interpolate from nearesr neighbor
        # # find nearest
        # qdiff = self.quat_lut.dot(x0)
        # near_idx = np.argmax(np.abs(qdiff))
        # assert (abs(qdiff[near_idx]) <= 1.0)
        # q_near = self.quat_lut[near_idx]
        # rot_near = RotationMatrix(Quaternion(q_near))

        # # make reference
        # for i in range(self.T_ref):
        #     rmat_i = RotationMatrix(
        #         AngleAxis(i * np.sign(self.target_angle) * self.dyaw_ref, self.target_axis)
        #     )
        #     quat_i = rot_near.multiply(rmat_i).ToQuaternion().wxyz()

        #     if np.dot(quat_i, x0) < 0:
        #         quat_i = -quat_i

        #     quat_ref[i] = quat_i

        # TODO(yongpeng): interpolate wxyz from nearest neighbor
        # # find nearest
        # qdiff = self.quat_lut.dot(x0)
        # near_idx = np.argmax(np.abs(qdiff))
        # assert (abs(qdiff[near_idx]) <= 1.0)

        # # make reference
        # for i in range(self.T_ref):
        #     if near_idx + i >= len(self.quat_lut):
        #         quat_i = self.quat_lut[-1]
        #     else:
        #         quat_i = self.quat_lut[near_idx + i]

        #     if np.dot(quat_i, x0) < 0:
        #         quat_i = -quat_i

        #     quat_ref[i] = quat_i

        # TODO(yongpeng): interpolate from x0
        # # calc difference
        # start_quat = Quaternion(x0)
        # target_quat = self.target
        # start_rot = RotationMatrix(start_quat)

        # dq = start_quat.inverse().multiply(target_quat)
        # angle_axis = RotationMatrix(dq).ToAngleAxis()

        # angle_ = angle_axis.angle()
        # axis_ = angle_axis.axis()

        # if abs(angle_) > 0.35:
        #     dyaw_ref = self.dyaw_ref
        # else:
        #     dyaw_ref = 0.25 * self.dyaw_ref

        # interpolate_steps = int(abs(angle_) / dyaw_ref)
        # print(f"error {angle_}deg, interp steps {interpolate_steps}")

        # for i in range(min(interpolate_steps, self.T_ref)):
        #     rmat_i = RotationMatrix(
        #         AngleAxis(i * np.sign(angle_) * dyaw_ref, axis_)
        #     )
        #     quat_i = start_rot.multiply(rmat_i).ToQuaternion().wxyz()

        #     # invert the sign if needed
        #     if np.dot(quat_i, x0) < 0:
        #         quat_i = -quat_i

        #     quat_ref[i] = quat_i

        # for i in range(interpolate_steps, self.T_ref):
        #     quat_ref[i] = self.target.wxyz() if np.dot(self.target.wxyz(), x0) >= 0 else -self.target.wxyz()

        # TODO(yongpeng): set reference to target
        for i in range(self.T_ref):
            quat_ref[i] = self.target.wxyz() if np.dot(self.target.wxyz(), x0) >= 0 else -self.target.wxyz()

        # TODO(yongpeng): continuous rotation
        # start_rot = RotationMatrix(Quaternion(x0))
        # for i in range(self.T_ref):
        #     rmat_i = RotationMatrix(
        #         AngleAxis(i * self.dyaw_ref, self.rand_axis)
        #     )
        #     quat_i = start_rot.multiply(rmat_i).ToQuaternion().wxyz()

        #     # invert the sign if needed
        #     if np.dot(quat_i, x0) < 0:
        #         quat_i = -quat_i

        #     quat_ref[i] = quat_i

        return quat_ref
    
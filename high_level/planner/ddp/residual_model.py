import numpy as np
import crocoddyl
import pinocchio as pin

from common.common_ddp import convert_rpy_to_matrix, convert_quat_to_matrix, get_angvel_over_euler_derivative, CalcNQdot2W


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


class ResidualModelFrameRotation(crocoddyl.ResidualModelAbstract):
    """
        Residual model for relative frame rotation.
        Reference: crocoddyl GitHub repo (https://github.com/loco-3d/crocoddyl/blob/9919619930878f6c4c015cdf94dc7346c986580a/include/crocoddyl/multibody/residuals/frame-rotation.hxx)
        :param Rref: reference frame rotation (3x3 matrix)
        :param xrot_slc: slice of frame rotation in state (default to euler xyz)
    """
    def __init__(self, state, nu, Rref, xrot_slc):
        crocoddyl.ResidualModelAbstract.__init__(self, state, 3, nu, True, True, False)

        self.Rref_ = Rref.copy()
        self.oRf_inv_ = Rref.copy().T

        if isinstance(xrot_slc, slice):
            if xrot_slc.stop - xrot_slc.start == 3:
                self.xrot_type_ = 'RPY'
            elif xrot_slc.stop - xrot_slc.start == 4:
                self.xrot_type_ = 'Quat'
            else:
                raise ValueError(f"Invalid xrot_slc {xrot_slc}")
        else:
            assert isinstance(xrot_slc, list)
            if len(xrot_slc) == 3:
                self.xrot_type_ = 'RPY'
            elif len(xrot_slc) == 4:
                self.xrot_type_ = 'Quat'
            else:
                raise ValueError(f"Invalid xrot_slc {xrot_slc}")
        
        self.xrot_slc_ = xrot_slc

    def set_reference(self, rotation):
        self.Rref_ = rotation.copy()
        self.oRf_inv_ = rotation.copy().T

    def calc(self, data, x, u):
        if self.xrot_type_ == 'RPY':
            rpy = x[self.xrot_slc_]
            oMf_rot = convert_rpy_to_matrix(rpy)
        elif self.xrot_type_ == 'Quat':
            quat = x[self.xrot_slc_]
            oMf_rot = convert_quat_to_matrix(quat)

        # data.rRf[:] = self.oRf_inv_ @ oMf_rot
        self.rRf = self.oRf_inv_ @ oMf_rot
        data.r[:] = pin.log3(self.rRf)

    def calcDiff(self, data, x, u):
        rJf = pin.Jlog3(self.rRf)

        if self.xrot_type_ == 'RPY':
            rpy = x[self.xrot_slc_]
            fJf = get_angvel_over_euler_derivative(rpy, seq='RPY')
        elif self.xrot_type_ == 'Quat':
            quat = x[self.xrot_slc_]
            fJf = CalcNQdot2W(quat)

        data.Rx[:, self.xrot_slc_] = rJf @ fJf

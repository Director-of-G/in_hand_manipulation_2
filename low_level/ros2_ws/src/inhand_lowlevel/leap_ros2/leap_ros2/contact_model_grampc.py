import numpy as np
from numpy import ndarray
from pygrampc import ProblemDescription
from scipy.interpolate import make_interp_spline
from scipy.linalg import block_diag
from leap_ros2.utils import compute_pose_error


class SoftContactV3(ProblemDescription):
    """
        The matrices K, J are rewritten in stacked version compared with SoftContact(V2)
        Please refer to the journal paper
    """
    def __init__(self, Qq, Qp, Qf, R, Kd, Kp, jac_func, k_func, fk_func, nc=1, Nhor=10, enable_coupling=False):
        ProblemDescription.__init__(self)
        nq = Kp.shape[0]
        self.nq = nq
        self.nc = nc

        self.Nx = 2 * nq + 3 * nc
        self.Nu = nq
        self.Np = 0
        self.Ng = 0
        self.Nh = 0
        self.NgT = 0
        self.NhT = 0
        self.Nhor = Nhor
        self.enable_coupling = enable_coupling

        self.Qq = Qq.copy()        # (nq, nq)
        self.Qp = Qp.copy()        # [(6, 6)]*4
        self.Qf = Qf.copy()        # (3*nc, 3*nc)
        self.R = R.copy()          # (nq, nq)
        self.Gmat = np.zeros((6, 3*nc))

        self.Kdinv = np.linalg.inv(Kd)
        self.Kp = Kp.copy()

        self.jac_func = jac_func
        self.k_func = k_func
        self.fk_func = fk_func

        # desired state trajectory
        self.xdes_spline_stored = None

        # stored (time-variant) jacobian, stiffness and desired link pos
        self.J_stored = {}                  # list of J
        self.K_stored = {}                  # list of K
        self.K_coup_stored = {}
        self.pdes_stored = {}

    def reset(self):
        """ Reset all stored data (must call before each MPC) """
        self.xdes_spline_stored = None
        self.J_stored.clear()
        self.K_stored.clear()
        self.K_coup_stored.clear()
        self.pdes_stored.clear()

    def initialize_xdes_spline(self, t_knots, xdes_knots):
        """
            :param t_knots: (Nhor,)
            :param xdes_knots: (Nhor, Nx)
        """
        xdes_spline = make_interp_spline(t_knots, xdes_knots, k=1)
        self.xdes_spline_stored = xdes_spline

    def get_xdes(self, t):
        return self.xdes_spline_stored(t)

    def get_jacobian(self, t, q, trans_only=False):
        """
            Get the jacobian of step t
            :param t: time (key for stored data)
            :param q: compute and store new data
            :param trans_only: if True, will return J[:3]
        """
        t_key = round(t, 4)
        J = np.zeros((6, self.nq))
        if t_key in self.J_stored:
            J = self.J_stored[t_key]
        else:
            J = self.jac_func(q)
            self.J_stored[t_key] = J

        if trans_only:
            return [jac[:3] for jac in J]
        else:
            return J
        
    def get_stiffness(self, t, J):
        """
            Get the resulting stiffness of step t
            :param t: time (key for stored data)
            :param J: compute and store new data
        """
        t_key = round(t, 4)
        if t_key in self.K_stored:
            K = self.K_stored[t_key]
        else:
            K = self.k_func(J)
            self.K_stored[t_key] = K
        return K
    
    def get_K_coup(self, t, K_bar):
        """
            Get the coupled stiffness
            :param t: time (key for stored data)
            :param K_bar: decoupled stiffness
        """
        t_key = round(t, 4)
        if t_key in self.K_coup_stored:
            K_coup = self.K_coup_stored[t_key]
        else:
            GKGT = self.Gmat[[0, 4, 5]] @ K_bar @ self.Gmat[[0, 4, 5]].T
            if np.linalg.matrix_rank(GKGT) < 3 or self.enable_coupling is False:
                K_coup = K_bar
            else:
                K_coup = K_bar + K_bar @ self.Gmat[[0, 4, 5]].T @ np.linalg.inv(GKGT) @ self.Gmat[[0, 4, 5]] @ K_bar
            self.K_coup_stored[t_key] = K_coup

        # print("rank: ", np.linalg.matrix_rank(self.Gmat @ K_bar @ self.Gmat.T))

        return K_coup
    
    def get_pdes(self, t, q):
        """
            Get the desired link pos of step t
            :param t: time (key for stored data)
            :param q: compute and store new data
        """
        t_key = round(t, 4)
        if t_key in self.pdes_stored:
            pdes = self.pdes_stored[t_key]
        else:
            pdes = self.fk_func(q)
            self.pdes_stored[t_key] = pdes
        return pdes
    
    def set_Qf(self, Qf):
        """ Reset Qf """
        self.Qf[:] = Qf.copy()

    def set_Qq(self, Qq):
        """ Reset Qq """
        self.Qq = Qq.copy()

    def set_Qp(self, Qp):
        """ Reset Qp """
        self.Qp = Qp.copy()

    def set_R(self, R):
        """ Reset R """
        self.R[:] = R.copy()

    def set_grasping_matrix(self, Gmat):
        """ Set grasping matrix """
        self.Gmat[:] = Gmat.copy()
    
    def get_Q(self):
        """ Return Qq, Qp, Qf based on which fingers are in contact """
        Qf = self.Qf.copy()
        return self.Qq, self.Qp, Qf
    
    def get_R(self):
        R = self.R.copy()
        return R

    def ffct(self, out: ndarray, t: float, x: ndarray, u: ndarray, p: ndarray):
        nq = self.nq
        q, qd, fext = x[:nq], x[nq:2*nq], x[2*nq:]

        J = self.get_jacobian(t, q, trans_only=True)            # list of J
        J_vcat = np.vstack(J)                                   # (3*nc, nq)

        K = self.get_stiffness(t, J)
        K_bar = block_diag(*K)                                  # (3*nc, 3*nc)
        K_coup = self.get_K_coup(t, K_bar)                      # (3*nc, 3*nc)
        
        # out (2*nq+3*nc)
        out[:nq] = u + self.Kdinv @ (self.Kp @ (qd - q) - J_vcat.T @ fext)
        out[nq:2*nq] = u
        out[2*nq:] = K_coup @ J_vcat @ u

    def dfdx_vec(self, out: ndarray, t: float, x: ndarray, vec: ndarray, u: ndarray, p: ndarray):
        nq = self.nq
        q = x[:nq]
        
        J = self.get_jacobian(t, q, trans_only=True)
        J_vcat = np.vstack(J)

        out[:] = np.block([-self.Kdinv@self.Kp, self.Kdinv@self.Kp, -self.Kdinv@J_vcat.T]).T @ vec[:nq]

    def dfdu_vec(self, out: ndarray, t: float, x: ndarray, vec: ndarray, u: ndarray, p: ndarray):
        nq = self.nq
        q = x[:nq]

        J = self.get_jacobian(t, q, trans_only=True)
        J_vcat = np.vstack(J)

        K = self.get_stiffness(t, J)
        K_bar = block_diag(*K)                                  # (3*nc, 3*nc)
        K_coup = self.get_K_coup(t, K_bar)                      # (3*nc, 3*nc)

        out[:] = np.block([[np.eye(nq,)],[np.eye(nq,)],[K_coup @ J_vcat]]).T @ vec

    def lfct(self, out: ndarray, t: float, x: ndarray, u: ndarray, p: ndarray, xdes: ndarray, udes: ndarray):
        nq = self.nq

        # get desired
        xdes_itp = self.get_xdes(t)
        qdes = xdes_itp[nq:2*nq]
        pdes = self.get_pdes(t, qdes)
        fdes = xdes_itp[2*nq:]

        # get current
        qd, fext = x[nq:2*nq], x[2*nq:]
        pd = self.fk_func(qd)

        # joint error and pose error
        Qq, Qp, Qf = self.get_Q()
        R = self.get_R()
        out[0] = (qd - qdes).T @ Qq @ (qd - qdes) + \
                    (fext - fdes).T @ Qf @ (fext - fdes) + \
                    u.T @ R @ u

        # pose error
        for i in range(self.nc):
            pe = compute_pose_error(
                t1=pdes[i].translation, R1=pdes[i].rotation,
                t2=pd[i].translation, R2=pd[i].rotation
            )
            out[0] += pe.T @ Qp[i] @ pe

    def dldx(self, out: ndarray, t: float, x: ndarray, u: ndarray, p: ndarray, xdes: ndarray, udes: ndarray):
        nq = self.nq

        # get desired
        xdes_itp = self.get_xdes(t)
        qdes = xdes_itp[nq:2*nq]
        pdes = self.get_pdes(t, qdes)
        fdes = xdes_itp[2*nq:]

        # get current
        qd, fext = x[nq:2*nq], x[2*nq:]
        pd = self.fk_func(qd)

        # get gradient
        Qq, Qp, Qf = self.get_Q()
        out[nq:2*nq] = 2 * Qq @ (qd - qdes)
        out[2*nq:] = 2 * Qf @ (fext - fdes)
        J = self.get_jacobian(t, qd)

        for i in range(self.nc):
            pe = compute_pose_error(
                t1=pdes[i].translation, R1=pdes[i].rotation,
                t2=pd[i].translation, R2=pd[i].rotation
            )
            # Jlinv = compute_orientation_error_jacobian_inv_lie_algebra(pe[3:])

            # Jerr = np.block([[np.eye(3), np.zeros((3, 3))], [np.zeros((3, 3)), Jlinv]])
            out[nq:2*nq] += 2 * (np.dot(Qp[i], pe)).T @ J[i]
            # out[nq:2*nq] += 2 * (Jerr.T @ Qp[i].T @ pe).T @ J[i]

    def dldu(self, out: ndarray, t: float, x: ndarray, u: ndarray, p: ndarray, xdes: ndarray, udes: ndarray):
        R = self.get_R()
        out[:] = 2 * R @ u

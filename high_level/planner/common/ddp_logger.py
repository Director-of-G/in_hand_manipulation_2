import numpy as np


class DDPLogger(object):
    def __init__(self, nx, nu, nr, num_contacts):
        self.nx = nx
        self.nu = nu
        self.nr = nr
        self.num_contacts = num_contacts

        self.data_ = {}

    def log_data(self, index=0, **kwargs):
        """
            log data for timestep i
        """
        for key, value in kwargs.items():
            if key not in self.data_:
                self.data_[key] = []
            self.data_[key].append((index, value))

    def set_data(self, **kwargs):
        """
            set data
        """
        for key, value in kwargs.items():
            self.data_[key] = value
    
    def get_data(self, key):
        """
            get data
        """
        return self.data_[key]

    def get_data_in_array(self, key):
        """
            get data and the index list
        """
        data = self.data_[key]
        index_list = [item[0] for item in data]
        value_list = [item[1] for item in data]
        value_list = np.array(value_list)
        return index_list, value_list


def test_ddp_matrix_computation(
    solver,
    verbose=True
):
    """
        test the computation of DDP matrices
        reference: [handwiki](https://handwiki.org/wiki/Differential_dynamic_programming)
    """
    problem = solver.problem
    T = problem.T

    Qx, Qu = solver.Qx, solver.Qu
    Qxx, Quu, Qxu, Quu_inv = solver.Qxx, solver.Quu, solver.Qxu, solver.Quu_inv
    Vx, Vxx = solver.Vx, solver.Vxx
    k = solver.k

    Quu_cond, Lxx_cond = [], []
    Q_error = []

    for i in range(T):
        Fx, Fu = problem.runningDatas[i].Fx, problem.runningDatas[i].Fu
        Lx, Lu = problem.runningDatas[i].Lx, problem.runningDatas[i].Lu
        Lxx = problem.runningDatas[i].Lxx
        Vxi, Vxxi = Vx[i+1], Vxx[i+1]

        Lxx, Luu, Lxu = problem.runningDatas[i].Lxx, problem.runningDatas[i].Luu, problem.runningDatas[i].Lxu

        Lxx_cond.append(np.linalg.cond(Lxx))

        # TEST1
        Qxi = Qx[i]
        Qxi_ = Lx + Fx.T.dot(Vxi)
        error_ = np.linalg.norm(Qxi-Qxi_, ord=2)/np.linalg.norm(Qxi_, ord=2)
        Q_error.append(error_)

        # TEST2
        Qui = Qu[i]
        Qui_ = Lu + Fu.T.dot(Vxi)
        error_ = np.linalg.norm(Qui-Qui_, ord=2)/np.linalg.norm(Qui_, ord=2)
        Q_error.append(error_)

        # TEST3
        Qxxi = Qxx[i]
        Qxxi_ = Lxx + Fx.T.dot(Vxxi.dot(Fx))
        error_ = np.linalg.norm(Qxxi-Qxxi_, ord="fro")/np.linalg.norm(Qxxi_, ord="fro")
        Q_error.append(error_)

        # TEST4
        Quui = Quu[i]
        Quui_ = Luu + Fu.T.dot(Vxxi.dot(Fu))
        error_ = np.linalg.norm(Quui-Quui_, ord="fro")/np.linalg.norm(Quui_, ord="fro")
        Q_error.append(error_)
        Quu_cond.append(np.linalg.cond(Quui))

        # TEST5
        Qxui = Qxu[i]
        Qxui_ = Lxu + Fx.T.dot(Vxxi.dot(Fu))
        error_ = np.linalg.norm(Qxui-Qxui_, ord="fro")/np.linalg.norm(Qxui_, ord="fro")
        Q_error.append(error_)
        
        # TEST6
        ki = k[i]
        Quu_invi = Quu_inv[i]
        ki_ = Quu_invi.dot(Qui)
        error_ = np.linalg.norm(ki-ki_, ord=2)/(np.linalg.norm(ki_, ord=2)+1e-5)
        Q_error.append(error_)

    Quu_cond, Lxx_cond = np.array(Quu_cond), np.array(Lxx_cond)
    Q_error = np.array(Q_error)

    if verbose:
        print("-------- error of ddp matrices --------")
        print("[Qx error]: ", Q_error[0::6].mean())
        print("[Qu error]: ", Q_error[1::6].mean())
        print("[Qxx error]: ", Q_error[2::6].mean())
        print("[Quu error]: ", Q_error[3::6].mean())
        print("[Qxu error]: ", Q_error[4::6].mean())
        print("[k error]: ", Q_error[5::6].mean())
        print("[Quu cond (first, max)]: ", (Quu_cond[0], Quu_cond.max()))
        print("[Lxx cond (first, max)]: ", (Lxx_cond[0], Lxx_cond.max()))
        print("---------------------------------------")

    return Q_error

import numpy as np
from scipy.spatial.transform import Rotation as SciR


def generate_quat_target():
    deg_limit = 135

    quat_target = None
    while True:
        rand_so3 = SciR.random()
        rvec = rand_so3.as_rotvec()
        ang = np.linalg.norm(rvec)
        if np.abs(ang) < np.deg2rad(deg_limit):
            quat_target = rand_so3.as_quat()[[3, 0, 1, 2]]
            break

    return quat_target

if __name__ == '__main__':
    N = 100
    quat_targets = np.zeros((N, 4))
    for i in range(N):
        quat_targets[i] = generate_quat_target()
    print("quat targets: ", quat_targets)
    np.save('./data/quat_targets.npy', quat_targets)

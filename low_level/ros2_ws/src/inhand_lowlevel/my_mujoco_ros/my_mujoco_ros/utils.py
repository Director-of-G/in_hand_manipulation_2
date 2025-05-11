import numpy as np
from scipy.spatial.transform import Rotation as SciR
from geometry_msgs.msg import Point, Wrench


def get_contact_message(contact_point, contact_force, contact_normal):
    pos_msg = Point()
    pos_msg.x = contact_point[0]
    pos_msg.y = contact_point[1]
    pos_msg.z = contact_point[2]

    wrench_msg = Wrench()
    wrench_msg.force.x = contact_force[0]
    wrench_msg.force.y = contact_force[1]
    wrench_msg.force.z = contact_force[2]

    # store contact normal in the torque field
    wrench_msg.torque.x = contact_normal[0]
    wrench_msg.torque.y = contact_normal[1]
    wrench_msg.torque.z = contact_normal[2]

    return pos_msg, wrench_msg

def transform_pos_quat(trans, pos, quat):
    """
        Apply a homogeneous transformation to a position and quaternion
        trans (4x4 homogeneous transformation)
        pos (xyz)
        quat (wxyz)
    """
    pos_homo = np.append(pos, 1)
    pos = (trans @ pos_homo)[:-1]

    rot_mat = SciR.from_quat(quat[[1, 2, 3, 0]]).as_matrix()    # wxyz -> xyzw
    rot_mat = trans[:3, :3] @ rot_mat
    quat = SciR.from_matrix(rot_mat).as_quat()[[3, 0, 1, 2]]    # xyzw -> wxyz

    return pos, quat

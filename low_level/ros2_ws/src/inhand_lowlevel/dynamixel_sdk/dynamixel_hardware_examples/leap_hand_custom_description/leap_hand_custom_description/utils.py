import numpy as np
from scipy.spatial.transform import Rotation as SciR
from sensor_msgs.msg import JointState, MultiDOFJointState
from geometry_msgs.msg import Point
from trajectory_msgs.msg import JointTrajectory
from moveit_msgs.msg import DisplayTrajectory, RobotState, RobotTrajectory
from visualization_msgs.msg import Marker, MarkerArray


def create_joint_state_from_pos(q, header, joint_names):
    """
    :param q: np.array of shape (N,), N is dof
    """
    joint_state = JointState()
    joint_state.header = header
    joint_state.header.frame_id = "world"
    joint_state.name = joint_names
    joint_state.position = q.tolist()
    joint_state.velocity = [0.0 * len(q)]

    return joint_state


def display_motion_in_rviz(traj:JointTrajectory):
    """
    Publish the hand motion as DisplayTrajectory to RViz
    :param q_arr: np.array of shape (T, N), T is time steps
    """

    def create_robot_state(traj:JointTrajectory):
        q0 = traj.points[0].positions
        robot_state = RobotState()
        robot_state.joint_state = create_joint_state_from_pos(q0, traj.header, traj.joint_names)
        _empty_multi_dof_joint_state = MultiDOFJointState()
        _empty_multi_dof_joint_state.header = robot_state.joint_state.header
        robot_state.multi_dof_joint_state = _empty_multi_dof_joint_state
        return robot_state

    def joint_traj_to_robot_traj(traj:JointTrajectory):
        robot_trajectory = RobotTrajectory()
        robot_trajectory.joint_trajectory = traj  # 直接将 JointTrajectory 消息赋值给 joint_trajectory 字段
        
        return robot_trajectory

    display_trajectory = DisplayTrajectory()
    display_trajectory.trajectory_start = create_robot_state(traj)
    robot_trajectory = joint_traj_to_robot_traj(traj)
    display_trajectory.trajectory.append(robot_trajectory)

    return display_trajectory


def get_force_vis_message(force, position, marker_id=0, rgba=(1.0, 0.0, 0.0, 1.0), base_frame="base"):
    marker = Marker()
    marker.header.frame_id = base_frame

    marker.ns = ""
    marker.id = marker_id

    marker.type = Marker.ARROW
    marker.action = Marker.ADD

    marker.points = []

    start_point = Point(x=position[0], y=position[1], z=position[2])
    
    end_point = Point(x=position[0]+force[0], y=position[1]+force[1], z=position[2]+force[2])
    
    marker.points.append(start_point)
    marker.points.append(end_point)

    marker.scale.x = 0.0025
    marker.scale.y = 0.01
    marker.scale.z = 0.01

    marker.color.a = rgba[3]
    marker.color.r = rgba[0]
    marker.color.g = rgba[1]
    marker.color.b = rgba[2]

    return marker


def get_contact_point_vis_message(position, marker_id=0, rgba=(1.0, 0.0, 0.0, 1.0), base_frame="base"):
    marker = Marker()
    marker.header.frame_id = base_frame

    marker.ns = ""
    marker.id = marker_id

    marker.type = Marker.SPHERE
    marker.action = Marker.ADD

    marker.pose.position = Point(x=position[0], y=position[1], z=position[2])
    
    marker.scale.x = 0.005
    marker.scale.y = 0.005
    marker.scale.z = 0.005

    marker.color.a = rgba[3]
    marker.color.r = rgba[0]
    marker.color.g = rgba[1]
    marker.color.b = rgba[2]

    return marker


def get_hfmc_vis_message(msg, stamp, frame_id, action):
    """ Get ellipsoid markers for HFMC visualization """
    def get_ellipsold_marker(msg, stamp, frame_id, action):
        point = msg.points[i]
        direction = np.array(msg.directions[i].data).reshape(3, 3)
        # check orthogonal
        assert np.allclose(direction.T @ direction, np.eye(3), atol=1e-3)

        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = frame_id

        marker.ns = ""
        marker.type = Marker.SPHERE
        marker.action = action

        marker.pose.position.x = point.x
        marker.pose.position.y = point.y
        marker.pose.position.z = point.z

        quat = SciR.from_matrix(direction).as_quat()
        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]

        return marker
    
    nc = len(msg.points)
    vis_markers = MarkerArray()
    for i in range(nc):
        w_force = np.array([msg.w_force[i].x, msg.w_force[i].y, msg.w_force[i].z])
        w_position = np.array([msg.w_position[i].x, msg.w_position[i].y, msg.w_position[i].z])

        position_marker = get_ellipsold_marker(msg, stamp, frame_id, action)
        force_marker = get_ellipsold_marker(msg, stamp, frame_id, action)
        
        position_marker.scale.x, position_marker.scale.y, position_marker.scale.z = \
            max(0.03*w_position[0], 0.003), max(0.03*w_position[1], 0.003), max(0.03*w_position[2], 0.003)
        position_marker.color.r, position_marker.color.g, position_marker.color.b, position_marker.color.a = 0.0, 0.0, 1.0, 1.0
        position_marker.id = i

        force_marker.scale.x, force_marker.scale.y, force_marker.scale.z = \
            max(0.03*w_force[0], 0.003), max(0.03*w_force[1], 0.003), max(0.03*w_force[2], 0.003)
        force_marker.color.r, force_marker.color.g, force_marker.color.b, force_marker.color.a = 0.0, 1.0, 0.0, 1.0
        force_marker.id = i + nc

        vis_markers.markers.append(position_marker)
        vis_markers.markers.append(force_marker)

    return vis_markers

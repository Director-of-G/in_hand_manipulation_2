# All hardware states

# header
std_msgs/Header header

# dof state
std_msgs/Float32MultiArray q

# dof vel
std_msgs/Float32MultiArray v

# measured contact force
std_msgs/Float32MultiArray w_meas

# measured contact point
std_msgs/Float32MultiArray p_meas

# 6D original pose (for special purposes)
geometry_msgs/Pose obj_pose

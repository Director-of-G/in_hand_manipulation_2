# States for meshcat visualization

std_msgs/Bool vis_trajectory

# dof states
std_msgs/Float32MultiArray q
std_msgs/Float32MultiArray q_goal

# contact states
std_msgs/Float32MultiArray p_world
std_msgs/Float32MultiArray f_world

# solution trajectory
std_msgs/Float32MultiArray[] q_traj

# message for visualizing position/force controlled
# directions in Hybrid-Force-Motion-Control (HFMC)

# header
std_msgs/Header header

# fingertip center
geometry_msgs/Vector3[] points

# directions
std_msgs/Float32MultiArray[] directions

# weights
geometry_msgs/Vector3[] w_position
geometry_msgs/Vector3[] w_force

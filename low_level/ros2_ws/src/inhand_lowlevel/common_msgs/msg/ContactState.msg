# contact sensor output in sim.
std_msgs/Header header

# contact names, usually geom names
string[] names

# contact points, represented in the link frame that geom belongs to
geometry_msgs/Point[] points

# contact forces (with zero torque), represented in the link frame that geom belongs to
geometry_msgs/Wrench[] wrenches

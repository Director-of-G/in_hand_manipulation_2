cmake_minimum_required(VERSION 3.5)
project(common_msgs)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rosidl_typesupport_c REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ContactState.msg"
  "msg/HardwareStates.msg"
  "msg/JointTrajectoryWrapper.msg"
  "msg/MeshcatVis.msg"
  "msg/HybridFMCVis.msg"
  "srv/ReadHardwareStates.srv"
  DEPENDENCIES std_msgs geometry_msgs trajectory_msgs
)

#############
## Install ##
#############

###################################
## ament specific configuration  ##
###################################
ament_export_dependencies(rosidl_typesupport_c)
ament_export_dependencies(rosidl_default_runtime)

ament_package()
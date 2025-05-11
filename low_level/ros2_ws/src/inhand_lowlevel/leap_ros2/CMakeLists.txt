cmake_minimum_required(VERSION 3.5)
project(leap_ros2)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(common_msgs REQUIRED)
find_package(my_mujoco_ros REQUIRED)

# Skip if only run simulation
find_package(apriltag_ros QUIET)
if(NOT apriltag_ros_FOUND)
    message(STATUS "apriltag_ros not found, skipping.")
endif()

find_package(rosidl_default_generators REQUIRED)
find_package(rosidl_typesupport_c REQUIRED)

# rosidl_generate_interfaces(${PROJECT_NAME}
#   "msg/HardwareStates.msg"
#   DEPENDENCIES std_msgs
# )

#############
## Install ##
#############

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python scripts in the scripts directory
file(GLOB SCRIPT_FILES scripts/*.py)
install(PROGRAMS ${SCRIPT_FILES}
  # DESTINATION share/${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME}
)

# # Install all Python scripts in the launch directory (if any)
# file(GLOB LAUNCH_SCRIPTS launch/*.py)
# install(PROGRAMS ${LAUNCH_SCRIPTS}
#   DESTINATION share/${PROJECT_NAME}
# )

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING PATTERN "*.launch" PATTERN "*.launch.py"
)

# Install config files
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
  FILES_MATCHING PATTERN "*.yaml" PATTERN "*.sdf" PATTERN "*.json"
)

# Install all model files
install(DIRECTORY models/
  DESTINATION share/${PROJECT_NAME}/models
)

###################################
## ament specific configuration  ##
###################################
ament_export_dependencies(rosidl_typesupport_c)
ament_export_dependencies(rosidl_default_runtime)

ament_package()

cmake_minimum_required(VERSION 3.5)
project(contact_rich_control)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(common_msgs REQUIRED)
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

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING PATTERN "*.launch" PATTERN "*.launch.py"
)

# Install config files
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
  FILES_MATCHING PATTERN "*.yaml"
)

###################################
## ament specific configuration  ##
###################################
ament_export_dependencies(rosidl_typesupport_c)
ament_export_dependencies(rosidl_default_runtime)

ament_package()
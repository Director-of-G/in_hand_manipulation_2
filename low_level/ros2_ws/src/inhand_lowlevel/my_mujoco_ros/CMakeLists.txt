cmake_minimum_required(VERSION 3.5)
project(my_mujoco_ros)

# Find ROS 2 packages
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)

find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(common_msgs REQUIRED)

# Find the required Python packages
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(rosidl_default_generators REQUIRED)
find_package(rosidl_typesupport_c REQUIRED)

# Install Python dependencies
ament_python_install_package(${PROJECT_NAME})

# Generate messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/ApplyLinkFT.srv"
  DEPENDENCIES std_msgs geometry_msgs
)

# Install launch files
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# Install Python scripts in the scripts directory
install(
  PROGRAMS
    scripts/mujoco_ros_core.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install config files
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
  FILES_MATCHING PATTERN "*.yaml"
)

ament_export_dependencies(rosidl_typesupport_c)
ament_export_dependencies(rosidl_default_runtime)

ament_package()
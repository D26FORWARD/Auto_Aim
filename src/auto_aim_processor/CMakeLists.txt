cmake_minimum_required(VERSION 3.8)
project(auto_aim_processor)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(auto_aim_interfaces REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

# Add your header file directory
include_directories(
    include
    ${Eigen3_INCLUDE_DIRS}
)

# Add the source files
set(NODE_SRCS
    src/processor_node.cpp
    src/my_detector.cpp
    src/my_tracker.cpp
)

# Add the executable
add_library(${PROJECT_NAME}_lib SHARED
  ${NODE_SRCS}
)

target_include_directories(${PROJECT_NAME}_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME}_lib
  ${Eigen3_LIBS}
  ${OpenCV_LIBS}
  rclcpp::rclcpp
  rclcpp_components::rclcpp_components
  sensor_msgs::sensor_msgs
  cv_bridge::cv_bridge
  auto_aim_interfaces::auto_aim_interfaces
)

add_executable(combined_processor_node src/processor_node.cpp)
ament_target_dependencies(combined_processor_node rclcpp sensor_msgs image_transport cv_bridge auto_aim_interfaces)
target_link_libraries(combined_processor_node ${OpenCV_LIBS} ${Eigen3_LIBS})

# Install the executable
install(TARGETS
    combined_processor_node
    DESTINATION lib/${PROJECT_NAME}
)

# Install header files
install(
  DIRECTORY include/
  DESTINATION include
)

# Install the parameter file
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# Install the model file
install(
  DIRECTORY model
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter tests
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_dependencies(rclcpp sensor_msgs image_transport cv_bridge auto_aim_interfaces Eigen3 OpenCV)
ament_package()

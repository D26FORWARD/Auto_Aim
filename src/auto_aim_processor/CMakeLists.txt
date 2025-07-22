cmake_minimum_required(VERSION 3.5)
project(auto_aim_processor)

# 设置 C++ 标准
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 查找所有依赖包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(auto_aim_interfaces REQUIRED)
find_package(cv_bridge REQUIRED) # 添加了这一行
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)

# 1. 先创建库
add_library(auto_aim_processor_lib SHARED
  src/my_detector.cpp
  src/my_tracker.cpp
)

# 2. 然后为库添加依赖和头文件目录
ament_target_dependencies(auto_aim_processor_lib
  rclcpp
  rclcpp_components
  std_msgs
  sensor_msgs
  auto_aim_interfaces
  tf2_ros
  tf2_geometry_msgs
  visualization_msgs
  Eigen3
  OpenCV
  fmt
)
target_include_directories(auto_aim_processor_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# 1. 先创建可执行文件
add_executable(processor_node src/processor_node.cpp)

# 2. 然后为可执行文件添加依赖、链接库和头文件目录
ament_target_dependencies(processor_node
  rclcpp
  rclcpp_components
  std_msgs
  sensor_msgs
  auto_aim_interfaces
  cv_bridge # 添加了这一行
  tf2_ros
  visualization_msgs
)
target_link_libraries(processor_node auto_aim_processor_lib)
target_include_directories(processor_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# 安装目标文件和头文件
install(TARGETS
  auto_aim_processor_lib
  processor_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

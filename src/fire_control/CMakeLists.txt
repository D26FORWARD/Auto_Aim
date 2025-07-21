cmake_minimum_required(VERSION 3.8)
project(fire_control)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 去掉 -Werror 或者添加 -Wno-sign-compare 忽略有符号和无符号整数比较警告
add_definitions(-Wall -Wno-sign-compare)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_auto REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(auto_aim_interfaces REQUIRED)

ament_auto_find_build_dependencies()

# 主库
ament_auto_add_library(${PROJECT_NAME} SHARED
  DIRECTORY src
)

add_executable(fire_control_node src/fire_control_node.cpp)

ament_target_dependencies(fire_control_node
  auto_aim_interfaces
  rclcpp
  Eigen3
  rcl_interfaces # 添加 rcl_interfaces 依赖
)

include_directories("/usr/include/eigen3")

target_include_directories(fire_control_node PUBLIC
  ${EIGEN3_INCLUDE_DIRS}
  /usr/include/eigen3
  ${rclcpp_INCLUDE_DIRS}
  ${rcl_interfaces_INCLUDE_DIRS}
)

target_link_libraries(fire_control_node
  Eigen3::Eigen
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()


install(TARGETS
  fire_control_node
  DESTINATION lib/${PROJECT_NAME}
)

# ament_auto_package(
#   INSTALL_TO_SHARE
#   model
# )  

ament_package()

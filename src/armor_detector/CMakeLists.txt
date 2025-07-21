cmake_minimum_required(VERSION 3.10)
project(armor_detector)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 去掉 -Werror 或者添加 -Wno-sign-compare 忽略有符号和无符号整数比较警告
add_definitions(-Wall -Wno-sign-compare)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#######################
## Find dependencies ##
#######################
find_package(ament_cmake_auto REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(auto_aim_interfaces REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
# 添加对 sensor_msgs 的查找
find_package(sensor_msgs REQUIRED)
# 添加对 cv_bridge 的查找
find_package(cv_bridge REQUIRED)
# 添加对 rcl_interfaces 的查找
find_package(rcl_interfaces REQUIRED)
ament_auto_find_build_dependencies()

###########
## Build ##
###########

# 主库
ament_auto_add_library(${PROJECT_NAME} SHARED
  DIRECTORY src
)

include_directories(${OpenCV_INCLUDE_DIRS})

target_include_directories(${PROJECT_NAME} PUBLIC 
  ${OpenCV_INCLUDE_DIRS}
  ${rclcpp_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
  ${cv_bridge_INCLUDE_DIRS}
  ${rcl_interfaces_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# rclcpp component 注册
rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN rm_auto_aim::ArmorDetectorNode
  EXECUTABLE armor_detector_node
)

# quaternion_to_tf_node 独立节点
add_executable(quaternion_to_tf_node src/odom_to_camera_node.cpp)

# add_executable(detection_hj_node src/detection_hj_node.cpp)
include_directories("/usr/include/eigen3")

include_directories(${PROJECT_SOURCE_DIR}/include/armor_detector)

ament_target_dependencies(quaternion_to_tf_node
  rclcpp
  auto_aim_interfaces
  tf2
  tf2_ros
  geometry_msgs
  Eigen3
  tf2_geometry_msgs
  sensor_msgs # 添加 sensor_msgs 依赖
  cv_bridge # 添加 cv_bridge 依赖
  rcl_interfaces # 添加 rcl_interfaces 依赖
)

target_include_directories(quaternion_to_tf_node PUBLIC
  ${EIGEN3_INCLUDE_DIRS}
  /usr/include/eigen3
  ${rclcpp_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
  ${cv_bridge_INCLUDE_DIRS}
  ${rcl_interfaces_INCLUDE_DIRS}
)

# target_include_directories(detection_hj_node PUBLIC
#   ${EIGEN3_INCLUDE_DIRS}
#   /usr/include/eigen3
#   ${OpenCV_INCLUDE_DIRS} # 确保 detection_hj_node 包含 OpenCV 头文件
#   ${rclcpp_INCLUDE_DIRS}
#   ${sensor_msgs_INCLUDE_DIRS}
#   ${cv_bridge_INCLUDE_DIRS}
#   ${rcl_interfaces_INCLUDE_DIRS}
# )

target_link_libraries(quaternion_to_tf_node
  Eigen3::Eigen
  ${OpenCV_LIBS} # 确保 quaternion_to_tf_node 链接 OpenCV 库
)

# target_link_libraries(detection_hj_node
#   ${OpenCV_LIBS} # 确保 detection_hj_node 链接 OpenCV 库
#   ${cv_bridge_LIBRARIES}
# )

#############
## Install ##
#############

install(TARGETS
  quaternion_to_tf_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_auto_package(
  INSTALL_TO_SHARE
  model
)    
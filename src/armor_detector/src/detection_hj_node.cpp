// #include "Armor_Detection/include/Armor_detection.h"
#include "../include/armor_detector/Armor_detection.hpp"
// STD
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>  // 添加 rclcpp 头文件
#include <rclcpp/duration.hpp>
#include <rclcpp/qos.hpp>
#include <sensor_msgs/msg/image.hpp>  // 添加 sensor_msgs 头文件
#include <cv_bridge/cv_bridge.h>  // 添加 cv_bridge 头文件




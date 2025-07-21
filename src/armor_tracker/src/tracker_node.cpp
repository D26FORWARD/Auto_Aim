// Copyright (C) 2022 ChenJun
// Copyright (C) 2024 Zheng Yu
// Licensed under the MIT License.

#include "armor_tracker/tracker_node.hpp"

// STD
#include <cmath>
#include <memory>
#include <vector>

//! 空气阻力模型参数
// robot_struct里面有，这么写不美观
#define GRAVITY 9.78
#define SMALL_AIR_K 0.01903
#define BIG_AIR_K 0.00556
#define BIG_LIGHT_AIR_K 0.00530

#define FIRE_DELAY 0.153
#define BULLET_V 22.3
#define ANGLE_V_THRESHOLD 1.5

namespace rm_auto_aim
{
ArmorTrackerNode::ArmorTrackerNode(const rclcpp::NodeOptions & options)
: Node("armor_tracker", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting TrackerNode!");

  // Maximum allowable armor distance in the XOY plane
  max_armor_distance_ = this->declare_parameter("max_armor_distance", 10.0);

  // Vision data subscriber
  vision_data_sub_ = this->create_subscription<auto_aim_interfaces::msg::Vision>(
    "/Vision_data", 10,
    std::bind(&ArmorTrackerNode::vision_callback, this, std::placeholders::_1));

  // ctrl msg publisher
  ctrl_pub_ = this->create_publisher<auto_aim_interfaces::msg::RobotCtrl>(
    "/Robot_ctrl_data", 10);

  // Tracker
  double max_match_distance = this->declare_parameter("tracker.max_match_distance", 0.15);
  double max_match_yaw_diff_ = this->declare_parameter("tracker.max_match_yaw_diff", 1.0);
  tracker_ = std::make_unique<Tracker>(max_match_distance, max_match_yaw_diff_);
  tracker_->tracking_thres = this->declare_parameter("tracker.tracking_thres", 5);
  lost_time_thres_ = this->declare_parameter("tracker.lost_time_thres", 0.3);

  // EKF
  // xa = x_armor, xc = x_robot_center
  // state: xc, v_xc, yc, v_yc, za, v_za, yaw, v_yaw, r
  // measurement: xa, ya, za, yaw
  // f - Process function
  auto f = [this](const Eigen::VectorXd & x) {
    Eigen::VectorXd x_new = x;
    x_new(0) += x(1) * dt_;
    x_new(2) += x(3) * dt_;
    x_new(4) += x(5) * dt_;
    x_new(6) += x(7) * dt_;
    return x_new;
  };
  // J_f - Jacobian of process function
  auto j_f = [this](const Eigen::VectorXd &) {
    Eigen::MatrixXd f(9, 9);
    // clang-format off
    f <<  1,   dt_, 0,   0,   0,   0,   0,   0,   0,
          0,   1,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   1,   dt_, 0,   0,   0,   0,   0, 
          0,   0,   0,   1,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   1,   dt_, 0,   0,   0,
          0,   0,   0,   0,   0,   1,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   1,   dt_, 0,
          0,   0,   0,   0,   0,   0,   0,   1,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   1;
    // clang-format on
    return f;
  };
  // h - Observation function
  auto h = [](const Eigen::VectorXd & x) {
    Eigen::VectorXd z(4);
    double xc = x(0), yc = x(2), yaw = x(6), r = x(8);
    z(0) = xc - r * cos(yaw);  // xa
    z(1) = yc - r * sin(yaw);  // ya
    z(2) = x(4);               // za
    z(3) = x(6);               // yaw
    return z;
  };
  // J_h - Jacobian of observation function
  auto j_h = [](const Eigen::VectorXd & x) {
    Eigen::MatrixXd h(4, 9);
    double yaw = x(6), r = x(8);
    // clang-format off
    //    xc   v_xc yc   v_yc za   v_za yaw         v_yaw r
    h <<  1,   0,   0,   0,   0,   0,   r*sin(yaw), 0,   -cos(yaw),
          0,   0,   1,   0,   0,   0,   -r*cos(yaw),0,   -sin(yaw),
          0,   0,   0,   0,   1,   0,   0,          0,   0,
          0,   0,   0,   0,   0,   0,   1,          0,   0;
    // clang-format on
    return h;
  };
  // update_Q - process noise covariance matrix
  s2qxyz_max_ = declare_parameter("ekf.sigma2_q_xyz_max", 0.1);
  s2qxyz_min_ = declare_parameter("ekf.sigma2_q_xyz_min", 0.05);
  s2qyaw_max_ = declare_parameter("ekf.sigma2_q_yaw_max", 10.0);
  s2qyaw_min_ = declare_parameter("ekf.sigma2_q_yaw_min", 5.0);
  s2qr_ = declare_parameter("ekf.sigma2_q_r", 80.0);
  auto u_q = [this](const Eigen::VectorXd & x_p) {
    double vx = x_p(1), vy = x_p(3), v_yaw = x_p(7);
    double dx = pow(pow(vx, 2) + pow(vy, 2), 0.5);
    double dy = abs(v_yaw);
    Eigen::MatrixXd q(9, 9);
    double x, y;
    x = exp(-dy) * (s2qxyz_max_ - s2qxyz_min_) + s2qxyz_min_;
    y = exp(-dx) * (s2qyaw_max_ - s2qyaw_min_) + s2qyaw_min_;
    double t = dt_, r = s2qr_;
    double q_x_x = pow(t, 4) / 4 * x, q_x_vx = pow(t, 3) / 2 * x, q_vx_vx = pow(t, 2) * x;
    double q_y_y = pow(t, 4) / 4 * y, q_y_vy = pow(t, 3) / 2 * x, q_vy_vy = pow(t, 2) * y;
    double q_r = pow(t, 4) / 4 * r;
    // clang-format off
    //    xc      v_xc    yc      v_yc    za      v_za    yaw     v_yaw   r
    q <<  q_x_x,  q_x_vx, 0,      0,      0,      0,      0,      0,      0,
          q_x_vx, q_vx_vx,0,      0,      0,      0,      0,      0,      0,
          0,      0,      q_x_x,  q_x_vx, 0,      0,      0,      0,      0,
          0,      0,      q_x_vx, q_vx_vx,0,      0,      0,      0,      0,
          0,      0,      0,      0,      q_x_x,  q_x_vx, 0,      0,      0,
          0,      0,      0,      0,      q_x_vx, q_vx_vx,0,      0,      0,
          0,      0,      0,      0,      0,      0,      q_y_y,  q_y_vy, 0,
          0,      0,      0,      0,      0,      0,      q_y_vy, q_vy_vy,0,
          0,      0,      0,      0,      0,      0,      0,      0,      q_r;
    // clang-format on
    return q;
  };
  // update_R - measurement noise covariance matrix
  r_xyz_factor = declare_parameter("ekf.r_xyz_factor", 0.05);
  r_yaw = declare_parameter("ekf.r_yaw", 0.02);
  auto u_r = [this](const Eigen::VectorXd & z) {
    Eigen::DiagonalMatrix<double, 4> r;
    double x = r_xyz_factor;
    r.diagonal() << abs(x * z[0]), abs(x * z[1]), abs(x * z[2]), r_yaw;
    return r;
  };
  // P - error estimate covariance matrix
  Eigen::DiagonalMatrix<double, 9> p0;
  p0.setIdentity();
  tracker_->ekf = ExtendedKalmanFilter{f, h, j_f, j_h, u_q, u_r, p0};

  // Reset tracker service
  using std::placeholders::_1;
  using std::placeholders::_2;
  using std::placeholders::_3;
  reset_tracker_srv_ = this->create_service<std_srvs::srv::Trigger>(
    "/tracker/reset", [this](
                        const std_srvs::srv::Trigger::Request::SharedPtr,
                        std_srvs::srv::Trigger::Response::SharedPtr response) {
      tracker_->tracker_state = Tracker::LOST;
      response->success = true;
      RCLCPP_INFO(this->get_logger(), "Tracker reset!");
      return;
    });

  // Change target service
  change_target_srv_ = this->create_service<std_srvs::srv::Trigger>(
    "/tracker/change", [this](
                         const std_srvs::srv::Trigger::Request::SharedPtr,
                         std_srvs::srv::Trigger::Response::SharedPtr response) {
      tracker_->tracker_state = Tracker::CHANGE_TARGET;
      response->success = true;
      RCLCPP_INFO(this->get_logger(), "Target change!");
      return;
    });

  // Subscriber with tf2 message_filter
  // tf2 relevant
  tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  // Create the timer interface before call to waitForTransform,
  // to avoid a tf2_ros::CreateTimerInterfaceException exception
  auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
    this->get_node_base_interface(), this->get_node_timers_interface());
  tf2_buffer_->setCreateTimerInterface(timer_interface);
  tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);
  // subscriber and filter
  armors_sub_.subscribe(this, "/detector/armors", rmw_qos_profile_sensor_data);
  target_frame_ = this->declare_parameter("target_frame", "odom");
  // target_frame_ = this->declare_parameter("target_frame", "camera_switch");

  tf2_filter_ = std::make_shared<tf2_filter>(
    armors_sub_, *tf2_buffer_, target_frame_, 10, this->get_node_logging_interface(),
    this->get_node_clock_interface(), std::chrono::duration<int>(1));
  // Register a callback with tf2_ros::MessageFilter to be called when transforms are available
  tf2_filter_->registerCallback(&ArmorTrackerNode::armorsCallback, this);

  // Measurement publisher (for debug usage)
  info_pub_ = this->create_publisher<auto_aim_interfaces::msg::TrackerInfo>("/tracker/info", 10);

  // Publisher
  target_pub_ = this->create_publisher<auto_aim_interfaces::msg::Target>(
    "/tracker/target", rclcpp::SensorDataQoS());

  // Visualization Marker Publisher
  // See http://wiki.ros.org/rviz/DisplayTypes/Marker
  position_marker_.ns = "position";
  position_marker_.type = visualization_msgs::msg::Marker::SPHERE;
  position_marker_.scale.x = position_marker_.scale.y = position_marker_.scale.z = 0.1;
  position_marker_.color.a = 1.0;
  position_marker_.color.g = 1.0;
  linear_v_marker_.type = visualization_msgs::msg::Marker::ARROW;
  linear_v_marker_.ns = "linear_v";
  linear_v_marker_.scale.x = 0.03;
  linear_v_marker_.scale.y = 0.05;
  linear_v_marker_.color.a = 1.0;
  linear_v_marker_.color.r = 1.0;
  linear_v_marker_.color.g = 1.0;
  angular_v_marker_.type = visualization_msgs::msg::Marker::ARROW;
  angular_v_marker_.ns = "angular_v";
  angular_v_marker_.scale.x = 0.03;
  angular_v_marker_.scale.y = 0.05;
  angular_v_marker_.color.a = 1.0;
  angular_v_marker_.color.b = 1.0;
  angular_v_marker_.color.g = 1.0;
  armor_marker_.ns = "armors";
  armor_marker_.type = visualization_msgs::msg::Marker::CUBE;
  armor_marker_.scale.x = 0.03;
  armor_marker_.scale.z = 0.125;
  armor_marker_.color.a = 1.0;
  armor_marker_.color.r = 1.0;
  marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/tracker/marker", 10);

  predict_position_marker_.ns = "predict_position";
  predict_position_marker_.type = visualization_msgs::msg::Marker::SPHERE;
  predict_position_marker_.scale.x = position_marker_.scale.y = position_marker_.scale.z = 0.1;
  predict_position_marker_.color.a = 1.0;
  predict_position_marker_.color.g = 1.0;
  predict_linear_v_marker_.type = visualization_msgs::msg::Marker::ARROW;
  predict_linear_v_marker_.ns = "predict_linear_v";
  predict_linear_v_marker_.scale.x = 0.03;
  predict_linear_v_marker_.scale.y = 0.05;
  predict_linear_v_marker_.color.a = 1.0;
  predict_linear_v_marker_.color.r = 1.0;
  predict_linear_v_marker_.color.g = 1.0;
  predict_angular_v_marker_.type = visualization_msgs::msg::Marker::ARROW;
  predict_angular_v_marker_.ns = "predict_angular_v";
  predict_angular_v_marker_.scale.x = 0.03;
  predict_angular_v_marker_.scale.y = 0.05;
  predict_angular_v_marker_.color.a = 1.0;
  predict_angular_v_marker_.color.b = 1.0;
  predict_angular_v_marker_.color.g = 1.0;
  predict_armor_marker_.ns = "predict_armors";
  predict_armor_marker_.type = visualization_msgs::msg::Marker::CUBE;
  predict_armor_marker_.scale.x = 0.03;
  predict_armor_marker_.scale.z = 0.125;
  predict_armor_marker_.color.a = 1.0;
  predict_armor_marker_.color.r = 1.0;
  predict_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/tracker/predict_marker", 10);
}


void ArmorTrackerNode::vision_callback(const auto_aim_interfaces::msg::Vision::ConstSharedPtr& vision_msg)
{
    now_pitch_ = vision_msg->pitch;  // pitch
    now_yaw_ = vision_msg->yaw;    // yaw
    // std::cout<<"NOW: pitch:"<< now_pitch_ << " yaw: "<< now_yaw_ <<std::endl;
}

void ArmorTrackerNode::armorsCallback(const auto_aim_interfaces::msg::Armors::SharedPtr armors_msg)
{
  // Tranform armor position from image frame to world coordinate
  ////////////// ADD
  // auto Robot_ctrl_t = std::make_shared<auto_aim_interfaces::msg::RobotCtrl>();
  // // double xc, yc, zc;
  // Eigen::Vector3d closest_armor_xyz;
  // double min_distance = 100;
  // int armor_num = 0;
  ////////////// ENDADD
  for (auto & armor : armors_msg->armors) {
    geometry_msgs::msg::PoseStamped ps;
    ps.header = armors_msg->header;
    ps.pose = armor.pose;
    try {
      armor.pose = tf2_buffer_->transform(ps, target_frame_).pose;
    } catch (const tf2::ExtrapolationException & ex) {
      RCLCPP_ERROR(get_logger(), "Error while transforming %s", ex.what());
      return;
    }
    ////////////// ADD
    // Eigen::Vector3d armor_xyz(armor.pose.position.x,  armor.pose.position.y, armor.pose.position.z);
    // double distance = (armor_xyz - Eigen::Vector3d(0, 0, 0)).norm();
    // if (distance < min_distance) {
    //   min_distance = distance;
    //   closest_armor_xyz = armor_xyz;
    // }
    // armor_num ++;
    ////////////// ENDADD
  }

  ////////////// ADD
  // 不应该在这里写火控，后面要改
  // if(armor_num > 0){
  //   Robot_ctrl_t->target_lock = 49;
  //   // 使用最近装甲板的位置计算枪管角度
  //   Eigen::Vector3d rpy = Barrel_Solve(closest_armor_xyz);
  //   Robot_ctrl_t->pitch = rpy[1];
  //   Robot_ctrl_t->yaw = rpy[2];
  // }else{
  //   // 如果没有敌人，控制量等于当前量并不开火
  //   Robot_ctrl_t->target_lock = 50;
  //   Robot_ctrl_t->pitch = now_pitch_;
  //   Robot_ctrl_t->yaw = now_yaw_;
  //   Robot_ctrl_t->fire_command = 0;
  // }
  ////////////// ENDADD

  // Filter abnormal armors
  armors_msg->armors.erase(
    std::remove_if(
      armors_msg->armors.begin(), armors_msg->armors.end(),
      [this](const auto_aim_interfaces::msg::Armor & armor) {
        return abs(armor.pose.position.z) > 1.2 ||
               Eigen::Vector2d(armor.pose.position.x, armor.pose.position.y).norm() >
                 max_armor_distance_;
      }),
    armors_msg->armors.end());

  // Init message
  auto_aim_interfaces::msg::TrackerInfo info_msg;
  auto_aim_interfaces::msg::Target target_msg;
  rclcpp::Time time = armors_msg->header.stamp;
  target_msg.header.stamp = time;
  target_msg.header.frame_id = target_frame_;

  // Update tracker
  if (tracker_->tracker_state == Tracker::LOST) {
    tracker_->init(armors_msg);
    target_msg.tracking = false;
  } else {
    dt_ = (time - last_time_).seconds();
    tracker_->lost_thres = static_cast<int>(lost_time_thres_ / dt_);

    tracker_->update(armors_msg);

    // Publish Info
    info_msg.position_diff = tracker_->info_position_diff;
    info_msg.yaw_diff = tracker_->info_yaw_diff;
    info_msg.position.x = tracker_->measurement(0);
    info_msg.position.y = tracker_->measurement(1);
    info_msg.position.z = tracker_->measurement(2);
    info_msg.yaw = tracker_->measurement(3);

    // std::cout<<"measurement(0): "<<tracker_->measurement(0)<<"  measurement(1): "<<tracker_->measurement(1)<<"  measurement(2): "<<tracker_->measurement(2)<<std::endl;

    info_pub_->publish(info_msg);

    if (tracker_->tracker_state == Tracker::DETECTING) {
      target_msg.tracking = false;
    } else if (
      tracker_->tracker_state == Tracker::TRACKING ||
      tracker_->tracker_state == Tracker::TEMP_LOST) {
      target_msg.tracking = true;
      // Fill target message
      const auto & state = tracker_->target_state;
      target_msg.id = tracker_->tracked_id;
      target_msg.armors_num = static_cast<int>(tracker_->tracked_armors_num);
      target_msg.position.x = state(0);
      target_msg.velocity.x = state(1);
      target_msg.position.y = state(2);
      target_msg.velocity.y = state(3);
      target_msg.position.z = state(4);
      target_msg.velocity.z = state(5);
      target_msg.yaw = state(6);
      target_msg.v_yaw = state(7);
      target_msg.radius_1 = state(8);
      target_msg.radius_2 = tracker_->another_r;
      target_msg.dz = tracker_->dz;
    } else if (tracker_->tracker_state == Tracker::CHANGE_TARGET) {
      target_msg.tracking = false;
    }
  }

  last_time_ = time;

  target_pub_->publish(target_msg);
  auto target_msg_ptr = std::make_shared<const auto_aim_interfaces::msg::Target>(target_msg);
  target_sub_callback(target_msg_ptr);
  publishMarkers(target_msg);
}

// This is the final visuable markers publisher function. 
void ArmorTrackerNode::publishMarkers(const auto_aim_interfaces::msg::Target & target_msg)
{
  // const double v_bullet = 20.7; // 固定弹速，单位：米每秒
  // // 计算子弹飞行时间
  // double distance_to_origin = std::sqrt(xc * xc + yc * yc + zc * zc);
  // double t = distance_to_origin / v_bullet + 0.02;

  // // 预测t秒后中心位置
  // double predicted_xc = xc + vx * t;
  // double predicted_yc = yc + vy * t;
  // double predicted_zc = zc + vz * t;
  // double predicted_yaw = target_msg->yaw + v_yaw * t;

  position_marker_.header = target_msg.header;
  linear_v_marker_.header = target_msg.header;
  angular_v_marker_.header = target_msg.header;
  armor_marker_.header = target_msg.header;

  visualization_msgs::msg::MarkerArray marker_array;
  if (target_msg.tracking) {
    double yaw = target_msg.yaw, r1 = target_msg.radius_1, r2 = target_msg.radius_2;
    double xc = target_msg.position.x, yc = target_msg.position.y, za = target_msg.position.z;
    double vx = target_msg.velocity.x, vy = target_msg.velocity.y, vz = target_msg.velocity.z;
    double dz = target_msg.dz;

    position_marker_.action = visualization_msgs::msg::Marker::ADD;
    position_marker_.pose.position.x = xc;
    position_marker_.pose.position.y = yc;
    position_marker_.pose.position.z = za + dz / 2;

    linear_v_marker_.action = visualization_msgs::msg::Marker::ADD;
    linear_v_marker_.points.clear();
    linear_v_marker_.points.emplace_back(position_marker_.pose.position);
    geometry_msgs::msg::Point arrow_end = position_marker_.pose.position;
    arrow_end.x += vx;
    arrow_end.y += vy;
    arrow_end.z += vz;
    linear_v_marker_.points.emplace_back(arrow_end);

    angular_v_marker_.action = visualization_msgs::msg::Marker::ADD;
    angular_v_marker_.points.clear();
    angular_v_marker_.points.emplace_back(position_marker_.pose.position);
    arrow_end = position_marker_.pose.position;
    arrow_end.z += target_msg.v_yaw / M_PI;
    angular_v_marker_.points.emplace_back(arrow_end);

    armor_marker_.action = visualization_msgs::msg::Marker::ADD;
    armor_marker_.scale.y = tracker_->tracked_armor.type == "small" ? 0.135 : 0.23;
    bool is_current_pair = true;
    size_t a_n = target_msg.armors_num;
    geometry_msgs::msg::Point p_a;
    double r = 0;
    for (size_t i = 0; i < a_n; i++) {
      double tmp_yaw = yaw + i * (2 * M_PI / a_n);
      // Only 4 armors has 2 radius and height
      if (a_n == 4) {
        r = is_current_pair ? r1 : r2;
        p_a.z = za + (is_current_pair ? 0 : dz);
        is_current_pair = !is_current_pair;
      } else {
        r = r1;
        p_a.z = za;
      }
      p_a.x = xc - r * cos(tmp_yaw);
      p_a.y = yc - r * sin(tmp_yaw);

      armor_marker_.id = i;
      armor_marker_.pose.position = p_a;
      tf2::Quaternion q;
      q.setRPY(0, target_msg.id == "outpost" ? -0.26 : 0.26, tmp_yaw);
      armor_marker_.pose.orientation = tf2::toMsg(q);
      marker_array.markers.emplace_back(armor_marker_);

      // ADD
      // 计算左端点和右端点的坐标
      double armor_width = 0.11; // 假设这是装甲板宽度
      geometry_msgs::msg::Point left_endpoint, right_endpoint;
      double half_width = armor_width / 2;
      left_endpoint.x = p_a.x - half_width * sin(tmp_yaw);
      left_endpoint.y = p_a.y + half_width * cos(tmp_yaw);
      left_endpoint.z = p_a.z;

      right_endpoint.x = p_a.x + half_width * sin(tmp_yaw);
      right_endpoint.y = p_a.y - half_width * cos(tmp_yaw);
      right_endpoint.z = p_a.z;

      // 创建用于显示左端点的标记
      visualization_msgs::msg::Marker left_marker;
      left_marker.header = target_msg.header;
      left_marker.ns = "armor_endpoints";
      left_marker.id = i * 2;
      left_marker.type = visualization_msgs::msg::Marker::SPHERE;
      left_marker.action = visualization_msgs::msg::Marker::ADD;
      left_marker.pose.position = left_endpoint;
      left_marker.pose.orientation.x = 0.0;
      left_marker.pose.orientation.y = 0.0;
      left_marker.pose.orientation.z = 0.0;
      left_marker.pose.orientation.w = 1.0;
      left_marker.scale.x = 0.05;
      left_marker.scale.y = 0.05;
      left_marker.scale.z = 0.05;
      left_marker.color.r = 1.0;
      left_marker.color.g = 0.0;
      left_marker.color.b = 0.0;
      left_marker.color.a = 1.0;
      marker_array.markers.emplace_back(left_marker);

      // 创建用于显示右端点的标记
      visualization_msgs::msg::Marker right_marker;
      right_marker.header = target_msg.header;
      right_marker.ns = "armor_endpoints";
      right_marker.id = i * 2 + 1;
      right_marker.type = visualization_msgs::msg::Marker::SPHERE;
      right_marker.action = visualization_msgs::msg::Marker::ADD;
      right_marker.pose.position = right_endpoint;
      right_marker.pose.orientation.x = 0.0;
      right_marker.pose.orientation.y = 0.0;
      right_marker.pose.orientation.z = 0.0;
      right_marker.pose.orientation.w = 1.0;
      right_marker.scale.x = 0.05;
      right_marker.scale.y = 0.05;
      right_marker.scale.z = 0.05;
      right_marker.color.r = 0.0;
      right_marker.color.g = 1.0;
      right_marker.color.b = 0.0;
      right_marker.color.a = 1.0;
      marker_array.markers.emplace_back(right_marker);

      // ADD
    }
  } else {
    position_marker_.action = visualization_msgs::msg::Marker::DELETEALL;
    linear_v_marker_.action = visualization_msgs::msg::Marker::DELETEALL;
    angular_v_marker_.action = visualization_msgs::msg::Marker::DELETEALL;

    armor_marker_.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_array.markers.emplace_back(armor_marker_);
  }

  marker_array.markers.emplace_back(position_marker_);
  marker_array.markers.emplace_back(linear_v_marker_);
  marker_array.markers.emplace_back(angular_v_marker_);
  marker_pub_->publish(marker_array);
}

/**
 *  函数名: Barrel_Solve
 *  传入: Eigen::Vector3d position
 *  传出: Vector3d rpy
 *  功能: 通过传入的坐标计算枪管移动角度
 */
Eigen::Vector3d ArmorTrackerNode::Barrel_Solve(Eigen::Vector3d position) {

  // std::cout<<"position[0]: "<<position[0]<<

  Eigen::Vector3d rpy;
  rpy[2] = -atan2(position[0],position[1]) / M_PI*180.0;
  rpy[1] = atan2(position[2],position[1]) / M_PI*180.0;
  rpy[0] = atan2(position[2],position[0]) / M_PI*180.0;

  // 单独计算抬升角度 | atan2()计算角度在车旋转180度之后会解算出[90-180]
  rpy[1] = airResistanceSolve(position);  // 暂时禁用

  // 添加枪管补偿
  rpy[0] += 0;
  rpy[1] += 0;
  rpy[2] += 0;

  return rpy;
}

/**
 *  函数名: BulletModel
 *  传入: float x, float v, float angle       (水平距离,弹速,抬升角度)
 *  传出: float y                             (落点高度)
 *  功能: 通过传入的坐标计算枪管移动角度
 *  弹道模型推导:
 *  https://robomaster-oss.github.io/rmoss_tutorials/#/rmoss_core/rmoss_projectile_motion/projectile_motion_iteration
 */
float ArmorTrackerNode::BulletModel(float x, float v, float angle) { //x:m,v:m/s,angle:rad
  float t,y;
#ifndef Hero
  t = (float)((exp(SMALL_AIR_K * x) - 1) / (SMALL_AIR_K * v * cos(angle)));
#else
  t = (float)((exp(BIG_AIR_K * x) - 1) / (BIG_AIR_K * v * cos(angle)));
#endif
  y = (float)(v * sin(angle) * t - GRAVITY * t * t/* * cos(ab_pitch)*/ / 2);
  return y;
}

float ArmorTrackerNode::airResistanceSolve(Eigen::Vector3d Pos) {

  // -----------要水平距离的融合，否则计算的距离会少，在视野边缘处误差会大----------
  float x = (float)sqrt(Pos[0]*Pos[0]+ Pos[1]*Pos[1]);
  float y = (float)Pos[2];

  float y_temp, y_actual, dy;
  float Solve_pitch;
  y_temp = y;

  float bullet_speed;
#ifndef Hero
  bullet_speed = 25;
#else
  bullet_speed = 15;
#endif
  // 迭代法 | 使用弹道模型对落点高度进行计算,直到落点高度达到要求,得到抬升角度
  for (int i = 0; i < 20; i++)
  {
      Solve_pitch = (float)atan2(y_temp, x);                        // 计算当前枪管抬升角度
      y_actual = BulletModel(x, bullet_speed, Solve_pitch);           // 得到落点高度
      dy = y - y_actual;                                               // 计算落点高度和目标高度的误差
      y_temp = y_temp + dy;                                            // 误差补偿

      //! 误差精度达到一定要求
      if (fabsf(dy) < 0.00001) {
          break;
      }
  }
  Solve_pitch = Solve_pitch*180.f/M_PI;  // 解算角度
  return Solve_pitch;
}

// fire control
// 最好不要在这里写火控，只是暂时因为和火控包通信没成功做此妥协
void ArmorTrackerNode::target_sub_callback(const auto_aim_interfaces::msg::Target::ConstSharedPtr& target_msg) 
{       
    RCLCPP_INFO(this->get_logger(), "INTO target_sub_callback");
    // std::cout<<"INTO target_sub_callback"<<std::endl;
    auto Robot_ctrl_t = std::make_shared<auto_aim_interfaces::msg::RobotCtrl>();
    // const double v_bullet = 20.7; // 固定弹速，单位：米每秒
    // const double angular_velocity_threshold = 1.5; // 角速度阈值，可根据实际情况调整
    // const double 

    if (target_msg->tracking) {
        Robot_ctrl_t->target_lock = 49;

        double xc = target_msg->position.x, yc = target_msg->position.y, zc = target_msg->position.z;
        double vx = target_msg->velocity.x, vy = target_msg->velocity.y, vz = target_msg->velocity.z;
        double v_yaw = target_msg->v_yaw;
        size_t a_n = target_msg->armors_num;
        double min_distance = std::numeric_limits<double>::max();
        Eigen::Vector3d closest_armor_xyz;

        // 计算子弹飞行时间
        double distance_to_origin = std::sqrt(xc * xc + yc * yc + zc * zc);
        double t = distance_to_origin / BULLET_V + FIRE_DELAY;

        // 预测t秒后中心位置
        double predicted_xc = xc + vx * t;
        double predicted_yc = yc + vy * t;
        double predicted_zc = zc + vz * t;
        double predicted_yaw = target_msg->yaw + v_yaw * t;

        // 遍历所有装甲板，找到距离最近的装甲板
        for (size_t i = 0; i < a_n; i++) {
            double yaw = predicted_yaw + i * (2 * M_PI / a_n);
            double r1 = target_msg->radius_1, r2 = target_msg->radius_2;
            double za = predicted_zc;
            double dz = target_msg->dz;
            double r = 0;
            double p_a_z = 0;
            // Only 4 armors has 2 radius and height
            if (a_n == 4) {
                bool is_current_pair = (i % 2 == 0);
                r = is_current_pair ? r1 : r2;
                p_a_z = za + (is_current_pair ? 0 : dz);
            } else {
                r = r1;
                p_a_z = za;
            }
            double p_a_x = predicted_xc - r * cos(yaw);
            double p_a_y = predicted_yc - r * sin(yaw);

            // std::cout<<"p_a_x: "<<p_a_x<<" p_a_y: "<<p_a_y<<"p_a_z: "<<p_a_z<<std::endl;

            Eigen::Vector3d armor_xyz(p_a_x, p_a_y, p_a_z);
            double distance = (armor_xyz - Eigen::Vector3d(0, 0, 0)).norm();

            if (distance < min_distance) {
                min_distance = distance;
                closest_armor_xyz = armor_xyz;
            }
        }

        // 使用最近装甲板的位置计算枪管角度
        Eigen::Vector3d rpy;

        // if (std::abs(v_yaw) > ANGLE_V_THRESHOLD) {
        //   // yaw轴对准四个装甲板的中心，但是pitch高度是预测装甲板的高度
        //   // Eigen::Vector3d fake_armor_xyz(predicted_xc, predicted_yc, closest_armor_xyz[2]);
        //   Eigen::Vector3d fake_armor_xyz(predicted_xc, predicted_yc, predicted_zc);

        //   rpy = Barrel_Solve(fake_armor_xyz);

        // }else{
        //   rpy = Barrel_Solve(closest_armor_xyz);
        // }

        rpy = Barrel_Solve(closest_armor_xyz);

        Robot_ctrl_t->pitch = rpy[1] + 0.5 ;
        // Robot_ctrl_t->pitch = rpy[1] + 0.9;
        Robot_ctrl_t->yaw = rpy[2] - 2.3; // 粗暴的偏置调整方法
        // 计算装甲板两端点的坐标和对应的yaw角
        double armor_width = 0.13; // 假设这是装甲板宽度
        double half_width = armor_width / 2;
        double closest_yaw = std::atan2(closest_armor_xyz[1] - predicted_yc, closest_armor_xyz[0] - predicted_xc);
        geometry_msgs::msg::Point left_endpoint, right_endpoint;
        left_endpoint.x = closest_armor_xyz[0] - half_width * sin(closest_yaw);
        left_endpoint.y = closest_armor_xyz[1] + half_width * cos(closest_yaw);
        left_endpoint.z = closest_armor_xyz[2];

        right_endpoint.x = closest_armor_xyz[0] + half_width * sin(closest_yaw);
        right_endpoint.y = closest_armor_xyz[1] - half_width * cos(closest_yaw);
        right_endpoint.z = closest_armor_xyz[2];

        Eigen::Vector3d left_xyz(left_endpoint.x, left_endpoint.y, left_endpoint.z);
        Eigen::Vector3d right_xyz(right_endpoint.x, right_endpoint.y, right_endpoint.z);

        Eigen::Vector3d left_rpy = Barrel_Solve(left_xyz);
        Eigen::Vector3d right_rpy = Barrel_Solve(right_xyz);

        double left_yaw = left_rpy[2];
        double right_yaw = right_rpy[2];

        double max_yaw = std::max(left_yaw, right_yaw);
        double min_yaw = std::min(left_yaw, right_yaw);

        // 考虑优劣弧判断当前yaw角是否在两端点的yaw角范围内
        bool is_firing = false;

        if((max_yaw-min_yaw)<((180-max_yaw)+min_yaw+180)){
          is_firing = (now_yaw_ >= min_yaw && now_yaw_ <= max_yaw);
        }else{
          is_firing = (now_yaw_<=180 && now_yaw_>= max_yaw)||(now_yaw_<=min_yaw && now_yaw_>= -180);
        }

        std::cout<<"min yaw: "<<min_yaw<<" max_yaw: "<<max_yaw<<" now yaw: "<<now_yaw_<<std::endl;

        Robot_ctrl_t->fire_command = is_firing ? 1 : 0;

    } else {
        // 如果没有敌人，控制量等于当前量并不开火
        Robot_ctrl_t->target_lock = 50;
        Robot_ctrl_t->pitch = now_pitch_;
        Robot_ctrl_t->yaw = now_yaw_;
        Robot_ctrl_t->fire_command = 0;
    }

    ctrl_pub_->publish(*Robot_ctrl_t);

    // std::cout<<"CTRL: pitch: "<< Robot_ctrl_t->pitch << " yaw: "<< Robot_ctrl_t->yaw << " fire: "<< (int)Robot_ctrl_t->fire_command << std::endl;
}   

}  // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::ArmorTrackerNode)
